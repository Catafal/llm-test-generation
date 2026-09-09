"""Execution traces as training-target material (D028 step 2).

Given a function's source and one call expression, run the call under
``sys.settrace`` inside the sandbox and record, per executed line, the line
text and the local variables that changed. The rendering is a compact
scratchpad the model can learn to emit before an assert: the *derivation* of
the expected value, grounded in a real execution (Scratchpads 2021, NExT 2024,
Self-Execution Simulation 2026).

    trace_call(source, "f(3, 4)") -> Trace(steps=[...], result=repr, error=None)
    render(trace, max_steps=12)   -> "# f(3, 4)\\n#  L3 total = 0 ... \\n#  -> 12"

Long traces are abbreviated (first and last steps kept) so a target stays
within budget; loops are summarised by iteration count. Only pure functions
reach this code (harvest purity filter), so tracing is safe and deterministic.
"""

import json
from dataclasses import dataclass, field

from testgen.harness.runner import run_suite

TRACE_FILE = "trace.json"
MAX_REPR = 60

# Runs inside the sandbox as a pytest "suite": traces the call, writes JSON.
_PROBE = """import json, sys, reprlib
import solution
# every name incl. private ones, so suite expressions evaluate as written
globals().update({{k: v for k, v in vars(solution).items() if not k.startswith("__")}})

_r = reprlib.Repr(); _r.maxstring = {max_repr}; _r.maxother = {max_repr}
_steps = []
_prev = {{}}

def _tracer(frame, event, arg):
    if frame.f_code.co_filename != solution.__file__:
        return None
    if event == "line":
        global _prev
        cur = {{k: _r.repr(v) for k, v in frame.f_locals.items() if not k.startswith("__")}}
        changed = {{k: v for k, v in cur.items() if _prev.get(k) != v}}
        _steps.append({{"line": frame.f_lineno, "vars": changed}})
        _prev = cur
    elif event == "return":
        _steps.append({{"line": frame.f_lineno, "ret": _r.repr(arg)}})
    return _tracer

def test_trace():
    out = {{"steps": [], "result": None, "error": None}}
    sys.settrace(_tracer)
    try:
        value = {call}
        out["result"] = _r.repr(value)
    except Exception as e:  # noqa: BLE001
        out["error"] = type(e).__name__ + ": " + str(e)[:80]
    finally:
        sys.settrace(None)
    out["steps"] = _steps[:{max_steps}]
    out["n_steps"] = len(_steps)
    with open("{trace_file}", "w") as f:
        json.dump(out, f)
"""


@dataclass
class Trace:
    call: str
    steps: list[dict] = field(default_factory=list)
    n_steps: int = 0
    result: str | None = None
    error: str | None = None


def trace_call(source: str, call: str, max_steps: int = 400) -> Trace:
    """Execute ``call`` against ``source`` in the sandbox.

    ``call`` is any expression as it appears in a suite, e.g. ``"f(3, 4)"`` or
    ``"len(f([1, 2]))"``; the probe star-imports ``solution`` so bare names
    resolve, and only lines inside ``solution`` are traced.
    """
    probe = _PROBE.format(call=call, max_repr=MAX_REPR, max_steps=max_steps, trace_file=TRACE_FILE)
    run = run_suite(probe, source, collect=(TRACE_FILE,))
    raw = run.artifacts.get(TRACE_FILE)
    if not raw:
        return Trace(call, error=f"probe {run.status}: {run.stderr_tail[-120:]}")
    data = json.loads(raw)
    return Trace(call, data["steps"], data["n_steps"], data["result"], data["error"])


def render(source: str, trace: Trace, max_lines: int = 10) -> str:
    """Compact comment block: the call, the changed variables per line, the result.

    Kept short on purpose: ``max_lines`` derivation lines, first and last
    kept with an elision marker, so a target with several asserts fits.
    """
    src_lines = source.splitlines()
    rows = []
    # A "line" event fires *before* that line runs, so the locals that changed
    # at event i are the effect of the line reported at event i-1. Shift them.
    lines = [st for st in trace.steps if "ret" not in st]
    for i, st in enumerate(lines):
        code = src_lines[st["line"] - 1].strip() if 0 < st["line"] <= len(src_lines) else "?"
        effect = lines[i + 1]["vars"] if i + 1 < len(lines) else {}
        vars_ = ", ".join(f"{k}={v}" for k, v in effect.items())
        rows.append(f"#   L{st['line']} {code[:50]}" + (f"  -> {vars_}" if vars_ else ""))
    if len(rows) > max_lines:
        keep = max_lines // 2
        rows = rows[:keep] + [f"#   ... {len(rows) - 2 * keep} more lines ..."] + rows[-keep:]
    tail = f"#   raises {trace.error}" if trace.error else f"#   returns {trace.result}"
    return "\n".join([f"# {trace.call}", *rows, tail])
