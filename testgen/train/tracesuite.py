"""Turn a valid suite into an execution-explanation target (D028 step 2).

For each ``assert <expr> == <literal>`` whose left side is a call to the
function under test (directly, or through a local assigned from such a call
earlier in the same test), the call is traced in the sandbox and rendered
as a scratchpad (``trace.render``). Two target formats:

  inline    the rendered block is inserted as comments immediately above the
            assert (the model emits derivations inside the suite at inference)
  prefix    all rendered blocks go in one explanation section before the
            fenced suite; at inference the section is stripped before scoring

Caps keep targets inside the training budget: at most ``max_asserts`` traced
asserts per suite and ``max_lines`` derivation lines per assert. Untraceable
asserts (non-call left sides, calls that raise) are left as written.
"""

import ast
from dataclasses import dataclass, field

from testgen.train.trace import render, trace_call

PREFIX_OPEN, PREFIX_CLOSE = "<derivation>", "</derivation>"


@dataclass
class TraceStats:
    sites: int = 0
    traced: int = 0
    skipped_untraceable: int = 0
    skipped_cap: int = 0
    trace_lines: list[int] = field(default_factory=list)


def _call_for(node: ast.Assert, fn_body: list[ast.stmt], target: str) -> str | None:
    """The call expression behind the assert's left side, or None."""
    left = node.test.left
    if isinstance(left, ast.Call) and target in ast.unparse(left.func):
        return ast.unparse(left)
    if isinstance(left, ast.Name):  # result = f(...); assert result == ...
        for stmt in fn_body:
            if stmt is node:
                break
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == left.id
                and isinstance(stmt.value, ast.Call)
                and target in ast.unparse(stmt.value.func)
            ):
                return ast.unparse(stmt.value)
    return None


def _is_site(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)):
        return False
    t = node.test
    if len(t.ops) != 1 or not isinstance(t.ops[0], ast.Eq):
        return False
    try:
        ast.literal_eval(t.comparators[0])
        return True
    except (ValueError, TypeError, SyntaxError):
        return False


def trace_sites(
    suite: str, source: str, target: str, max_asserts: int = 4, max_lines: int = 6
) -> tuple[list[tuple[int, str]], TraceStats]:
    """(assert line number, rendered block) for traceable literal asserts, in order."""
    tree = ast.parse(suite)
    stats, out = TraceStats(), []
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("test")):
            continue
        for node in ast.walk(fn):
            if not _is_site(node):
                continue
            stats.sites += 1
            call = _call_for(node, fn.body, target)
            if call is None:
                stats.skipped_untraceable += 1
                continue
            if len(out) >= max_asserts:
                stats.skipped_cap += 1
                continue
            # The suite imports `target` from solution; the probe imports solution.
            tr = trace_call(source, "solution." + call.replace(target + "(", target + "(", 1))
            if tr.error and not tr.steps:
                stats.skipped_untraceable += 1
                continue
            block = render(source, tr, max_lines=max_lines).replace("solution.", "", 1)
            stats.traced += 1
            stats.trace_lines.append(block.count("\n") + 1)
            out.append((node.lineno, block))
    return out, stats


def inline_target(suite: str, sites: list[tuple[int, str]]) -> str:
    """Insert each block as comment lines above its assert, preserving indentation."""
    lines = suite.splitlines()
    for lineno, block in sorted(sites, reverse=True):
        indent = lines[lineno - 1][: len(lines[lineno - 1]) - len(lines[lineno - 1].lstrip())]
        lines[lineno - 1 : lineno - 1] = [indent + ln for ln in block.splitlines()]
    return "\n".join(lines) + "\n"


def prefix_target(suite: str, sites: list[tuple[int, str]]) -> str:
    """Explanation section, then the fenced suite (stripped at inference)."""
    body = "\n\n".join(block for _, block in sites)
    return f"{PREFIX_OPEN}\n{body}\n{PREFIX_CLOSE}\n```python\n{suite}```"
