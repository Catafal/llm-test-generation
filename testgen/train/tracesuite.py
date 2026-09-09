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


def _calls_target(expr: ast.AST, target: str) -> bool:
    return any(isinstance(n, ast.Call) and target in ast.unparse(n.func) for n in ast.walk(expr))


class _Substitute(ast.NodeTransformer):
    """Replace local names with the expressions they were assigned from."""

    def __init__(self, bindings: dict[str, ast.expr]) -> None:
        self.bindings = bindings

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.bindings:
            return self.bindings[node.id]
        return node


def _bindings(module: ast.Module, fn: ast.FunctionDef, before: int) -> dict[str, ast.expr]:
    """name -> assigned expression, for module-level constants and locals set before ``before``."""
    out: dict[str, ast.expr] = {}
    stmts = [st for st in module.body if isinstance(st, ast.Assign)]
    stmts += [st for st in ast.walk(fn) if isinstance(st, ast.Assign) and st.lineno < before]
    for stmt in sorted(stmts, key=lambda st: st.lineno):
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            out[stmt.targets[0].id] = stmt.value
    return out


def _call_for(node: ast.Assert, fn: ast.FunctionDef, module: ast.Module, target: str) -> str | None:
    """The self-contained expression to trace behind the assert's left side, or None.

    Locals and module constants referenced by the left side are replaced by
    the expressions they were assigned from (to a fixed point), so
    ``row = {...}; r = f(row); assert r['k'] == 1`` traces ``f({...})['k']``.
    Anything still unresolved (helpers, fixtures) fails in the probe and is
    reported as untraceable.
    """
    bindings = _bindings(module, fn, node.lineno)
    expr = ast.parse(ast.unparse(node.test.left), mode="eval").body
    for _ in range(4):  # bounded fixed point over chained assignments
        new = _Substitute(bindings).visit(ast.parse(ast.unparse(expr), mode="eval").body)
        if ast.unparse(new) == ast.unparse(expr):
            break
        expr = new
    return ast.unparse(expr) if _calls_target(expr, target) else None


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
            call = _call_for(node, fn, tree, target)
            if call is None:
                stats.skipped_untraceable += 1
                continue
            if len(out) >= max_asserts:
                stats.skipped_cap += 1
                continue
            tr = trace_call(source, call)
            if tr.error and not tr.steps:
                stats.skipped_untraceable += 1
                continue
            block = render(source, tr, max_lines=max_lines)
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
