"""Extract self-contained, pure, top-level functions from a Python module (D006, D017).

A candidate must: be a module-level ``def`` with a docstring; reference only
its own parameters/locals, builtins, and names imported from an allow-listed
set of pure stdlib modules; and use no I/O, nondeterminism, globals, or
async. The extracted ``source`` includes the stdlib imports it needs, so it
runs as ``solution.py`` on its own.

The allow-list is deliberately a whitelist: anything not listed is impure
until someone argues otherwise in the decision log.
"""

import ast
import builtins
from dataclasses import dataclass

PURE_STDLIB = frozenset(
    {
        "math",
        "re",
        "itertools",
        "functools",
        "collections",
        "string",
        "operator",
        "typing",
        "dataclasses",
        "enum",
        "fractions",
        "decimal",
        "statistics",
        "heapq",
        "bisect",
        "json",
        "textwrap",
        "unicodedata",
        "copy",
        "numbers",
        "abc",
        "array",
        "cmath",
        "difflib",
        "html",
        "base64",
        "binascii",
        "struct",
        "zlib",
        "hashlib",
    }
)
# Builtins that make a function non-pure or non-deterministic.
FORBIDDEN_BUILTINS = frozenset(
    {
        "open",
        "input",
        "print",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "hash",
        "id",
        "breakpoint",
        "exit",
        "quit",
    }
)
_BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str  # imports + function, runnable standalone
    docstring: str
    lineno: int
    end_lineno: int


def _module_imports(tree: ast.Module) -> dict[str, tuple[str, ast.stmt]]:
    """Map each bound name to (top-level module, import statement) for module-level imports."""
    bound: dict[str, tuple[str, ast.stmt]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[(alias.asname or alias.name).split(".")[0]] = (alias.name.split(".")[0], node)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                bound[alias.asname or alias.name] = (node.module.split(".")[0], node)
    return bound


def _bound_inside(fn: ast.FunctionDef) -> set[str]:
    """Names the function binds itself: params, assignments, loops, comprehensions, nested defs."""
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            # parameters of this function and of every nested function/lambda
            a = node.args
            names |= {p.arg for p in a.args + a.kwonlyargs + a.posonlyargs}
            names |= {p.arg for p in (a.vararg, a.kwarg) if p}
            if not isinstance(node, ast.Lambda):
                names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def _reject_reason(fn: ast.FunctionDef, imports: dict[str, tuple[str, ast.stmt]]) -> str | None:
    """None if the function is acceptable, else a short reason for the datasheet."""
    if not ast.get_docstring(fn):
        return "no docstring"
    local = _bound_inside(fn)
    for node in ast.walk(fn):
        if isinstance(node, ast.Global | ast.Nonlocal):
            return "global/nonlocal"
        if isinstance(node, ast.Await | ast.AsyncFor | ast.AsyncWith | ast.AsyncFunctionDef):
            return "async"
        if isinstance(node, ast.Import | ast.ImportFrom):
            mod = (node.module if isinstance(node, ast.ImportFrom) else node.names[0].name) or ""
            if mod.split(".")[0] not in PURE_STDLIB:
                return f"impure import inside function: {mod}"
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            n = node.id
            if n in FORBIDDEN_BUILTINS:
                return f"forbidden builtin: {n}"
            if n in local or n in _BUILTIN_NAMES:
                continue
            if n in imports:
                if imports[n][0] not in PURE_STDLIB:
                    return f"impure import: {imports[n][0]}"
                continue
            return f"free name: {n}"
    return None


def extract_candidates(module_source: str) -> tuple[list[Candidate], dict[str, str]]:
    """Return (accepted candidates, {function name: reject reason}) for one module."""
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return [], {"<module>": "syntax error"}
    imports = _module_imports(tree)
    accepted, rejected = [], {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        reason = _reject_reason(node, imports)
        if reason:
            rejected[node.name] = reason
            continue
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & set(imports)
        import_stmts = {ast.unparse(imports[n][1]) for n in used}
        header = "\n".join(sorted(import_stmts))
        body = ast.get_source_segment(module_source, node) or ast.unparse(node)
        source = (header + "\n\n\n" if header else "") + body + "\n"
        accepted.append(
            Candidate(
                node.name,
                source,
                ast.get_docstring(node) or "",
                node.lineno,
                node.end_lineno or node.lineno,
            )
        )
    return accepted, rejected
