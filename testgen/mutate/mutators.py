"""Per-node mutation catalogue (D016). Mirrors mutmut's operator set plus
cosmic-ray's zero-iteration loop and exception replacer, each tagged with a
category so a profile can switch it on or off.

Every entry in ``sites(node)`` is ``(category, variant, mutate)`` where
``mutate`` edits a node **in place on a deep copy** and returns a one-line
description. Mutators never touch the original tree.
"""

import ast
import re
from collections.abc import Callable

Mutator = Callable[[ast.AST], str]
Site = tuple[str, str, Mutator]

_COMPARE = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
# Held-out probe (D015): arithmetic only. Never used in training curation.
_ARITH = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Div,
    ast.Mod: ast.Div,
    ast.Pow: ast.Mult,
}
_BITWISE = {
    ast.BitAnd: ast.BitOr,
    ast.BitOr: ast.BitAnd,
    ast.BitXor: ast.BitAnd,
    ast.LShift: ast.RShift,
    ast.RShift: ast.LShift,
}
_STR_METHODS = {
    "lower": "upper",
    "upper": "lower",
    "lstrip": "rstrip",
    "rstrip": "lstrip",
    "find": "rfind",
    "rfind": "find",
    "ljust": "rjust",
    "rjust": "ljust",
    "index": "rindex",
    "rindex": "index",
    "removeprefix": "removesuffix",
    "removesuffix": "removeprefix",
    "partition": "rpartition",
    "rpartition": "partition",
    "split": "rsplit",
    "rsplit": "split",
}
_NON_ESCAPE = re.compile(r"((?<!\\)[^\\]+)")


def _swap_op(table: dict) -> Mutator:
    def mutate(node: ast.AST) -> str:
        before = type(node.op).__name__  # type: ignore[attr-defined]
        node.op = table[type(node.op)]()  # type: ignore[attr-defined]
        return f"{before} -> {type(node.op).__name__}"  # type: ignore[attr-defined]

    return mutate


def _set(attr: str, value_factory: Callable[[], ast.AST], desc: str) -> Mutator:
    def mutate(node: ast.AST) -> str:
        setattr(node, attr, value_factory())
        return desc

    return mutate


def _compare(node: ast.Compare) -> list[Site]:
    if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARE:
        return []

    def mutate(n: ast.AST) -> str:
        before = type(n.ops[0]).__name__
        n.ops[0] = _COMPARE[type(n.ops[0])]()
        return f"{before} -> {type(n.ops[0]).__name__}"

    return [("compare", type(node.ops[0]).__name__, mutate)]


def _constant(node: ast.Constant, is_docstring: bool) -> list[Site]:
    v = node.value
    if isinstance(v, bool):
        return [("boolean", "flip", _set("value", lambda: not v, f"{v} -> {not v}"))]
    if isinstance(v, int | float | complex):
        bump = 1j if isinstance(v, complex) else 1
        return [("boundary", "plus1", _set("value", lambda: v + bump, f"{v!r} -> {v + bump!r}"))]
    if isinstance(v, str) and not is_docstring:
        return _string_constant(v)
    return []


def _string_constant(v: str) -> list[Site]:
    variants = [
        ("xx", "XX" + v + "XX"),
        ("lower", _NON_ESCAPE.sub(lambda m: m.group(1).lower(), v)),
        ("upper", _NON_ESCAPE.sub(lambda m: m.group(1).upper(), v)),
    ]
    # Skip no-ops (e.g. upper-casing an already upper string): trivially equivalent.
    return [
        ("string", name, _set("value", lambda nv=nv: nv, f"{v!r} -> {nv!r}"))
        for name, nv in variants
        if nv != v
    ]


def _boolop(node: ast.BoolOp) -> list[Site]:
    return [("boolean", type(node.op).__name__, _swap_op({ast.And: ast.Or, ast.Or: ast.And}))]


def _unary(node: ast.UnaryOp) -> list[Site]:
    # Removing the operator = replacing the node by its operand. Done by
    # turning the UnaryOp into a no-op: we cannot replace a node in place via
    # ast.walk, so we rewrite it as `+operand` (UAdd) which unparses to +x.
    # For `not x` and `-x`/`~x` that is a behaviour change; for `+x` it is a
    # no-op, so we do not enumerate UAdd.
    if isinstance(node.op, ast.UAdd):
        return []
    cat = "boolean" if isinstance(node.op, ast.Not) else "unary"
    name = type(node.op).__name__
    return [(cat, f"remove_{name}", _set("op", ast.UAdd, f"remove {name}"))]


def _return(node: ast.Return) -> list[Site]:
    if node.value is None:
        return []
    return [
        (
            "return",
            "none",
            _set("value", lambda: ast.Constant(None), "return <expr> -> return None"),
        )
    ]


def _binop(node: ast.BinOp | ast.AugAssign) -> list[Site]:
    t = type(node.op)
    if t in _ARITH:
        return [("arith", t.__name__, _swap_op(_ARITH))]
    if t in _BITWISE:
        return [("bitwise", t.__name__, _swap_op(_BITWISE))]
    return []


def _augassign(node: ast.AugAssign) -> list[Site]:
    # `a += b` -> `a = b`: modelled as replacing the op with a marker handled
    # at unparse time is awkward; instead rewrite value to drop the target:
    # we express it by turning the node into `target = value` via __class__ swap.
    def mutate(n: ast.AST) -> str:
        n.__class__ = ast.Assign
        n.targets = [n.target]  # type: ignore[attr-defined]
        del n.target, n.op  # type: ignore[attr-defined]
        n.type_comment = None  # type: ignore[attr-defined]
        return "augassign -> assign"

    return _binop(node) + [("assign", "drop_aug", mutate)]


def _assign(node: ast.Assign | ast.AnnAssign) -> list[Site]:
    if node.value is None:
        return []
    if isinstance(node.value, ast.Constant) and node.value.value is None:
        return [("assign", "none_to_empty", _set("value", lambda: ast.Constant(""), "None -> ''"))]
    return [("assign", "to_none", _set("value", lambda: ast.Constant(None), "<expr> -> None"))]


def _for(node: ast.For) -> list[Site]:
    return [
        (
            "loop",
            "zero_iter",
            _set("iter", lambda: ast.List(elts=[], ctx=ast.Load()), "for ... in []"),
        )
    ]


def _break_continue(node: ast.Break | ast.Continue) -> list[Site]:
    target = ast.Continue if isinstance(node, ast.Break) else ast.Break

    def mutate(n: ast.AST) -> str:
        n.__class__ = target
        return f"{type(node).__name__} -> {target.__name__}"

    return [("loop", type(node).__name__.lower(), mutate)]


def _call(node: ast.Call) -> list[Site]:
    sites: list[Site] = []
    if isinstance(node.func, ast.Attribute) and node.func.attr in _STR_METHODS:
        attr, new = node.func.attr, _STR_METHODS[node.func.attr]
        # mutmut: split<->rsplit only differ with a maxsplit, so only mutate then.
        if (
            attr not in ("split", "rsplit")
            or len(node.args) == 2
            or any(k.arg == "maxsplit" for k in node.keywords)
        ):
            sites.append(
                (
                    "string",
                    f"{attr}_to_{new}",
                    lambda n, new=new: (setattr(n.func, "attr", new), f".{attr} -> .{new}")[1],
                )
            )
    for i, arg in enumerate(node.args):
        if not (isinstance(arg, ast.Constant) and arg.value is None):
            sites.append(
                (
                    "call",
                    f"arg{i}_none",
                    lambda n, i=i: (n.args.__setitem__(i, ast.Constant(None)), f"arg {i} -> None")[
                        1
                    ],
                )
            )
    for i, kw in enumerate(node.keywords):
        if kw.arg is not None:
            sites.append(
                (
                    "call",
                    f"kw{i}_xx",
                    lambda n, i=i: (
                        setattr(n.keywords[i], "arg", n.keywords[i].arg + "XX"),
                        "kwarg name + XX",
                    )[1],
                )
            )
    return sites


def _lambda(node: ast.Lambda) -> list[Site]:
    if isinstance(node.body, ast.Constant) and node.body.value is None:
        return []
    return [
        ("lambda", "body_none", _set("body", lambda: ast.Constant(None), "lambda body -> None"))
    ]


def _except(node: ast.ExceptHandler) -> list[Site]:
    if node.type is None:
        return []
    # `except ():` is valid Python and catches nothing: the handler never fires.
    return [
        (
            "exception",
            "never",
            _set("type", lambda: ast.Tuple(elts=[], ctx=ast.Load()), "except X -> except ()"),
        )
    ]


def _funcdef(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Site]:
    return [
        (
            "decorator",
            f"remove{i}",
            lambda n, i=i: (n.decorator_list.pop(i), f"remove decorator {i}")[1],
        )
        for i in range(len(node.decorator_list))
    ]


def _match(node: ast.Match) -> list[Site]:
    if len(node.cases) < 2:
        return []
    return [
        ("match", f"drop_case{i}", lambda n, i=i: (n.cases.pop(i), f"drop case {i}")[1])
        for i in range(len(node.cases))
    ]


def _ifexp(node: ast.IfExp) -> list[Site]:
    # mutmut: force each arm by neutralising the condition: (c) and False / (c) or True.
    def force(op: type, lit: bool) -> Mutator:
        def mutate(n: ast.AST) -> str:
            n.test = ast.BoolOp(op=op(), values=[n.test, ast.Constant(lit)])
            return f"ternary condition {'and False' if not lit else 'or True'}"

        return mutate

    return [
        ("boolean", "force_else", force(ast.And, False)),
        ("boolean", "force_if", force(ast.Or, True)),
    ]


def sites(node: ast.AST, is_docstring: bool = False) -> list[Site]:
    """All mutations this node admits, in catalogue order."""
    match node:
        case ast.Compare():
            return _compare(node)
        case ast.Constant():
            return _constant(node, is_docstring)
        case ast.BoolOp():
            return _boolop(node)
        case ast.UnaryOp():
            return _unary(node)
        case ast.Return():
            return _return(node)
        case ast.AugAssign():
            return _augassign(node)
        case ast.BinOp():
            return _binop(node)
        case ast.Assign() | ast.AnnAssign():
            return _assign(node)
        case ast.For():
            return _for(node)
        case ast.Break() | ast.Continue():
            return _break_continue(node)
        case ast.Call():
            return _call(node)
        case ast.Lambda():
            return _lambda(node)
        case ast.ExceptHandler():
            return _except(node)
        case ast.FunctionDef() | ast.AsyncFunctionDef():
            return _funcdef(node)
        case ast.Match():
            return _match(node)
        case ast.IfExp():
            return _ifexp(node)
    return []
