"""AST mutation operators: one mutant per site, deterministic ids (D005, D015).

Five categories. Four are used in training-data curation; ``arith`` is the
held-out probe and must never feed a curation signal:

    compare   <  <->  <=,  >  <->  >=,  ==  <->  !=
    boundary  integer constant  n  ->  n + 1
    boolean   and <-> or,  ``not x``  ->  x
    return    ``return expr``  ->  ``return None``
    arith     +  <->  -,  *  <->  //,  /  ->  *   (also augmented assigns)   [PROBE]

Why one mutant per site with no cap: a cap quantises per-function score and
starves rare categories (Opus review, 2026-09-07). Callers that need fewer
(training curation) use ``sample_mutants`` with their own seed.

Site enumeration uses ``ast.walk`` on the original tree and again on a deep
copy; the two traversals visit nodes in the same order, so index ``k`` names
the same node in both. Mutant ids are ``<category>:<line>:<col>:<variant>``
and are stable as long as the source is unchanged.
"""

import ast
import copy
import random
from collections.abc import Callable
from dataclasses import dataclass

TRAINING_CATEGORIES = ("compare", "boundary", "boolean", "return")
PROBE_CATEGORY = "arith"
ALL_CATEGORIES = (*TRAINING_CATEGORIES, PROBE_CATEGORY)

_COMPARE_FLIP = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}
_ARITH_SWAP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
    ast.Div: ast.Mult,
}


@dataclass(frozen=True)
class Mutant:
    id: str
    category: str
    line: int
    description: str
    source: str


# A site is (node index, category, variant name, mutator). The mutator edits
# the node in place on a *copy* of the tree and returns a human description.
_Site = tuple[int, str, str, Callable[[ast.AST], str]]


def _flip_compare(node: ast.Compare) -> str:
    before = type(node.ops[0]).__name__
    node.ops[0] = _COMPARE_FLIP[type(node.ops[0])]()
    return f"{before} -> {type(node.ops[0]).__name__}"


def _bump_constant(node: ast.Constant) -> str:
    before = node.value
    node.value = before + 1
    return f"{before} -> {node.value}"


def _swap_boolop(node: ast.BoolOp) -> str:
    before = type(node.op).__name__
    node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    return f"{before} -> {type(node.op).__name__}"


def _return_none(node: ast.Return) -> str:
    node.value = ast.Constant(value=None)
    return "return <expr> -> return None"


def _swap_arith(node: ast.BinOp | ast.AugAssign) -> str:
    before = type(node.op).__name__
    node.op = _ARITH_SWAP[type(node.op)]()
    return f"{before} -> {type(node.op).__name__}"


def _sites_for(index: int, node: ast.AST) -> list[_Site]:
    """Every mutation this node admits. Kept flat so adding an operator is one branch."""
    sites: list[_Site] = []
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE_FLIP:
        sites.append((index, "compare", type(node.ops[0]).__name__, _flip_compare))
    if isinstance(node, ast.Constant) and type(node.value) is int:  # excludes bool
        sites.append((index, "boundary", "plus1", _bump_constant))
    if isinstance(node, ast.BoolOp):
        sites.append((index, "boolean", type(node.op).__name__, _swap_boolop))
    if isinstance(node, ast.Return) and node.value is not None:
        sites.append((index, "return", "none", _return_none))
    if isinstance(node, ast.BinOp | ast.AugAssign) and type(node.op) in _ARITH_SWAP:
        sites.append((index, "arith", type(node.op).__name__, _swap_arith))
    return sites


def _apply(tree: ast.Module, site: _Site) -> Mutant:
    index, category, variant, mutate = site
    mutated = copy.deepcopy(tree)
    node = list(ast.walk(mutated))[index]
    description = mutate(node)
    line = getattr(node, "lineno", 0)
    col = getattr(node, "col_offset", 0)
    return Mutant(
        id=f"{category}:{line}:{col}:{variant}",
        category=category,
        line=line,
        description=description,
        source=ast.unparse(ast.fix_missing_locations(mutated)) + "\n",
    )


def generate_mutants(source: str, categories: tuple[str, ...] = ALL_CATEGORIES) -> list[Mutant]:
    """All mutants of ``source`` in the given categories, one per site, in source order."""
    tree = ast.parse(source)
    sites = [
        s
        for i, node in enumerate(ast.walk(tree))
        for s in _sites_for(i, node)
        if s[1] in categories
    ]
    # ast.walk is breadth-first; sort so ids and order follow the source text.
    return sorted((_apply(tree, s) for s in sites), key=lambda m: (m.line, int(m.id.split(":")[2])))


def sample_mutants(mutants: list[Mutant], cap: int, seed: int) -> list[Mutant]:
    """Seeded subsample for training curation only (D015). Eval never caps."""
    if len(mutants) <= cap:
        return list(mutants)
    rng = random.Random(seed)
    return sorted(rng.sample(mutants, cap), key=mutants.index)
