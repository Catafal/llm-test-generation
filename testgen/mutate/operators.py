"""Mutant generation: one mutant per (site, variant), deterministic ids (D015, D016).

Site enumeration uses ``ast.walk`` on the original tree and again on a deep
copy; both visit nodes in the same order, so index ``k`` names the same node
in both. Mutant ids are ``<category>:<line>:<col>:<variant>`` and are stable
while the source is unchanged. Results are sorted by source position.

Docstrings (first statement of a module/function/class) are never mutated.
"""

import ast
import copy
import random
from dataclasses import dataclass

from testgen.mutate import mutators
from testgen.mutate.profiles import ALL_CATEGORIES, PROBE_CATEGORY, active_categories

__all__ = ["ALL_CATEGORIES", "PROBE_CATEGORY", "Mutant", "generate_mutants", "sample_mutants"]


@dataclass(frozen=True)
class Mutant:
    id: str
    category: str
    line: int
    description: str
    source: str


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    ids.add(id(first.value))
    return ids


def _apply(tree: ast.Module, index: int, site: mutators.Site) -> Mutant:
    category, variant, mutate = site
    mutated = copy.deepcopy(tree)
    node = list(ast.walk(mutated))[index]
    line, col = getattr(node, "lineno", 0), getattr(node, "col_offset", 0)
    description = mutate(node)
    return Mutant(
        id=f"{category}:{line}:{col}:{variant}",
        category=category,
        line=line,
        description=description,
        source=ast.unparse(ast.fix_missing_locations(mutated)) + "\n",
    )


def generate_mutants(
    source: str, categories: tuple[str, ...] = active_categories()
) -> list[Mutant]:
    """All mutants of ``source`` in the given categories, sorted by source position."""
    tree = ast.parse(source)
    docstrings = _docstring_ids(tree)
    found = []
    for index, node in enumerate(ast.walk(tree)):
        for site in mutators.sites(node, is_docstring=id(node) in docstrings):
            if site[0] in categories:
                found.append(_apply(tree, index, site))
    return sorted(found, key=lambda m: (m.line, int(m.id.split(":")[2])))


def sample_mutants(mutants: list[Mutant], cap: int, seed: int) -> list[Mutant]:
    """Seeded subsample for training curation only (D015). Eval never caps."""
    if len(mutants) <= cap:
        return list(mutants)
    rng = random.Random(seed)
    return sorted(rng.sample(mutants, cap), key=mutants.index)
