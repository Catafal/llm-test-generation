"""Frozen mutant sets (D015): the eval mutant set is generated once, written to
disk with a version stamp, and every arm reads the same file. Regenerating on
the fly would let an operator edit silently change the eval mid-experiment.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from testgen.mutate.equivalence import split_equivalent
from testgen.mutate.operators import ALL_CATEGORIES, Mutant, generate_mutants

OPERATORS_VERSION = "2026-09-07.1"  # bump whenever operators.py changes behaviour


def freeze(function_id: str, source: str, categories: tuple[str, ...] = ALL_CATEGORIES) -> dict:
    mutants = generate_mutants(source, categories)
    live, equivalent = split_equivalent(source, mutants)
    return {
        "function_id": function_id,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "operators_version": OPERATORS_VERSION,
        "categories": list(categories),
        "mutants": [asdict(m) for m in mutants],
        "trivially_equivalent": sorted(equivalent),
        "live_count": len(live),
    }


def write(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=1))


def read(path: Path) -> tuple[list[Mutant], set[str]]:
    """Load a frozen set; raise if the operators version no longer matches."""
    data = json.loads(path.read_text())
    if data["operators_version"] != OPERATORS_VERSION:
        raise ValueError(
            f"{path}: frozen with operators {data['operators_version']}, "
            f"current is {OPERATORS_VERSION}; regenerate deliberately or pin the version"
        )
    return [Mutant(**m) for m in data["mutants"]], set(data["trivially_equivalent"])
