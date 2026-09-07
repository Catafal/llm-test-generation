import pytest

from testgen.mutate import artifact
from testgen.mutate.equivalence import is_trivially_equivalent, split_equivalent
from testgen.mutate.operators import generate_mutants

DEAD_BRANCH = "def f(x):\n    if False:\n        return 5\n    return x\n"


def test_constant_in_dead_branch_is_trivially_equivalent():
    # CPython drops `if False:` bodies at compile time, so bumping 5 -> 6 changes nothing.
    ms = [m for m in generate_mutants(DEAD_BRANCH) if m.description == "5 -> 6"]
    assert ms and is_trivially_equivalent(DEAD_BRANCH, ms[0].source)


def test_live_mutant_is_not_equivalent():
    ms = [m for m in generate_mutants(DEAD_BRANCH) if m.category == "return" and m.line == 4]
    assert ms and not is_trivially_equivalent(DEAD_BRANCH, ms[0].source)


def test_split_reports_equivalent_ids():
    live, eq = split_equivalent(DEAD_BRANCH, generate_mutants(DEAD_BRANCH))
    assert any("boundary:3" in i for i in eq)
    assert all(m.id not in eq for m in live)


def test_freeze_write_read_roundtrip(tmp_path):
    frozen = artifact.freeze("f1", DEAD_BRANCH)
    p = tmp_path / "f1.json"
    artifact.write(frozen, p)
    mutants, eq = artifact.read(p)
    assert len(mutants) == len(frozen["mutants"]) and eq == set(frozen["trivially_equivalent"])
    assert frozen["live_count"] == len(mutants) - len(eq)


def test_read_refuses_stale_operators_version(tmp_path):
    frozen = artifact.freeze("f1", DEAD_BRANCH)
    frozen["operators_version"] = "0000-00-00.0"
    p = tmp_path / "f1.json"
    artifact.write(frozen, p)
    with pytest.raises(ValueError, match="operators"):
        artifact.read(p)
