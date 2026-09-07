"""Execution filter (T3): oracle-fill then gate; best candidate per function."""

from testgen.mutate.equivalence import split_equivalent
from testgen.mutate.operators import generate_mutants
from testgen.mutate.profiles import training_categories
from testgen.train.filter import best_per_function, score_candidate, to_chat

REF = '''
def clamp(x, lo, hi):
    """Clamp x into [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
'''

WRONG_VALUES = """```python
from solution import clamp

def test_low():
    assert clamp(-5, 0, 10) == 1

def test_high():
    assert clamp(50, 0, 10) == 11

def test_mid():
    assert clamp(5, 0, 10) == 5
```"""

BAD_INPUT = """```python
from solution import clamp

def test_crash():
    assert clamp(1) == 1
```"""


def _live():
    live, _ = split_equivalent(REF, generate_mutants(REF, training_categories()))
    return live


def test_oracle_rescues_wrong_values_and_counts_kills():
    rec = score_candidate(WRONG_VALUES, REF, _live())
    assert rec["parsed"] and not rec["valid_before"] and rec["valid"]
    assert rec["oracle"]["replaced"] == 2 and rec["kills"] > 0
    assert "clamp(-5, 0, 10) == 0" in rec["suite"]


def test_bad_inputs_are_not_rescued():
    rec = score_candidate(BAD_INPUT, REF, _live())
    assert rec["parsed"] and not rec["valid"] and rec["kills"] == 0


def test_best_prefers_score_then_fewer_tests():
    a = {"valid": True, "kills": 3, "mutation_score": 0.5, "n_tests": 8}
    b = {"valid": True, "kills": 3, "mutation_score": 0.5, "n_tests": 4}
    c = {"valid": True, "kills": 0, "mutation_score": 0.0, "n_tests": 1}
    assert best_per_function([a, b, c]) is b
    assert best_per_function([c]) is None


def test_chat_target_round_trips_through_extract_suite():
    from testgen.generate.prompts import extract_suite

    suite = "from solution import clamp\n\n\ndef test_x():\n    assert clamp(1, 0, 2) == 1\n"
    rec = to_chat(REF, suite)
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    got, flags = extract_suite(rec["messages"][-1]["content"])
    assert got == suite and not flags["truncated"]
