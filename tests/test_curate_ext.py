"""D032 stage-1 filters: one pure function, clean test imports, enough mutants."""

from testgen.train.curate_ext import stage_a_row, stage_b_row

SOLUTION = '''def letter_grade(score):
    """Map a 0-100 score to a letter grade."""
    if score < 0 or score > 100:
        raise ValueError("out of range")
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"
'''
TEST = """import pytest
from solution import letter_grade

def test_a():
    assert letter_grade(95) == "A"

def test_boundaries():
    assert letter_grade(90) == "A"
    assert letter_grade(89) == "B"
    assert letter_grade(60) == "D"

def test_fail():
    assert letter_grade(0) == "F"
    with pytest.raises(ValueError):
        letter_grade(101)
"""


def _row(solution=SOLUTION, test=TEST, subset="Docs", qid="Docs_1_I"):
    return {
        "solution": solution,
        "test": test,
        "subset": subset,
        "question_id": qid,
        "gpt_difficulty": "easy",
        "gpt_pass_percentage": 1.0,
    }


def test_stage_a_accepts_single_pure_function_and_groups_variants():
    rec = stage_a_row(_row())
    assert rec is not None and rec["function"] == "letter_grade" and rec["n_tests"] == 3
    assert rec["family"] == "Docs/Docs_1" and rec["live_mutants"] >= 8
    assert stage_a_row(_row(qid="Docs_1_II"))["family"] == rec["family"]


def test_stage_a_rejects_helpers_and_foreign_imports():
    assert stage_a_row(_row(solution=SOLUTION + "\ndef helper():\n    return 1\n")) is None
    assert stage_a_row(_row(test="import numpy\n" + TEST)) is None
    assert stage_a_row(_row(solution="def f(:\n")) is None


def test_stage_b_requires_passing_and_killing_suite():
    rec = stage_a_row(_row())
    kept = stage_b_row(rec)
    assert kept is not None and kept["kills"] >= 1 and 0 < kept["mutation_score"] <= 1
    bad = stage_a_row(_row(test=TEST.replace('== "F"', '== "A"', 1)))
    assert stage_b_row(bad) is None
