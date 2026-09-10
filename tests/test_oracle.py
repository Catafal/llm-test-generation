"""Oracle fill (D023): wrong literals are rewritten from execution, nothing else moves."""

from testgen.harness.runner import run_suite
from testgen.train.oracle import assertion_styles, fill, sites

REF = '''
def area(w, h):
    """Rectangle area."""
    return w * h
'''

SUITE = """import pytest
from solution import area


def test_wrong_value():
    assert area(3, 4) == 13


def test_right_value():
    assert area(2, 2) == 4


def test_local_name():
    result = area(5, 5)
    assert result == 24


def test_not_touched():
    assert area(1, 1) > 0
    assert area(2, 3) == area(3, 2)
    assert area(1.5, 2) == pytest.approx(3.0)
"""


def test_sites_are_literal_equalities_only():
    assert len(sites(SUITE)) == 3


def test_fill_rewrites_only_wrong_literals():
    filled, stats, _ = fill(SUITE, REF)
    assert stats.sites == 3 and stats.recorded == 3
    assert stats.replaced == 2 and stats.already_correct == 1
    assert "area(3, 4) == 12" in filled
    assert "result == 25" in filled
    assert "area(2, 2) == 4" in filled
    assert "area(2, 3) == area(3, 2)" in filled  # non-literal RHS untouched
    # The filled suite is now valid on the reference.
    run = run_suite(filled, REF)
    assert run.status == "ok" and not run.any_failed and len(run.tests) == 4


def test_fill_skips_long_values():
    suite = "from solution import area\n\ndef test_big():\n    assert area(10**45, 3) == 1\n"
    filled, stats, _ = fill(suite, REF)
    assert stats.too_long == 1 and stats.replaced == 0
    assert "== 1\n" in filled


def test_fill_without_cap_rewrites_long_values():
    """D030: evaluation passes max_repr=None; the harness owns the value."""
    suite = "from solution import area\n\ndef test_big():\n    assert area(10**45, 3) == 1\n"
    filled, stats, _ = fill(suite, REF, max_repr=None)
    assert stats.replaced == 1 and stats.too_long == 0
    assert f"== {3 * 10**45}\n" in filled


def test_fill_survives_crashing_expression():
    suite = (
        "from solution import area\n\n"
        "def test_crash():\n    assert area(1) == 1\n\n"
        "def test_ok():\n    assert area(2, 5) == 9\n"
    )
    filled, stats, _ = fill(suite, REF)
    assert stats.recorded == 1 and stats.replaced == 1
    assert "area(2, 5) == 10" in filled


def test_assertion_styles():
    s = assertion_styles(SUITE)
    assert s == {
        "literal_eq": 3,
        "approx": 1,
        "relational": 1,
        "membership": 0,
        "expr_eq": 1,
        "other": 0,
    }
