import ast

from testgen.mutate.operators import (
    ALL_CATEGORIES,
    PROBE_CATEGORY,
    TRAINING_CATEGORIES,
    generate_mutants,
    sample_mutants,
)

CLAMP = '''def clamp(x, lo, hi):
    """Limit x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
'''


def by_category(source, cat):
    return [m for m in generate_mutants(source) if m.category == cat]


def test_compare_flips_each_operator_once():
    ms = by_category(CLAMP, "compare")
    assert [m.description for m in ms] == ["Lt -> LtE", "Gt -> GtE"]
    assert "if x <= lo:" in ms[0].source and "if x > hi:" in ms[0].source


def test_return_mutants_one_per_return():
    ms = by_category(CLAMP, "return")
    assert len(ms) == 3
    assert all("return None" in m.source for m in ms)


def test_boundary_bumps_int_constants_but_not_bools_or_docstrings():
    src = "def f(a):\n    if a is True:\n        return 0\n    return len('abc') + 2\n"
    ms = by_category(src, "boundary")
    assert sorted(m.description for m in ms) == ["0 -> 1", "2 -> 3"]


def test_boolean_swaps_and_strips_not():
    src = "def f(a, b):\n    return a and b\n"
    ms = by_category(src, "boolean")
    assert ms[0].description == "And -> Or" and "a or b" in ms[0].source


def test_arith_probe_covers_binop_and_augassign():
    src = "def f(a, b):\n    a += b\n    return a * 2 - 1\n"
    ms = by_category(src, PROBE_CATEGORY)
    # source order: line 2 augassign, then line 3 outer Sub before inner Mult (same column)
    assert [m.description for m in ms] == ["Add -> Sub", "Sub -> Add", "Mult -> FloorDiv"]


def test_category_filter_excludes_probe_for_training():
    ms = generate_mutants(CLAMP, TRAINING_CATEGORIES)
    assert PROBE_CATEGORY not in {m.category for m in ms}
    assert set(ALL_CATEGORIES) >= {m.category for m in generate_mutants(CLAMP)}


def test_every_mutant_parses_and_differs_from_source():
    for m in generate_mutants(CLAMP):
        ast.parse(m.source)
        assert m.source != CLAMP


def test_ids_are_stable_and_unique():
    a = [m.id for m in generate_mutants(CLAMP)]
    b = [m.id for m in generate_mutants(CLAMP)]
    assert a == b and len(set(a)) == len(a)


def test_sample_is_seeded_and_keeps_source_order():
    ms = generate_mutants(CLAMP)
    s1, s2 = sample_mutants(ms, 3, seed=7), sample_mutants(ms, 3, seed=7)
    assert s1 == s2 and len(s1) == 3
    assert [ms.index(m) for m in s1] == sorted(ms.index(m) for m in s1)
