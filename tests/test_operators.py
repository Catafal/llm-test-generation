import ast

import pytest

from testgen.mutate.operators import PROBE_CATEGORY, generate_mutants, sample_mutants
from testgen.mutate.profiles import (
    ALL_CATEGORIES,
    EVAL_PROFILE,
    active_categories,
    training_categories,
)

CLAMP = '''def clamp(x, lo, hi):
    """Limit x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
'''


def by_category(source, cat, categories=ALL_CATEGORIES):
    return [m for m in generate_mutants(source, categories) if m.category == cat]


def descs(source, cat, categories=ALL_CATEGORIES):
    return [m.description for m in by_category(source, cat, categories)]


# ---- one test per category: the catalogue is the spec ----------------------


def test_compare_flips_incl_membership_and_identity():
    src = "def f(a, b):\n    return a < b or a in b or a is b\n"
    assert descs(src, "compare") == ["Lt -> LtE", "In -> NotIn", "Is -> IsNot"]
    assert "if x <= lo:" in by_category(CLAMP, "compare")[0].source


def test_boundary_bumps_numbers_not_bools_not_docstrings():
    src = "def f(a):\n    if a is True:\n        return 0\n    return len('abc') + 2.5\n"
    assert sorted(descs(src, "boundary")) == ["0 -> 1", "2.5 -> 3.5"]
    assert descs(CLAMP, "boundary") == []  # docstring untouched, no numbers


def test_boolean_covers_andor_not_literal_and_ternary():
    src = "def f(a, b):\n    c = True\n    return (a and not b) if c else b\n"
    assert set(descs(src, "boolean")) == {
        "True -> False",
        "And -> Or",
        "remove Not",
        "ternary condition and False",
        "ternary condition or True",
    }


def test_return_one_per_return():
    ms = by_category(CLAMP, "return")
    assert len(ms) == 3 and all("return None" in m.source for m in ms)


def test_string_constants_and_method_swaps_skip_noops():
    src = "def f(s):\n    return s.strip().lower() + 'ABC' + s.split(',', 1)[0]\n"
    d = descs(src, "string")
    assert ".lower -> .upper" in d and ".split -> .rsplit" in d
    assert "'ABC' -> 'XXABCXX'" in d and "'ABC' -> 'abc'" in d
    assert "'ABC' -> 'ABC'" not in d  # upper-casing an upper string is a no-op: skipped
    assert ".strip" not in " ".join(d)  # strip has no swap partner


def test_split_without_maxsplit_not_mutated():
    assert ".split -> .rsplit" not in descs("def f(s):\n    return s.split(',')\n", "string")


def test_loop_zero_iteration_and_break_continue():
    src = (
        "def f(xs):\n    for x in xs:\n        if x:\n            break\n"
        "        continue\n    return 1\n"
    )
    assert descs(src, "loop") == ["for ... in []", "Break -> Continue", "Continue -> Break"]
    assert "for x in []:" in by_category(src, "loop")[0].source


def test_unary_minus_and_invert_removed():
    src = "def f(a):\n    return -a + ~a\n"
    assert descs(src, "unary") == ["remove USub", "remove Invert"]
    assert "+a" in by_category(src, "unary")[0].source


def test_assign_variants():
    src = "def f(a):\n    b = a\n    c = None\n    b += 1\n    return b\n"
    assert descs(src, "assign") == ["<expr> -> None", "None -> ''", "augassign -> assign"]
    assert "b = 1" in by_category(src, "assign")[2].source


def test_bitwise_separate_from_arith_probe():
    src = "def f(a, b):\n    return (a & b) << 1\n"
    assert descs(src, "bitwise") == ["LShift -> RShift", "BitAnd -> BitOr"]
    assert descs(src, PROBE_CATEGORY) == []


def test_arith_probe_covers_binop_and_augassign():
    src = "def f(a, b):\n    a += b\n    return a * 2 - 1 % 3 ** 2\n"
    d = descs(src, PROBE_CATEGORY)
    assert d[0] == "Add -> Sub"  # augassign first (line 2)
    assert set(d) == {"Add -> Sub", "Sub -> Add", "Mult -> Div", "Mod -> Div", "Pow -> Mult"}


def test_exception_handler_never_matches():
    src = "def f(a):\n    try:\n        return int(a)\n    except ValueError:\n        return 0\n"
    ms = by_category(src, "exception")
    assert ms[0].description == "except X -> except ()" and "except ():" in ms[0].source


def test_decorator_removed_and_match_case_dropped():
    src = (
        "import functools\n\n@functools.cache\ndef f(a):\n    match a:\n"
        "        case 1:\n            return 'one'\n        case _:\n            return 'other'\n"
    )
    assert descs(src, "decorator") == ["remove decorator 0"]
    assert descs(src, "match") == ["drop case 0", "drop case 1"]


def test_call_and_lambda_exist_but_are_off_in_eval_profile():
    src = "def f(xs):\n    return sorted(xs, key=lambda x: x, reverse=True)\n"
    assert descs(src, "call") == ["arg 0 -> None", "kwarg name + XX", "kwarg name + XX"]
    assert descs(src, "lambda") == ["lambda body -> None"]
    assert not EVAL_PROFILE["call"][0] and not EVAL_PROFILE["lambda"][0]
    assert "call" not in active_categories() and "lambda" not in active_categories()


# ---- invariants -------------------------------------------------------------


def test_training_profile_excludes_probe_only():
    assert set(active_categories()) - set(training_categories()) == {PROBE_CATEGORY}


def test_default_generation_uses_eval_profile():
    src = "def f(xs):\n    return sorted(xs, key=lambda x: x)\n"
    assert {m.category for m in generate_mutants(src)} <= set(active_categories())


@pytest.mark.parametrize("src", [CLAMP, "def f(s):\n    return s.lower() if s else 'X'\n"])
def test_every_mutant_parses_and_differs(src):
    for m in generate_mutants(src, ALL_CATEGORIES):
        ast.parse(m.source)
        assert m.source != src, m.id


def test_ids_stable_unique_and_source_ordered():
    a = generate_mutants(CLAMP, ALL_CATEGORIES)
    b = generate_mutants(CLAMP, ALL_CATEGORIES)
    assert [m.id for m in a] == [m.id for m in b]
    assert len({m.id for m in a}) == len(a)
    assert [m.line for m in a] == sorted(m.line for m in a)


def test_sample_is_seeded_and_keeps_source_order():
    ms = generate_mutants(CLAMP, ALL_CATEGORIES)
    s1, s2 = sample_mutants(ms, 3, seed=7), sample_mutants(ms, 3, seed=7)
    assert s1 == s2 and len(s1) == 3
    assert [ms.index(m) for m in s1] == sorted(ms.index(m) for m in s1)
