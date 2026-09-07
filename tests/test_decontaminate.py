"""Positive controls: a planted copy of a training function must be caught."""

from testgen.data.decontaminate import _families, _flags_against

TRAIN = [
    {
        "id": "mbpp:1",
        "family": "mbpp:1",
        "source": (
            "def min_cost(cost, m, n):\n"
            "    tc = [[0 for x in range(n + 1)] for x in range(m + 1)]\n"
            "    tc[0][0] = cost[0][0]\n    for i in range(1, m + 1):\n"
            "        tc[i][0] = tc[i - 1][0] + cost[i][0]\n    return tc[m][n]\n"
        ),
    }
]


def fake_embed(texts):
    # identical texts -> identical vectors; otherwise orthogonal-ish
    return [[1.0, 0.0] if "min_cost" in t else [0.0, 1.0] for t in texts]


def test_verbatim_copy_is_flagged_by_every_layer():
    held = [{"id": "gh:copy", "repo": "r1", "source": TRAIN[0]["source"]}]
    reasons = _flags_against(held, TRAIN, fake_embed)["gh:copy"]
    kinds = {r.split()[0] for r in reasons}
    assert {"ast-exact", "ngram10", "ast-jaccard", "cosine"} <= kinds


def test_renamed_copy_is_flagged_structurally_not_lexically():
    renamed = TRAIN[0]["source"].replace("min_cost", "cheapest").replace("tc", "table")
    held = [{"id": "gh:renamed", "repo": "r1", "source": renamed}]
    reasons = _flags_against(held, TRAIN, None)["gh:renamed"]
    kinds = {r.split()[0] for r in reasons}
    assert "ast-exact" in kinds  # n-gram may also fire: two renames leave long shared runs


def test_unrelated_function_is_clean():
    held = [
        {"id": "gh:new", "repo": "r2", "source": "def f(a, b):\n    return a if a > b else b\n"}
    ]
    assert _flags_against(held, TRAIN, None).get("gh:new", []) == []


def test_families_merge_same_repo_and_flagged_pairs():
    held = [
        {"id": "a", "repo": "r1", "source": "def f(x):\n    return x + 1\n"},
        {"id": "b", "repo": "r1", "source": "def g(y):\n    return y * 2\n"},
        {"id": "c", "repo": "r2", "source": "def h(z):\n    return z + 1\n"},  # ast-equal to a
        {"id": "d", "repo": "r3", "source": "def k(q):\n    return [q] * 3\n"},
    ]
    fam, log = _families(held, None)
    assert fam["a"] == fam["b"] == fam["c"] and fam["d"] != fam["a"]
    assert any("ast-exact" in m and "c" in m for m in log)
