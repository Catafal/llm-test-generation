"""Known-answer tests for the paired statistics behind every headline number (FT15)."""

from testgen.stats import compare, mcnemar_exact, paired_bootstrap


def _rec(valid: bool, score: float | None, g_valid: bool | None = None, g_score=None) -> dict:
    """A minimal outputs.jsonl record; grounded block only when g_valid is given."""
    sc = {"valid": valid, "mutation_score": score}
    r = {"parsed": True, "score": sc}
    if g_valid is not None:
        r["grounded"] = {"score": {"valid": g_valid, "mutation_score": g_score}}
    return r


def test_mcnemar_exact_known_values():
    # discordant 5 vs 1: two-sided exact binomial p = 2 * P(X <= 1 | n=6, 1/2) = 2 * 7/64
    assert abs(mcnemar_exact(5, 1) - 14 / 64) < 1e-12
    assert mcnemar_exact(0, 0) == 1.0  # no discordant pairs: nothing to test
    assert mcnemar_exact(3, 3) == 1.0  # perfectly balanced: p capped at 1
    assert mcnemar_exact(10, 0) == 2 * (1 / 2) ** 10


def test_paired_bootstrap_is_exact_on_constant_differences():
    mean, lo, hi = paired_bootstrap([0.1] * 50, reps=200, seed=1)
    assert (mean, lo, hi) == (0.1, 0.1, 0.1)


def test_paired_bootstrap_interval_brackets_the_mean_and_is_seeded():
    diffs = [(-1) ** i * 0.2 for i in range(40)]  # mean 0, symmetric
    a = paired_bootstrap(diffs, reps=500, seed=7)
    b = paired_bootstrap(diffs, reps=500, seed=7)
    assert a == b and a[1] <= a[0] <= a[2] and a[1] < 0 < a[2]


def test_compare_counts_discordant_pairs_and_both_valid_scores():
    a = {
        "f1": _rec(True, 0.5),
        "f2": _rec(True, 1.0),
        "f3": _rec(False, None),
        "f4": _rec(True, 0.8),
    }
    b = {
        "f1": _rec(True, 0.5),
        "f2": _rec(False, None),
        "f3": _rec(True, 0.9),
        "f4": _rec(True, 0.6),
    }
    out = compare(a, b, "a", "b")
    assert out["n"] == 4 and out["validity"] == {"a": 0.75, "b": 0.75}
    assert out["discordant"] == {"only_a": 1, "only_b": 1}
    assert out["both_valid"] == 2  # f1 and f4
    assert abs(out["mutation_score_diff_on_both_valid"]["mean"] - 0.1) < 1e-12  # (0 + 0.2) / 2
    assert "grounded_score" not in out


def test_compare_grounded_reads_the_filled_suite_and_scores_invalid_as_zero():
    # unaided: both invalid; grounded: a valid (0.5), b still invalid -> grounded score 0
    a = {"f1": _rec(False, None, g_valid=True, g_score=0.5), "f2": _rec(True, 1.0, True, 1.0)}
    b = {"f1": _rec(False, None, g_valid=False, g_score=None), "f2": _rec(True, 1.0, True, 0.5)}
    out = compare(a, b, "a", "b", grounded=True)
    assert out["validity"] == {"a": 1.0, "b": 0.5}  # grounded validity
    assert out["grounded_score"] == {"a": 0.75, "b": 0.25}  # (0.5+1)/2 vs (0+0.5)/2
    assert abs(out["grounded_score_diff_all_functions"]["mean"] - 0.5) < 1e-12
    # the identity every results table must satisfy: score = validity x mean score on valid
    assert abs(out["grounded_score"]["a"] - out["validity"]["a"] * 0.75) < 1e-12
