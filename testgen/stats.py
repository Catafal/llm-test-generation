"""Paired comparisons between arms on the same functions (FT15, D015).

    uv run python -m testgen.stats runs/baselines-test-<id>/outputs.jsonl 9b/zero 4b/zero
    uv run python -m testgen.stats --grounded <outputs.jsonl> 4b/zero 4b/few   # D030

Every arm scores the same functions, so comparisons are within-subject:
- validity: paired binary outcome -> McNemar (exact binomial on discordant
  pairs) and a paired bootstrap CI on the difference in validity rate.
- mutation score: computed over functions where *both* arms are valid,
  paired bootstrap CI on the mean difference (so the comparison is not
  confounded by which functions each arm happened to get valid).
- grounded (D030, ``--grounded``): validity and mutation score are read from
  the oracle-filled suite (``record["grounded"]``), and the primary row is
  the paired bootstrap CI on the mean *grounded score* difference over ALL
  functions (0 for an invalid or unparsed suite), so the comparison is not
  conditioned on validity.
Pure Python; no scipy.
"""

import json
import math
import random
import sys
from collections import defaultdict


def load_arms(path: str) -> dict[str, dict[str, dict]]:
    arms: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        key = _short(r["model"]) + "/" + r["condition"]
        arms[key][r["id"]] = r
    return arms


def _short(model_id: str) -> str:
    name = model_id.split("/")[-1].lower()
    if "qwen3.5-9b" in name:
        return "9b"
    if "qwen3.5-4b" in name:
        return "4b"
    if "coder-7b" in name:
        return "coder7b"
    return name


def _sc(r: dict, grounded: bool) -> dict | None:
    """The SuiteScore dict to read: the oracle-filled suite's (D030) or the unaided one."""
    if grounded:
        return r["grounded"]["score"] if r.get("grounded") else None
    return r.get("score")


def _valid(r: dict, grounded: bool = False) -> bool:
    s = _sc(r, grounded)
    return bool(r["parsed"] and s and s["valid"])


def _score(r: dict, grounded: bool = False) -> float | None:
    return _sc(r, grounded)["mutation_score"] if _valid(r, grounded) else None


def _grounded_score(r: dict) -> float:
    """D030 primary metric: mutation score of the filled suite, 0 if invalid or unparsed."""
    return _score(r, grounded=True) or 0.0


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap(
    diffs: list[float], reps: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """(mean diff, 2.5th, 97.5th percentile) resampling functions with replacement."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return sum(diffs) / n, means[int(0.025 * reps)], means[int(0.975 * reps)]


def compare(
    a: dict[str, dict], b: dict[str, dict], name_a: str, name_b: str, grounded: bool = False
) -> dict:
    """Paired comparison of two arms keyed by function id (see module docstring)."""
    ids = sorted(set(a) & set(b))
    va = [_valid(a[i], grounded) for i in ids]
    vb = [_valid(b[i], grounded) for i in ids]
    only_a = sum(x and not y for x, y in zip(va, vb, strict=True))
    only_b = sum(y and not x for x, y in zip(va, vb, strict=True))
    v_mean, v_lo, v_hi = paired_bootstrap(
        [float(x) - float(y) for x, y in zip(va, vb, strict=True)]
    )
    both = [i for i in ids if _valid(a[i], grounded) and _valid(b[i], grounded)]
    out = {
        "arms": f"{name_a} vs {name_b}",
        "n": len(ids),
        "validity": {name_a: sum(va) / len(ids), name_b: sum(vb) / len(ids)},
        "validity_diff": {"mean": v_mean, "ci95": [v_lo, v_hi]},
        "discordant": {f"only_{name_a}": only_a, f"only_{name_b}": only_b},
        "mcnemar_p": mcnemar_exact(only_a, only_b),
        "both_valid": len(both),
    }
    if both:
        d = [_score(a[i], grounded) - _score(b[i], grounded) for i in both]
        s_mean, s_lo, s_hi = paired_bootstrap(d)
        out["mutation_score_diff_on_both_valid"] = {"mean": s_mean, "ci95": [s_lo, s_hi]}
    if grounded:
        ga = [_grounded_score(a[i]) for i in ids]
        gb = [_grounded_score(b[i]) for i in ids]
        g_mean, g_lo, g_hi = paired_bootstrap([x - y for x, y in zip(ga, gb, strict=True)])
        out["grounded_score"] = {name_a: sum(ga) / len(ids), name_b: sum(gb) / len(ids)}
        out["grounded_score_diff_all_functions"] = {"mean": g_mean, "ci95": [g_lo, g_hi]}
    return out


def main(argv: list[str]) -> int:
    grounded = "--grounded" in argv
    path, name_a, name_b = [a for a in argv if a != "--grounded"]
    arms = load_arms(path)
    print(json.dumps(compare(arms[name_a], arms[name_b], name_a, name_b, grounded), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
