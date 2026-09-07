"""Turn run results into the three numbers the project reports.

Decision 5 (T2): a mutant is *killed* if any test fails, errors, or the run
times out or crashes on it. A suite is *valid* only if it runs cleanly on the
reference with every test passing. An invalid suite gets **no** mutation
score (None), not zero, so a model cannot game the metric with tests that
always fail. Validity and false-failure are reported separately (FT17).
"""

from dataclasses import dataclass, field

from testgen.harness.runner import RunResult


@dataclass
class SuiteScore:
    valid: bool
    # one of: reference_timeout | reference_crash | no_tests | false_failure
    invalid_reason: str | None
    n_tests: int
    false_failures: int  # tests failing on the reference implementation
    mutants_total: int
    mutants_equivalent: int  # excluded from the denominator
    mutants_killed: int
    mutation_score: float | None  # killed / (total - equivalent); None if invalid or no mutants
    kills: dict[str, list[str]] = field(default_factory=dict)  # mutant id -> tests that failed


def mutant_killed(run: RunResult) -> bool:
    """Timeouts and crashes count as kills: the mutant changed behaviour."""
    return run.status != "ok" or run.any_failed


def _killing_tests(run: RunResult) -> list[str]:
    if run.status != "ok":
        return [f"<{run.status}>"]
    return [t.name for t in run.tests if t.outcome != "passed"]


def score_suite(
    reference: RunResult,
    mutants: dict[str, RunResult],
    equivalent: set[str] | None = None,
) -> SuiteScore:
    """Score one suite given its run on the reference and on each mutant.

    ``mutants`` maps mutant id -> run result. ``equivalent`` ids are excluded
    from the denominator (D013) and should not appear in ``mutants``; if they
    do, they are ignored.
    """
    equivalent = equivalent or set()
    n_tests = len(reference.tests)
    false_failures = sum(t.outcome != "passed" for t in reference.tests)

    if reference.status == "timeout":
        reason = "reference_timeout"
    elif reference.status == "crash":
        reason = "reference_crash"
    elif n_tests == 0:
        reason = "no_tests"
    elif false_failures:
        reason = "false_failure"
    else:
        reason = None

    live = {mid: run for mid, run in mutants.items() if mid not in equivalent}
    total = len(live) + len(equivalent & set(mutants))
    base = SuiteScore(
        valid=reason is None,
        invalid_reason=reason,
        n_tests=n_tests,
        false_failures=false_failures,
        mutants_total=total,
        mutants_equivalent=total - len(live),
        mutants_killed=0,
        mutation_score=None,
    )
    if reason is not None or not live:
        return base

    kills = {mid: _killing_tests(run) for mid, run in live.items() if mutant_killed(run)}
    base.mutants_killed = len(kills)
    base.mutation_score = len(kills) / len(live)
    base.kills = kills
    return base


def aggregate(scores: list[SuiteScore]) -> dict[str, float | int]:
    """Pool suite scores into the headline table. Denominators are explicit."""
    n = len(scores)
    valid = [s for s in scores if s.valid]
    scored = [s for s in valid if s.mutation_score is not None]
    return {
        "suites": n,
        "valid": len(valid),
        "validity_rate": len(valid) / n if n else 0.0,
        "false_failure_suites": sum(1 for s in scores if s.invalid_reason == "false_failure"),
        "mutants_killed": sum(s.mutants_killed for s in scored),
        "mutants_live_total": sum(s.mutants_total - s.mutants_equivalent for s in scored),
        "mean_mutation_score": (
            sum(s.mutation_score for s in scored) / len(scored) if scored else 0.0
        ),
    }
