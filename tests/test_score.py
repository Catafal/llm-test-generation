from testgen.harness.runner import RunResult, TestResult
from testgen.harness.score import aggregate, mutant_killed, score_suite


def ok(*outcomes: str) -> RunResult:
    return RunResult("ok", [TestResult(f"t{i}", o) for i, o in enumerate(outcomes)])


def test_valid_suite_scores_killed_over_live():
    s = score_suite(
        reference=ok("passed", "passed"),
        mutants={
            "m1": ok("failed", "passed"),
            "m2": ok("passed", "passed"),
            "m3": RunResult("timeout"),
        },
    )
    assert s.valid and s.invalid_reason is None
    assert s.mutants_total == 3 and s.mutants_killed == 2
    assert s.mutation_score == 2 / 3
    assert s.kills == {"m1": ["t0"], "m3": ["<timeout>"]}


def test_equivalent_mutants_leave_denominator():
    s = score_suite(ok("passed"), {"m1": ok("failed"), "eq": ok("passed")}, equivalent={"eq"})
    assert s.mutants_total == 2 and s.mutants_equivalent == 1
    assert s.mutation_score == 1.0


def test_false_failure_makes_suite_invalid_with_no_score():
    s = score_suite(ok("passed", "failed"), {"m1": ok("failed", "failed")})
    assert not s.valid and s.invalid_reason == "false_failure"
    assert s.false_failures == 1
    assert s.mutation_score is None and s.mutants_killed == 0


def test_reference_timeout_and_crash_and_empty_are_invalid():
    assert score_suite(RunResult("timeout"), {}).invalid_reason == "reference_timeout"
    assert score_suite(RunResult("crash"), {}).invalid_reason == "reference_crash"
    assert score_suite(RunResult("ok", []), {}).invalid_reason == "no_tests"


def test_crash_on_mutant_counts_as_kill():
    assert mutant_killed(RunResult("crash"))
    assert mutant_killed(ok("error"))
    assert not mutant_killed(ok("passed"))


def test_aggregate_reports_denominators():
    scores = [
        score_suite(ok("passed"), {"m1": ok("failed"), "m2": ok("passed")}),
        score_suite(ok("failed"), {"m1": ok("failed")}),
    ]
    a = aggregate(scores)
    assert a["suites"] == 2 and a["valid"] == 1 and a["validity_rate"] == 0.5
    assert a["false_failure_suites"] == 1
    assert a["mutants_killed"] == 1 and a["mutants_live_total"] == 2
    assert a["mean_mutation_score"] == 0.5
