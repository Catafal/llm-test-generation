"""D030: grounded aggregation over scored records, unparsed suites count as 0."""

from testgen.baselines import aggregate_records


def _rec(valid: bool, score: float | None, g_valid: bool, g_score: float | None) -> dict:
    base = {
        "valid": valid,
        "invalid_reason": None if valid else "false_failure",
        "n_tests": 1,
        "false_failures": 0 if valid else 1,
        "mutants_total": 2,
        "mutants_equivalent": 0,
        "mutants_killed": 0,
        "mutation_score": score,
        "kills": {},
    }
    return {
        "parsed": True,
        "score": base,
        "grounded": {
            "suite": "",
            "oracle": {"replaced": 1},
            "score": {**base, "valid": g_valid, "mutation_score": g_score},
        },
        "truncated": False,
        "pytest_import": False,
        "generation": {"completion_tokens": 10, "seconds": 1.0},
    }


def test_grounded_rows_span_all_records():
    records = [
        _rec(False, None, True, 1.0),  # rescued by the oracle
        _rec(True, 0.5, True, 0.5),
        {
            "parsed": False,
            "score": None,
            "truncated": True,
            "pytest_import": False,
            "generation": {"completion_tokens": 10, "seconds": 1.0},
        },
    ]
    a = aggregate_records(records, grounded=True)
    assert a["validity_rate"] == 0.5  # unaided: over the two parsed suites, as before
    assert a["grounded_validity_rate"] == 2 / 3  # over every record
    assert abs(a["mean_grounded_score"] - 0.5) < 1e-9  # (1.0 + 0.5 + 0) / 3
    assert a["grounded_literals_replaced"] == 2
