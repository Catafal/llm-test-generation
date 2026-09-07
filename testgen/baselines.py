"""T7: prompting baselines for every candidate model on a function pool.

    uv run python -m testgen.baselines --pool pilot                 # 10 pilot cases
    uv run python -m testgen.baselines --pool heldout --models 9b   # held-out pool, one model

For each model x condition (zero-shot, few-shot) the script generates one
suite per function (batched, greedy, thinking off), truncates to the test
budget, runs the suite against the reference and every live mutant, and
scores it. Raw generations and per-function scores go to
runs/<run>/outputs.jsonl (gitignored); the manifest and the aggregate table
go to runs/<run>/manifest.json (committed).

Few-shot exemplars are the STRONG suites of two pilot cases (clamp,
letter_grade). They are never part of the held-out pool. When the pool *is*
the pilot, those two cases are excluded from few-shot scoring to avoid
self-exemplar leakage.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from config import MAX_TESTS_PER_SUITE, ROOT
from testgen.harness import manifest
from testgen.harness.runner import run_many, run_suite
from testgen.harness.score import aggregate, score_suite
from testgen.models import MODELS  # noqa: E402  (registry lives with the storage manager)
from testgen.mutate.equivalence import split_equivalent
from testgen.mutate.operators import generate_mutants
from testgen.pilot import load_cases

SHOT_CASES = ("clamp", "letter_grade")
BATCH = 8


def load_pool(name: str) -> list[dict]:
    """Rows: {id, source, equivalent}. Pilot rows carry hand-labelled equivalents."""
    if name == "pilot":
        return [
            {"id": c.name, "source": c.source, "equivalent": set(c.equivalent)}
            for c in load_cases()
        ]
    path = ROOT / "data" / "heldout" / "pool.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return [{"id": r["id"], "source": r["source"], "equivalent": set()} for r in rows]


def few_shots() -> list[tuple[str, str]]:
    cases = {c.name: c for c in load_cases()}
    return [(cases[n].source, cases[n].strong) for n in SHOT_CASES]


def score_generation(text: str, source: str, equivalent: set[str]) -> dict:
    """Extract, budget, run, score. Returns a JSON-able record."""
    from testgen.generate.prompts import enforce_test_budget, extract_suite

    suite, flags = extract_suite(text)
    if suite is None:
        return {"parsed": False, "score": None, **flags}
    suite, n_tests = enforce_test_budget(suite, MAX_TESTS_PER_SUITE)
    mutants = generate_mutants(source)
    live, trivial = split_equivalent(source, mutants)
    excluded = trivial | equivalent
    live = [m for m in live if m.id not in excluded]
    ref = run_suite(suite, source)
    runs = run_many(suite, {m.id: m.source for m in live})
    s = score_suite(ref, runs, excluded)
    return {
        "parsed": True,
        "suite": suite,
        "n_tests_generated": n_tests,
        "score": asdict(s),
        **flags,
    }


def run_condition(backend, pool: list[dict], shots: list | None, run_dir: Path, tag: str) -> dict:
    from testgen.generate.prompts import build_messages
    from testgen.harness.score import SuiteScore

    records, scores = [], []
    for i in range(0, len(pool), BATCH):
        chunk = pool[i : i + BATCH]
        gens = backend.generate_many(
            [build_messages(r["source"], MAX_TESTS_PER_SUITE, shots) for r in chunk]
        )
        for row, g in zip(chunk, gens, strict=True):
            rec = {
                "id": row["id"],
                "condition": tag,
                "model": backend.model_id,
                "generation": asdict(g),
            }
            rec.update(score_generation(g.text, row["source"], row["equivalent"]))
            records.append(rec)
            if rec["score"] is not None:
                scores.append(SuiteScore(**rec["score"]))
        print(f"  {tag}: {min(i + BATCH, len(pool))}/{len(pool)}", flush=True)
    with (run_dir / "outputs.jsonl").open("a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    agg = aggregate(scores)
    agg["unparsed"] = sum(1 for r in records if not r["parsed"])
    agg["truncated"] = sum(1 for r in records if r.get("truncated"))
    agg["pytest_import_added"] = sum(1 for r in records if r.get("pytest_import"))
    agg["mean_completion_tokens"] = sum(
        r["generation"]["completion_tokens"] for r in records
    ) / len(records)
    agg["mean_seconds"] = sum(r["generation"]["seconds"] for r in records) / len(records)
    return agg


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["pilot", "heldout"], default="pilot")
    ap.add_argument("--models", default="9b,4b,coder7b")
    ap.add_argument("--conditions", default="zero,few")
    args = ap.parse_args(argv)

    from testgen.generate.mlx_backend import Backend

    pool = load_pool(args.pool)
    shots = few_shots()
    run_dir = manifest.new_run(
        f"baselines-{args.pool}",
        pool=args.pool,
        n_functions=len(pool),
        models=[MODELS[m] for m in args.models.split(",")],
        conditions=args.conditions.split(","),
        few_shot_cases=list(SHOT_CASES),
    )
    table: dict[str, dict] = {}
    for key in args.models.split(","):
        backend = Backend(MODELS[key])
        print(f"model {MODELS[key]}", flush=True)
        for cond in args.conditions.split(","):
            fs = shots if cond == "few" else None
            rows = (
                pool
                if not (cond == "few" and args.pool == "pilot")
                else [r for r in pool if r["id"] not in SHOT_CASES]
            )
            table[f"{key}/{cond}"] = run_condition(backend, rows, fs, run_dir, cond)
        del backend
    manifest.update(run_dir, results=table)
    print(
        f"\n{'arm':<14}{'n':>4}{'valid':>7}{'mut.score':>10}{'unparsed':>9}{'trunc':>6}"
        f"{'+pytest':>8}{'tok':>6}{'s':>6}"
    )
    for arm, a in table.items():
        print(
            f"{arm:<14}{a['suites']:>4}{a['validity_rate']:>7.2f}{a['mean_mutation_score']:>10.3f}"
            f"{a['unparsed']:>9}{a['truncated']:>6}{a['pytest_import_added']:>8}"
            f"{a['mean_completion_tokens']:>6.0f}{a['mean_seconds']:>6.1f}"
        )
    print(f"\nrun: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
