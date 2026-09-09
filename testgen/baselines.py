"""T7: prompting baselines for every candidate model on a function pool.

    uv run python -m testgen.baselines --pool pilot                 # 10 pilot cases
    uv run python -m testgen.baselines --pool heldout --models 9b   # held-out pool, one model
    uv run python -m testgen.baselines --pool dev --limit 60 --models 4b-bf16 \
        --conditions zero --adapter models/adapters/<run>          # fine-tune, dev slice (FT12)

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
BEST_OF_K = 4  # D023 ceiling row: K temperature samples, first one valid on the reference wins
BEST_OF_TEMP = 0.7


def load_pool(name: str) -> list[dict]:
    """Rows: {id, source, equivalent}. Pilot rows carry hand-labelled equivalents.

    ``test`` / ``dev`` select the D020 split of data/heldout/pool.jsonl.
    """
    if name == "pilot":
        return [
            {"id": c.name, "source": c.source, "equivalent": set(c.equivalent)}
            for c in load_cases()
        ]
    path = ROOT / "data" / "heldout" / "pool.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return [
        {"id": r["id"], "source": r["source"], "equivalent": set()}
        for r in rows
        if r["split"] == name
    ]


def few_shots() -> list[tuple[str, str]]:
    cases = {c.name: c for c in load_cases()}
    return [(cases[n].source, cases[n].strong) for n in SHOT_CASES]


def score_generation(text: str, source: str, equivalent: set[str]) -> dict:
    """Extract, budget, run, score. Returns a JSON-able record."""
    from testgen.generate.prompts import enforce_test_budget, extract_suite

    try:
        suite, flags = extract_suite(text)
        if suite is None:
            return {"parsed": False, "score": None, **flags}
        suite, n_tests = enforce_test_budget(suite, MAX_TESTS_PER_SUITE)
    except RecursionError:  # pathological nesting blows ast.unparse; same rule for every arm
        return {"parsed": False, "score": None, "truncated": False, "pytest_import": False}
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


def best_of(backend, row: dict, shots: list | None) -> dict:
    """Ceiling row (D023): K samples at temperature; the first that is valid on the
    reference is scored (a caller holding the implementation can run the suite and
    resample). Budget per sample is unchanged; total cost is K x, reported as such."""
    from testgen.generate.mlx_backend import Generation  # lazy: mlx only where needed
    from testgen.generate.prompts import build_messages

    msgs = build_messages(row["source"], MAX_TESTS_PER_SUITE, shots)
    gens = backend.generate_many([msgs] * BEST_OF_K, temperature=BEST_OF_TEMP)
    chosen, rec = gens[0], None
    for g in gens:
        rec = score_generation(g.text, row["source"], row["equivalent"])
        chosen = g
        if rec["score"] is not None and rec["score"]["valid"]:
            break
    total = Generation(
        chosen.text,
        chosen.prompt_tokens,
        sum(g.completion_tokens for g in gens),
        sum(g.seconds for g in gens),
    )
    return {"generation": asdict(total), "samples": BEST_OF_K, **rec}


def run_condition(backend, pool: list[dict], shots: list | None, run_dir: Path, tag: str) -> dict:
    from testgen.generate.prompts import build_messages
    from testgen.harness.score import SuiteScore

    records, scores = [], []
    out = (run_dir / "outputs.jsonl").open("a")  # incremental: a crash keeps what was scored
    for i in range(0, len(pool), BATCH):
        chunk = pool[i : i + BATCH]
        if tag == "bestof":
            gens = [None] * len(chunk)
        else:
            style = "shape" if tag == "shape" else "default"
            gens = backend.generate_many(
                [
                    build_messages(r["source"], MAX_TESTS_PER_SUITE, shots, style=style)
                    for r in chunk
                ]
            )
        for row, g in zip(chunk, gens, strict=True):
            rec = {"id": row["id"], "condition": tag, "model": backend.model_id}
            if tag == "bestof":
                rec.update(best_of(backend, row, shots))
            else:
                rec["generation"] = asdict(g)
                rec.update(score_generation(g.text, row["source"], row["equivalent"]))
            records.append(rec)
            out.write(json.dumps(rec) + "\n")
            if rec["score"] is not None:
                scores.append(SuiteScore(**rec["score"]))
        out.flush()
        print(f"  {tag}: {min(i + BATCH, len(pool))}/{len(pool)}", flush=True)
    out.close()
    agg = aggregate(scores)
    agg["unparsed"] = sum(1 for r in records if not r["parsed"])
    agg["truncated"] = sum(1 for r in records if r.get("truncated"))
    agg["pytest_import_added"] = sum(1 for r in records if r.get("pytest_import"))
    agg["mean_completion_tokens"] = sum(
        r["generation"]["completion_tokens"] for r in records
    ) / len(records)
    agg["mean_seconds"] = sum(r["generation"]["seconds"] for r in records) / len(records)
    agg["mean_thinking_tokens"] = sum(
        r["generation"].get("thinking_tokens", 0) for r in records
    ) / len(records)
    return agg


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["pilot", "test", "dev"], default="pilot")
    ap.add_argument("--models", default="9b,4b,coder7b")
    ap.add_argument(
        "--conditions",
        default="zero,few",
        help="zero, few, bestof (ceiling row), shape (D028 oracle-shape prompt)",
    )
    ap.add_argument("--limit", type=int, default=0, help="first N functions of the pool only")
    ap.add_argument("--adapter", default="", help="LoRA adapter dir applied to every model")
    ap.add_argument("--tag", default="", help="run-name suffix, e.g. the checkpoint id")
    ap.add_argument("--thinking", action="store_true", help="reasoning on (budget unchanged)")
    args = ap.parse_args(argv)

    from testgen.generate.mlx_backend import Backend

    pool = load_pool(args.pool)
    if args.limit:
        pool = pool[: args.limit]  # pool order is fixed by the committed artifact
    shots = few_shots()
    name = f"baselines-{args.pool}" + (f"-{args.tag}" if args.tag else "")
    run_dir = manifest.new_run(
        name,
        pool=args.pool,
        n_functions=len(pool),
        models=[MODELS[m] for m in args.models.split(",")],
        conditions=args.conditions.split(","),
        few_shot_cases=list(SHOT_CASES),
        adapter=args.adapter or None,
        thinking=args.thinking,
    )
    table: dict[str, dict] = {}
    for key in args.models.split(","):
        backend = Backend(MODELS[key], adapter_path=args.adapter or None, thinking=args.thinking)
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
