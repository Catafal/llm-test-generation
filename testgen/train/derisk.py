"""De-risk run for D023: does oracle filling lift yield over plain rejection?

    uv run python -m testgen.train.derisk --n 40 --k 4 --temp 0.7 --batch 16

On a difficulty-stratified sample of *dev* functions (never test), the 4B
proposes K suites per function at temperature. Every candidate is scored two
ways on the same run:

  A  rejection only   valid on the reference as generated, kills >= 1
                      training-category mutant.
  B  oracle-filled    wrong literal expected values rewritten from execution
                      (``oracle.fill``), then the same validity + kill gate.

Reported per difficulty stratum: functions covered (>= 1 kept candidate)
under A and B, candidate-level validity, how many literals were rewritten and
how long they were, the assertion-style mix, and generation throughput at the
chosen batch size (the "parallelism" lever on one GPU is the batch, not a
second process). Raw records go to runs/derisk-*/outputs.jsonl.
"""

import argparse
import json
import random
import sys
import time
from dataclasses import asdict

from config import MAX_TESTS_PER_SUITE, ROOT
from testgen.generate.prompts import build_messages, enforce_test_budget, extract_suite
from testgen.harness import manifest
from testgen.harness.runner import RunResult, run_many, run_suite
from testgen.harness.score import mutant_killed
from testgen.models import MODELS
from testgen.mutate.equivalence import split_equivalent
from testgen.mutate.operators import generate_mutants
from testgen.mutate.profiles import training_categories
from testgen.train.oracle import assertion_styles, fill

STRATA = 4  # quantile bins of live-mutant count


def stratified_dev(n: int, seed: int) -> list[dict]:
    """``n`` dev functions, equal counts per live-mutant quantile, seeded."""
    path = ROOT / "data" / "heldout" / "pool.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    dev = sorted((r for r in rows if r["split"] == "dev"), key=lambda r: r["live_mutants"])
    rng, per, picked = random.Random(seed), n // STRATA, []
    for s in range(STRATA):
        bin_ = dev[s * len(dev) // STRATA : (s + 1) * len(dev) // STRATA]
        for r in rng.sample(bin_, per):
            picked.append({**r, "stratum": s})
    return picked


def _valid(run: RunResult) -> bool:
    return run.status == "ok" and bool(run.tests) and not run.any_failed


def _kills(suite: str, live: list) -> int:
    runs = run_many(suite, {m.id: m.source for m in live})
    return sum(mutant_killed(r) for r in runs.values())


def score_candidate(text: str, source: str) -> dict:
    """Both pipelines on one generation. Mutants: training categories only (D015)."""
    suite, flags = extract_suite(text)
    if suite is None:
        return {"parsed": False, **flags}
    suite, n_tests = enforce_test_budget(suite, MAX_TESTS_PER_SUITE)
    live, _ = split_equivalent(source, generate_mutants(source, training_categories()))
    rec = {"parsed": True, "n_tests": n_tests, "styles": assertion_styles(suite), **flags}
    # A: as generated.
    rec["a_valid"] = _valid(run_suite(suite, source))
    rec["a_kills"] = _kills(suite, live) if rec["a_valid"] else 0
    # B: oracle-filled, then the identical gate.
    filled, stats, _ = fill(suite, source)
    rec["oracle"] = asdict(stats)
    rec["b_valid"] = _valid(run_suite(filled, source))
    rec["b_kills"] = _kills(filled, live) if rec["b_valid"] else 0
    rec["live_mutants"] = len(live)
    rec["suite"], rec["filled"] = suite, filled
    return rec


def generate(backend, rows: list[dict], k: int, temp: float, batch: int) -> list[dict]:
    """K samples per function, packed ``batch`` prompts at a time. Returns raw records."""
    jobs = [(r, j) for r in rows for j in range(k)]
    out, tokens, seconds = [], 0, 0.0
    for i in range(0, len(jobs), batch):
        chunk = jobs[i : i + batch]
        start = time.monotonic()
        gens = backend.generate_many(
            [build_messages(r["source"], MAX_TESTS_PER_SUITE, None) for r, _ in chunk],
            temperature=temp,
        )
        seconds += time.monotonic() - start
        for (r, j), g in zip(chunk, gens, strict=True):
            tokens += g.completion_tokens
            out.append({"id": r["id"], "stratum": r["stratum"], "sample": j, "text": g.text})
        print(
            f"  generated {min(i + batch, len(jobs))}/{len(jobs)}  {tokens / seconds:.0f} tok/s",
            flush=True,
        )
    return out


def summarise(records: list[dict], rows: list[dict]) -> dict:
    by_fn: dict[str, list[dict]] = {}
    for rec in records:
        by_fn.setdefault(rec["id"], []).append(rec)
    strata = {}
    for s in range(STRATA):
        ids = [r["id"] for r in rows if r["stratum"] == s]
        cov_a = sum(any(c.get("a_valid") and c["a_kills"] for c in by_fn[i]) for i in ids)
        cov_b = sum(any(c.get("b_valid") and c["b_kills"] for c in by_fn[i]) for i in ids)
        strata[s] = {"functions": len(ids), "covered_a": cov_a, "covered_b": cov_b}
    parsed = [r for r in records if r["parsed"]]
    styles = dict.fromkeys(parsed[0]["styles"], 0) if parsed else {}
    for r in parsed:
        for key, v in r["styles"].items():
            styles[key] += v
    keys = ("sites", "recorded", "replaced", "already_correct", "too_long", "unliteral")
    oracle = dict.fromkeys(keys, 0)
    lengths = []
    for r in parsed:
        for key in oracle:
            oracle[key] += r["oracle"][key]
        lengths += r["oracle"]["repr_lengths"]
    return {
        "candidates": len(records),
        "unparsed": len(records) - len(parsed),
        "a_valid_rate": sum(r["a_valid"] for r in parsed) / max(1, len(parsed)),
        "b_valid_rate": sum(r["b_valid"] for r in parsed) / max(1, len(parsed)),
        "coverage_by_stratum": strata,
        "assertion_styles": styles,
        "oracle": oracle,
        "replaced_repr_lengths": sorted(lengths),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="4b")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import mlx.core as mx

    from testgen.generate.mlx_backend import Backend

    mx.random.seed(args.seed)
    rows = stratified_dev(args.n, args.seed)
    run_dir = manifest.new_run(
        "derisk",
        model=MODELS[args.model],
        n_functions=len(rows),
        k=args.k,
        temperature=args.temp,
        batch=args.batch,
        seed=args.seed,
        function_ids=[r["id"] for r in rows],
    )
    backend = Backend(MODELS[args.model])
    raw = generate(backend, rows, args.k, args.temp, args.batch)
    del backend
    src = {r["id"]: r["source"] for r in rows}
    records = []
    with (run_dir / "outputs.jsonl").open("a") as f:
        for i, g in enumerate(raw, 1):
            rec = {**g, **score_candidate(g["text"], src[g["id"]])}
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            if i % 20 == 0:
                print(f"  scored {i}/{len(raw)}", flush=True)
    summary = summarise(records, rows)
    manifest.update(run_dir, results=summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "replaced_repr_lengths"}, indent=1))
    print(f"\nrun: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
