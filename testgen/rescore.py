"""D030 step G1: re-score saved generations under the grounded metric. No GPU.

    uv run python -m testgen.rescore runs/baselines-test-<id> [--limit N]

Reads  runs/<run>/outputs.jsonl        raw generations + unaided scores
Writes runs/<run>-grounded/outputs.jsonl  same records, re-scored with
       ``score_generation(grounded=True)``: the unaided ``score`` is
       recomputed (identical harness, so it must match) and ``grounded`` is
       added. Manifest carries ``rescored_from``.

Exploratory by construction (D030): the metric was chosen after these
generations existed. The confirmatory arm is a fresh evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

from testgen.baselines import aggregate_records, load_pool, score_generation
from testgen.harness import manifest


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="runs/<run> directory with outputs.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="first N records only (smoke test)")
    args = ap.parse_args(argv)

    src = Path(args.run)
    old = json.loads((src / "manifest.json").read_text())
    records = [json.loads(ln) for ln in (src / "outputs.jsonl").read_text().splitlines() if ln]
    if args.limit:
        records = records[: args.limit]
    pool = {r["id"]: r for r in load_pool(old["pool"])}

    run_dir = manifest.new_run(
        f"{src.name.split('-2026')[0]}-grounded",
        rescored_from=src.name,
        pool=old["pool"],
        models=old.get("models"),
        conditions=old.get("conditions"),
        adapter=old.get("adapter"),
        grounded=True,
    )
    out = (run_dir / "outputs.jsonl").open("w")
    by_arm: dict[str, list[dict]] = {}
    for i, r in enumerate(records, 1):
        row = pool[r["id"]]
        text = r["generation"]["text"]
        new = {k: r[k] for k in ("id", "condition", "model", "generation")}
        if "samples" in r:  # best-of ceiling row: the chosen sample's text was saved
            new["samples"] = r["samples"]
        new.update(score_generation(text, row["source"], row["equivalent"], grounded=True))
        if bool(new["score"] and new["score"]["valid"]) != bool(r["score"] and r["score"]["valid"]):
            new["validity_changed_on_rescore"] = True  # harness drift; must stay rare
        by_arm.setdefault(new["condition"], []).append(new)
        out.write(json.dumps(new) + "\n")
        if i % 25 == 0:
            out.flush()
            print(f"  rescored {i}/{len(records)}", flush=True)
    out.close()
    table = {arm: aggregate_records(recs, grounded=True) for arm, recs in by_arm.items()}
    manifest.update(run_dir, results=table)
    for arm, a in table.items():
        print(
            f"{arm:<8} n={a['suites']:<4} valid={a['validity_rate']:.3f} "
            f"g.valid={a['grounded_validity_rate']:.3f} "
            f"mut(g.valid)={a['grounded_mean_mutation_score']:.3f} "
            f"g.score={a['mean_grounded_score']:.3f}"
        )
    print(f"run: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
