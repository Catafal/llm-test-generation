"""Build execution-explanation SFT sets (D028 step 2) in both formats.

    uv run --group models python -m testgen.train.tracetargets --max-asserts 4 --max-lines 6

Source: data/train/sft/scored.jsonl, the best unaided-valid killing suite per
function (the model's own, no oracle fill). For each, trace the literal
asserts and write:
  data/train/trace/inline/{train,valid}.jsonl   comments above each assert
  data/train/trace/prefix/{train,valid}.jsonl   <derivation> section + suite
  data/train/trace/stats.json
Split 95/5 by family. Suites with no traceable assert are kept as plain
targets (the model must still learn ordinary suites), counted separately.
"""

import argparse
import json
import random
import sys
from collections import Counter

from config import MAX_TESTS_PER_SUITE, ROOT
from testgen.generate.prompts import build_messages
from testgen.train.propose import load_pool
from testgen.train.tracesuite import inline_target, prefix_target, trace_sites

SCORED = ROOT / "data" / "train" / "sft" / "scored.jsonl"
OUT = ROOT / "data" / "train" / "trace"
VALID_FRACTION = 0.05


def best_unaided(scored_path=SCORED) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for ln in scored_path.read_text().splitlines():
        if not ln.strip():
            continue
        c = json.loads(ln)
        if c.get("parsed") and c.get("valid_before") and c.get("kills", 0) > 0:
            if c["id"] not in best or c["mutation_score"] > best[c["id"]]["mutation_score"]:
                best[c["id"]] = c
    return best


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-asserts", type=int, default=4)
    ap.add_argument("--max-lines", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args(argv)
    pool = {r["id"]: r for r in load_pool()}
    best = best_unaided()
    stats = Counter()
    records = []
    for i, (fid, c) in enumerate(best.items(), 1):
        src, fn = pool[fid]["source"], pool[fid]["function"]
        sites, st = trace_sites(c["suite_unaided"], src, fn, args.max_asserts, args.max_lines)
        stats["functions"] += 1
        stats["traced_asserts"] += st.traced
        stats["with_trace"] += bool(sites)
        msgs = build_messages(src, MAX_TESTS_PER_SUITE, None)
        inline = inline_target(c["suite_unaided"], sites) if sites else c["suite_unaided"]
        prefix = (
            prefix_target(c["suite_unaided"], sites)
            if sites
            else f"```python\n{c['suite_unaided']}```"
        )
        records.append(
            {
                "family": pool[fid]["family"],
                "inline": {
                    "messages": [*msgs, {"role": "assistant", "content": f"```python\n{inline}```"}]
                },
                "prefix": {"messages": [*msgs, {"role": "assistant", "content": prefix}]},
            }
        )
        if i % 50 == 0:
            print(f"  {i}/{len(best)}", flush=True)
    families = sorted({r["family"] for r in records})
    rng = random.Random(args.seed)
    rng.shuffle(families)
    valid_fams = set(families[: max(1, round(len(families) * VALID_FRACTION))])
    for fmt in ("inline", "prefix"):
        d = OUT / fmt
        d.mkdir(parents=True, exist_ok=True)
        with (d / "train.jsonl").open("w") as tr, (d / "valid.jsonl").open("w") as va:
            for r in records:
                (va if r["family"] in valid_fams else tr).write(json.dumps(r[fmt]) + "\n")
    stats["train"] = sum(1 for r in records if r["family"] not in valid_fams)
    stats["valid"] = len(records) - stats["train"]
    stats.update({"max_asserts": args.max_asserts, "max_lines": args.max_lines})
    (OUT / "stats.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
