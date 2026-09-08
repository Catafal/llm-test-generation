"""T3: execution filter -> SFT set (D003, D018, D023, FT5, FT6).

    uv run python -m testgen.train.filter --run runs/propose-<ts>

Per candidate: extract + test budget (D018), oracle-fill wrong literal expected
values (D023), run on the reference, and score kills on the *training*
mutant categories only (the ``arith`` probe never feeds curation, D015).
Per function keep the best valid candidate that kills >= 1 mutant: highest
mutation score, then fewer tests (verbosity is a liability under the fixed
budget). Functions with no keeper are dropped.

Writes data/train/sft/train.jsonl, valid.jsonl   mlx-lm chat format
       data/train/sft/yield.json                  the yield table + oracle stats
       data/train/sft/kept.jsonl                  per-kept-example provenance
       data/train/sft/scored.jsonl                every candidate's scores (D026 pairs)

Split is 95/5 by *family* (D020 logic), seeded. The assistant turn is the
filled suite in a python fence, i.e. exactly what extract_suite() expects at
inference (FT10).
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from config import MAX_TESTS_PER_SUITE, MAX_TRAIN_TOKENS, ROOT
from testgen.generate.prompts import build_messages, enforce_test_budget, extract_suite
from testgen.harness.runner import RunResult, run_many, run_suite
from testgen.harness.score import mutant_killed
from testgen.models import MODELS
from testgen.mutate.equivalence import split_equivalent
from testgen.mutate.operators import generate_mutants
from testgen.mutate.profiles import training_categories
from testgen.train.oracle import assertion_styles, fill
from testgen.train.propose import load_pool

SFT_DIR = ROOT / "data" / "train" / "sft"
VALID_FRACTION = 0.05


def _valid(run: RunResult) -> bool:
    return run.status == "ok" and bool(run.tests) and not run.any_failed


def score_candidate(text: str, source: str, live: list) -> dict:
    """Extract, budget, oracle-fill, validate, count training-category kills."""
    suite, flags = extract_suite(text)
    if suite is None:
        return {"parsed": False, **flags}
    suite, n_tests = enforce_test_budget(suite, MAX_TESTS_PER_SUITE)
    rec = {"parsed": True, "n_tests": min(n_tests, MAX_TESTS_PER_SUITE), **flags}
    rec["valid_before"] = _valid(run_suite(suite, source))
    rec["suite_unaided"] = suite
    filled, stats, _ = fill(suite, source)
    rec["oracle"] = asdict(stats)
    rec["valid"] = _valid(run_suite(filled, source))
    rec["kills"] = 0
    if rec["valid"]:
        runs = run_many(filled, {m.id: m.source for m in live})
        rec["kills"] = sum(mutant_killed(r) for r in runs.values())
    rec["mutation_score"] = rec["kills"] / len(live) if live else 0.0
    rec["suite"], rec["styles"] = filled, assertion_styles(filled)
    return rec


def best_per_function(cands: list[dict], max_tokens: int = MAX_TRAIN_TOKENS) -> dict | None:
    """Valid, killing, and short enough to train on untruncated (D024)."""
    keep = [c for c in cands if c.get("valid") and c["kills"] > 0 and c["n_tokens"] <= max_tokens]
    if not keep:
        return None
    return max(keep, key=lambda c: (c["mutation_score"], -c["n_tests"]))


def token_length(tokenizer, chat: dict) -> int:
    """Tokens the trainer will see for this example (chat template applied)."""
    text = tokenizer.apply_chat_template(chat["messages"], tokenize=False)
    return len(tokenizer(text).input_ids)


def to_chat(source: str, suite: str) -> dict:
    msgs = build_messages(source, MAX_TESTS_PER_SUITE, None)
    return {"messages": [*msgs, {"role": "assistant", "content": f"```python\n{suite}```"}]}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/propose-* directory")
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODELS["4b-bf16"])
    pool = {r["id"]: r for r in load_pool()}
    raw = [
        json.loads(ln)
        for ln in (Path(args.run) / "outputs.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    by_fn: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_fn[r["id"]].append(r)

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    scored_out = (SFT_DIR / "scored.jsonl").open("w")
    scored: dict[str, list[dict]] = {}
    totals: Counter = Counter()
    oracle: Counter = Counter()
    for i, (fid, cands) in enumerate(by_fn.items(), 1):
        source = pool[fid]["source"]
        live, _ = split_equivalent(source, generate_mutants(source, training_categories()))
        scored[fid] = [{**c, **score_candidate(c["text"], source, live)} for c in cands]
        for c in scored[fid]:
            c["n_tokens"] = (
                token_length(tokenizer, to_chat(source, c["suite"])) if c["parsed"] else 0
            )
            # D026 pairs need every candidate, unaided text included; drop the raw reply.
            scored_out.write(json.dumps({k: v for k, v in c.items() if k != "text"}) + "\n")
            totals["candidates"] += 1
            totals["parsed"] += c["parsed"]
            totals["valid_before_oracle"] += c.get("valid_before", False)
            totals["valid_after_oracle"] += c.get("valid", False)
            totals["valid_and_killing"] += bool(c.get("valid") and c["kills"])
            totals["valid_killing_fits"] += bool(
                c.get("valid") and c["kills"] and c["n_tokens"] <= MAX_TRAIN_TOKENS
            )
            for k, v in c.get("oracle", {}).items():
                if k != "repr_lengths":
                    oracle[k] += v
        if i % 25 == 0:
            print(f"  scored {i}/{len(by_fn)} functions", flush=True)

    scored_out.close()
    kept = []
    for fid, cands in scored.items():
        best = best_per_function(cands)
        if best is not None:
            kept.append(
                {
                    "id": fid,
                    "family": pool[fid]["family"],
                    "stratum_live": pool[fid]["live_mutants"],
                    **best,
                }
            )
    totals["functions"] = len(by_fn)
    totals["functions_kept"] = len(kept)

    # 95/5 by family, seeded (D020 logic): near-duplicates never straddle train/valid.
    families = sorted({k["family"] for k in kept})
    rng = random.Random(args.seed)
    rng.shuffle(families)
    valid_fams = set(families[: max(1, round(len(families) * VALID_FRACTION))])
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    with (SFT_DIR / "train.jsonl").open("w") as tr, (SFT_DIR / "valid.jsonl").open("w") as va:
        for k in kept:
            line = json.dumps(to_chat(pool[k["id"]]["source"], k["suite"])) + "\n"
            (va if k["family"] in valid_fams else tr).write(line)
    with (SFT_DIR / "kept.jsonl").open("w") as f:
        for k in kept:
            f.write(json.dumps({key: v for key, v in k.items() if key != "text"}) + "\n")
    styles: Counter = Counter()
    for k in kept:
        styles.update(k["styles"])
    summary = {
        "propose_run": Path(args.run).name,
        "yield": dict(totals),
        "oracle": dict(oracle),
        "kept_assertion_styles": dict(styles),
        "kept_mean_mutation_score": sum(k["mutation_score"] for k in kept) / max(1, len(kept)),
        "kept_mean_tests": sum(k["n_tests"] for k in kept) / max(1, len(kept)),
        "kept_mean_tokens": sum(k["n_tokens"] for k in kept) / max(1, len(kept)),
        "max_train_tokens": MAX_TRAIN_TOKENS,
        "train": sum(1 for k in kept if k["family"] not in valid_fams),
        "valid": sum(1 for k in kept if k["family"] in valid_fams),
        "seed": args.seed,
    }
    (SFT_DIR / "yield.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
