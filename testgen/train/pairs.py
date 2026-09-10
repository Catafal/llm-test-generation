"""D026: preference pairs from the model's own scored self-samples.

    uv run --group models python -m testgen.train.pairs

Reads  data/train/sft/scored.jsonl   (every candidate: pass/fail, kills, texts)
       data/train/pool.jsonl         (families for the split)
Writes data/train/dpo/train.jsonl, valid.jsonl   mlx-lm-lora DPO records
       data/train/dpo/pairs.json                 counts and selection stats

A pair is (same function, same prompt):
  chosen    passed the reference UNAIDED (no oracle fill) and kills >= 1
            training-category mutant. The model produced it as-is.
  rejected  parsed but failed on the reference. Preferred order:
            (1) "good tests, wrong values": rescued by the oracle and killing
                (the exact failure mode), (2) any other parsed failure.
--grounded (D030): the harness fills the values at inference, so the pair
targets what the model still owns, inputs and structure:
  chosen    valid AFTER oracle fill with the most training-category kills
            (ties: fewer tests); the assistant turn is still the UNAIDED
            text the model wrote, never the corrected one (D025 lesson).
  rejected  (1) parsed but invalid after fill (bad inputs, wrong arity,
            wrong non-literal asserts), (2) valid with zero kills, (3) the
            weakest valid candidate if it kills at least GROUNDED_KILL_GAP
            fewer mutants than the chosen. Output dir data/train/dpo-grounded/.
Both sides must fit MAX_TRAIN_TOKENS. At most PAIRS_PER_FUNCTION per
function, distinct chosen/rejected where possible, so no function dominates.
Record format follows mlx_lm_lora.trainer.datasets.DPODataset:
{"system", "prompt", "chosen", "rejected"} as strings; the trainer applies
the chat template. The assistant content is the fenced suite, identical to
what extract_suite() expects at inference (FT10).
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict

from config import MAX_TESTS_PER_SUITE, MAX_TRAIN_TOKENS, ROOT
from testgen.generate.prompts import build_messages
from testgen.models import MODELS
from testgen.train.propose import load_pool

SCORED = ROOT / "data" / "train" / "sft" / "scored.jsonl"
DPO_DIR = ROOT / "data" / "train" / "dpo"
DPO_GROUNDED_DIR = ROOT / "data" / "train" / "dpo-grounded"
PAIRS_PER_FUNCTION = 2
VALID_FRACTION = 0.05
GROUNDED_KILL_GAP = 2  # a "weak" rejected must trail the chosen by this many kills


def fence(suite: str) -> str:
    return f"```python\n{suite}```"


def _fits(tokenizer, system: str, prompt: str, suite: str) -> bool:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": fence(suite)},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    return len(tokenizer(text).input_ids) <= MAX_TRAIN_TOKENS


def _rejected_rank(c: dict) -> tuple:
    """Lower sorts first: oracle-rescued killing suites are the hardest negatives."""
    rescued = bool(c.get("valid") and c.get("kills", 0) > 0)
    return (0 if rescued else 1, -c.get("kills", 0), -c.get("n_tests", 0))


def build_pairs(cands: list[dict], tokenizer, system: str, prompt: str) -> list[tuple[dict, dict]]:
    parsed = [c for c in cands if c.get("parsed")]
    chosen = [c for c in parsed if c.get("valid_before") and c.get("kills", 0) > 0]
    rejected = sorted((c for c in parsed if not c.get("valid_before")), key=_rejected_rank)
    chosen = [c for c in chosen if _fits(tokenizer, system, prompt, c["suite_unaided"])]
    rejected = [c for c in rejected if _fits(tokenizer, system, prompt, c["suite_unaided"])]
    chosen.sort(key=lambda c: (-c["mutation_score"], c["n_tests"]))
    pairs = []
    for i in range(min(PAIRS_PER_FUNCTION, len(chosen), len(rejected))):
        pairs.append((chosen[i], rejected[i]))
    return pairs


def _grounded_rejected_type(c: dict) -> int:
    """0 invalid after fill, 1 valid but kills nothing, 2 valid and weak."""
    if not c.get("valid"):
        return 0
    return 1 if c.get("kills", 0) == 0 else 2


def build_pairs_grounded(
    cands: list[dict], tokenizer, system: str, prompt: str
) -> list[tuple[dict, dict]]:
    parsed = [c for c in cands if c.get("parsed")]
    parsed = [c for c in parsed if _fits(tokenizer, system, prompt, c["suite_unaided"])]
    chosen = sorted(
        (c for c in parsed if c.get("valid") and c.get("kills", 0) > 0),
        key=lambda c: (-c["kills"], c["n_tests"]),
    )
    if not chosen:
        return []
    top_kills = chosen[0]["kills"]
    rejected = [
        c
        for c in parsed
        if c is not chosen[0]
        and (_grounded_rejected_type(c) < 2 or c["kills"] <= top_kills - GROUNDED_KILL_GAP)
    ]
    rejected.sort(key=lambda c: (_grounded_rejected_type(c), c.get("kills", 0)))
    return [
        (chosen[i], rejected[i]) for i in range(min(PAIRS_PER_FUNCTION, len(chosen), len(rejected)))
    ]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--grounded", action="store_true", help="D030 pairs -> dpo-grounded/")
    args = ap.parse_args(argv)
    out_dir = DPO_GROUNDED_DIR if args.grounded else DPO_DIR
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODELS["4b-bf16"])
    pool = {r["id"]: r for r in load_pool()}
    by_fn: dict[str, list[dict]] = defaultdict(list)
    for ln in SCORED.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            by_fn[r["id"]].append(r)

    records, stats = [], Counter()
    for fid, cands in by_fn.items():
        msgs = build_messages(pool[fid]["source"], MAX_TESTS_PER_SUITE, None)
        system, prompt = msgs[0]["content"], msgs[-1]["content"]
        builder = build_pairs_grounded if args.grounded else build_pairs
        pairs = builder(cands, tokenizer, system, prompt)
        stats["functions"] += 1
        stats["functions_with_pairs"] += bool(pairs)
        for ch, rj in pairs:
            stats["pairs"] += 1
            stats["rejected_oracle_rescued"] += bool(rj.get("valid") and rj.get("kills", 0) > 0)
            if args.grounded:
                kind = ("invalid_after_fill", "zero_kills", "weak")[_grounded_rejected_type(rj)]
                stats[f"rejected_{kind}"] += 1
                stats["chosen_kills_total"] += ch["kills"]
                stats["rejected_kills_total"] += rj.get("kills", 0)
            records.append(
                {
                    "id": fid,
                    "family": pool[fid]["family"],
                    "system": system,
                    "prompt": prompt,
                    "chosen": fence(ch["suite_unaided"]),
                    "rejected": fence(rj["suite_unaided"]),
                }
            )

    families = sorted({r["family"] for r in records})
    rng = random.Random(args.seed)
    rng.shuffle(families)
    valid_fams = set(families[: max(1, round(len(families) * VALID_FRACTION))])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train.jsonl").open("w") as tr, (out_dir / "valid.jsonl").open("w") as va:
        for r in records:
            out = {k: r[k] for k in ("system", "prompt", "chosen", "rejected")}
            (va if r["family"] in valid_fams else tr).write(json.dumps(out) + "\n")
    stats["train"] = sum(1 for r in records if r["family"] not in valid_fams)
    stats["valid"] = len(records) - stats["train"]
    stats["max_train_tokens"] = MAX_TRAIN_TOKENS
    stats["pairs_per_function"] = PAIRS_PER_FUNCTION
    stats["grounded"] = args.grounded
    (out_dir / "pairs.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
