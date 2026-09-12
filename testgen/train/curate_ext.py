"""D032 stage 1: curate an external (function, pytest) dataset through our harness.

    uv run python -m testgen.train.curate_ext --stage a               # cheap filters, all rows
    uv run python -m testgen.train.curate_ext --stage b --limit 12000 # execute + mutants
    uv run python -m testgen.train.curate_ext --stage c --keep 4000   # decontaminate, SFT set

Source: KodCode-V1 (HF ``KodCode/KodCode-V1``, CC BY-NC 4.0). Each row is a
GPT-4o solution module and a pytest module that imports it with
``from solution import <name>``, which is exactly our inference format.

Stage A (AST only, every row of the chosen subsets):
  solution = imports + exactly one top-level function that passes the
  held-out purity filter; test imports only ``solution``/``pytest``/stdlib;
  test budgeted to MAX_TESTS_PER_SUITE; >= MIN_LIVE_MUTANTS live mutants.
  -> data/train/ext/stage_a.jsonl (gitignored)
Stage B (sandbox, a seeded random sample of stage-A survivors):
  suite passes on the solution; kills >= 1 training-category mutant.
  -> data/train/ext/candidates.jsonl (gitignored; pool schema + suite fields)
Stage C: layers 1 and 3 of the decontamination vs both held-out splits
  (n-gram, AST hash, AST Jaccard; layer-2 embeddings skipped: the source is
  2025 synthetic, the held-out pool is post-2026-06 GitHub); AST-exact
  self-dedup; families from the KodCode question id; top --keep by
  (mutation score, fewer tests); 95/5 family split.
  -> data/train/ext/{train,valid}.jsonl, pool.jsonl, yield.json,
     DECONTAMINATION.md, NOTICE.md
"""

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from config import MAX_TESTS_PER_SUITE, MAX_TRAIN_TOKENS, ROOT
from testgen.data.decontaminate import _flags_against
from testgen.data.purity import extract_candidates
from testgen.data.similarity import ast_hash
from testgen.generate.prompts import enforce_test_budget
from testgen.harness.runner import run_many, run_suite
from testgen.harness.score import mutant_killed
from testgen.models import MODELS
from testgen.mutate.equivalence import split_equivalent
from testgen.mutate.operators import generate_mutants
from testgen.mutate.profiles import training_categories
from testgen.train.filter import _valid, to_chat, token_length
from testgen.train.oracle import assertion_styles

EXT_DIR = ROOT / "data" / "train" / "ext"
HELDOUT_POOL = ROOT / "data" / "heldout" / "pool.jsonl"
DATASET = "KodCode/KodCode-V1"
DATASET_URL = "https://huggingface.co/datasets/KodCode/KodCode-V1"
LICENCE = "CC BY-NC 4.0"
# Subsets whose questions read like real utility functions, not puzzles.
# Prefill is seeded from public benchmarks (MBPP/HumanEval style) and is excluded.
SUBSETS = ("Docs", "Package", "Filter", "Algorithm", "Data_Structure", "Evol")
MIN_LIVE_MUTANTS = 8  # same floor as the held-out harvest (D017)
TEST_IMPORT_OK = {
    "solution",
    "pytest",
    "math",
    "re",
    "itertools",
    "collections",
    "functools",
    "string",
    "random",
    "datetime",
    "json",
    "typing",
    "fractions",
    "decimal",
    "operator",
    "heapq",
    "bisect",
    "copy",
    "statistics",
}
VALID_FRACTION = 0.05
_ROMAN = re.compile(r"_(I|II|III|IV|V|VI|VII|VIII|IX|X)$")


def _test_imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


def stage_a_row(row: dict) -> dict | None:
    """Cheap AST filters. Returns a pool-schema record or None with no execution."""
    try:
        sol_tree, test_tree = ast.parse(row["solution"]), ast.parse(row["test"])
    except (SyntaxError, ValueError):
        return None
    defs = [n for n in sol_tree.body if isinstance(n, ast.FunctionDef)]
    others = [
        n for n in sol_tree.body if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))
    ]
    if len(defs) != 1 or others:
        return None  # one function, no helpers, classes or module state
    accepted, _ = extract_candidates(row["solution"])
    if len(accepted) != 1:
        return None
    if not _test_imports(test_tree) <= TEST_IMPORT_OK:
        return None
    suite, n_tests = enforce_test_budget(row["test"], MAX_TESTS_PER_SUITE)
    if n_tests == 0:
        return None
    cand = accepted[0]
    live, _ = split_equivalent(cand.source, generate_mutants(cand.source))
    if len(live) < MIN_LIVE_MUTANTS:
        return None
    qid = str(row["question_id"])
    return {
        "id": f"kodcode:{row['subset']}/{qid}::{cand.name}",
        "function": cand.name,
        "repo": DATASET_URL,
        "licence": LICENCE,
        "path": f"{row['subset']}/{qid}",
        "family": f"{row['subset']}/{_ROMAN.sub('', qid)}",  # _I/_II variants share a family
        "live_mutants": len(live),
        "source_sha256": hashlib.sha256(cand.source.encode()).hexdigest(),
        "docstring": cand.docstring,
        "source": cand.source,
        "suite": suite,
        "n_tests": min(n_tests, MAX_TESTS_PER_SUITE),
        "gpt_difficulty": row.get("gpt_difficulty"),
        "gpt_pass_percentage": row.get("gpt_pass_percentage"),
    }


def stage_a(args) -> int:
    snap = Path(
        snapshot_download(DATASET, repo_type="dataset", allow_patterns=["data/train-*.parquet"])
    )
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    out = (EXT_DIR / "stage_a.jsonl").open("w")
    seen: Counter = Counter()
    for shard in sorted(snap.glob("data/train-*.parquet")):
        table = pq.read_table(
            shard,
            columns=[
                "subset",
                "question_id",
                "solution",
                "test",
                "gpt_difficulty",
                "gpt_pass_percentage",
            ],
        )
        for row in table.to_pylist():
            if row["subset"] not in SUBSETS:
                continue
            seen[f"rows/{row['subset']}"] += 1
            rec = stage_a_row(row)
            if rec is None:
                continue
            seen[f"kept/{row['subset']}"] += 1
            out.write(json.dumps(rec) + "\n")
        print(f"  {shard.name}: {dict(seen)}", flush=True)
    out.close()
    (EXT_DIR / "stage_a.json").write_text(json.dumps(dict(seen), indent=1))
    print(json.dumps(dict(seen), indent=1))
    return 0


def stage_b_row(rec: dict) -> dict | None:
    """Execute: suite must pass on the solution and kill >= 1 training-category mutant."""
    if not _valid(run_suite(rec["suite"], rec["source"])):
        return None
    live, _ = split_equivalent(
        rec["source"], generate_mutants(rec["source"], training_categories())
    )
    if not live:
        return None
    runs = run_many(rec["suite"], {m.id: m.source for m in live})
    kills = sum(mutant_killed(r) for r in runs.values())
    if kills == 0:
        return None
    return {
        **rec,
        "kills": kills,
        "mutation_score": kills / len(live),
        "styles": assertion_styles(rec["suite"]),
    }


def stage_b(args) -> int:
    rows = [json.loads(ln) for ln in (EXT_DIR / "stage_a.jsonl").read_text().splitlines() if ln]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit] if args.limit else rows
    out_path = EXT_DIR / "candidates.jsonl"
    done = set()
    if args.resume and out_path.exists():
        done = {json.loads(ln)["id"] for ln in out_path.read_text().splitlines() if ln}
    out = out_path.open("a" if args.resume else "w")
    stats = Counter(tried=0, kept=0, resumed=len(done))
    for i, rec in enumerate(rows, 1):
        if rec["id"] in done:
            continue
        stats["tried"] += 1
        kept = stage_b_row(rec)
        if kept is not None:
            stats["kept"] += 1
            out.write(json.dumps(kept) + "\n")
            out.flush()
        if i % 200 == 0:
            print(f"  {i}/{len(rows)} {dict(stats)}", flush=True)
    out.close()
    (EXT_DIR / "stage_b.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats), indent=1))
    return 0


def stage_c(args) -> int:
    from transformers import AutoTokenizer

    out_dir = Path(args.out) if args.out else EXT_DIR  # a second cut must not overwrite the first
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = [json.loads(ln) for ln in (EXT_DIR / "candidates.jsonl").read_text().splitlines() if ln]
    held = [json.loads(ln) for ln in HELDOUT_POOL.read_text().splitlines() if ln]
    # Layers 1 + 3 against BOTH held-out splits (D022 rule); the helper flags the
    # first list against the second, so candidates go first.
    flagged = _flags_against(cands, held, embed=None)
    hard = {i for i, r in flagged.items() if any(not x.startswith("REVIEW") for x in r)}
    clean, seen_hash = [], set()
    for c in cands:
        if c["id"] in hard:
            continue
        h = ast_hash(c["source"])
        if h in seen_hash:
            continue  # exact AST duplicate inside the dataset
        seen_hash.add(h)
        clean.append(c)
    tokenizer = AutoTokenizer.from_pretrained(MODELS["4b-bf16"])
    for c in clean:
        c["n_tokens"] = token_length(tokenizer, to_chat(c["source"], c["suite"]))
    fits = [c for c in clean if c["n_tokens"] <= MAX_TRAIN_TOKENS]
    fits.sort(key=lambda c: (-c["mutation_score"], c["n_tests"], c["id"]))
    kept = fits[: args.keep] if args.keep else fits

    families = sorted({k["family"] for k in kept})
    rng = random.Random(args.seed)
    rng.shuffle(families)
    valid_fams = set(families[: max(1, round(len(families) * VALID_FRACTION))])
    with (out_dir / "train.jsonl").open("w") as tr, (out_dir / "valid.jsonl").open("w") as va:
        for k in kept:
            line = json.dumps(to_chat(k["source"], k["suite"])) + "\n"
            (va if k["family"] in valid_fams else tr).write(line)
    (out_dir / "pool.jsonl").write_text("".join(json.dumps(k) + "\n" for k in kept))
    styles: Counter = Counter()
    for k in kept:
        styles.update(k["styles"])
    summary = {
        "dataset": DATASET,
        "licence": LICENCE,
        "subsets": list(SUBSETS),
        "candidates": len(cands),
        "removed_vs_heldout": len(hard),
        "ast_exact_duplicates": len(cands) - len(hard) - len(clean),
        "fit_max_train_tokens": len(fits),
        "kept": len(kept),
        "families": len(families),
        "train": sum(1 for k in kept if k["family"] not in valid_fams),
        "valid": sum(1 for k in kept if k["family"] in valid_fams),
        "kept_mean_mutation_score": sum(k["mutation_score"] for k in kept) / max(1, len(kept)),
        "kept_mean_tests": sum(k["n_tests"] for k in kept) / max(1, len(kept)),
        "kept_mean_tokens": sum(k["n_tokens"] for k in kept) / max(1, len(kept)),
        "kept_assertion_styles": dict(styles),
        "kept_by_subset": dict(Counter(k["path"].split("/")[0] for k in kept)),
        "seed": args.seed,
    }
    (out_dir / "yield.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "NOTICE.md").write_text(
        f"# Source\n\nFunctions and tests in pool.jsonl are reproduced from {DATASET_URL} "
        f"({LICENCE}; solutions and tests generated by GPT-4o-0513, KodCode 2025). "
        "Non-commercial: any adapter trained on this set inherits that restriction.\n"
    )
    (out_dir / "DECONTAMINATION.md").write_text(
        "# Decontamination of the external training set\n\n"
        f"Candidates: {len(cands)}. Reference: both held-out splits of data/heldout/pool.jsonl "
        f"({len(held)} functions). Layers: 1 (n-gram) and 3 (AST hash, AST Jaccard); "
        "layer 2 (code embeddings) skipped: the source is 2025 synthetic data, the "
        "held-out pool is post-2026-06 GitHub, so only generic near-duplicates are at risk "
        "and layers 1/3 catch those.\n\n"
        f"Removed as matching a held-out function: {len(hard)}\n\n"
        + "".join(f"- {i}: {'; '.join(flagged[i])}\n" for i in sorted(hard))
    )
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "c"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="stage b: rows to execute (seeded sample)")
    ap.add_argument("--resume", action="store_true", help="stage b: continue candidates.jsonl")
    ap.add_argument("--keep", type=int, default=4000, help="stage c: training examples to keep")
    ap.add_argument("--out", default="", help="stage c: output dir (default data/train/ext)")
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args(argv)
    return {"a": stage_a, "b": stage_b, "c": stage_c}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
