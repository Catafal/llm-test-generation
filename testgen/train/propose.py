"""T2: self-sampling. The 4B proposes K candidate suites per training function (D023).

    uv run --group models python -m testgen.train.propose --k 8 --batch 16

Reads  data/train/pool.jsonl
Writes runs/propose-<ts>/outputs.jsonl   {id, sample, text, completion_tokens}
       runs/propose-<ts>/manifest.json

Same system/user prompt as evaluation (FT10), thinking off, temperature 0.7,
budget MAX_NEW_TOKENS. ``--resume <run_dir>`` continues an interrupted run:
(id, sample) pairs already on disk are skipped. Batching is the parallelism:
one model, ``--batch`` prompts per forward pass.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from config import MAX_TESTS_PER_SUITE, ROOT
from testgen.generate.prompts import build_messages
from testgen.harness import manifest
from testgen.models import MODELS

POOL = ROOT / "data" / "train" / "pool.jsonl"


def load_pool(path: Path = POOL) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _done(run_dir: Path) -> set[tuple[str, int]]:
    out = run_dir / "outputs.jsonl"
    if not out.exists():
        return set()
    return {
        (r["id"], r["sample"])
        for r in (json.loads(ln) for ln in out.read_text().splitlines() if ln.strip())
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="4b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N functions only (smoke test)")
    ap.add_argument("--resume", default="", help="existing runs/propose-* directory")
    args = ap.parse_args(argv)

    import mlx.core as mx

    from testgen.generate.mlx_backend import Backend

    pool = load_pool()
    if args.limit:
        pool = pool[: args.limit]
    run_dir = (
        Path(args.resume)
        if args.resume
        else manifest.new_run(
            "propose",
            model=MODELS[args.model],
            n_functions=len(pool),
            k=args.k,
            temperature=args.temp,
            batch=args.batch,
            seed=args.seed,
            pool_sha256=__import__("hashlib").sha256(POOL.read_bytes()).hexdigest(),
        )
    )
    done = _done(run_dir)
    jobs = [(r, j) for r in pool for j in range(args.k) if (r["id"], j) not in done]
    print(f"{len(jobs)} generations to do ({len(done)} already on disk) -> {run_dir}", flush=True)
    mx.random.seed(args.seed + len(done))  # resumed runs do not replay the same samples
    backend = Backend(MODELS[args.model])
    tokens, seconds = 0, 0.0
    with (run_dir / "outputs.jsonl").open("a") as f:
        for i in range(0, len(jobs), args.batch):
            chunk = jobs[i : i + args.batch]
            start = time.monotonic()
            gens = backend.generate_many(
                [build_messages(r["source"], MAX_TESTS_PER_SUITE, None) for r, _ in chunk],
                temperature=args.temp,
            )
            seconds += time.monotonic() - start
            for (r, j), g in zip(chunk, gens, strict=True):
                tokens += g.completion_tokens
                rec = {
                    "id": r["id"],
                    "sample": j,
                    "text": g.text,
                    "completion_tokens": g.completion_tokens,
                }
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(
                f"  {min(i + args.batch, len(jobs))}/{len(jobs)}  {tokens / seconds:.0f} tok/s",
                flush=True,
            )
    manifest.update(
        run_dir, generations=len(done) + len(jobs), tokens_per_second=tokens / max(seconds, 1e-9)
    )
    print(f"run: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
