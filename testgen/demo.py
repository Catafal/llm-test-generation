"""90-second demo: one held-out function, base vs fine-tuned suite, harness, mutants.

    uv run --group models python -m testgen.demo --pick          # list good demo functions
    uv run --group models python -m testgen.demo --id <fn id>    # live generation, both arms
    uv run --group models python -m testgen.demo --id <fn id> --from-runs   # replay saved outputs

Steps shown, in order:
  1. the function under test (held-out, post-2026-06 GitHub);
  2. the base model's suite as written -> run on the reference (failing asserts)
     -> harness fills expected values by execution -> valid? -> mutants killed;
  3. the fine-tuned suite, same treatment;
  4. one mutant only the fine-tuned suite catches: the mutated line and the
     test that fails on it.

``--pick`` scans the two committed test runs (base zero-shot and the 12k
adapter, both grounded) for functions where the fine-tune kills a mutant the
base misses, shortest source first, so the recording fits one screen.
Live generation is greedy under the evaluation budget, exactly as scored.
"""

import argparse
import ast
import difflib
import json
import sys
from pathlib import Path

from config import MAX_TESTS_PER_SUITE, RUNS_DIR
from testgen.baselines import load_pool
from testgen.generate.prompts import build_messages, enforce_test_budget, extract_suite
from testgen.harness.runner import run_many, run_suite
from testgen.harness.score import mutant_killed, score_suite
from testgen.models import MODELS
from testgen.mutate.equivalence import split_equivalent
from testgen.mutate.operators import generate_mutants
from testgen.train.oracle import fill

BASE_RUN = "baselines-test-bf16base-grounded-20260910T203314Z"
TUNED_RUN = "baselines-test-finetune-20260914T115239Z"
ADAPTER = "models/adapters/lora-4b-ext12k/ckpt-0002400"
BAR = "─" * 72


def _load_run(name: str) -> dict[str, dict]:
    path = RUNS_DIR / name / "outputs.jsonl"
    return {json.loads(ln)["id"]: json.loads(ln) for ln in path.read_text().splitlines() if ln}


def pick(pool: list[dict], max_lines: int = 30) -> list[tuple[str, int, int, int]]:
    """(id, source lines, base kills, tuned kills) where the tune kills a mutant the base misses."""
    base, tuned = _load_run(BASE_RUN), _load_run(TUNED_RUN)
    out = []
    for row in pool:
        b, t = base.get(row["id"], {}).get("grounded"), tuned.get(row["id"], {}).get("grounded")
        if not (b and t and b["score"]["valid"] and t["score"]["valid"]):
            continue
        only_tuned = set(t["score"]["kills"]) - set(b["score"]["kills"])
        n_lines = row["source"].count("\n")
        if only_tuned and n_lines <= max_lines:
            out.append((row["id"], n_lines, len(b["score"]["kills"]), len(t["score"]["kills"])))
    return sorted(out, key=lambda x: (x[1], -(x[3] - x[2])))


def generate(backend, source: str) -> str:
    msgs = build_messages(source, MAX_TESTS_PER_SUITE, None)
    return backend.generate_many([msgs])[0].text


def show_arm(label: str, text: str, source: str, live: list) -> tuple[str, dict]:
    """Print the suite, the unaided run, the fill, the grounded run and the kills."""
    print(f"\n{BAR}\n{label}\n{BAR}")
    suite, _ = extract_suite(text)
    suite, _ = enforce_test_budget(suite, MAX_TESTS_PER_SUITE)
    print(suite.rstrip())
    ref = run_suite(suite, source)
    failing = [t.name for t in ref.tests if t.outcome != "passed"]
    passed = len(ref.tests) - len(failing)
    note = f"   (failing: {', '.join(failing)})" if failing else ""
    print(f"\n→ as written, on the correct function: {passed}/{len(ref.tests)} tests pass{note}")
    filled, stats, _ = fill(suite, source, max_repr=None)
    print(f"→ harness executes the function and rewrites {stats.replaced} expected value(s)")
    ref2 = run_suite(filled, source)
    runs = run_many(filled, {m.id: m.source for m in live})
    score = score_suite(ref2, runs)
    verdict = "VALID" if score.valid else f"still invalid ({score.invalid_reason})"
    print(f"→ after filling: {verdict}; mutants killed: {score.mutants_killed}/{len(live)}")
    return filled, {"kills": score.kills, "suite": filled}


def show_unique_kill(source: str, live: list, base: dict, tuned: dict) -> None:
    only = [m for m in live if m.id in tuned["kills"] and m.id not in base["kills"]]
    print(f"\n{BAR}\nMutants only the fine-tuned suite catches: {len(only)}\n{BAR}")
    if not only:
        return
    # Mutants are ast.unparse'd, so diff against the unparsed original: only the
    # mutation shows, not quote-style noise. Pick the mutant with the smallest diff.
    original = ast.unparse(ast.parse(source)).splitlines()

    def _diff(m):
        return [
            ln
            for ln in difflib.unified_diff(
                original, m.source.splitlines(), "", "", lineterm="", n=0
            )
            if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
        ]

    m = min(only, key=lambda m: len(_diff(m)))
    print(f"{m.id}  ({m.description})\n" + "\n".join(_diff(m)))
    print(f"\ncaught by: {', '.join(tuned['kills'][m.id])}")
    runs = run_many(tuned["suite"], {m.id: m.source})
    assert mutant_killed(runs[m.id])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="", help="held-out function id (test split)")
    ap.add_argument("--pick", action="store_true", help="list functions that make a good demo")
    ap.add_argument("--from-runs", action="store_true", help="replay the saved test generations")
    ap.add_argument("--adapter", default=ADAPTER)
    args = ap.parse_args(argv)

    pool = load_pool("test")
    if args.pick:
        for fid, n, kb, kt in pick(pool)[:15]:
            print(f"{n:>3} lines  base {kb:>2} / tuned {kt:>2} kills   {fid}")
        return 0
    row = next(r for r in pool if r["id"] == args.id)
    source = row["source"]
    mutants = generate_mutants(source)
    live, _ = split_equivalent(source, mutants)
    print(f"{BAR}\nFunction under test  ({args.id})\n{BAR}\n{source.rstrip()}")
    print(f"\n{len(mutants)} mutants generated, {len(live)} live after the equivalence filter")

    if args.from_runs:
        texts = (
            _load_run(BASE_RUN)[args.id]["generation"]["text"],
            _load_run(TUNED_RUN)[args.id]["generation"]["text"],
        )
    else:
        from testgen.generate.mlx_backend import Backend  # mlx only when generating

        base = Backend(MODELS["4b-bf16"])
        t_base = generate(base, source)
        del base
        tuned = Backend(MODELS["4b-bf16"], adapter_path=args.adapter)
        t_tuned = generate(tuned, source)
        del tuned
        texts = (t_base, t_tuned)

    _, base_res = show_arm("Qwen3.5-4B, prompted", texts[0], source, live)
    _, tuned_res = show_arm(
        f"Qwen3.5-4B + {Path(args.adapter).parent.name}", texts[1], source, live
    )
    show_unique_kill(source, live, base_res, tuned_res)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
