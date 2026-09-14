"""90-second demo, as a terminal UI: one held-out function, base vs fine-tune, harness.

    uv run --group models python -m testgen.demo --pick          # list good demo functions
    uv run --group models python -m testgen.demo --id <fn id>    # live generation, both arms
    uv run --group models python -m testgen.demo --id <fn id> --from-runs   # replay saved outputs

Screen, top to bottom (rich):
  1. the function under test (held-out, post-2026-06 GitHub) and its mutant count;
  2. per arm: spinner while the model writes, the suite with syntax colour, then
     three harness lines: pass/fail as written on the reference, values the
     harness rewrote from execution, valid-after-fill and mutants killed;
  3. a scoreboard table for the two arms;
  4. one mutant only the fine-tuned suite catches: mutated line and the test.

``--pick`` scans the two committed test runs (base zero-shot and the 12k
adapter, both grounded) for functions where the fine-tune kills a mutant the
base misses, shortest source first. Live generation is greedy under the
evaluation budget, exactly as scored (batch-1 numerics can differ slightly
from the scored batch-8 runs; ``--from-runs`` replays the scored text).
"""

import argparse
import ast
import difflib
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from config import MAX_TESTS_PER_SUITE, RUNS_DIR
from testgen.baselines import load_pool
from testgen.generate.prompts import build_messages, enforce_test_budget, extract_suite
from testgen.harness.runner import run_many, run_suite
from testgen.harness.score import score_suite
from testgen.models import MODELS
from testgen.mutate.equivalence import split_equivalent
from testgen.mutate.operators import generate_mutants
from testgen.train.oracle import fill

BASE_RUN = "baselines-test-bf16base-grounded-20260910T203314Z"
TUNED_RUN = "baselines-test-finetune-20260914T115239Z"
ADAPTER = "models/adapters/lora-4b-ext12k/ckpt-0002400"
console = Console()


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


def generate_live(source: str, adapter: str) -> tuple[str, str]:
    """Load base, write; load base + adapter, write. One model in memory at a time."""
    from testgen.generate.mlx_backend import Backend  # mlx only when generating

    msgs = build_messages(source, MAX_TESTS_PER_SUITE, None)
    texts = []
    for label, path in (("base model", None), ("fine-tuned adapter", adapter)):
        with console.status(f"[bold]loading {label}…"):
            backend = Backend(MODELS["4b-bf16"], adapter_path=path)
        with console.status(f"[bold]{label} is writing the test suite…"):
            texts.append(backend.generate_many([msgs])[0].text)
        del backend
    return texts[0], texts[1]


def run_arm(label: str, text: str, source: str, live: list, style: str) -> dict:
    """Show the suite and the three harness steps; return the numbers for the scoreboard."""
    suite, _ = extract_suite(text)
    suite, _ = enforce_test_budget(suite, MAX_TESTS_PER_SUITE)
    console.print(
        Panel(Syntax(suite.rstrip(), "python", theme="ansi_dark"), title=label, border_style=style)
    )
    with console.status("running the suite on the correct function…"):
        ref = run_suite(suite, source)
    failing = [t.name for t in ref.tests if t.outcome != "passed"]
    passed = len(ref.tests) - len(failing)
    colour = "green" if not failing else "yellow"
    console.print(
        f"  [{colour}]▸ as written: {passed}/{len(ref.tests)} tests pass on the correct function"
        + (f"  (failing: {', '.join(failing)})" if failing else "")
    )
    with console.status("harness executes the function and fills expected values…"):
        filled, stats, _ = fill(suite, source, max_repr=None)
    console.print(f"  [cyan]▸ harness rewrote {stats.replaced} expected value(s) from execution")
    with console.status(f"running the filled suite on {len(live)} mutants…"):
        runs = run_many(filled, {m.id: m.source for m in live})
        score = score_suite(run_suite(filled, source), runs)
    if score.valid:
        console.print(
            f"  [bold green]▸ valid after filling · kills {score.mutants_killed}/{len(live)}"
        )
    else:
        console.print(
            f"  [bold red]▸ still invalid after filling ({score.invalid_reason}) · 0 kills"
        )
    console.print()
    return {
        "tests": len(ref.tests),
        "passed": passed,
        "filled": stats.replaced,
        "valid": score.valid,
        "kills": score.kills,
        "n_live": len(live),
        "suite": filled,
    }


def scoreboard(base: dict, tuned: dict) -> None:
    t = Table(title="Scoreboard", show_lines=False)
    t.add_column("")
    t.add_column("prompted", justify="right")
    t.add_column("fine-tuned", justify="right", style="bold")
    t.add_row("tests written", str(base["tests"]), str(tuned["tests"]))
    t.add_row(
        "pass as written",
        f"{base['passed']}/{base['tests']}",
        f"{tuned['passed']}/{tuned['tests']}",
    )
    t.add_row("values filled by harness", str(base["filled"]), str(tuned["filled"]))
    t.add_row(
        "valid after filling", "yes" if base["valid"] else "no", "yes" if tuned["valid"] else "no"
    )
    t.add_row(
        "mutants killed",
        f"{len(base['kills'])}/{base['n_live']}",
        f"{len(tuned['kills'])}/{tuned['n_live']}",
    )
    console.print(t)


def unique_kill(source: str, live: list, base: dict, tuned: dict) -> None:
    only = [m for m in live if m.id in tuned["kills"] and m.id not in base["kills"]]
    if not only:
        console.print("[dim]no mutant is caught by the fine-tuned suite alone[/dim]")
        return
    # Mutants are ast.unparse'd; diff against the unparsed original so only the
    # mutation shows. Pick the mutant with the smallest diff.
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
    console.print(
        Panel(
            Syntax("\n".join(_diff(m)), "diff", theme="ansi_dark"),
            title=f"{len(only)} mutant(s) only the fine-tuned suite catches · e.g. {m.description}",
            subtitle=f"caught by: {', '.join(tuned['kills'][m.id])}",
            border_style="green",
        )
    )


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
            console.print(f"{n:>3} lines  base {kb:>2} / tuned {kt:>2} kills   {fid}")
        return 0
    row = next(r for r in pool if r["id"] == args.id)
    source = row["source"]
    mutants = generate_mutants(source)
    live, _ = split_equivalent(source, mutants)
    console.print(
        Panel(
            Syntax(source.rstrip(), "python", theme="ansi_dark"),
            title=f"Function under test · {args.id.split(':')[0]}",
            subtitle=f"{len(mutants)} mutants · {len(live)} live after the equivalence filter",
            border_style="white",
        )
    )
    if args.from_runs:
        texts = (
            _load_run(BASE_RUN)[args.id]["generation"]["text"],
            _load_run(TUNED_RUN)[args.id]["generation"]["text"],
        )
    else:
        texts = generate_live(source, args.adapter)
    base = run_arm("Qwen3.5-4B · prompted", texts[0], source, live, "grey50")
    tuned = run_arm(
        f"Qwen3.5-4B · + {Path(args.adapter).parent.name}", texts[1], source, live, "cyan"
    )
    scoreboard(base, tuned)
    unique_kill(source, live, base, tuned)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
