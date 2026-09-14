"""Figures for the write-up and README, generated from run artifacts.

    uv run python -m testgen.figure          # -> docs/figures/results.png, devcurves.png

results.png    grounded score per arm on the test split (n=315) with the paired
               bootstrap 95% CI of the difference vs base zero-shot, drawn around
               each arm's mean. Arms whose raw generations are on disk are
               plotted; the two behind the headline comparison are committed.
devcurves.png  dev-171 grounded score per checkpoint for the KodCode adapters,
               against the base's dev score.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from config import ROOT, RUNS_DIR
from testgen import stats

FIG_DIR = ROOT / "docs" / "figures"
BASE = "baselines-test-bf16base-grounded-20260910T203314Z"
# label -> run directory; order is the plotting order
ARMS = {
    "base zero-shot": BASE,
    "base few-shot": "baselines-test-bf16base-few-grounded-20260910T205151Z",
    "it.1 SFT\n388 self": "baselines-test-finetune-grounded-20260910T211110Z",
    "it.4 DPO\n645 self": "baselines-test-finetune-20260911T075927Z",
    "it.5 SFT\n4k KodCode": "baselines-test-finetune-20260912T130009Z",
    "it.5 SFT\n12k KodCode": "baselines-test-finetune-20260914T115239Z",
}
DEV = {
    "4k KodCode": ROOT / "models" / "adapters" / "lora-4b-ext" / "devcurve.json",
    "12k KodCode": ROOT / "models" / "adapters" / "lora-4b-ext12k" / "devcurve.json",
}
DEV_BASE = 0.621  # baselines-dev-bf16base171-grounded (G1 rescore)


def _records(run: str) -> dict[str, dict] | None:
    path = RUNS_DIR / run / "outputs.jsonl"
    if not path.exists():
        return None
    return {json.loads(ln)["id"]: json.loads(ln) for ln in path.read_text().splitlines() if ln}


def results_figure() -> Path:
    stats.GROUNDED = True
    base = _records(BASE)
    labels, means, lo, hi = [], [], [], []
    for label, run in ARMS.items():
        recs = _records(run)
        if recs is None:
            continue
        cmp = stats.compare(recs, base, "arm", "base")
        d = cmp["grounded_score_diff_all_functions"]
        m = cmp["grounded_score"]["arm"]
        labels.append(label)
        means.append(m)
        lo.append(m - (d["mean"] - d["ci95"][0]))
        hi.append(m + (d["ci95"][1] - d["mean"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    colors = ["#888" if lab.startswith("base") else "#2a6f97" for lab in labels]
    ax.bar(x, means, color=colors, width=0.6)
    ax.errorbar(
        x,
        means,
        yerr=[
            [m - a for m, a in zip(means, lo, strict=True)],
            [b - m for m, b in zip(means, hi, strict=True)],
        ],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.axhline(means[0], color="#888", ls="--", lw=1)
    ax.set_xticks(list(x), labels, fontsize=9)
    ax.set_ylabel("grounded score (kills / live mutants, 0 if invalid)")
    ax.set_ylim(0.5, 0.72)
    ax.set_title("Qwen3.5-4B, test split n=315; bars = mean, whiskers = paired 95% CI vs base")
    for i, (m, b) in enumerate(zip(means, hi, strict=True)):
        ax.annotate(f"{m:.3f}", (i, b + 0.004), ha="center", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "results.png"
    fig.savefig(out, dpi=160)
    return out


def devcurve_figure() -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, path in DEV.items():
        if not path.exists():
            continue
        curve = json.loads(path.read_text())
        pts = sorted((int(k), v["grounded_score"]) for k, v in curve.items() if k != "best")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
    ax.axhline(DEV_BASE, color="#888", ls="--", lw=1, label="base zero-shot")
    ax.set_xlabel("training iteration (micro-batches)")
    ax.set_ylabel("dev-171 grounded score")
    ax.set_title("Checkpoint selection: dev split, grounded harness")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "devcurves.png"
    fig.savefig(out, dpi=160)
    return out


def main(argv: list[str]) -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(results_figure())
    print(devcurve_figure())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
