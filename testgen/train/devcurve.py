"""T5 early stopping signal (FT12): harness score of every checkpoint on a dev slice.

    uv run --group models python -m testgen.train.devcurve --run models/adapters/<run> --limit 60

For each ``NNNNNNN_adapters.safetensors`` in the adapter directory a
checkpoint directory ``ckpt-NNNNNNN/`` is created (adapter_config.json copied,
weights symlinked) so mlx-lm can load it, then the evaluator runs zero-shot on
the first ``--limit`` dev functions under the eval budget. Results go to
``<run>/devcurve.json``; the best checkpoint by validity (ties: mutation
score) is named there. With ``--grounded`` (D030) the evaluator also
oracle-fills, and the best checkpoint is the one with the highest mean
grounded score. Validation loss is logged by the trainer but is not
the selection signal.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from config import RUNS_DIR
from testgen import baselines

CKPT = re.compile(r"^(\d{7})_adapters\.safetensors$")


def checkpoints(run: Path) -> list[tuple[str, Path]]:
    out = []
    for f in sorted(run.iterdir()):
        if m := CKPT.match(f.name):
            d = run / f"ckpt-{m.group(1)}"
            d.mkdir(exist_ok=True)
            shutil.copy(run / "adapter_config.json", d / "adapter_config.json")
            link = d / "adapters.safetensors"
            if not link.exists():
                link.symlink_to(f.resolve())
            out.append((m.group(1), d))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--model", default="4b-bf16")
    ap.add_argument("--grounded", action="store_true", help="D030: select by grounded score")
    ap.add_argument("--every", type=int, default=1, help="evaluate every Nth checkpoint only")
    args = ap.parse_args(argv)
    run = Path(args.run)
    curve_path = run / "devcurve.json"
    curve = json.loads(curve_path.read_text()) if curve_path.exists() else {}
    for i, (step, d) in enumerate(checkpoints(run), 1):
        if step in curve or i % args.every:
            continue  # resumable; --every thins the curve when checkpoints are dense
        print(f"== checkpoint {step}", flush=True)
        baselines.main(
            [
                "--pool",
                "dev",
                "--limit",
                str(args.limit),
                "--models",
                args.model,
                "--conditions",
                "zero",
                "--adapter",
                str(d),
                "--tag",
                f"{run.name}-ckpt{step}",
                *(["--grounded"] if args.grounded else []),
            ]
        )
        latest = max(RUNS_DIR.glob(f"baselines-dev-{run.name}-ckpt{step}-*"))
        res = json.loads((latest / "manifest.json").read_text())["results"][f"{args.model}/zero"]
        curve[step] = {
            "run": latest.name,
            "validity": res["validity_rate"],
            "mutation_score": res["mean_mutation_score"],
            "grounded_score": res.get("mean_grounded_score"),
        }
        curve_path.write_text(json.dumps(curve, indent=1))
    if args.grounded:
        best = max(curve, key=lambda s: curve[s]["grounded_score"] or 0.0)
    else:
        best = max(curve, key=lambda s: (curve[s]["validity"], curve[s]["mutation_score"]))
    curve_path.write_text(json.dumps({**curve, "best": best}, indent=1))
    print(json.dumps(curve, indent=1), f"\nbest checkpoint: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
