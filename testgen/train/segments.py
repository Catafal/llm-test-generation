"""Segmented LoRA training around the mlx-lm Metal descriptor leak (D024 addendum).

    uv run --group models python -m testgen.train.segments --run models/adapters/lora-4b-r16

mlx-lm#1185: LoRA training on qwen3_5 leaks Metal buffer descriptors and dies
with ``[metal::malloc] Resource limit (499000) exceeded`` after ~220
micro-iterations regardless of memory. Each segment here is a fresh process
of ``segment`` micro-iterations that resumes the previous adapter weights, so
the descriptor count restarts from zero.

What resuming loses: mlx-lm saves adapter weights only, so Adam moments
restart every segment and the YAML lr_schedule would restart too. The driver
therefore computes the *schedule's* learning rate at each segment's midpoint
(warmup then cosine decay over the total optimizer steps) and trains the
segment at that constant rate; segment 0 keeps the YAML schedule so warmup
happens normally. Checkpoints land in ``<run>/NNNNNNN_adapters.safetensors``
exactly as a single run would write them, so ``devcurve.py`` needs no change.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from config import ROOT

CONFIG = ROOT / "configs" / "lora-4b.yaml"


def schedule_lr(step: float, cfg: dict) -> float:
    """Learning rate the YAML schedule would give at optimizer step ``step``."""
    init, decay_steps, end = cfg["lr_schedule"]["arguments"]
    warmup = cfg["lr_schedule"].get("warmup", 0)
    if step < warmup:
        return init * step / warmup
    t = min(step - warmup, decay_steps)
    return end + (init - end) * 0.5 * (1 + math.cos(math.pi * t / decay_steps))


def run_segment(k: int, run: Path, cfg: dict, segment: int, accum: int) -> Path:
    """Train micro-iterations [k*segment, (k+1)*segment) in a fresh process."""
    seg_dir = run / f"seg-{k}"
    seg_cfg = dict(cfg)
    seg_cfg.update({"iters": segment, "adapter_path": str(seg_dir), "save_every": segment})
    if k > 0:
        step_mid = (k * segment + segment / 2) / accum
        seg_cfg.pop("lr_schedule", None)
        seg_cfg["learning_rate"] = schedule_lr(step_mid, cfg)
        seg_cfg["resume_adapter_file"] = str(run / f"{k * segment:07d}_adapters.safetensors")
    cfg_path = run / f"seg-{k}.yaml"
    cfg_path.write_text(yaml.safe_dump(seg_cfg))
    lr = seg_cfg.get("learning_rate")
    print(f"== segment {k}: iters {k * segment}-{(k + 1) * segment}, lr {lr}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "testgen.train.train", "-c", str(cfg_path)], cwd=ROOT, check=True
    )
    out = run / f"{(k + 1) * segment:07d}_adapters.safetensors"
    shutil.copy(seg_dir / "adapters.safetensors", out)
    shutil.copy(seg_dir / "adapters.safetensors", run / "adapters.safetensors")
    shutil.copy(seg_dir / "adapter_config.json", run / "adapter_config.json")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--segment", type=int, default=120)
    ap.add_argument("--start", type=int, default=0, help="first segment index (resume)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(CONFIG.read_text())
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    total, accum = cfg["iters"], cfg["grad_accumulation_steps"]
    n_segments = math.ceil(total / args.segment)
    for k in range(args.start, n_segments):
        run_segment(k, run, cfg, args.segment, accum)
    (run / "segments.json").write_text(
        json.dumps({"segment": args.segment, "n_segments": n_segments, "reason": "mlx-lm#1185"})
    )
    print("SEGMENTS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
