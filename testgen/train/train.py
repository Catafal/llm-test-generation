"""T5 entry point: mlx-lm LoRA with a fast path for frozen Gated DeltaNet layers.

    uv run --group models python -m testgen.train.train -c configs/lora-4b.yaml \
        --adapter-path models/adapters/<run>

Why this wrapper exists (D024). In training mode mlx-lm's Qwen3.5 DeltaNet
layer abandons its Metal kernel for a per-token Python loop whose float32
recurrent state is kept for every step for the backward pass. On a 4B with
32 layers that is >24 GB and ~12 tok/s at batch 1 (measured 2026-09-07;
upstream: ml-explore/mlx#3539, mlx-lm#1206). Layers *below* the first LoRA
adapter never need a gradient, so they can keep the kernel. This wrapper:

1. loads the YAML config exactly as ``mlx_lm.lora`` does;
2. after mlx-lm inserts the adapters into the last ``num_layers`` layers,
   swaps every frozen DeltaNet layer to a subclass that reports
   ``training=False`` (the only effect: ``use_kernel=True``; the layer has no
   dropout);
3. writes a run manifest next to the adapter (config sha, dataset sha, base
   model, git commit) per FT3;
4. disables mx.compile. mlx-lm compiles the whole training step; at batch 1
   every distinct sequence length is a new shape-specialised compilation of a
   graph holding the per-token DeltaNet loop, and the cached compilations
   accumulate Metal buffers until ``[metal::malloc] Resource limit (499000)
   exceeded`` (mlx-lm#1185; died at micro-iterations 225 and 95 here).
"""

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import mlx.core as mx
import yaml
from mlx_lm import lora
from mlx_lm.models.qwen3_5 import GatedDeltaNet

from config import ROOT


class FrozenGatedDeltaNet(GatedDeltaNet):
    """A DeltaNet layer that always takes the inference kernel (no gradient needed)."""

    @property
    def training(self) -> bool:
        return False


def fast_frozen_layers(model, num_layers: int) -> int:
    """Swap frozen DeltaNet layers to the kernel path. Returns how many were swapped."""
    layers = model.layers
    frozen = layers[: len(layers) - num_layers] if num_layers > 0 else []
    n = 0
    for layer in frozen:
        if getattr(layer, "is_linear", False):
            layer.linear_attn.__class__ = FrozenGatedDeltaNet
            n += 1
    return n


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(args) -> None:
    data = Path(args.data)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    manifest = {
        "config": args.config,
        "config_sha256": _sha(Path(args.config)) if args.config else None,
        "model": args.model,
        "data": str(data),
        "train_sha256": _sha(data / "train.jsonl") if (data / "train.jsonl").exists() else None,
        "num_layers": args.num_layers,
        "lora_parameters": args.lora_parameters,
        "iters": args.iters,
        "seed": args.seed,
        "git_sha": git.stdout.strip(),
    }
    Path(args.adapter_path).mkdir(parents=True, exist_ok=True)
    (Path(args.adapter_path) / "manifest.json").write_text(json.dumps(manifest, indent=1))


def load_args(argv: list[str]) -> types.SimpleNamespace:
    """Same precedence as mlx_lm.lora.main: CLI > YAML > CONFIG_DEFAULTS."""
    parser: argparse.ArgumentParser = lora.build_parser()
    parser.add_argument(
        "--keep-compile", action="store_true", help="dense models: keep mx.compile (faster)"
    )
    args = vars(parser.parse_args(argv))
    if args.get("config"):
        with open(args["config"]) as f:
            for k, v in yaml.load(f, lora.yaml_loader).items():
                if args.get(k) is None:
                    args[k] = v
    for k, v in lora.CONFIG_DEFAULTS.items():
        if args.get(k) is None:
            args[k] = v
    return types.SimpleNamespace(**args)


def main(argv: list[str]) -> int:
    args = load_args(argv)
    write_manifest(args)
    if not args.keep_compile:
        mx.disable_compile()  # see module docstring, point 4 (Qwen3.5 hybrid only)

    def train_model(a, model, train_set, valid_set, cb=None):
        # mlx-lm inserts adapters inside; swap frozen DeltaNet layers first, since
        # nothing below the first adapter is on the gradient tape.
        n = fast_frozen_layers(model, a.num_layers)
        print(f"frozen DeltaNet layers on the kernel path: {n}", flush=True)
        return _original_train_model(a, model, train_set, valid_set, cb)

    lora.train_model = train_model
    lora.run(args)
    return 0


_original_train_model = lora.train_model

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
