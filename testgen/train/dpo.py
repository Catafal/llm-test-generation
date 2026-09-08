"""D026 entry point: DPO with mlx-lm-lora on the model's own pass/fail pairs.

    uv run --group models python -m testgen.train.dpo -c configs/dpo-4b.yaml \
        --adapter-path models/adapters/<run>

Wraps ``mlx_lm_lora.train`` the way ``train.py`` wraps ``mlx_lm.lora``, for
the same Qwen3.5 reasons (D024):
1. ``mx.disable_compile()``: mlx-lm-lora compiles the DPO step too, and the
   per-shape compilations leak Metal buffers on this model.
2. After the adapters are inserted, frozen DeltaNet layers below the first
   adapter are swapped to the inference kernel (no gradient needed there).
3. Optionally ``--chunked-recurrence``: mlx-lm-lora's checkpointed chunked
   recurrence for the adapted DeltaNet layers (memory for longer sequences).
4. A run manifest next to the adapter (FT3).

The reference model is a second frozen copy of the base (mlx-lm-lora loads it
when no reference path is given); it runs forward-only on the kernel path.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import yaml
from mlx_lm_lora import train as lora_train

from config import ROOT
from testgen.train.train import fast_frozen_layers

_original_from_pretrained = lora_train.from_pretrained


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(args) -> None:
    data = Path(args.data)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    manifest = {
        "mode": args.train_mode,
        "config": args.config,
        "config_sha256": _sha(Path(args.config)) if args.config else None,
        "model": args.model,
        "data": str(data),
        "train_sha256": _sha(data / "train.jsonl") if (data / "train.jsonl").exists() else None,
        "num_layers": args.num_layers,
        "lora_parameters": args.lora_parameters,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "iters": args.iters,
        "seed": args.seed,
        "git_sha": git.stdout.strip(),
    }
    Path(args.adapter_path).mkdir(parents=True, exist_ok=True)
    (Path(args.adapter_path) / "manifest.json").write_text(json.dumps(manifest, indent=1))


def load_args(argv: list[str]):
    """Precedence: explicit CLI flag > YAML > parser default > CONFIG_DEFAULTS.

    mlx-lm-lora's parser has real defaults (train_mode "sft", ...), so its own
    "fill YAML only where None" rule silently ignores most YAML keys; here a
    YAML key loses only to a flag actually present on the command line.
    """
    parser: argparse.ArgumentParser = lora_train.build_parser()
    parser.add_argument("--chunked-recurrence", type=int, default=0, help="chunk size, 0 = off")
    args = parser.parse_args(argv)
    explicit = {a.dest for a in parser._actions if any(opt in argv for opt in a.option_strings)}
    if args.config:
        with open(args.config) as f:
            for k, v in yaml.load(f, Loader=lora_train.yaml_loader).items():
                if k not in explicit:
                    setattr(args, k, v)
    for k, v in lora_train.CONFIG_DEFAULTS.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def main(argv: list[str]) -> int:
    args = load_args(argv)
    write_manifest(args)
    mx.disable_compile()
    if args.chunked_recurrence:
        from mlx_lm_lora.recurrent_patch import enable_memory_safe_recurrences

        enable_memory_safe_recurrences(args.chunked_recurrence)

    def from_pretrained(**kwargs):
        model, tokenizer, adapter_file = _original_from_pretrained(**kwargs)
        n = fast_frozen_layers(model, args.num_layers)
        print(f"frozen DeltaNet layers on the kernel path: {n}", flush=True)
        return model, tokenizer, adapter_file

    lora_train.from_pretrained = from_pretrained
    lora_train.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
