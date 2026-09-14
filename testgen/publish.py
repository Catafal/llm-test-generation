"""Publish the shipped adapters to the Hugging Face Hub, or pull them back.

    uv run python -m testgen.publish push [--user <hf user>]    # needs `hf auth login`
    uv run python -m testgen.publish pull [--user <hf user>]    # anonymous download

Each adapter becomes one model repo <user>/<name> holding the checkpoint's
``adapters.safetensors`` + ``adapter_config.json`` (mlx-lm format), the
training manifest, and a README made of a YAML header (base model, licence
CC BY-NC 4.0 inherited from KodCode) plus ``docs/model-card.md``. ``pull``
restores them under ``models/adapters/<run>/ckpt-<step>/`` so ``make demo``
and ``make eval`` work on a fresh clone.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, whoami

from config import ROOT

ADAPTERS = {
    # hub repo name -> (local run dir, checkpoint step)
    "qwen3.5-4b-testgen-lora-ext12k": ("lora-4b-ext12k", "0002400"),
    "qwen3.5-4b-testgen-lora-ext4k": ("lora-4b-ext", "0003200"),
}
HEADER = """---
base_model: mlx-community/Qwen3.5-4B-bf16
library_name: mlx
license: cc-by-nc-4.0
tags: [lora, mlx, pytest, test-generation, mutation-testing]
datasets: [KodCode/KodCode-V1]
---

# {name}

LoRA adapter for Qwen3.5-4B that writes pytest suites, evaluated behind an
execution-grounded harness. Load with `mlx_lm.load("mlx-community/Qwen3.5-4B-bf16",
adapter_path=<this repo>)`. Code, harness and evaluation:
https://github.com/Catafal/llm-test-generation . Non-commercial (KodCode licence).

"""


def _ckpt_dir(run: str, step: str) -> Path:
    return ROOT / "models" / "adapters" / run / f"ckpt-{step}"


def push(user: str) -> int:
    api = HfApi()
    card = (ROOT / "docs" / "model-card.md").read_text()
    for name, (run, step) in ADAPTERS.items():
        src = _ckpt_dir(run, step)
        repo = f"{user}/{name}"
        api.create_repo(repo, repo_type="model", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            shutil.copy(src / "adapters.safetensors", stage / "adapters.safetensors")
            shutil.copy(src / "adapter_config.json", stage / "adapter_config.json")
            shutil.copy(src.parent / "manifest.json", stage / "training-manifest.json")
            (stage / "README.md").write_text(HEADER.format(name=name) + card)
            api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="model")
        print(f"pushed {src.relative_to(ROOT)} -> https://huggingface.co/{repo}")
    return 0


def pull(user: str) -> int:
    for name, (run, step) in ADAPTERS.items():
        dst = _ckpt_dir(run, step)
        dst.mkdir(parents=True, exist_ok=True)
        snap = Path(
            snapshot_download(
                f"{user}/{name}", allow_patterns=["adapters.safetensors", "adapter_config.json"]
            )
        )
        for f in ("adapters.safetensors", "adapter_config.json"):
            shutil.copy(snap / f, dst / f)
        print(f"pulled {user}/{name} -> {dst.relative_to(ROOT)}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--user", default="", help="Hub user; default: the logged-in account")
    args = ap.parse_args(argv)
    user = args.user or whoami()["name"]
    return (push if args.action == "push" else pull)(user)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
