"""Model registry and storage manager.

    uv run python -m testgen.models list           # what is on disk, with sizes
    uv run python -m testgen.models pull 9b 4b     # download into models/hf/
    uv run python -m testgen.models rm 4b          # delete one
    uv run python -m testgen.models rm --all       # free everything

All weights live under models/hf/ inside the repo (gitignored). Deleting the
folder loses nothing that cannot be re-pulled; the manifest of every run
records the exact model id used.
"""

import argparse
import shutil
import sys
from pathlib import Path

from config import MODEL_CACHE_DIR

# key -> Hugging Face repo id. Keys are what the CLI and baselines use.
MODELS = {
    "9b": "mlx-community/Qwen3.5-9B-4bit",  # D010 target, inference-only 4-bit
    "4b": "mlx-community/Qwen3.5-4B-4bit",  # D021 target, inference/proposer (4-bit)
    "4b-bf16": "mlx-community/Qwen3.5-4B-bf16",  # D021 fine-tune base (bf16 LoRA, FT9)
    "coder7b": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",  # baseline-only reference row
    "embed": "jinaai/jina-embeddings-v2-base-code",  # decontamination layer 2
}
# jina-v2-code's remote code pulls this sibling repo; delete it with `embed`.
_DEPENDENT = {"embed": ["jinaai/jina-bert-v2-qk-post-norm"]}


def _folder(repo_id: str) -> Path:
    return MODEL_CACHE_DIR / ("models--" + repo_id.replace("/", "--"))


def _size(path: Path) -> int:
    # snapshots/ holds symlinks into blobs/; skip them or every file counts twice
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def cmd_list() -> None:
    total = 0
    print(f"{'key':<9}{'repo':<48}{'on disk':>9}")
    for key, repo in MODELS.items():
        size = _size(_folder(repo)) + sum(_size(_folder(d)) for d in _DEPENDENT.get(key, []))
        total += size
        print(f"{key:<9}{repo:<48}{size / 1e9:>8.1f}G" if size else f"{key:<9}{repo:<48}{'-':>9}")
    print(f"{'':<9}{'total in ' + str(MODEL_CACHE_DIR):<48}{total / 1e9:>8.1f}G")


def cmd_pull(keys: list[str]) -> None:
    from huggingface_hub import snapshot_download  # lazy: honours HF_HUB_CACHE from config

    for key in keys:
        print(f"pulling {key} = {MODELS[key]} -> {MODEL_CACHE_DIR}")
        snapshot_download(MODELS[key])
        for dep in _DEPENDENT.get(key, []):
            snapshot_download(dep)


def cmd_rm(keys: list[str], everything: bool) -> None:
    targets = list(MODELS) if everything else keys
    for key in targets:
        for repo in [MODELS[key], *_DEPENDENT.get(key, [])]:
            folder = _folder(repo)
            if folder.exists():
                shutil.rmtree(folder)
                print(f"removed {folder}")
    if everything and MODEL_CACHE_DIR.exists() and not any(MODEL_CACHE_DIR.iterdir()):
        MODEL_CACHE_DIR.rmdir()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("pull")
    p.add_argument("keys", nargs="+", choices=list(MODELS))
    r = sub.add_parser("rm")
    r.add_argument("keys", nargs="*", choices=list(MODELS))
    r.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        cmd_list()
    elif a.cmd == "pull":
        cmd_pull(a.keys)
    else:
        cmd_rm(a.keys, a.all)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
