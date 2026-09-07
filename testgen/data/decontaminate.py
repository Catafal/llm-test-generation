"""Decontaminate harvested candidates against MBPP and against themselves (D012, D017).

    uv run python -m testgen.data.decontaminate            # real embeddings (--group decontam)
    uv run python -m testgen.data.decontaminate --no-embed # layers 1 and 3 only
    uv run python -m testgen.data.decontaminate --dir data/train --against heldout  # D022

Reads  <dir>/candidates.jsonl
Writes <dir>/pool.jsonl           kept functions + family id
       <dir>/DECONTAMINATION.md   per-layer thresholds, counts, borderline pairs
       <dir>/NOTICE.md            every source repo and licence
       <dir>/manifest.json        floor, counts, thresholds, reference sha, model

``--against mbpp`` (default, held-out pool) or ``--against heldout`` (training
pool, D022: flagged against *either* split of data/heldout/pool.jsonl).
Families group by repository *owner*, not just repository (D022).

Removal rules: a candidate flagged against *any* MBPP function by any layer is
dropped. Among candidates, near-duplicates are merged into one family and only
the first (by id) is kept. Review-band cosine pairs are kept but listed.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from config import ROOT
from testgen.data import mbpp
from testgen.data.harvest import is_test_code
from testgen.data.similarity import (
    AST_JACCARD_FLAG,
    COSINE_FLAG,
    COSINE_REVIEW,
    NGRAM_N,
    Embedder,
    ast_hash,
    ast_shingles,
    cosine_matrix,
    jaccard,
    ngram_overlap,
    ngrams,
)

HELDOUT = ROOT / "data" / "heldout"


class UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def _flags_against(
    held: list[dict], train: list[dict], embed: Embedder | None
) -> dict[str, list[str]]:
    """held id -> reasons it matches something in train (empty list = clean)."""
    reasons: dict[str, list[str]] = defaultdict(list)
    t_ng = [ngrams(t["source"]) for t in train]
    t_hash = {ast_hash(t["source"]): t["id"] for t in train}
    t_sh = [ast_shingles(t["source"]) for t in train]
    for h in held:
        h_ng, h_hash, h_sh = ngrams(h["source"]), ast_hash(h["source"]), ast_shingles(h["source"])
        if h_hash in t_hash:
            reasons[h["id"]].append(f"ast-exact {t_hash[h_hash]}")
        for t, ng, sh in zip(train, t_ng, t_sh, strict=True):
            if k := ngram_overlap(h_ng, ng):
                reasons[h["id"]].append(f"ngram{NGRAM_N} x{k} {t['id']}")
            if (j := jaccard(h_sh, sh)) >= AST_JACCARD_FLAG:
                reasons[h["id"]].append(f"ast-jaccard {j:.2f} {t['id']}")
    if embed is not None:
        cos = cosine_matrix(embed([h["source"] for h in held]), embed([t["source"] for t in train]))
        for i, h in enumerate(held):
            for j, t in enumerate(train):
                if cos[i][j] >= COSINE_FLAG:
                    reasons[h["id"]].append(f"cosine {cos[i][j]:.2f} {t['id']}")
                elif cos[i][j] >= COSINE_REVIEW:
                    reasons[h["id"]].append(f"REVIEW cosine {cos[i][j]:.2f} {t['id']}")
    return reasons


def _owner(repo_url: str) -> str:
    """``https://github.com/<owner>/<repo>`` -> owner; anything shorter is its own owner."""
    parts = repo_url.rstrip("/").split("/")
    return parts[3] if len(parts) > 4 else repo_url


def _families(held: list[dict], embed: Embedder | None) -> tuple[dict[str, str], list[str]]:
    """Union-find: same owner, or any layer flags the pair. Returns (id -> family, merge log)."""
    uf, log = UnionFind([h["id"] for h in held]), []
    by_owner: dict[str, str] = {}
    for h in held:
        owner = _owner(h["repo"])
        if owner in by_owner:
            uf.union(h["id"], by_owner[owner])
        by_owner.setdefault(owner, h["id"])
    ng = [ngrams(h["source"]) for h in held]
    sh = [ast_shingles(h["source"]) for h in held]
    hs = [ast_hash(h["source"]) for h in held]
    cos = None
    if embed is not None:
        e = embed([h["source"] for h in held])
        cos = cosine_matrix(e, e)
    for i in range(len(held)):
        for j in range(i + 1, len(held)):
            why = None
            if hs[i] == hs[j]:
                why = "ast-exact"
            elif ngram_overlap(ng[i], ng[j]):
                why = f"ngram{NGRAM_N}"
            elif (jac := jaccard(sh[i], sh[j])) >= AST_JACCARD_FLAG:
                why = f"ast-jaccard {jac:.2f}"
            elif cos is not None and cos[i][j] >= COSINE_FLAG:
                why = f"cosine {cos[i][j]:.2f}"
            if why:
                uf.union(held[i]["id"], held[j]["id"])
                log.append(f"{why}: {held[i]['id']}  ~  {held[j]['id']}")
    return {h["id"]: uf.find(h["id"]) for h in held}, log


def _write_report(
    path: Path, stats: dict, flagged: dict[str, list[str]], merges: list[str]
) -> None:
    hard = {k: [r for r in v if not r.startswith("REVIEW")] for k, v in flagged.items()}
    review = {k: [r for r in v if r.startswith("REVIEW")] for k, v in flagged.items()}
    lines = [
        f"# Decontamination Report — {stats['dir']} vs {stats['against']}",
        "",
        f"Generated {date.today().isoformat()}. Floor date {stats['floor']}. "
        f"Candidates {stats['candidates']}, kept {stats['kept']}, families {stats['families']}.",
        "",
        "## Layers and thresholds",
        "",
        f"- n-gram: word-level, n={NGRAM_N}; any shared n-gram with a reference function flags.",
        f"- AST: identifiers→slots, docstrings dropped, constants→type; exact hash flags; "
        f"k=5 node shingles Jaccard ≥ {AST_JACCARD_FLAG} flags.",
        f"- embedding: {stats['embedding_model'] or 'not run'}; cosine ≥ {COSINE_FLAG} flags, "
        f"{COSINE_REVIEW}–{COSINE_FLAG} listed for human review.",
        "- family: same repository owner, plus union-find over every flagged candidate pair; "
        "one function kept per family cluster of near-duplicates.",
        "",
        f"## Removed against {stats['against']}",
        "",
        f"{sum(1 for v in hard.values() if v)} candidates removed.",
        "",
        *[f"- `{k}`: " + "; ".join(v[:3]) for k, v in sorted(hard.items()) if v],
        "",
        "## Review band (kept, human check requested)",
        "",
        *([f"- `{k}`: " + "; ".join(v[:3]) for k, v in sorted(review.items()) if v] or ["none"]),
        "",
        "## Near-duplicate merges within the pool",
        "",
        *([f"- {m}" for m in merges] or ["none"]),
        "",
        "## Limitations",
        "",
        "- Layer 1 and 3 are exact/structural; a semantic rewrite passes both.",
        "- The embedding model has blind spots; the review band exists for that reason.",
        "- The cutoff for Qwen3.5 is year-only; the floor is a margin, not a proof.",
        "- Renames are not followed in the introducing-commit check (see harvest.py).",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true", help="skip layer 2")
    ap.add_argument("--floor", default="2026-06-01")
    ap.add_argument("--dir", default=str(HELDOUT), help="pool directory holding candidates.jsonl")
    ap.add_argument("--against", choices=["mbpp", "heldout"], default="mbpp")
    args = ap.parse_args(argv)
    out_dir = Path(args.dir).resolve()

    held = [
        json.loads(ln)
        for ln in (out_dir / "candidates.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    n_raw = len(held)
    held = [h for h in held if not is_test_code(h["path"], h["function"])]  # defensive re-filter
    print(f"dropped {n_raw - len(held)} test-code candidates")
    if args.against == "mbpp":
        ref_path, ref_sha = mbpp.fetch()
        train = mbpp.load(ref_path)
    else:  # both held-out splits: the training pool must be disjoint from test *and* dev
        ref_path = HELDOUT / "pool.jsonl"
        train = [json.loads(ln) for ln in ref_path.read_text().splitlines() if ln.strip()]
        ref_sha = __import__("hashlib").sha256(ref_path.read_bytes()).hexdigest()
    embed = (
        None
        if args.no_embed
        else __import__("testgen.data.similarity", fromlist=["jina_embedder"]).jina_embedder()
    )
    print(
        f"{len(held)} candidates vs {len(train)} {args.against} functions; "
        f"embeddings {'off' if embed is None else 'on'}"
    )

    flagged = _flags_against(held, train, embed)
    clean = [
        h for h in held if not any(not r.startswith("REVIEW") for r in flagged.get(h["id"], []))
    ]
    families, merges = _families(clean, embed)
    kept, seen_dup_cluster = [], set()
    dup_cluster = {}  # id -> cluster root among *near-duplicate* merges only (not repo grouping)
    dup_uf = UnionFind([h["id"] for h in clean])
    for m in merges:
        a, b = m.split(": ", 1)[1].split("  ~  ")
        dup_uf.union(a, b)
    for h in sorted(clean, key=lambda h: h["id"]):
        root = dup_uf.find(h["id"])
        if root in seen_dup_cluster:
            continue
        seen_dup_cluster.add(root)
        dup_cluster[h["id"]] = root
        kept.append({**h, "family": families[h["id"]]})

    out_dir.mkdir(exist_ok=True)
    (out_dir / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    repos = sorted({(r["repo"], r["licence"]) for r in kept})
    (out_dir / "NOTICE.md").write_text(
        "# Sources\n\nFunction bodies in pool.jsonl are reproduced verbatim under their "
        "original licences.\n\n" + "".join(f"- {u} — {lic}\n" for u, lic in repos)
    )
    stats = {
        "dir": out_dir.relative_to(ROOT).as_posix(),
        "against": args.against,
        "floor": args.floor,
        "candidates": len(held),
        "removed_vs_reference": len(held) - len(clean),
        "merged_duplicates": len(clean) - len(kept),
        "kept": len(kept),
        "families": len(set(r["family"] for r in kept)),
        "repos": len(repos),
        "reference": mbpp.MBPP_URL
        if args.against == "mbpp"
        else ref_path.relative_to(ROOT).as_posix(),
        "reference_sha256": ref_sha,
        "thresholds": {
            "ngram_n": NGRAM_N,
            "ast_jaccard": AST_JACCARD_FLAG,
            "cosine_flag": COSINE_FLAG,
            "cosine_review": COSINE_REVIEW,
        },
        "embedding_model": None if embed is None else "jinaai/jina-embeddings-v2-base-code",
        "generated": date.today().isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(stats, indent=1))
    _write_report(out_dir / "DECONTAMINATION.md", stats, flagged, merges)
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k not in ("thresholds", "mbpp_url")}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
