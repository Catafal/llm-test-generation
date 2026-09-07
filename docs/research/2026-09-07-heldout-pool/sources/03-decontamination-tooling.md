# Source report 03 — Three-layer decontamination tooling

Research agent report (Claude Sonnet), 2026-09-07. Unedited except for this header.
Claims marked unverified by the agent remain unverified.

---

## 1. N-gram / token overlap layer

- **GPT-3**: **13-gram** overlap as the contamination signal ([Survey on Data Contamination](https://arxiv.org/pdf/2502.14425), [BigCodeBench](https://arxiv.org/pdf/2406.15877)). **GPT-4**: 50-character substring overlap.
- **Qwen2.5-Coder**: **10-gram word-level overlap**; "any training data with a 10-gram overlap with the test data was removed" ([Qwen2.5-Coder Technical Report](https://arxiv.org/html/2409.12186v3)).
- **BigCodeBench**: both 10-gram (strict) and 13-gram (looser) against ODEX/StackOverflow; ≤2.5% overlap under 10-gram.
- **BigCode/StarCoder**: (a) exact-substring `find_substrings.py` in `bigcode-dataset/decontamination` ([repo](https://github.com/bigcode-project/bigcode-dataset/tree/main/decontamination)); (b) MinHash+LSH near-dedup for train-train ([HF blog](https://huggingface.co/blog/dedup)). MinHash params: regex split on non-alphanumerics, `ngram_size=5`, `num_perm=256`, Jaccard `threshold=0.7`, union-find clusters ([source](https://github.com/bigcode-project/bigcode-dataset/blob/main/near_deduplication/minhash_deduplication.py)).
- **SWE-bench-style**: SHA-256 file hash + 15-gram code-subsequence check against gold patches, 0.5 Jaccard on issue text ([field guide](https://medium.com/@adnanmasood/code-generation-repository-level-software-engineering-benchmarks-a-field-guide-to-llm-benchmarks-330bc3015d80)).
- Implementations: `datasketch` (MinHash/LSH), `text-dedup` (MinHash, SimHash, suffix-array exact substring; defaults `ngram_size=5, num_perm=240, threshold=0.7`) ([repo](https://github.com/ChenghaoMou/text-dedup)).
- **At 1,300 short functions, brute-force pairwise n-gram Jaccard is tractable; LSH is unnecessary.**

## 2. Embedding cosine-similarity layer

| Model | Params | Dim | Licence | Notes |
|---|---|---|---|---|
| `jinaai/jina-embeddings-v2-base-code` | 161M | 768 | Apache 2.0 | 30 languages incl. Python; CPU-runnable ([HF](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)) |
| `nomic-ai/nomic-embed-code` | 7B | not confirmed | Apache 2.0 | strong but heavy ([HF](https://huggingface.co/nomic-ai/nomic-embed-code)) |
| `codesage/codesage-small-v2` | 130M | 1024 | not confirmed | CPU-friendly alternative ([HF](https://huggingface.co/codesage/codesage-small-v2)) |
| `Qodo/Qodo-Embed-1-1.5B` | 1.5B | not confirmed | not confirmed | ([HF](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B)) |
| Voyage-code-3 | — | — | API-only | excluded |

**Recommendation:** `jina-embeddings-v2-base-code`. **Thresholds:** 0.8 is a common near-dup convention (Open-Platypus attribution unverified); for short Python functions the near-dup band sits higher. **≥0.90 flag, 0.80–0.90 review band** ([emergentmind survey](https://www.emergentmind.com/topics/cosine-similarity-threshold)).

## 3. AST / structural similarity layer

- **Dolos** (tree-sitter, winnowing, identifier masking) ([paper](https://arxiv.org/pdf/2402.10853)); **JPlag** (Greedy String Tiling); **MOSS** (hosted, legacy); **tree-sitter k-gram fingerprinting**; **plain `ast` normalisation** (`pycode_similar` does this) ([GitHub](https://github.com/fyrestone/pycode_similar)); **CodeBLEU** AST-match component ([overview](https://www.emergentmind.com/topics/codebleu)).
- **MVP path:** parse with `ast`; rename identifiers to positional slots; strip docstrings; bucket constants; (a) hash normalised `ast.dump()` for exact structural match; (b) k-shingle (k≈4–6) the normalised node-type sequence, Jaccard/MinHash for near-structural match. **Threshold:** exact hash = duplicate; **Jaccard ≥0.85** = near-duplicate (no canonical number in literature).

## 4. Family-split logic

- **MBPP**: `task_id`, `source_file`, `prompt`, `code`, `test_list`; `sanitized-mbpp.json` 427-problem subset ([README](https://github.com/google-research/google-research/blob/master/mbpp/README.md), [HF](https://huggingface.co/datasets/google-research-datasets/mbpp)). `task_id` is close to atomic.
- **SWE-bench style**: forks and related repos grouped into one partition; content-based checks layered on top because "repository names alone are insufficient."
- **Rule:** family = `(source_dataset, source_file_or_module)` merged transitively with any layer-2/3 flagged pair via union-find; split unit is the connected component.

## 5. Reporting

- DataComp-LM requires a decontamination report with per-benchmark contamination stats ([paper](https://arxiv.org/pdf/2406.11794)); Dolma states exact rule and magnitude ([paper](https://arxiv.org/pdf/2402.00159)); "developers should report train-test overlap" as first-class ([arXiv 2410.08385](https://arxiv.org/pdf/2410.08385)).

**Template:**
```
# Decontamination Report — train (n=…) vs held-out (n=…)
## 1. Layer 1 — n-gram: method, n, pairs flagged/removed, 2 borderline examples
## 2. Layer 2 — embedding: model, thresholds, flagged/reviewed/removed, 2 borderline examples
## 3. Layer 3 — AST: method, thresholds, flagged/removed, 2 borderline examples
## 4. Family split: definition, families collapsed across boundary, final counts
## 5. Summary: removed from train (n, %), removed from held-out (n, %), final counts, limitations
```

## Recommended configuration

| Layer | Tool | Params | Threshold | Runtime, M4 Pro, 1,300 fns |
|---|---|---|---|---|
| n-gram | pure Python set intersection | word-level n=10 (Qwen2.5-Coder); n=13 secondary | any shared n-gram ⇒ flag | seconds |
| embedding | `jina-embeddings-v2-base-code` via sentence-transformers | 768-dim | ≥0.90 flag; 0.80–0.90 review | < 1 min |
| AST | `ast` normaliser + hash + k=5 shingles | identifier→var_N, strip docstrings/constants | exact hash = dup; Jaccard ≥0.85 | seconds |
| family | union-find over flagged pairs + provenance | — | — | seconds |

**Total: under 5 minutes end-to-end.**

## Could not verify
1. "Open-Platypus used 0.8 cosine" — no primary source found.
2. `nomic-embed-code` embedding dimension.
3. `Qodo-Embed-1-1.5B` dimension.
4. LiveCodeBench's exact n-gram parameters.
5. CodeSage licence.
