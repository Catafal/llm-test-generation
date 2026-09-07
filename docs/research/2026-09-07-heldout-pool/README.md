# Held-Out Pool: Cutoff, Sources and Decontamination (T5)

**Date:** 2026-09-07
**Question:** How do we assemble 150–300 pure Python functions that Qwen3.5-9B
cannot have seen, with licences that allow committing them, and prove it?
**Method:** Three parallel Sonnet research passes; raw reports in `sources/`.
This is the synthesis and the pipeline proposal for D012's implementation.

| Source | Covers |
|---|---|
| [01-qwen35-training-cutoff.md](sources/01-qwen35-training-cutoff.md) | What is known about Qwen3.5's data cutoff |
| [02-post-cutoff-function-sources.md](sources/02-post-cutoff-function-sources.md) | Where post-cutoff functions can come from, with licences |
| [03-decontamination-tooling.md](sources/03-decontamination-tooling.md) | n-gram, embedding, AST layers; family split; report template |

---

## 1. The cutoff is year-only, and that changes the design

No primary source states a month-level cutoff for Qwen3.5. The model cards,
the blog, and the only Qwen3.5-branded arXiv paper (the Omni report) are
silent. The strongest signal is Alibaba's own API system prompt saying the
model "claims 2026 knowledge", via a third-party tracker. Asking the model is
unreliable; Qwen models hallucinate their cutoff. Release dates are firm:
the 9B and 4B shipped on 2026-03-02.

Consequence: a "post-cutoff" pool cannot be dated to a month. The honest
construction is a **safe floor with margin** and a written statement of the
uncertainty. The cutoff report proposes **2026-06-01** as the floor, three
months past the last release, or 2026-09-01 for maximum conservatism. The
later floor leaves a one-week window and no usable yield, so the practical
choice is June 1 with the caveat recorded in the datasheet.

For the baseline-only reference row, Qwen2.5-Coder-7B-Instruct's cutoff is
June 2024 per a Qwen maintainer, so anything after June 2026 clears it by
two years.

## 2. Where the functions can come from

The date-stamped benchmarks are all stale for this purpose: LiveCodeBench v6
ends April 2025, CodeElo and LiveBench-coding similarly. They cannot supply
2026 code today. The puzzle sites fail on licence (LeetCode, Codewars),
non-commercial terms (Project Euler), or redistribution norms (Advent of
Code). **Exercism is permissively licensed but its exercises predate the
cutoff**, so it is not a supplement either, except for exercises added after
June 2026, which will be few.

That leaves two live sources:

1. **GitHub** — repositories created after the floor, MIT/BSD/Apache, Python.
   Highest yield. The known trap: a new repository is not proof of new code
   (forks, templates, vendored utilities). The proof must be **per function**:
   the commit that introduced the file, dated after the floor, plus a novelty
   check against the training pool.
2. **PyPI** — packages first released after the floor, via the
   `pypi-json-data` mirror. Production-style code, but licence classifiers
   are unreliable and purity yield is low because packages do I/O.

No off-the-shelf purity filter exists. It is one custom AST pass: stdlib-only
imports, no `global`/`nonlocal`, no I/O or nondeterministic builtins, and a
docstring present. Complexity should not be proxied by branch count when the
real criterion is available: **run the mutation engine under eval-v1 and
require ≥8 live mutants**, the pilot's saturation rule.

## 3. Decontamination configuration

All three layers run in under five minutes on the Mac at this scale, so
there is no reason to approximate.

| Layer | Config | Flag rule | Precedent |
|---|---|---|---|
| n-gram | word-level, n=10; n=13 as secondary | any shared 10-gram between a held-out function and any training function | Qwen2.5-Coder, GPT-3, BigCodeBench |
| embedding | `jina-embeddings-v2-base-code` (161M, Apache 2.0, 768-dim) | cosine ≥0.90 flag; 0.80–0.90 human review | near-dup convention, raised for short code |
| AST | identifiers → slots, docstrings stripped, constants bucketed; exact hash + k=5 node shingles | exact hash = duplicate; Jaccard ≥0.85 = near-duplicate | Dolos/JPlag principle, MVP via `ast` |
| family | `(source repo, module)` merged by union-find over every flagged pair | split unit is the connected component | SWE-bench grouping |

Held-out is also checked against itself for near-duplicates. The report
follows the DataComp-LM / Dolma pattern: per-layer thresholds, counts
removed, two borderline examples per layer, and a limitations section.

## 4. Proposed pipeline (for Jordi's decisions)

1. **Floor:** 2026-06-01. Recorded with the uncertainty statement.
2. **Harvest:** GitHub Search API, `language:Python created:>=2026-06-01
   license:mit OR apache-2.0 OR bsd-3-clause`, sorted by stars, a few hundred
   repos. Clone shallow, extract every top-level function with a docstring
   via `ast`. Record repo URL, licence, file path, and the introducing
   commit's author date from `git log --diff-filter=A --follow`.
3. **Filter:** purity pass; introducing-commit date ≥ floor; ≥8 live mutants
   under eval-v1; not a test file; no third-party imports.
4. **Decontaminate:** three layers against MBPP (the training pool) and
   against itself; family split; report written.
5. **Datasheet:** per function: URL, commit SHA, licence name and URL, author
   date and why it proves post-floor authorship, AST fingerprint, live-mutant
   count.
6. **Commit:** function bodies verbatim, under a `NOTICE` file listing every
   source repo and licence. MIT/BSD/Apache permit this with attribution.

Expected yield: the purity plus ≥8-mutant filter will discard most functions;
reaching 150–300 keepers likely means scanning 200–500 repositories. Estimated
4–6 hours including a manual spot-check of the keepers.

## 5. What this research did not settle

- Month-level cutoff for Qwen3.5. Nobody publishes it.
- Whether LiveCodeBench rows carry reference solutions. Irrelevant now, since
  its dates are too old, but relevant for a future refresh.
- The exact yield of the GitHub route; the estimate is the agent's.
- Whether a function harvested from a real repo is *correct*. The reference
  is the source as written; mutation testing measures whether tests pin that
  behaviour, not whether the behaviour matches the docstring. Stated in
  limitations.
