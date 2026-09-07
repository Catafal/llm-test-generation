# Source report 02 — Sources of post-cutoff pure Python functions

Research agent report (Claude Sonnet), 2026-09-07. Unedited except for this header.
Claims marked unverified by the agent remain unverified.

---

## 1. LiveCodeBench and date-stamped benchmarks

**LiveCodeBench** (github.com/livecodebench/livecodebench, MIT-licensed repo code): problems pulled from LeetCode/AtCoder/Codeforces and tagged with a release date; releases stack cumulatively — v1 (May 2023–Mar 2024, 400 problems) through v6 (May 2023–Apr 2025, 1055 problems). Filter with `--start_date`/`--end_date` in `lcb_runner.evaluation.compute_scores`. Critically: **the released dataset (`code_generation_lite` on Hugging Face, license tag "cc") does not clearly bundle canonical Python solutions** — only "hidden test cases" are described; needs direct row inspection. As of 2026-09-07, v6's window tops out at **April 2025** — over a year stale relative to an early-2026 cutoff. [GitHub](https://github.com/livecodebench/livecodebench) | [HF dataset](https://huggingface.co/datasets/livecodebench/code_generation_lite) | [site](https://livecodebench.github.io/)

**CodeElo** (arxiv 2501.01257): Codeforces contests May–Nov 2024. Same staleness; no clear redistribution licence for reference solutions.

**LiveBench/coding** (github.com/livebench/livebench, HF `livebench/coding`): latest tracked release 2025-04-25; no confirmation of bundled canonical solutions; no explicit licence for the coding subset.

**Verdict: none of these benchmarks currently reach into 2026.** Candidates for a later refresh, not a right-now pull.

## 2. GitHub (GH Archive / BigQuery / Search API)

GH Archive is mirrored into public BigQuery tables, updated hourly; `bigquery-public-data.github_repos` (files, commits, licenses) exists. The GitHub Search API supports `created:>2026-01-01 language:Python license:mit` qualifiers directly; code search is capped at ~10 req/min, repo search under the 5,000 req/hr ceiling with secondary-rate-limit throttling. **"Created after date" is a weak proxy for "code written after date"**: forks, templates, and vendored utilities routinely land in new repos. A second filter is needed (blame/commit date on the specific function, or a novelty check against big code corpora).

## 3. PyPI

`pypi-data/pypi-json-data` (GitHub) mirrors the full PyPI JSON API into a git repo plus a daily SQLite bundle with `upload_time` per version — filter "first release after date" without hitting the live API. `bigquery-public-data.pypi.distribution_metadata` is the BigQuery equivalent. Package-level licence classifiers are often incomplete/wrong; re-derive from the sdist's LICENSE file.

## 4. Puzzle/kata sites — licensing is the blocker

- **Exercism**: MIT-licensed exercises at `github.com/exercism/python`. Cleanest licence.
- **Rosetta Code**: GFDL 1.2 — awkward for source redistribution.
- **Advent of Code**: policy asks not to redistribute puzzle text or inputs; solution code is not restricted, but you must write your own.
- **Project Euler**: CC BY-NC-SA 4.0 — NonCommercial clause.
- **Codewars**: problem statements are Codewars' IP; no blanket grant.
- **LeetCode**: ToS claims all content including solutions. Hardest no.
- **Codeforces/AtCoder**: no explicit redistribution terms found either way; implicit precedent via LCB/CodeElo.

**Exercism is the only source with an unambiguous, permissive, function-level licence.**

## 5. Extraction tooling

CodeSearchNet `function_parser` (tree-sitter, Python among 6 languages) walks a repo and emits function+docstring pairs; maintained fork at `ncoop57/function_parser`. **No off-the-shelf purity filter exists**: custom AST analysis — non-stdlib imports, `global`/`nonlocal`, impure builtins (`open`, `input`, `print`, unseeded `random`, `time`, `os`, `datetime.now`), and a branch/operator count threshold to ensure ≥8 mutants. One AST pass covers purity and complexity.

## 6. Licence/redistribution and datasheet

- **Safe to commit verbatim**: MIT/BSD/Apache GitHub repos (per-file licence in metadata), Exercism.
- **Reference by URL+hash only**: competitive-programming-derived problems, Advent of Code text/inputs, Project Euler statements.
- **Write-your-own-solution, cite the prompt**: Advent of Code, Project Euler.

Datasheet per function: source URL, commit/release SHA or PyPI version, licence (name + URL to the licence text), extraction date, author date (git commit date / PyPI upload_time / benchmark release date) with a note on why that date proves post-cutoff authorship, and an AST fingerprint for cross-source dedup.

## Ranked recommendation

1. **GitHub via Search API + creation/commit-date filter, MIT/BSD/Apache only** — highest yield; needs the two-stage filter (per-function blame date + novelty/dedup check) and a custom purity+complexity AST filter. A few hours to script; manual spot-check of ~300 candidates to reach 150–300 keepers. Biggest risk: licence misclassification and "new repo, old code."
2. **Exercism** — zero licensing ambiguity, pre-existing docstrings/tests, but finite and many exercises are toy one-liners; likely well under 150 branchy functions. Supplement, not primary.
3. **PyPI new-package extraction via pypi-json-data** — volume and production-style code, but licence re-verification per sdist and low purity yield (packages do I/O).

LiveCodeBench/CodeElo/LiveBench are **not currently viable** for an early-2026 cutoff.

## Unverified
- Whether LiveCodeBench's `code_generation_lite` rows contain canonical solutions vs tests-only.
- CodeElo/LiveBench exact licence terms for reference solutions.
- Codeforces' and AtCoder's ToS stance on solution redistribution.
- GitHub Search API's current secondary-rate-limit thresholds.
- Whether Rosetta Code has enough non-GFDL per-entry licences to be worth mining.
