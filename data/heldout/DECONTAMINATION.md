# Decontamination Report — held-out pool vs MBPP

Generated 2026-09-07. Floor date 2026-06-01. Candidates 151, kept 144, families 22.

## Layers and thresholds

- n-gram: word-level, n=10; any shared n-gram with an MBPP function flags.
- AST: identifiers→slots, docstrings dropped, constants→type; exact hash flags; k=5 node shingles Jaccard ≥ 0.85 flags.
- embedding: jinaai/jina-embeddings-v2-base-code; cosine ≥ 0.9 flags, 0.8–0.9 listed for human review.
- family: same source repo, plus union-find over every flagged candidate pair; one function kept per family cluster of near-duplicates.

## Removed against MBPP

0 candidates removed.


## Review band (kept, human check requested)

none

## Near-duplicate merges within the pool

- ast-exact: guillaumemeyer/watermarks-remover:service/scripts/bench_synthid_text.py::_numbers_preserved  ~  guillaumemeyer/watermarks-remover:service/scripts/bench_synthid_text.py::_urls_preserved
- ngram10: andrewyng/openworker:coworker/attachments.py::content_to_text  ~  andrewyng/openworker:coworker/compaction.py::_text_of
- ngram10: shepherd-agents/shepherd:commons-vcs/src/commons_vcs/_types.py::_normalize_json_value  ~  shepherd-agents/shepherd:commons-vcs/src/commons_vcs/canonical.py::_validate_json_primitive
- ngram10: shepherd-agents/shepherd:commons-vcs/src/commons_vcs/_types.py::_normalize_json_value  ~  shepherd-agents/shepherd:shepherd/packages/core/src/shepherd_core/effects/commons_vcs.py::normalize_commons_value
- ast-exact: shepherd-agents/shepherd:commons-vcs/src/commons_vcs/backends/git.py::_digest_to_segment  ~  shepherd-agents/shepherd:commons-vcs/src/commons_vcs/backends/git.py::_segment_to_digest
- ngram10: shepherd-agents/shepherd:commons-vcs/src/commons_vcs/canonical.py::_validate_json_primitive  ~  shepherd-agents/shepherd:shepherd/packages/core/src/shepherd_core/effects/commons_vcs.py::normalize_commons_value
- ngram10: MatrAIx-ai/MatrAIx-Persona-8B:application/playground/backend/service/job_aggregation.py::_distribution_directive_dedupe_marker  ~  MatrAIx-ai/MatrAIx-Persona-8B:application/playground/backend/service/job_aggregation.py::_distribution_directive_dimensions
- ast-jaccard 0.86: ShenSeanChen/waku-agent:evals/deterministic/test_experimental_toggle.py::test_turning_it_off_is_not_swallowed  ~  ShenSeanChen/waku-agent:evals/deterministic/test_graph_flag.py::test_turning_it_off_is_not_swallowed

## Limitations

- Layer 1 and 3 are exact/structural; a semantic rewrite passes both.
- The embedding model has blind spots; the review band exists for that reason.
- The cutoff for Qwen3.5 is year-only; the floor is a margin, not a proof.
- Renames are not followed in the introducing-commit check (see harvest.py).
