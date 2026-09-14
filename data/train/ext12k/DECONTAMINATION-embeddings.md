# Layer-2 (code-embedding) decontamination of the 12k KodCode set vs the held-out pool

Run 2026-09-14 after peer review (D035): `testgen.data.decontaminate --dir <copy of ext12k/pool.jsonl> --against heldout --no-families --device cpu`. jina-embeddings-v2-base-code, cosine flag ≥ 0.90, review band 0.80–0.90. The two flagged rows were NOT removed from the training set the published adapters were trained on; this report discloses them. The defensive test-code filter dropped 244 rows whose function name starts with `test_` before embedding.


Generated 2026-09-14. Floor date 2026-06-01. Candidates 11756, kept 11754, families 11754.

## Layers and thresholds

- n-gram: word-level, n=10; any shared n-gram with a reference function flags.
- AST: identifiers→slots, docstrings dropped, constants→type; exact hash flags; k=5 node shingles Jaccard ≥ 0.85 flags.
- embedding: jinaai/jina-embeddings-v2-base-code; cosine ≥ 0.9 flags, 0.8–0.9 listed for human review.
- family: same repository owner, plus union-find over every flagged candidate pair; one function kept per family cluster of near-duplicates.

## Removed against heldout

2 candidates removed.

- `kodcode:Algorithm/Algorithm_45007_I::rgb_to_hex`: cosine 0.91 ferdinandobons/brand-docs:scripts/brandkit/common/color.py::rgb_to_hex
- `kodcode:Filter/Filter_34041_I::rgb_to_hex`: cosine 0.91 ferdinandobons/brand-docs:scripts/brandkit/common/color.py::rgb_to_hex

## Review band (kept, human check requested)

- `kodcode:Algorithm/Algorithm_386_I::format_file_size`: REVIEW cosine 0.90 GangTailorUpgrade/undress-service:coomertool/utils.py::format_size
- `kodcode:Filter/Filter_10810_I::rgb`: REVIEW cosine 0.86 ferdinandobons/brand-docs:scripts/brandkit/common/color.py::rgb_to_hex
- `kodcode:Filter/Filter_29785_I::rgb_to_hex`: REVIEW cosine 0.88 ferdinandobons/brand-docs:scripts/brandkit/common/color.py::rgb_to_hex
- `kodcode:Filter/Filter_37745_I::rgb_to_hex`: REVIEW cosine 0.88 ferdinandobons/brand-docs:scripts/brandkit/common/color.py::rgb_to_hex
- `kodcode:Filter/Filter_50966_I::bytes_to_human_readable`: REVIEW cosine 0.82 GangTailorUpgrade/undress-service:coomertool/utils.py::format_size
- `kodcode:Filter/Filter_66844_I::rgb_to_hex`: REVIEW cosine 0.90 ferdinandobons/brand-docs:scripts/brandkit/common/color.py::rgb_to_hex
- `kodcode:Filter/Filter_74648_I::string_to_slug`: REVIEW cosine 0.86 michellzappa/headroom:host/accounts.py::slugify; REVIEW cosine 0.82 pyang5166/gbro-collage-broll:scripts/generate_video.py::slugify
- `kodcode:Filter/Filter_77422_I::convert_seconds_to_hms`: REVIEW cosine 0.80 HUANGCHIHHUNGLeo/claude-real-video:src/claude_real_video/core.py::_hhmmss

## Near-duplicate merges within the pool

none

## Limitations

- Layer 1 and 3 are exact/structural; a semantic rewrite passes both.
- The embedding model has blind spots; the review band exists for that reason.
- The cutoff for Qwen3.5 is year-only; the floor is a margin, not a proof.
- Renames are not followed in the introducing-commit check (see harvest.py).
