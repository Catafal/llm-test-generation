# llm-test-generation

Build a small model that **writes unit tests which expose bugs**, and prove it
with a measured comparison against prompting.

- One pure, self-contained Python function in; one executable pytest module out.
- "Expose bugs" is measured: the suite must pass on the correct function and
  fail on hidden faulty variants (mutants), under a fixed generation budget.
- Same model, three conditions: zero-shot, few-shot, LoRA fine-tune.
  Equal budget, same held-out functions, paired comparison.

Status: evaluation harness in progress. No training has run.

## Layout

```
testgen/harness/   run a suite against an implementation; score it
testgen/mutate/    AST mutation operators; equivalence filter
testgen/data/      function pools; decontamination
testgen/generate/  prompts and mlx-lm generation under a fixed budget
tests/             pytest for the harness itself
data/              pilot cases and held-out pool (raw gitignored, manifests committed)
runs/              raw outputs (gitignored) and run manifests (committed)
docs/research/     the research trail behind each decision
```

## Commands

```
make setup      # uv sync
make test       # harness unit tests
make lint       # ruff
make pilot      # 10-case sanity check of the metric
make baselines  # prompting baselines for the candidate models
make eval       # score a run against the held-out pool
```

## Research trail

- `docs/research/2026-09-06-base-model-selection/` — why Qwen3.5-9B
- `docs/research/2026-09-07-fine-tuning-principles/` — the principles, stack and rigor this repo follows
