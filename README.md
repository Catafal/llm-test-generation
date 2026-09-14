# llm-test-generation

Build a small model that **writes unit tests which expose bugs**, and prove it
with a measured comparison against prompting.

- One pure, self-contained Python function in; one executable pytest module out.
- "Expose bugs" is measured: the suite must pass on the correct function and
  fail on hidden faulty variants (mutants), under a fixed generation budget.
- Same model, three conditions: zero-shot, few-shot, LoRA fine-tune.
  Equal budget, same held-out functions, paired comparison.

Status: five fine-tuning iterations done; results in `docs/results/weekend-2.md`, model card in `docs/model-card.md`.

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

## Storage

Everything large lives inside the repo and is gitignored, so the whole
experiment can be removed without hunting through home-directory caches:

| path | what | remove with |
|---|---|---|
| `models/hf/` | every model weight this project downloads | `make models-rm ALL=1` |
| `.cache/harvest/` | cloned GitHub repos used to build the held-out pool | `make clean-harvest` |
| `.cache/mbpp/` | MBPP training pool (small) | `rm -rf .cache/mbpp` |
| `runs/*/outputs.jsonl` | raw generations (manifests stay committed) | delete the run folder |

`make models-list` shows what is on disk; `make models-pull KEYS="9b"` re-downloads.

## Research trail

- `docs/research/2026-09-06-base-model-selection/` — why Qwen3.5-9B
- `docs/research/2026-09-07-fine-tuning-principles/` — the principles, stack and rigor this repo follows
