# Model card and datasheet: `lora-4b-ext` / `lora-4b-ext12k`

LoRA adapters for **Qwen3.5-4B** that write pytest suites for a single pure
Python function, meant to run behind an **execution-grounded harness** that
fills expected values by executing the function. Evaluated on 315 held-out
real-world functions with paired statistics, pre-registered.

| | |
|---|---|
| Base model | `mlx-community/Qwen3.5-4B-bf16` (hybrid Gated DeltaNet + attention, 4.2B) |
| Adapters | [`jorcagra/qwen3.5-4b-testgen-lora-ext12k`](https://huggingface.co/jorcagra/qwen3.5-4b-testgen-lora-ext12k) (12,000 examples, ckpt 2400) and [`jorcagra/qwen3.5-4b-testgen-lora-ext4k`](https://huggingface.co/jorcagra/qwen3.5-4b-testgen-lora-ext4k) (4,000 examples, ckpt 3200); locally `models/adapters/lora-4b-ext12k/ckpt-0002400` and `models/adapters/lora-4b-ext/ckpt-0003200` |
| Method | LoRA rank 16, scale 2.0, dropout 0.05, all linear projections of the last 16 of 32 layers, completion-only loss, seq 1,024, LR 1e-4 cosine, one epoch |
| Training data | KodCode-V1 (GPT-4o solutions and tests), curated through this repo's harness |
| Licence | Adapters: **CC BY-NC 4.0** (inherited from KodCode). Code: see repo LICENSE. Non-commercial. |
| Hardware | One Mac M4 Pro 48 GB, mlx-lm 0.31.3; ~9.5 h (4k) and ~30 h (12k) training |
| Result | Grounded score +0.059 [+0.013, +0.105] (4k) and +0.062 [+0.015, +0.106] (12k) vs the base prompted, n = 315, paired bootstrap |

## Intended use

Generate a pytest module for one pure, self-contained Python function, then
**run the module through the harness** (`testgen/train/oracle.fill`), which
executes the reference to fill `assert f(x) == <literal>` sites, and score it
against mutants. The adapters were selected and evaluated only in that
setting. Without the harness the adapters are statistically no better than
the base at writing suites that pass as written (unaided validity 0.46 vs
0.44, CI includes zero).

**Out of scope.** Functions with I/O, classes, fixtures, mocks, or external
dependencies; languages other than Python; commercial use (licence);
treating the filled expected values as ground truth for a function that is
itself wrong (the harness snapshots the reference, so a reference bug is
baked into the suite).

## How to use

```bash
make setup && make sync-models && make models-pull KEYS="4b-bf16"   # base model (public, mlx)
make adapters-pull                                                   # both adapters from the Hub
make demo ID="NanmiCoder/open-image-prompts:retrieval/engine.py::weighted_tag_similarity" FROM_RUNS=1
# score the adapter on the held-out test split, oracle-filled in both arms
make eval ADAPTER=models/adapters/lora-4b-ext12k/ckpt-0002400 GROUNDED=1
make baselines POOL=test MODELS=4b-bf16 CONDITIONS=zero,few   # the base arms
uv run python -m testgen.stats --grounded <outputs.jsonl> 4b/zero 4b/few
```

Generation needs Apple Silicon (mlx); the harness, statistics, figures and
the replay demo run anywhere. Prompt, budget and decoding are fixed by `testgen/generate/prompts.py` and
`config.py`: 2,048 new tokens, at most 8 tests per suite, greedy, thinking
off. Every arm in every table below used the same.

## Metric

- **Validity (unaided):** the suite, as written, passes on the reference.
- **Grounded validity:** the suite passes on the reference after the harness
  rewrites literal expected values from execution.
- **Mutation score:** fraction of live mutants (AST operators, equivalent
  mutants removed by bytecode fingerprint) on which at least one test fails.
- **Grounded score (primary, pre-registered in D030):** mutation score of the
  filled suite, **0 if it is still invalid**, averaged over every function.
  A suite cannot score by writing fewer, safer tests.

Statistics: paired bootstrap CI (2,000 resamples) on the per-function
difference; exact McNemar on paired validity. n = 315 resolves a grounded-
score difference of about ±0.045.

## Results (test split, n = 315, all Qwen3.5-4B bf16)

| arm | unaided validity | grounded validity | mutation score, grounded-valid | grounded score | Δ vs base zero, CI 95% |
|---|---|---|---|---|---|
| base zero-shot | 0.438 | 0.717 | 0.842 | 0.604 | — |
| base few-shot | 0.420 | 0.702 | 0.854 | 0.599 | −0.005 [−0.051, +0.041] |
| it. 1 SFT, 388 self-generated, oracle-corrected | 0.397 | 0.743 | 0.846 | 0.629 | +0.024 [−0.021, +0.070] |
| it. 2 DPO, 497 own pass/fail pairs | 0.441 | 0.698 | 0.854 | 0.596 | −0.008 |
| it. 4 DPO, 645 own grounded pairs | 0.413 | 0.740 | 0.857 | 0.629 | +0.024 [−0.013, +0.064] |
| **it. 5a SFT, 4,000 KodCode (`lora-4b-ext`)** | 0.460 | 0.803 | 0.868 | **0.664** | **+0.059 [+0.013, +0.105]** |
| **it. 5b SFT, 12,000 KodCode (`lora-4b-ext12k`)** | 0.467 | 0.822 | 0.843 | **0.667** | **+0.062 [+0.015, +0.106]** |

Iteration 3 (execution-trace targets) ran on the dense Qwen3-4B-2507 and is
in `docs/results/weekend-2.md`: validity +0.054 (p = 0.11) from brevity,
mutation score −0.070, derivations fabricated.

Additional rows for the two shipped adapters, vs base zero-shot:
grounded validity +0.086 (p = 0.002) and +0.105 (p = 0.0002); mutation
score on both-valid functions −0.023 [−0.048, +0.001] and −0.024 [−0.049,
0.000]; unaided validity +0.022 and +0.029 (both unresolved); held-out
`arith` mutation category (never used in curation): 258 and 259 kills vs
206, no operator drift. 12k vs 4k: +0.003 [−0.027, +0.034].

Dev-171 curves (checkpoint selection, grounded score): 4k 0.601 / 0.589 /
0.623 / **0.630** at 800/1600/2400/3200; 12k **0.646** / 0.596 / 0.633 /
0.596 at 2400/4800/7200/9600; base 0.621.

## What the adapters learned (mechanism)

Style transfer from the teacher data, not execution prediction. Compared
with the base, the adapters write 4.7–4.9 tests per suite (base 7.7), 85–89%
literal-equality asserts (base 72%), fewer membership asserts, fewer
truncated suites. Under a harness that fills literals, that is the optimal
style; the harness rewrote 520–798 literals per 315 suites (base 412).
Unaided validity did not resolve, so the model did not become better at
predicting outputs. Four earlier iterations on 388–645 self-generated
examples, three signals and two bases, never moved per-assert value accuracy
either (`docs/results/weekend-2.md`, D025–D031).

## Training data (datasheet)

**Source.** [KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1)
(Xu et al. 2025, CC BY-NC 4.0): 487K synthetic coding questions with a
GPT-4o-0513 solution and a pytest module verified by execution. Subsets
used: Filter, Algorithm, Evol, Package, Docs (Prefill excluded: seeded from
public benchmarks; competition subsets excluded: I/O-judge style).

**Curation** (`testgen/train/curate_ext.py`, `make curate-ext`):
1. Stage A, AST only, 169K rows of the five subsets: solution = imports +
   exactly one top-level function passing the held-out purity filter (no
   I/O, no globals, no non-stdlib imports); test imports only `solution`,
   `pytest`, stdlib; test cut to the first 8 test functions; ≥ 8 live
   mutants. **44,527 kept.**
2. Stage B, sandbox, seeded sample (seed 20260911): suite passes on its
   solution and kills ≥ 1 training-category mutant. 24,000 sampled →
   **21,085 kept** (88%).
3. Stage C: n-gram-10 and AST-hash/AST-Jaccard decontamination against both
   held-out splits (14 removed), exact-AST dedup (7,025), ≤ 1,024 tokens with
   chat template (13,429), then the top-k by (mutation score, fewer tests).
   4k cut: mean mutation score 0.991, 5.7 tests, 644 tokens. 12k cut: 0.937,
   5.8 tests, 655 tokens. 95/5 split by KodCode question id.

Data files: `data/train/ext/` and `data/train/ext12k/` (`train.jsonl`,
`valid.jsonl`, `pool.jsonl`, `yield.json`, `DECONTAMINATION.md`,
`NOTICE.md`). Layer-2 embedding decontamination was skipped for this set
(2025 synthetic source vs post-2026-06 GitHub held-out); the self-generated
pool (`data/train/pool.jsonl`, 550 functions, 91 repos) had all three
layers.

**Known properties.** Synthetic, GPT-4o style; selection on mutation score
favours simpler functions; 89% of asserts are exact-value literals; the
teacher's values are correct by construction (execution-verified).

## Evaluation data (datasheet)

`data/heldout/pool.jsonl`: 486 pure Python functions from 95 public GitHub
repositories, each function introduced by a commit dated on or after
**2026-06-01** (after the base model's training cutoff), ≥ 8 live mutants,
per-repo cap 10, decontaminated against MBPP with n-gram, code-embedding
(jina-embeddings-v2-base-code, cosine ≥ 0.90) and AST layers (0 removed),
near-duplicates merged into 86 families, split **by family** into test 315
/ dev 171 (seed 20260907). `data/heldout/DECONTAMINATION.md` and
`NOTICE.md` list thresholds and every source licence. Function bodies are
reproduced verbatim under their licences. The training pools are
decontaminated against both splits.

## Limitations

- **Non-commercial** adapters (KodCode licence).
- The gain exists **only under the grounded harness**; as a stand-alone
  test writer the adapters are not measurably better than the base.
- Filled asserts snapshot the reference (oracle tautology): the metric
  measures input and coverage quality, not oracle reasoning.
- One test evaluation per adapter, checkpoint chosen on 171 dev functions;
  the intervals are wide (half-width ≈ 0.045). The 4k and 12k results
  are independent replications of each other, not of a third run.
- No dose-response from 4k to 12k; saturation and the lower selection
  quality of the larger cut are confounded.
- Held-out functions are pure and self-contained by construction; nothing
  here speaks to classes, fixtures, or repository-level tests.
- Small mutation-score loss on functions both arms get valid (shorter
  suites); the net over all functions is what the primary metric reports.
- Trained and evaluated on one machine with one seed per run.

## Negative results this card depends on

Iterations 1–4 (`docs/results/weekend-2.md`): oracle-corrected self-SFT
(−0.041 unaided validity), DPO on own pairs (null), execution-trace
scratchpad targets (brevity gain, kill loss, fabricated derivations),
grounded DPO on own pairs (+0.024, unresolved). The practice review that
predicted the lever is in `docs/research/2026-09-11-fine-tuning-practice/`.

## Reproducibility

Every number traces to a run manifest (`runs/<run>/manifest.json`: git sha,
model, budget, adapter, results) or an adapter manifest
(`models/adapters/<run>/manifest.json`: config sha, data sha, LoRA
parameters, seed). Configs: `configs/lora-4b-ext.yaml`,
`configs/lora-4b-ext12k.yaml`. Key runs: base
`baselines-test-bf16base-grounded-20260910T203314Z`, few-shot
`baselines-test-bf16base-few-grounded-20260910T205151Z`, 4k
`baselines-test-finetune-20260912T130009Z`, 12k
`baselines-test-finetune-20260914T115239Z`. Raw generations are gitignored;
manifests, data manifests and decontamination reports are committed.
