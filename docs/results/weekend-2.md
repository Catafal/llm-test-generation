# Self-distillation with oracle-filled expected values did not beat prompting

Weekend two of `llm-test-generation`. Question: does fine-tuning Qwen3.5-4B
on its own execution-verified test suites raise suite validity and mutation
score over zero-shot and few-shot prompting of the same model at equal
budget? Answer, on 315 post-cutoff held-out functions: **no**. Validity fell
by 4.1 points against zero-shot (95% CI −9.8 to +1.3) and 2.2 points
against few-shot (CI −8.3 to +3.5); mutation score on both-valid
functions was unchanged. The interval includes zero, so the effect is
unresolved at this sample size; the point estimate is negative.

## Setup

| Item | Value |
|---|---|
| Base model | `mlx-community/Qwen3.5-4B-bf16`, thinking off |
| Training functions | 550, GitHub repos created after 2026-06-01, not used for held-out, owners disjoint, ≥8 live mutants, decontaminated vs both held-out splits (`data/train/DECONTAMINATION.md`) |
| Proposer | the same 4B, temp 0.7, K=8, same prompt as evaluation |
| Targets | wrong literal expected values rewritten from execution (`testgen/train/oracle.py`, repr ≤ 40 chars), then pass on reference and kill ≥1 training-category mutant; best per function; ≤1024 tokens |
| SFT set | 412 functions → 388 train / 24 valid by family (`data/train/sft/yield.json`) |
| Adapter | LoRA rank 16, scale 2.0, dropout 0.05, all linear projections in the last 16 layers, bf16, batch 1 × accumulation 8, LR 1e-4 cosine, 3 epochs (1200 micro-iterations), mx.compile off |
| Checkpoint selection | harness validity on 60 dev functions over 10 checkpoints; step 120 chosen |
| Evaluation | test split n=315, zero-shot, greedy, 2048 new tokens, ≤8 tests, same normalisation for every arm (D018) |

## Results (test split, n=315)

| Arm | Validity | Mutation score, valid suites | Mean tokens | Tests generated | Truncated |
|---|---|---|---|---|---|
| bf16 base, zero-shot | 0.438 | 0.862 | 758 | 12.2 | 31 |
| bf16 base, few-shot | 0.419 | 0.889 | 442 | 6.5 | 1 |
| 4-bit base, zero-shot (weekend 1) | 0.400 | 0.869 | | | |
| 4-bit base, best-of-4 + reference filter, first 100 fns | 0.540 | 0.819 | 2649 (4 samples) | | |
| **Fine-tune, checkpoint 120, zero-shot** | **0.397** | **0.862** | 537 | 8.1 | 15 |

Paired, fine-tune vs bf16 base zero-shot (few-shot in the second column):

| Statistic | vs zero-shot | vs few-shot |
|---|---|---|
| Validity difference (bootstrap 95% CI) | −0.041 [−0.098, +0.013] | −0.022 [−0.083, +0.035] |
| Discordant functions (base only / fine-tune only) | 45 / 32 | 50 / 43 |
| Exact McNemar p | 0.17 | 0.53 |
| Mutation score difference on both-valid (CI) | −0.017 [−0.041, +0.003] (n=93) | −0.004 [−0.023, +0.011] (n=82) |

Per-category kill rate on the 93 both-valid functions is slightly higher for
the fine-tune in every category, including the `arith` probe that was held
out of all training curation (0.743 vs 0.714): no operator overfitting.

Validity by live-mutant quartile (valid functions of ~79): base 45/30/32/31,
fine-tune 42/28/27/28. The fine-tune is lower in every quartile.

## What the model learned

- **Brevity.** 8 tests instead of 12, 30% fewer tokens, half the truncations.
  Few-shot prompting gets the same brevity for free (6.5 tests, 442 tokens)
  with better validity, so brevity is not what was missing.
- **More literal assertions.** Share of `== <literal>` asserts rose from
  0.724 to 0.771; membership asserts fell from 0.207 to 0.177.
- **Not arithmetic.** Nearly every invalid suite in both arms is a false
  failure on a wrong expected value (base 176, fine-tune 190 of 315).

The mechanism is consistent with the risk recorded before training: targets
whose literals were corrected by execution teach the model to state exact
values with confidence, and at inference it still cannot compute them.

## Dev curve and its lesson

Harness validity on 60 dev functions over the ten checkpoints:
0.433, 0.367, 0.433, 0.400, 0.417, 0.383, 0.383, 0.383, 0.383, 0.400;
bf16 base on the same 60: 0.320. Token validation loss bottomed at step
400 (0.156) and rose to 0.193 by step 1200. The +7-function dev advantage
of the chosen checkpoint did not transfer to test: selecting the best of ten
on 60 functions is a winner's curse. Dev at this size is too small for
checkpoint selection (D020 revisit triggered).

## Other measured findings

- **Quantisation confound.** The bf16 base scores 0.438 where the 4-bit
  base scored 0.400 on the same split; comparisons must hold the weights
  format fixed.
- **Best-of-4 ceiling.** Resampling with reference filtering reaches 0.54,
  not the 0.87 independent samples would give: failures are per-function.
- **Oracle filling** lifted candidate validity from 0.28 to 0.49 and
  function coverage from 21 to 31 of 40 on dev, but the hardest quartile did
  not move (`runs/derisk-*`).

## Limitations

- n=315 resolves ~6 validity points; this result is unresolved, not a
  demonstrated harm.
- Single seed, single training run; no rank or data-size ablation yet.
- Dev-60 checkpoint selection; the pre-registered 171-function tie re-score
  was not executed before the test evaluation (process slip, recorded).
- LoRA on the last 16 layers and ≤1024-token targets, forced by mlx-lm's
  Gated DeltaNet training path (D024); all-layer LoRA is untested.
- Equivalent mutants: bytecode filter only; hand-labelling pending.
- "Post-cutoff" is a margin, not a proof (year-only cutoff).

## Next hypothesis

Teach assertion strategies the model can compute rather than literals it
cannot: keep only targets whose literals were already correct unaided, or
reweight toward relational and round-trip assertions; one epoch, rank 8;
select on all 171 dev functions.

Every number traces to a manifest under `runs/` or `models/adapters/`.
