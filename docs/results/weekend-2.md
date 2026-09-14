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

## Iteration 2: preference learning on the model's own pass/fail pairs (2026-09-09)

Same base, one variable changed: the signal. 497 pairs from 303 training
functions (chosen = passed unaided and killed ≥1 mutant; rejected = the
same function's failing suite, oracle-rescued killing suites first), DPO
with `mlx-lm-lora` (rank 8, last 16 layers, beta 0.1, lr 5e-6, 2 epochs,
117 optimizer steps), checkpoint chosen on all 171 dev functions.

| Arm | Validity | Mutation score, valid suites | Mean tokens |
|---|---|---|---|
| bf16 base, zero-shot | 0.438 | 0.862 | 758 |
| bf16 base, few-shot | 0.419 | 0.889 | 442 |
| SFT (iteration 1), checkpoint 120 | 0.397 | 0.862 | 537 |
| **DPO (iteration 2), checkpoint 900** | **0.441** | **0.874** | 548 |

Paired, DPO vs zero-shot: validity +0.003 [−0.048, +0.051], p = 1.00;
mutation score on 105 both-valid +0.012 [−0.002, +0.028]. Vs few-shot:
+0.022 [−0.035, +0.076], p = 0.49. Vs SFT: +0.044 [−0.013, +0.098],
p = 0.13. Dev-171: base 0.40; checkpoints 0.374–0.415. During training,
accuracy on 27 held-out pairs stayed at chance (0.44–0.61) while training
accuracy reached 0.9.

Reading: DPO removed the SFT harm (assertion mix back to the base's,
literal-eq 0.721; false failures 176/315, identical to the base) and kept
brevity, but did not raise validity. Telling a right literal from a wrong
one for the same input needs the computation the model lacks, so the
preference signal did not generalise even to unseen pairs. Two clean nulls
now bracket the same conclusion: any recipe that leaves the model guessing
expected values in one forward pass is capped near the base rate.

## Iteration 3: execution-explanation targets on a dense base (2026-09-10)

Two variables changed with a stated reason: base to dense
`Qwen3-4B-Instruct-2507` (the hybrid stack caps training at 1,024 tokens;
new baselines run), and the target format: the model's own unaided-valid
suites with a scratchpad comment block above each literal assert, rendered
from a real execution trace in the sandbox (`testgen/train/trace.py`,
`tracesuite.py`; cap 4 asserts, 6 lines). 358 train examples, all 36
layers, rank 8, sequence 2,048, 2 epochs; checkpoint 250 chosen on dev-171
(0.468 vs the dense base's 0.29).

| Arm (dense base), test n=315 | Validity | Mutation score, valid suites | Tests generated | Tokens | Truncated |
|---|---|---|---|---|---|
| zero-shot | 0.346 | 0.870 | 8.4 | 613 | 5 |
| few-shot | 0.368 | 0.893 | | 429 | 2 |
| **trace-target SFT, ckpt 250** | **0.400** | **0.763** | 6.7 | 1,270 | 83 |

Paired vs zero-shot: validity +0.054 [−0.010, +0.114], p = 0.11; mutation
score on 66 both-valid −0.070 [−0.131, −0.010]. Vs few-shot: +0.032,
p = 0.39; mutation −0.101.

Mechanism, measured on the raw generations: 261 of 315 suites contain
derivations, about five each, 60% of the tokens; per-test false-failure rate
0.267 for tests *with* a derivation vs 0.251 without. The model learned to
write fluent, trace-shaped derivations that are invented, and to write fewer
tests; 37 of the 83 truncated suites are valid because truncation removed
tests. The validity gain is brevity, the kill-rate loss is the same brevity.

## Iteration 4: changing the task, not the model (2026-09-11, D030)

Three iterations showed a 4B model does not learn to predict outputs. So the
task was changed: the harness fills every `assert <expr> == <literal>` from
the reference's execution, **in every arm, base included**, and the model is
judged on whether its inputs expose the mutants. The metric is the
*grounded score*: kills over live mutants, 0 if the suite still fails after
filling, averaged over all 315 test functions. Pre-registered success rule:
paired bootstrap CI on the difference vs base zero-shot excludes 0.

**Exploratory (existing generations re-scored under the new metric)** and
**confirmatory (grounded DPO, one run, checkpoint by dev-171)**, Qwen3.5-4B
bf16, same prompt, budget and mutants:

| arm | unaided validity | grounded validity | mutation score (grounded-valid) | grounded score | vs base zero (CI 95%) |
|---|---|---|---|---|---|
| base zero-shot | 0.438 | 0.717 | 0.842 | 0.604 | — |
| base few-shot | 0.420 | 0.702 | 0.854 | 0.599 | −0.005 [−0.051, +0.041] |
| SFT ckpt120 (iteration 1), re-scored | 0.397 | 0.743 | 0.846 | 0.629 | +0.024 [−0.021, +0.070] |
| DPO ckpt900 (iteration 2), re-scored | 0.441 | 0.698 | 0.854 | 0.596 | −0.008 |
| **grounded DPO ckpt1000 (confirmatory)** | 0.413 | 0.740 | 0.850 | **0.629** | **+0.024 [−0.013, +0.064]** |

Grounded DPO vs few-shot: +0.029 [−0.015, +0.073]. Grounded DPO vs the
re-scored SFT: 0.000 [−0.044, +0.042]. Grounded validity 0.740 vs 0.717
(McNemar p=0.41; 30 functions only the fine-tune, 23 only the base).
Mutation score on the 203 both-valid functions: +0.005 [−0.008, +0.017].
Unaided validity 0.413 vs 0.438 (no collapse). `arith` probe: 197 kills vs
206, no operator drift. Dev-171 curve, six checkpoints: 0.594–0.614, base
0.621, flat.

**Verdict.** The pre-registered bar (CI lower bound > 0) is not met. The
point estimate is +0.024 in both training arms, the same size as the
iteration-1 re-score, which suggests a small real effect that n=315 cannot
resolve (half-width ≈ 0.04); a definitive answer would need ~1,500
functions. The training data were learnable (validation preference
accuracy 0.50 → 0.70, margin 0 → 0.45) but the preference did not transfer
into better inputs on new functions.

**What the harness did.** Filling alone lifts every arm from ~0.43 to
~0.71 validity by rewriting ~1.4 literals per suite (412–584 per arm). The
remaining ~26% grounded-invalid suites fail on average 2.5 tests each,
on inputs and non-literal asserts, not values. That is the product result:
an execution-grounded harness is worth 28 validity points; a fine-tune on
top of it is worth at most a few.

**The principle we skipped, priced.** Every published small-model gain on
code used tens of thousands of examples from a stronger teacher. This
project banned an external teacher (D023) to keep the story self-contained
and stayed at 388–645 self-generated examples. Local distillation from a
30B-class coder in 4-bit fits the Mac and is the one untried lever; it is
recorded, not run.

**Limitation stated plainly.** Filled asserts snapshot the reference
(oracle tautology, Konstantinou 2024): the grounded metric measures input
and coverage quality, not oracle reasoning, and a reference bug would be
baked in. That is what a CI test-writing tool does; the write-up and the
demo say so.

## Iteration 5: 4,000 execution-verified examples from a frontier teacher (2026-09-12, D032 stage 1)

Same base (Qwen3.5-4B bf16), same LoRA recipe as iteration 1, same
grounded evaluation as iteration 4. The only change is the data: 4,000
(function, pytest) pairs curated from KodCode-V1 (GPT-4o solutions and
tests, CC BY-NC 4.0) through our own harness: one pure function, at least
8 live mutants, suite passes on its function, kills at least one
training-category mutant, n-gram and AST decontamination against both
held-out splits, exact-AST dedup, top 4,000 by mutation score and brevity.
One epoch, checkpoint 3200 chosen on dev-171 by grounded score
(0.630 vs base 0.621; four checkpoints, 0.589–0.630).

| arm | unaided validity | grounded validity | mut. score (grounded-valid) | grounded score | vs base zero (CI 95%) |
|---|---|---|---|---|---|
| base zero-shot | 0.438 | 0.717 | 0.842 | 0.604 | — |
| base few-shot | 0.420 | 0.702 | 0.854 | 0.599 | −0.005 |
| grounded DPO (iteration 4) | 0.413 | 0.740 | 0.850 | 0.629 | +0.024 [−0.013, +0.064] |
| **KodCode SFT ckpt3200** | 0.460 | **0.803** | 0.826 | **0.664** | **+0.059 [+0.013, +0.105]** |

Against few-shot: +0.064 [+0.016, +0.114]. Grounded validity +0.086
[+0.032, +0.137], McNemar p = 0.002 (51 functions only the fine-tune, 24
only the base). Mutation score on the 202 both-valid functions −0.023
[−0.048, +0.001]. Unaided validity +0.022 [−0.038, +0.083], unresolved.
`arith` probe: 258 kills vs 206, no operator drift. **The pre-registered
success rule (CI lower bound > 0 on the grounded score) is met.**

**Mechanism.** The model adopted the teacher's style: 4.9 tests per suite
(base 7.7), 85% literal-equality asserts (base 72%), 6% membership asserts
(base 21%), 19 truncated suites (base 31). Mutation score on grounded-valid
suites is lower (0.826 vs 0.842; 12k: 0.811): the net gain is validity.
(Table corrected 2026-09-14: the mutation-score column for the fine-tune
rows had shown the unaided values.) The harness rewrote 798
literals (base 412). Under an execution-grounded harness that is the
optimal style: short suites of exact-value asserts, every value supplied
by execution. What remains grounded-invalid fell from 89 to 62 suites.
Unaided validity did not resolve, so the model did **not** learn to
predict values better; it learned to write suites the harness can
complete. The small mutation-score loss on both-valid functions is the
price of shorter suites; the net over all functions is +5.9 points.

**What this settles.** Four iterations on 388–645 self-generated examples
were null; the fifth, on 4,000 frontier-teacher examples through the same
harness, is a resolved gain on the pre-registered metric. The lever was
the data source and scale, as the practice review predicted
(`docs/research/2026-09-11-fine-tuning-practice/`). The result holds
under the grounded harness, which is the product setting; it does not
show that a 4B model learned execution prediction.

**Limitations.** KodCode is CC BY-NC 4.0, so the adapter is non-commercial.
One epoch, one checkpoint selection on 171 functions, one test run; the
interval is wide (half-width ≈ 0.046) and the effect could be half or
twice the point estimate. Data selection on mutation score favours simpler
functions. A second training point (12,000 examples) would give a
dose-response curve; not run.

**Second point: 12,000 examples (2026-09-14).** Same recipe, same base,
12,000 examples cut from 21,085 candidates (mean mutation score of the
cut 0.937 vs 0.991 for the 4,000 cut), one epoch, checkpoint 2400 by
dev-171 (0.646). Test: grounded score **0.667**, +0.062 [+0.015, +0.106]
vs base zero-shot; grounded validity 0.822 (+0.105, p = 0.0002); unaided
validity 0.467 (+0.029, unresolved); arith 259 kills. **Against the
4,000-example adapter: +0.003 [−0.027, +0.034]**, grounded validity +0.019
(p = 0.38), unaided +0.006. The gain replicates on an independent run and
does not grow with three times the data. The 12k adapter needs less help
from the harness (520 literals rewritten vs 798) and writes 4.7 tests per
suite at 89% literal-equality asserts. Either the effect saturates at a
few thousand examples of this kind, or the lower selection quality of the
larger cut offsets its size; the two are confounded here.

**Control (2026-09-14, D035/D036, after peer review).** Base Qwen3.5-4B,
no training, prompted for the teacher's style (`SYSTEM_TEACHER`: one short
test per behaviour, exact return value with `==`, at most 5 tests; two
KodCode examples as exemplars), grounded harness, same 315 functions:
grounded validity 0.860, mutation score on grounded-valid 0.791, grounded
score **0.680**, +0.076 [+0.034, +0.118] vs base zero-shot (56/11
discordant); unaided validity 0.571, +0.133 [+0.073, +0.190]. Adapters vs
the control: 12k −0.014 [−0.051, +0.022], 4k −0.017 [−0.057, +0.022].
4.65 tests per suite, 82% literal asserts, 247 literals filled. Run
`baselines-test-control-teacher-20260914T143758Z`. **Reading, per the
rule fixed before the run:** the style is worth the six points and
prompting for it gets all of them; the adapters match the prompt and do
not beat it.

**Embedding decontamination of the 12k set (2026-09-14).** Run after
peer review: 2 of 11,756 rows flagged at cosine ≥ 0.90 vs held-out (both
`rgb_to_hex`), 9 in the 0.80–0.90 review band; 4 held-out functions
touched; rows not removed from the trained set; report in
`data/train/ext12k/DECONTAMINATION-embeddings.md`.

**Adapter + style prompt (2026-09-14, D037).** 12k adapter under
`SYSTEM_TEACHER`: grounded validity 0.867, mutation score on grounded-valid
0.769, grounded score 0.667; vs control −0.014 [−0.044, +0.019]; vs the
adapter under the plain prompt 0.000 [−0.031, +0.032]. The prompt raises
the adapter's validity and lowers its kills by the same amount. Redundant.

## What five iterations settle

Three signals (execution-corrected literals, the model's own pass/fail
preferences, execution-grounded derivations), two bases, one metric, and
per-assert value accuracy never moved. A fourth iteration removed values
from the task and trained on grounded preferences; the gain is +0.024,
unresolved at n=315. A fifth kept the harness and replaced the data with
4,000 execution-verified frontier-teacher examples: +0.059 [+0.013,
+0.105], resolved, and repeated at 12,000 examples (+0.062 [+0.015,
+0.106]) with no further gain from the extra data. The control run after
peer review then showed the base prompted for the same style at +0.076,
above both adapters. What the fine-tune learned was a style, and the style
can be asked for. The harness is the lever; the recipe was never the
problem, and neither, in the end, was the data. At a few hundred examples with LoRA,
a 4B model does not acquire execution prediction; the one published success
at 3B used ~80M traces. The bottleneck is a base-model capability, not a
recipe choice within this budget. The recipe, harness, decontaminated pools
and paired evaluation are the deliverable; the negative result is measured
at every step.

## Next hypothesis

Outside this project's budget: trace-explanation pretraining at scale
(millions of traced functions), or a base that already predicts outputs
(CRUXEval-O ≥ 70). The task change (execution-grounded expected values) was
run as iteration 4 and is now the product recommendation; the remaining
untried training lever is local teacher distillation at 10k+ examples.

Every number traces to a manifest under `runs/` or `models/adapters/`.
