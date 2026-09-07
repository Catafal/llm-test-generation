# Reviewer 3 — hiring-panel skeptic lens (Opus, independent, no file access)

Question: P9 and P10 together; what can the write-up honestly claim?

## 1. Each combination and its attack

(b)+(a) GitHub pool, 9B teacher. Headline: fine-tuning on filtered samples from a same-family larger model raises validity over prompting at fixed budget. Attacks: standard rejection-sampling distillation, and nothing separates the teacher's contribution from the filter's. The 9B and 4B share tokenizer, pretraining and likely post-training data, so did you measure distillation or recover capability the 4B already had? Yield is selection: functions where the 9B never passes are the hard-arithmetic ones, so the training set is the easy half and any held-out gain may concentrate there.

(b)+(c) 9B then Claude top-up. The top-up is precisely the residual where local yield is zero, so a frontier model supplies supervision on the hardest examples: the highest-information slice, not a minor patch. Honest headline is distillation from a frontier teacher. Attacks: claim contaminated by a teacher you cannot characterise or reproduce; unverified ToS on bulk subscription use, which a hiring panel reads as judgment; and the counterfactual of calling the frontier model at inference.

(a)+(a) MBPP pool. Worst. No docstrings, mostly fewer than eight mutants, so the training prompt is structurally different from the evaluation prompt; any effect confounds with format shift and a null is uninterpretable. Deeply pre-cutoff, so teacher yield is inflated by memorised solutions.

(b)+(b) Claude only. Cleanest data, least defensible claim: "distil Claude into a 4B", not the stated hypothesis, on an unverified terms question.

Alternative 1, self-distillation from the 4B. At validity 0.40, K=8 yields at least one valid suite for roughly 98 percent of functions before the mutant condition bites (independence assumption). Removes teacher contamination, terms exposure and same-family ambiguity in one move; the gain comes from the filter and the harness.

Alternative 2, oracle-computed expected values. The harness already executes the reference. Filling asserts by execution gives 100 percent yield including on the hard functions the teacher drops, eliminates the easy-function bias at its root, and makes the target exactly the thing the model fails at. Counter: the model never learns to compute values it cannot execute, so you must show transfer on held-out functions where no oracle exists. That is the honest experiment.

## 2. Ranking the attacks

Fatal to the portfolio's purpose:
1. ToS exposure on bulk teacher generation. A judgment problem, not a science problem; the one that ends a candidacy.
2. Prompt-format mismatch from the MBPP pool. Makes the headline uninterpretable.
3. Unmeasured teacher-versus-filter attribution. "The filter guarantees quality" cannot be defended as written unless the design separates them.

Limitations to state, not fatal:
4. Selection bias toward easy functions, survivable if yield and held-out gain are reported stratified by difficulty.
5. Pre-cutoff memorisation of the training pool, since evaluation is post-floor and family-disjoint.
6. Same-family distillation from the 9B, survivable if named explicitly.
7. Best-of-K at inference, survivable if the fixed-budget contract is stated as deliberate and the K-sample ceiling is measured.

## 3. One extra control under (b)+(c)

Measure best-of-4 zero-shot with harness filtering on a difficulty-stratified subsample of about 100 held-out functions and report it as a ceiling next to the single-greedy arms. Answers the sharpest question without a second training run. About 2 to 3 hours of machine time.

## 4. What impresses, what stops the read

Impressive: three-layer decontamination with a post-cutoff floor, the held-out arith probe, and above all targeting validity after measuring that mutation score on valid suites is saturated. Naming the smallest resolvable effect before running anything is the single strongest signal in the plan.

Stops the read: a headline gain sourced from a frontier teacher without the word distillation in the claim; an unverified terms question left standing; training on toy functions while evaluating on docstringed ones with no format ablation; any effect near the resolution floor without the paired test attached.

## 5. Recommendation

Pool: the GitHub harvest. The training prompt must have the same shape as the evaluation prompt; the toy corpus supplies no docstrings and too few live mutants.

Teacher: drop the frontier model entirely and self-distil from the target model. Fill the residual hard cases by having the harness execute the reference to compute expected values. Keep the 9B only as a labelled ablation arm, never as the data source. Criteria: no external supervision, no terms exposure, yield on the hard tail, reproducibility by a reviewer with the same hardware.

Defensible sentence: "fine-tuning Qwen3.5-4B on its own harness-filtered samples raises suite validity from 0.40 to X on 315 post-cutoff held-out functions at a fixed 2048-token greedy budget, with no supervision from any stronger model, and the gain holds on the arithmetic operator category held out of all training curation."

**Verdicts.** P9 -> (b) because train and evaluation prompts must share a format. P10 -> (a) modified to self-distillation from the 4B with oracle-filled residuals, because it removes the terms exposure and credits the filter rather than a teacher.
