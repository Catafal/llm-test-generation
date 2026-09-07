# Reviewer 2 — teacher / training-signal lens (Opus, independent, no file access)

Question: P10, who generates the candidate suites?

## 1. Alternatives, including what is missing from the list

- (a) 9B thinking-ON, local. Optimises independence and cost. Risks low yield on hard functions and a reasoning-style teacher whose surface form the thinking-off student never sees.
- (b) Claude via subscription. Optimises yield. Risks terms-of-service exposure and, more damaging, contamination of the headline claim.
- (c) (a) then (b) top-up. Inherits both risks and adds one: the top-up lands exactly on the hard functions, so teacher identity correlates with difficulty in the training set.
- (d) Self-distillation from the 4B, high K. The student already produces a valid suite 40 percent of the time; at K=8 with temperature, per-function coverage should be high. Optimises claim purity: no external capability enters the pipeline. Risks a capability ceiling and mode collapse onto existing habits.
- (e) Oracle-assisted targets. Not on the list, and the one that matches the diagnosis. Any model proposes inputs and suite structure; the harness executes the reference on those inputs and writes the expected values into the target. Every target is correct by construction, yield approaches 100 percent, the mutation gate still prunes weak suites. The only option that attacks the failure mode rather than sampling around it.
- (f) Execution-feedback repair. A slow, lossy approximation of (e). Useful only as a yield multiplier if (e) is rejected.
- (g) Property and invariant tests. Sidesteps hand-computed values. Risks teaching the student to dodge the hard part; property suites typically kill fewer mutants. Allow as a minority, never as the escape hatch.
- (h) Coder-7B or mixed teachers. Weakest baseline at 0.25 validity; diversity is cheaper from temperature. Skip.

## 2. What the student is supposed to learn

The measured deficit is expected-value prediction, not test design. Mutation score on valid suites is already 0.87 to 0.91, so the student knows what to test; it cannot reliably compute the answer.

Distilling from a teacher that also predicts values by hand teaches a better prior over values, nothing more: from a 40 percent guesser to perhaps a 65 percent guesser, mechanism opaque. Oracle-filled targets give a training distribution where the value is always right.

The real objection to (e): supervision the student cannot in principle derive teaches confident guessing (hallucination from unattainable labels). It has teeth. Constrain input selection toward inputs whose outputs are short and mentally computable, keep the mutation gate, do not hand the student a 15-digit float.

Teacher identity matters for the claim. A Claude-derived training set turns "fine-tune vs prompting of the same model" into a distillation result with a fine-tuning wrapper. Same-family Qwen is materially safer; the student itself is safest.

Thinking-ON teacher into thinking-OFF student is a real format hazard with a cheap fix: strip reasoning, keep only the final suite, render every target through the exact inference template.

## 3. Objections you will face

- Terms of service. Bulk programmatic generation through a consumer subscription gets asked about in an interview. Any answer other than "I did not do it" costs you.
- "You distilled Claude." Fatal to the framing. Applies to (b) and (c).
- Why not best-of-K at inference. The sharpest objection. The prompt contains the reference, so a caller can run the suite and resample on failure; that plausibly reaches 0.9 validity with no training. Defence is the fixed budget: single sample, greedy, no execution at inference. State it, and report the resample ceiling as an upper bound rather than letting a reviewer find it.
- Filter-induced easy-function bias. Rejection sampling yields most on easy functions. Option (e) removes this almost entirely, since yield stops depending on the proposer getting values right. Report per-category and per-difficulty coverage either way.
- Arith holdout leakage. Confirm no proposed input or docstring drags arithmetic-category patterns into the curated set.

## 4. Ranked recommendation

Criteria: attacks the measured failure mode; keeps the headline clean; no spend; no ToS exposure; hours to first SFT set.

1. Oracle-assisted, proposer is the local 9B thinking on, values computed by the harness. Mitigate the unattainable-label risk by constraining inputs so reference outputs stay short.
2. Oracle-assisted with the 4B itself as proposer. Cleanest possible claim, slightly weaker test design.
3. Option (c) as specified, only if oracle-assisted is blocked; disclose the teacher mix.
4. Option (a) alone.
5. Option (b) as primary. Do not.

Flip signal: if oracle-filled targets produce a fine-tune whose validity gain fails to transfer to the arith holdout while a predicted-value teacher transfers, the unattainable-label objection was right; fall back to (c).

## 5. One cheap experiment, under an hour

40 dev functions stratified by category and difficulty. Pipeline A: 9B thinking on, K=4, values predicted, rejection filtered. Pipeline B: same generations with expected values overwritten by executing the reference, then mutation gated. Compare accepted examples, per-stratum coverage, and the fraction of oracle-filled values short enough to compute without a tool. If B's coverage is materially flatter across strata, commit to oracle-assisted and never spend the three to five hours on teacher sampling.

**Verdict.** P10 -> oracle-assisted targets with a local Qwen proposer, because the failure is arithmetic the teacher cannot fix and the harness can, and it keeps the claim a fine-tuning result rather than a distillation one.
