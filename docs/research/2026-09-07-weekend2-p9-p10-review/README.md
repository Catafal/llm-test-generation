# Weekend-2 open decisions P9 and P10 — three-way Opus review (2026-09-07)

Three independent Opus reviewers, each with a different lens and no access to
the repo, reviewed the two open decisions in the weekend-2 plan. Sources:

- `01-data-lens.md` — where the training functions come from (P9).
- `02-teacher-lens.md` — who writes the candidate suites (P10).
- `03-skeptic-lens.md` — hiring-panel reading of P9 and P10 together.

## Where the reviewers agree

1. **MBPP is out as a training source.** All three. The training prompt must
   have the same shape as the evaluation prompt (signature + docstring +
   reference). MBPP has no docstrings and mostly fewer than eight live
   mutants, so any effect confounds with format shift, and a null would be
   uninterpretable. Mixing (option c) makes nothing attributable.
2. **No Claude, no `claude -p`, in any role.** Reviewers 2 and 3 both call
   it fatal: an unverified terms-of-service question on bulk subscription
   use reads as judgment, not paperwork, and a frontier-teacher training set
   turns the claim into "distilled Claude" with a fine-tuning wrapper. The
   top-up variant (c) is worse than it looks: the top-up lands exactly on the
   hardest functions, so the frontier model supplies the highest-information
   slice of the data.
3. **The diagnosis changes the teacher question.** Weekend one measured
   that invalid suites are wrong hand-computed expected values, not bad test
   design (mutation score on valid suites is already 0.87–0.91). A teacher
   that also predicts values by hand teaches a slightly better prior, not the
   skill. The harness already executes the reference, so it can *compute* the
   expected values: oracle-filled targets, correct by construction, near-100%
   yield including the hard tail that rejection sampling drops.
4. **Self-distillation is viable and the cleanest claim.** At 0.40 validity,
   the 4B's own samples at K=4–8 plus oracle filling clear the 500-example
   target with no external supervision. The gain is then attributable to
   the filter and the harness, which is the claim the project wants.
5. **Measure the best-of-K ceiling.** The sharpest reviewer question is
   "why not resample at inference, the prompt has the reference?". Answer it
   with the fixed-budget contract *and* a measured row: best-of-4 with
   harness filtering on ~100 held-out functions, reported as an upper bound.
6. **Report yield and held-out gain stratified by difficulty** (live-mutant
   count) and the assertion-style mix (literal vs relational/round-trip/
   invariant) before and after. That is how the easy-function selection bias
   and the "did it learn arithmetic or change strategy" question get answered.

## Where they disagree

- **P9: pre-floor (b) vs post-floor overflow (b').** Reviewer 1 proposes
  harvesting repositories created *after* the 2026-06-01 floor that were not
  drawn for the held-out pool: same distribution, and neither the teacher
  nor the student can have memorised them, which removes a confound (b)
  concedes for no gain. Reviewer 3 takes (b) as sufficient because the
  family-disjoint post-cutoff eval already carries the leakage argument.
  Resolution: (b') if supply holds, (b) as top-up below ~350 functions.
  Either way, extend families to repository owner, not just content.
- **P10: proposer is the 9B thinking-on (reviewer 2) or the 4B itself
  (reviewer 3).** Reviewer 2 wants the 9B's better test design; reviewer 3
  wants the purest claim and notes the 9B is a same-family distillation arm
  that needs its own label. Resolution: 4B as the primary proposer (thinking
  off, same prompt, temp 0.7), 9B as an ablation arm only if T8 has slack.

## Risks the recommendation accepts

- **Unattainable labels.** Oracle-filled values the model has no path to
  compute can teach confident guessing. Mitigation: only fill literals whose
  `repr` is short (recorded threshold), keep the mutation gate, and treat
  transfer to the held-out test split and the `arith` probe as the honest
  test of whether the objection bites.
- **Capability ceiling of self-distillation.** The student cannot learn a
  test-design idea it never samples. Accepted: test design is not the
  bottleneck weekend one measured.
- **Oracle filling is ~150 lines of new harness code** (AST rewrite of
  `assert call(...) == literal`, sandboxed evaluation of the call). It adds a
  weekend-2 task and a decision entry.

## Recommended decisions (for Jordi)

- **P9 → (b')**: post-floor overflow GitHub harvest, repositories and owners
  disjoint from the held-out pool, same purity / ≥8-live-mutant filter, same
  three-layer decontamination against both held-out splits; pre-floor (b) as
  top-up if under ~350; MBPP never in training.
- **P10 → self-distillation with oracle-filled expected values**: proposer =
  Qwen3.5-4B, thinking off, temp 0.7, K=4, same prompt as inference; harness
  runs each suite on the reference, rewrites wrong short literals with the
  executed value, then gates on ≥1 training-category kill and keeps the best
  per function. No Claude. 9B as a labelled ablation arm only. Amends D008.
- **Add to T6**: best-of-4 + harness-filter ceiling row on ~100 test
  functions; difficulty-stratified and assertion-style breakdowns.
- **De-risk first (≤1 h)**: 40 dev functions, 4B K=4; pipeline A rejection
  only vs pipeline B oracle-filled then gated; record yield per difficulty
  stratum, fraction of literals corrected, and `repr` length distribution.
