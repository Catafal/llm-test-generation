# Decisions

Every decision in the project, in the order it was made, with the options
that were on the table and the reason for the choice. This is a curated copy
of the working log I kept during the nine days; the private version also
holds session notes and pending questions, and nothing of substance is
removed here. Results that came out of a decision are stated with the
decision, so a reader can see what each one cost and what it settled.
Reversals are kept as written: three of these were overturned by review or
by measurement, and those are the ones I would point at first.

A "D" number is referenced from the series, the model card and the research
folders under `docs/research/`, which hold the sourced material behind the
choices that needed it.

## Day one: scope and the metric (6 September)

**D001. A standalone repo, with the reasoning trail kept private until curated.**
Building inside my notes vault would mix personal material with a public
artifact. The decision log stays private during the work because it also
tracks how the project fits a job search; this file is the extraction it
promised.

**D002. Evaluation first, training second.** Two weekends were budgeted and
the risk was spending them training against a metric that measured nothing.
Weekend one had to produce the harness, the mutation operators, a ten-case
pilot and the prompting baselines, and training could not start until the
pilot showed the metric separated good suites from bad ones. Rejected:
training first with an ad-hoc eval, which reaches a number faster and would
not survive a technical interview.

**D003. Training signal: execution-verified distillation.** Sample candidate
suites, run each through the harness, keep the ones that pass on the
reference and kill mutants, fine-tune on the kept ones. Rejected: raw
distillation (teaches the teacher's mistakes), mining repository tests
(licence and alignment work beyond the budget), reinforcement learning
(out of scope at the start). The trade-off accepted: the filter uses the
same harness as the final evaluation, so the student is optimised toward
the metric; mitigated by held-out functions and by reporting validity
separately.

**D004. MBPP for training, HumanEval held out.** Overturned by D012 a day
later: both sets are in every candidate model's pretraining.

**D005. A small custom set of AST mutation operators.** Comparison flips,
boundary constants, boolean swaps, return changes; equivalent mutants
excluded. Rejected: `mutmut` and `cosmic-ray`, larger operator sets that
are harder to audit and produce more equivalent mutants. Superseded by D016.

**D006. Scope: unit tests for one pure, self-contained function at a time.**
Deterministic, plain inputs and outputs, standard library only. Bugs are the
planted mutant classes, and the write-up says mutation score is a proxy.
Rejected: integration tests between functions (no clean ground truth) and
anything with I/O, randomness, time or third-party dependencies (unsafe or
non-deterministic in the sandbox). Narrow and measured beats broad and
unverifiable.

**D007. The model sees signature, docstring and the reference
implementation.** That is what a coding agent has when asked to "add tests
for this". Rejected: docstring only, because the docstrings in the candidate
pools are too thin and failures would measure specification ambiguity rather
than the model.

**D008. Teacher: a frontier model through a subscription SDK.** Overturned by
D023 after review: bulk programmatic use through a subscription is an
unresolved terms-of-service question, and a frontier teacher turns
"fine-tune versus prompting of the same model" into a distillation result.

**D009. Python only.** A TypeScript transfer condition was interesting and
would have added a second runner and mutation set for a secondary claim.
The public claim is a recipe demonstrated in Python.

## Day two: the model, the machine, the pool (7 September)

**D010. Base model: Qwen3.5-9B, thinking off, with the 4B as fallback.**
Research in `docs/research/2026-09-06-base-model-selection/`. Rejected:
Qwen2.5-Coder-7B (a 2024 model with weaker reasoning), 1.5B models (reason
too little), base variants without a prompting baseline, and thinking mode
anywhere (reasoning tokens consume the fixed budget and confound SFT
formatting). Trade-offs accepted: no published non-thinking coding numbers
for Qwen3.5 and no published Mac training run for the 9B. Addendum the same
day: bf16 LoRA, not 4-bit, following the Unsloth warning on Qwen3.5.

**D011. Compute: the Mac first, no spend by default.** The tooling research
recommended cloud-first for the 9B. I chose the M4 Pro 48 GB, with Colab as
the only fallback and no rented GPU. Rejected: RunPod or Lambda at 5 to 20
dollars a run, cheap but spend I did not want to normalise. The consequence,
accepted: if the local path failed, the honest options were a 4B on Colab or
a new decision.

**D012. Held-out pool: post-cutoff functions, family split, three-layer
decontamination.** HumanEval and MBPP are contaminated for every candidate
model and their published scores have not reproduced. Held-out evaluation
uses pure Python functions written after the model's training cutoff, split
by function family, decontaminated against the training pool by n-gram,
embedding and AST similarity. Rejected: keeping HumanEval and stating
contamination as a limitation, cheaper and weaker.

**D013. Keep the custom operators; add equivalence filtering and a held-out
operator category.** Rigor research preferred an established tool's
operators for comparability. I kept the auditable custom set and added
bytecode-equivalence filtering plus one operator category held out of every
training signal as the operator-overfitting probe.

**D014. Build mode: paired.** I make the architectural choices, the assistant
implements, and both are recorded. Ownership is satisfied by being able to
explain every choice, not by typing it alone. Rejected: building the harness
unassisted, which coupled a separate goal to the critical path.

**D015. Mutant set design after three independent reviews.** My proposals
(cap five mutants per function; boundary edits as the held-out probe) went
to three reviewers and all three rejected both. Adopted: uncapped mutants,
one per site, frozen to a versioned artifact shared byte-identically across
arms; the probe is a fifth operator, arithmetic swaps, never used in any
training signal; equivalence filtering kept but stated to catch little.
Trail: `docs/research/2026-09-07-mutation-design-review/`.

**D016. The engine implements the full mutmut-equivalent catalogue; the
active set is a recorded profile.** I challenged my own earlier table: if
the recipe is meant to apply to any Python function, the engine should know
every bug class the established tools know. Which categories are active is
a named profile recorded in every artifact. Two categories that kill by
crashing rather than by logic are off in the evaluation profile.

**D017. Held-out pool construction.** Floor date 1 June 2026, three months
after the model's release; GitHub repositories under permissive licences
created after the floor; proof of authorship per function from the commit
that introduced the file; complexity defined as at least eight live mutants
under our engine; function bodies committed verbatim under a NOTICE file.
Rejected: date-stamped benchmarks (stale), puzzle sites (licence),
reference-by-URL (a reviewer could not run the eval). Accepted: "post-cutoff"
is a margin, not a proof.

**D018. Output normalisation, identical in every arm.** An unclosed code
fence is trimmed to the last parseable statement and flagged; a missing
`import pytest` is injected and flagged. Rejected: treating both as failures,
which would make validity a proxy for prompt compliance.

**D019. Generation budget 2,048 tokens; all weights inside the repo.** The
9B truncated at 1,024 on the pilot. Weights live in gitignored folders with
pull and remove targets, so the experiment can be deleted cleanly.

**D020. Pool of 503 split by family into test and dev.** Evaluating every arm
on all 503 cost about seven hours a pass. Whole families assigned with a
fixed seed: test 315 after final filtering, dev 171. Test is touched only for
final numbers. Accepted: 315 resolves medium effects only; a few-point
difference is reported as unresolved, not claimed.

**D021. Fine-tune target: Qwen3.5-4B.** Held-out baselines put the 9B and the
4B at the same validity within the interval. The smallest model with an
equivalent baseline is the target; the 9B stays as a baseline row. A weaker
headline model, offset by the finding that size bought nothing here.

**D022. Training functions from a second post-cutoff harvest, owners disjoint
from the held-out pool.** Three reviewers unanimously rejected MBPP as the
training pool: no docstrings, few mutants, a different prompt shape, so a
null would be uninterpretable. Trail:
`docs/research/2026-09-07-weekend2-p9-p10-review/`.

**D023. No external teacher: the model proposes its own suites, and the
harness fills the expected values.** Two of three reviewers called the
subscription-SDK teacher fatal rather than a limitation. The proposer became
Qwen3.5-4B itself, and for every candidate the harness rewrote wrong literal
expected values with the executed value before filtering. Accepted risk,
stated in advance: filled values the model cannot compute may teach
confident guessing. That risk is exactly what D025 measured.

**D024. Training path on mlx-lm.** The hybrid Qwen3.5 stack trains its
DeltaNet layers with a per-token loop, so all-layer LoRA ran out of memory
at any useful sequence length. LoRA on the last 16 layers, sequence 1,024,
an inference-kernel subclass for the frozen layers (four times the
throughput), and after two crashes a root cause: the compiled training step
re-specialised on every sequence length and leaked Metal buffers.
Disabling compilation fixed it and cut peak memory. Rejected: writing a
proper backward kernel, a day of work with correctness risk, recorded as the
first thing to do after an interpretable result.

## Iterations one to three (8 to 10 September)

**D025. Result: SFT on oracle-corrected self-samples does not raise
validity.** Validity 0.397 against 0.438 zero-shot, interval including zero.
The mechanism, measured: nearly every invalid suite is a wrong expected
value; the corrected targets were ones the model could not have produced, so
it learned to state exact values with more confidence without computing them
better. The dev-60 gain that selected the checkpoint was noise over ten
checkpoints. What did work: oracle filling as a data lever, brevity, the
probe, and the fact that the negative was detected before any claim.

**D026. Iteration two: preference learning on the model's own pass and fail
pairs.** Positives-only SFT cannot teach what not to assert. One variable
changed, the signal; the base stayed so the baselines remained comparable.
Rejected for now: switching to a dense base (plumbing, not the fix), more
SFT data, trace targets (blocked on this base by sequence length).

**D027. Result: a null on validity.** Plus 0.003 against zero-shot. The
preference was learned on the training pairs and stayed at chance on held-out
pairs: distinguishing the model's own wrong literals from right ones needs
the computation it lacks. It did remove the SFT harm.

**D028. Iteration three in two steps: an oracle-shape prompt, then
execution-trace targets on a dense base.** I asked for the problem, the
odds and the wider option space before choosing, and chose both in that
order with the odds written down first (about 45 percent and 30 percent).
Step one failed on dev. A refined diagnosis on the base's invalid suites:
three quarters fail only on wrong values, one quarter on inputs the
function rejects, and 63 of 177 are one failing test from valid. Step two
moved to the dense Qwen3-4B because scratchpad targets did not fit the
hybrid path's sequence cap; new baselines were run for it.

**D029. Result: trace targets teach the form of a derivation, not the
computation.** Validity unresolved, mutation score significantly down. The
derivations were fluent and invented; per-test false-failure rate was no
better with a derivation than without, and the validity gain came from
shorter suites. Three signals, two bases, and the per-assert value accuracy
never moved. The decision at the time: close the training programme and
ship the negative result with its mechanisms.

## Iteration four: change the task (10 to 11 September)

**D030. Execution-grounded expected values in every arm, base included;
train for kills.** Options: (A) local teacher distillation at scale, the
principle I had skipped, odds 35 to 45 percent because the student still
guesses values at inference; (B) let the harness fill every literal-equality
assert from execution in both arms and measure input choice, odds 55 to 65
percent; (C) ship the null. I chose B and pre-registered it: primary metric
the mean grounded score paired over all 315 functions, success only if the
interval's lower bound clears zero, two training arms and no more, stop
after the second whatever the result. Oracle tautology accepted as a stated
limitation: filled asserts snapshot the reference.

**D031. Result: the harness is the gain; the fine-tune on top is
unresolved.** Filling lifted every arm from about 0.43 to about 0.71
validity. Grounded DPO scored +0.024 with an interval including zero, and
the re-scored SFT adapter landed on the identical point estimate. The rule
said stop, and I stopped: no third arm, no metric change.

## Iteration five: the teacher I had refused (11 to 14 September)

**D032. A three-stage ladder, cheapest first, stop at the first resolved
gain.** Research (`docs/research/2026-09-11-fine-tuning-practice/`) said the
nulls were a self-generated-data ceiling, not a model ceiling. Stage one: an
existing public dataset curated through my harness, go rule at least 5,000
clean examples. Stage two: a local 30B-class teacher. Stage three: online RL
with the harness as reward. Base kept as Qwen3.5-4B despite the slower
training path, because switching would confound the comparison with four
earlier iterations; the cost accepted was a cap of 4,000 examples.

**D033. Stage one clears the bar.** KodCode-V1 (GPT-4o solutions, execution
verified), 4,000 examples: +0.059 with interval +0.013 to +0.105,
pre-registered. Mechanism: style transfer, short suites of exact-value
asserts that the harness can complete. Unaided validity unresolved; the
model did not learn to predict values. Ladder stopped at stage one.

**D034. A second point at 12,000 examples, registered before it ran.** The
first objection to a 4,000-example result is that it is small, and the
curation yield table said the same pipeline had 12,000 rows at the same
bar. Result +0.062, and +0.003 against the 4k adapter: replication, no
dose-response.

## The control, and what it changed (14 September)

**D035. A control after peer review, pre-registered.** Three reviewers read
the draft and converged on one question: the adapters write the teacher's
style, which is what the harness repairs, and the base was never prompted
for it. The reviewers also caught a column in the results table that showed
unaided values in the grounded position; corrected with a note. Control:
the base with a system prompt asking for one short test per behaviour and
exact return values, two training examples as exemplars, no training. The
reading rule was written before the run: if the control's gain clears zero
and the adapters' gain over it does not, the headline becomes the style, and
prompting gets it.

**D036. The control beats the fine-tune.** Grounded score 0.680, +0.076 over
zero-shot; both adapters below it with intervals including zero; unaided
validity 0.571, the only thing in the series that moved it. Headline
rewritten everywhere. The reviewers were right, and the control was the
cheapest run in the series.

**D037. The best adapter under the best prompt, pre-registered.** 0.667, the
same number as under the plain prompt, and −0.014 against the control. The
adapter and the prompt teach the same thing and do not stack.

**D038. What is left in the harness, measured rather than built.** Instead
of building a harness variant, I bucketed every test still failing after
the fill. More than half are the function raising on the model's inputs,
which no fill repairs; a third are assert shapes the fill does not touch;
pruning failing tests would put every arm near 0.975 validity and erase the
quantity being measured, so it is recorded as a variant to pre-register,
not a fix. Per-call cost was also measured: the prompted base is the fastest
arm, which removed the last production argument for the adapters.
