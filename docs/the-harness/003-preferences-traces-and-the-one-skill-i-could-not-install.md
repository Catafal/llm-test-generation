# Preferences, traces, and the one skill I could not install

Experiment 003 · 10 September 2026 · measured · 8 min read

**Summary.** Two more signals aimed at the same bottleneck. Preference pairs built from the model's own passing and failing suites: no measurable change, with an interval that would have missed a gain below five points, and the assertion mix returned to the base's. Execution traces rendered above each assert as a scratchpad: a validity gain that turned out to be brevity, and derivations that were fluent and invented. Three signals, two bases, and the per-assert accuracy never moved.

After 002 I wrote down what I believed. Any method that leaves the model guessing expected values in a single forward pass is capped near the base rate. So the lever had to do one of two things: teach the model what not to assert, which needs a signal with negatives in it, or give it a way to compute at inference, which means reasoning tokens. I ran one of each.

## Iteration 2: the model's own preferences

Supervised fine-tuning on good examples never contains a wrong answer. The data from 002 did contain them: for most functions the model had written both a suite that passed and a suite that failed, on the same input, differing in a literal. That is a preference pair, and DPO trains on pairs.

497 pairs from 303 functions. Chosen: a suite that passed on the reference as written, no oracle help, and killed at least one mutant. Rejected: a suite for the same function that failed, and where possible the ones the oracle had rescued, because those are the exact failure mode, good tests with wrong values. Both sides under 1,024 tokens. DPO through `mlx-lm-lora`, rank 8, the same sixteen layers, beta 0.1, learning rate 5e-6, two epochs, the reference model a frozen copy of the base. Checkpoint chosen on all 171 dev functions this time.

| arm | validity | mutation score on valid | tokens |
|---|---|---|---|
| base, zero-shot | 0.438 | 0.862 | 758 |
| base, few-shot | 0.419 | 0.889 | 442 |
| SFT, entry 002 | 0.397 | 0.862 | 537 |
| DPO, checkpoint 900 | 0.441 | 0.874 | 548 |

Against zero-shot: validity +0.003, [−0.048, +0.051], p = 1.00. Against the SFT: +0.044, p = 0.13. The assertion mix went back to the base's, literal-equality share 0.721, and the false-failure count went back to exactly the base's 176. Brevity stayed.

So DPO returned the assertion mix and the false-failure count to the base's and added nothing the interval could resolve. The most informative number is one from training: accuracy on the 27 held-out pairs stayed at chance, between 0.44 and 0.61, while training accuracy reached 0.9. The model could memorise which of two suites was right. It could not tell, for a function it had not seen, which of two literals was the right one, because telling requires the computation it does not do.

## Iteration 3: give it somewhere to compute

The one published result I found where a model this size learned to predict outputs used about eighty million execution traces in pretraining. I had a laptop and a weekend. The cheap version: let the model reason before each assert, in the output, where reasoning costs tokens but is at least possible. And train that reasoning on real traces, not on the model's own guesses.

Two changes at once here, and I want the reason for each on the record. The base moved to the dense `Qwen3-4B-Instruct-2507`, because the hybrid stack's training path caps examples at 1,024 tokens and a suite with a scratchpad above every assert does not fit. New baselines were run for the dense base: 0.346 zero-shot, 0.368 few-shot, a weaker model than the hybrid, and every comparison in this iteration is against those. The target format changed to the model's own unaided-valid suites with a comment block above each literal assert, rendered from a trace recorded with `sys.settrace` while the sandbox ran the function on that exact input. At most four traced asserts per suite, six lines per trace. 358 training examples, of which 289 carried at least one trace.

Before that, a cheaper probe: the same prompt, but asking the model for assertions it could be certain of without computing outputs, membership, types, relations between calls, exact values only when readable off the code. On dev it scored 0.357 against 0.398. Asking for fewer exact values produced fewer exact values and more failures on bad inputs. I also turned the model's own thinking mode on with the same budget: 0.333 against 0.32. Nothing there either.

The trace-target model trained on all 36 layers, rank 8, sequence 2,048, two epochs. Validation loss went from 0.631 to 0.171. On dev-171 the best checkpoint scored 0.468 against the dense base's 0.29, and for a day it looked like the answer.

| arm, dense base | validity | mutation score on valid | tests | tokens | truncated |
|---|---|---|---|---|---|
| zero-shot | 0.346 | 0.870 | 8.4 | 613 | 5 |
| few-shot | 0.368 | 0.893 | | 429 | 2 |
| trace-target SFT, checkpoint 250 | 0.400 | 0.763 | 6.7 | 1,270 | 83 |

Validity +0.054, [−0.010, +0.114], p = 0.11. Mutation score on the 66 both-valid functions: −0.070, [−0.131, −0.010]. Unresolved on validity, a significant loss on kills.

## Reading the generations

The validity gain came with a kill-rate loss, and the two have the same cause, which I could see only by reading what the model wrote.

261 of 315 suites contained derivations, about five each, 60 percent of the tokens in a suite. 83 suites hit the 2,048-token budget. Then the number that decided it: the per-test false-failure rate was 0.267 for tests with a derivation above them and 0.251 for tests without. The derivations did not make the assert under them more likely to be right. They were fluent, shaped exactly like the traces in the training data, and invented. The model had learned to narrate an execution it was not performing.

The validity gain was shorter suites. 6.7 tests instead of 8.4, and 37 of the 83 truncated suites were valid because truncation had removed tests. The kill-rate loss was the same brevity from the other side.

I should say that I never ran the variant where the scratchpad is stripped at inference. The per-test analysis says the derivations do not help the asserts they precede, so stripping them would recover the budget and nothing else, and I chose to spend the remaining time on a different question instead.

## What this settles

Three signals: literals corrected by execution, the model's own pass and fail preferences, derivations grounded in real traces. Two bases. One metric. The share of asserts with a wrong expected value did not move once. What moved each time was form: how many tests, how confident the literals, whether there was a comment block above them.

At a few hundred examples through LoRA, this model does not acquire output prediction, and the one result that did acquire it at this size used five orders of magnitude more traces. That is a capability of the base checkpoint, not a choice in my recipe. I closed the programme on that sentence, wrote it down as a decision, and the next entry is what happened after I had accepted it.

## What this does not say

Not that DPO or scratchpads are useless. DPO here did precisely what a preference signal should on a model that can tell the pairs apart, and this model could not. Scratchpads need a model that can execute in its head at least sometimes, which is the thing being tested for.

Not that a dense base is worse than a hybrid one. The dense base scored lower on this task at zero-shot, and it was chosen for a training-path reason, not a capability one.

## Threats to validity

Iteration 3 changed two things at once, base and target, with the reasons above. It means the trace result cannot be separated into "the dense base" and "the format" from this run alone; the per-test analysis is what carries the mechanism claim, not the aggregate. The dev-171 curve for the trace model, 0.468 against 0.29, is the biggest dev gap in the series and the smallest test gap, and I take that as a lesson about dev curves rather than about the model.

Single seed, single run, in every iteration. n = 315 resolves about six validity points and none of these effects reached that.

## How to check this instead of trusting me

```
git clone https://github.com/Catafal/llm-test-generation
make pairs && make dpo          # iteration 2 data and training
uv run python -m testgen.train.tracetargets   # iteration 3 targets from traces
```

The pairs file, the trace targets with their statistics (1,068 traced asserts over 373 functions), every dev-curve manifest and both test-run manifests are committed. The per-test false-failure analysis is a script over the committed generations.

## What I take from this

I had been looking for the recipe that would install a skill, and I spent three iterations learning that the skill was not installable at this budget. Each iteration measured its own mechanism, which is the part I would keep: not "it did not work", but "it learned the form and not the computation", three times, with the numbers that show the difference.

The model writes tests well. What it cannot do is tell me what its own tests should expect. I had been trying to fix that in the model. The next entry stops.

← The Harness
