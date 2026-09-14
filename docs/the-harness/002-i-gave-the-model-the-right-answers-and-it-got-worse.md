# I gave the model the right answers and it got worse

Experiment 002 · 8 September 2026 · measured · 9 min read

**Summary.** The first fine-tune trained on the model's own suites with the wrong expected values corrected by execution. Validity fell 4 points against zero-shot prompting, on an interval that includes zero. The mechanism was in the data: I had taught the model to state exact values with confidence, and at inference it still could not compute them.

Entry 001 ended with the bottleneck named. Six suites in ten fail because the model predicts a wrong output for an input it chose itself. So the training data had to carry correct expected values, and the obvious source of correct values was sitting in the harness already: it executes the function.

## The two decisions that defined the data

I had two open questions and I put both through a three-way review before deciding, with the alternatives written down.

**Where do the training functions come from?** The held-out pool had used every post-cutoff repository I harvested up to a cap of ten functions each. The training pool is the overflow: repositories created after the same 1 June 2026 floor, never drawn for held-out, with repository owners disjoint from the held-out owners, the same purity filter, the same requirement of at least eight live mutants, and the same three-layer decontamination against both held-out splits. 586 candidates, 11 removed as too close to a held-out function, 25 merged as near-duplicates, 550 kept from 91 repositories. MBPP was never a training source. I rejected it because the evaluation had been decontaminated against it and I did not want to depend on that check being perfect.

**Where do the training suites come from?** The alternatives were a stronger teacher model writing suites, or the 4B writing its own. I chose the 4B, and I want to record why, because this decision is the one the whole series turns on. A stronger teacher would have made the story "distillation works", which everyone already knows. Self-distillation with execution as the only external signal would have made it "a small model can improve itself on a checkable task", which is the claim I actually wanted to test. The teacher stayed out.

The recipe, then. The 4B proposes eight suites per training function at temperature 0.7, same prompt as evaluation. For each candidate, the harness finds every `assert <expr> == <literal>`, runs the suite with those asserts replaced by recorders, and rewrites any literal whose recorded value differs, if the correct value's repr is at most 40 characters. A fifteen-digit float is a value the model could never compute, and I did not want to teach it to emit one. Then the filled suite has to pass on the reference and kill at least one training-category mutant. Best candidate per function, by mutation score and then by fewer tests.

A de-risk run on 40 dev functions first, because the filling step was the new idea. Plain rejection sampling covered 21 of the 40 functions with at least one valid, killing suite. Filling covered 31. Candidate validity went from 0.25 to 0.49. The hardest quartile by mutant count did not move at all. I recorded that last sentence and went ahead anyway; it was the first sign of what was coming and I read it as a yield problem.

The full run: 4,400 candidates over 550 functions, 4,345 parsed, 1,247 valid as written, 2,165 valid after filling. 412 functions ended with a keeper: 388 for training, 24 held back by family for validation loss.

## Where the weekend actually went

Qwen3.5-4B is a hybrid: eight attention layers and twenty-four Gated DeltaNet layers. The inference kernel for the DeltaNet recurrence has no gradient in mlx-lm, so training falls back to a per-token Python loop. At batch one and all layers adapted, that was 12 tokens per second and 24.9 GB of memory before an out-of-memory on an 1,100-token example. Sequence length 1,536 ran out of memory even with eight adapted layers.

What I settled on, after a day: adapt only the last sixteen of thirty-two layers, swap the frozen DeltaNet layers to a subclass that always takes the inference kernel, cap training examples at 1,024 tokens so nothing is truncated, and turn off the compilation of the training step, which had been building a fresh compiled graph for every sequence length and exhausting a Metal resource limit at around 500 of them. That got 37 tokens per second at 25 GB. Also the adapter's `scale` parameter in mlx-lm is a direct multiplier on the adapter output, not the alpha of the usual formula; I set it to 32 first, the loss diverged to 13 within a few dozen steps, and it went back to 2.0, which is alpha 32 over rank 16 in the convention I had assumed.

Before the real run: overfit eight examples to near-zero loss. Validation loss 0.212 to 0.015 in forty iterations. The pipeline could learn. That check takes ten minutes and would have saved me a weekend if it had failed, so it runs before every training in this series.

Then LoRA rank 16 on all linear projections of those sixteen layers, batch one with gradient accumulation eight, learning rate 1e-4 with cosine decay, three epochs, a checkpoint every 120 micro-batches. Ten checkpoints, each scored by the harness on 60 dev functions, because validation loss is a proxy and the harness is the thing.

## The result

Test split, 315 functions, everything paired.

| arm | validity | mutation score on valid suites | tokens | tests generated |
|---|---|---|---|---|
| base, bf16, zero-shot | 0.438 | 0.862 | 758 | 12.2 |
| base, bf16, few-shot | 0.419 | 0.889 | 442 | 6.5 |
| fine-tune, checkpoint 120 | 0.397 | 0.862 | 537 | 8.1 |

Fine-tune against zero-shot: validity −0.041, 95% interval [−0.098, +0.013], McNemar p = 0.17, with 45 functions valid only for the base and 32 only for the fine-tune. Against few-shot: −0.022, [−0.083, +0.035]. Mutation score on the 93 functions both arms got valid: −0.017, [−0.041, +0.003]. Kill rate on the held-out arithmetic category, which no curation ever saw: 0.743 for the fine-tune against 0.714 for the base. No operator overfitting.

The interval includes zero, so by the rule I wrote in 001 this is unresolved, not a demonstrated harm. The point estimate is negative, and the dev curve had said the opposite.

## What the model learned

I read the generations rather than the aggregate, because the aggregate was a null and a null on its own teaches nothing.

It learned brevity. Eight tests instead of twelve, 30 percent fewer tokens, half the truncations. Few-shot prompting gets the same brevity for free, with better validity, so brevity was not what was missing.

It learned to assert exact values. The share of `== <literal>` asserts rose from 0.724 to 0.771 and membership asserts fell to match. That is the mechanism, and it was predictable from the data: every training target had its literals corrected by execution, so every target was a suite where confident exact values were right. The model cannot see why they were right. It learned that the right answer looks like a confident exact value, and at inference it produces confident exact values that are wrong at the same rate as before. 190 false-failure suites against the base's 176.

I had written the risk down before training: "correcting the literals produces targets the model could not have produced itself". I wrote it as a risk and treated it as a yield detail. It was the whole result.

## The dev curve, and the mistake in it

Harness validity on 60 dev functions across the ten checkpoints: 0.433, 0.367, 0.433, 0.400, 0.417, 0.383, 0.383, 0.383, 0.383, 0.400. The base scored 0.320 on the same 60. Every checkpoint beat the base on dev. None did on test.

Picking the best of ten checkpoints on 60 functions is a winner's curse, and I knew that in the abstract and did it anyway because 60 was what fit in the time. The +7-function advantage of the chosen checkpoint was selection noise. From this entry on, checkpoint selection uses all 171 dev functions, and I report the whole curve rather than the winner.

## What this does not say

It does not say self-distillation cannot work. It says that self-distillation with the labels corrected by an oracle the model does not have at inference does not work, on this task, at this scale.

It does not say the 4B cannot learn from execution feedback at all. Entry 003 tries a signal that carries negatives.

## Threats to validity

Single seed, single training run. The dev-60 checkpoint selection, above. LoRA on sixteen layers and 1,024-token targets were forced by the training path, not chosen, and all-layer adaptation was never tested on this base. The bf16 weights scored 0.438 where the 4-bit weights had scored 0.400 on the same split in 001, so the baseline moved under me between weekends; every comparison here holds the format fixed at bf16.

## How to check this instead of trusting me

```
git clone https://github.com/Catafal/llm-test-generation
make setup && make sync-models
uv run python -m testgen.stats runs/<test run>/outputs.jsonl 4b/zero 4b/few
```

The training data, the yield table, the oracle statistics (33,546 literal sites, 4,157 rewritten, 727 skipped as too long) and the per-checkpoint dev manifests are committed. The adapter manifest records the config hash, the data hash and the seed.

## What I take from this

The oracle worked as a data lever and failed as a teaching lever, and those are different things. It doubled the number of usable training suites, and every one of those suites carried a correction the model had no way to reproduce. I had assumed that more correct examples of the target behaviour would move the model toward the target behaviour. What moved was the surface form of the target behaviour.

The uncomfortable version: I had spent a weekend building a way to hand the model right answers, and the model learned what right answers look like.

← The Harness
