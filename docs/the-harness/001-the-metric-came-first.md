# The metric came first

Experiment 001 · 7 September 2026 · measured · 14 min read

**Summary.** Before any training I built the thing that would judge it: a sandbox that runs a generated suite against the correct function and against dozens of deliberately broken copies, a pool of real-world functions written after the model's training cutoff, and paired statistics. The pilot said the metric could tell a good suite from a bad one. The baselines said validity, not bug-finding, was the headroom.

I wanted a work sample that shows one thing: that I can take a small model, fine-tune it for a task, and say honestly whether it got better. Not a demo where the fine-tune wins because the baseline was never tried properly. So the first weekend went entirely on the judge, and no training ran until the judge had been checked against something outside itself.

The task is narrow on purpose. One pure Python function goes in, with its docstring and its implementation. One pytest module comes out. A good module passes on the function as written and fails on as many hidden faulty variants of it as possible, under a fixed budget of tokens and test count. That is the whole task. No classes, no fixtures, no mocks, no I/O, no repository context. I chose the narrowest version of "writes tests that find bugs" that still has a measurable answer, because everything wider adds a variable I could not control on a laptop.

## Decisions I made before seeing a number

I keep a decision log in the repository, and by the end of the first weekend it had twenty-one entries. The ones that shaped everything after are these.

**Evaluation first, training second.** The harness, the mutation operators and the held-out pool were built and frozen before the training pipeline existed. The reason is the same as in every other entry on this notebook: a metric chosen after you have seen how your model does on it is not a metric.

**The model sees the implementation.** I considered giving the model only the signature and docstring, which is closer to test-driven development and a harder task. I rejected it because it makes the metric ambiguous: a suite can be wrong because the model misread a spec or because the spec was underspecified, and I cannot separate the two. With the implementation visible, a wrong expected value is a wrong expected value.

**Mutation score, with validity reported next to it.** A mutant is a copy of the function with one small change: a comparison flipped, a constant off by one, a boundary moved, an arithmetic operator swapped. A suite kills a mutant if at least one test fails on it. The metric is killed over live, where live excludes mutants that compile to identical bytecode. A suite that fails on the correct function gets no mutation score at all, not zero. That rule matters: without it a model can score by writing tests that always fail.

**One mutation category held out of everything.** Arithmetic-operator swaps are generated and scored but never used to select training data. If a fine-tune's kill rate on that category drifts away from the others, it learned my operators rather than the task. It never drifted, in any of the five iterations, and I am glad I could say that with a number.

**Post-cutoff functions only.** The held-out pool is pure functions harvested from public GitHub repositories, each introduced by a commit dated on or after 1 June 2026, after the base model's training data ends. 538 candidates, decontaminated against MBPP with n-gram, code-embedding and AST layers (nothing was removed, which is what you expect from post-cutoff code, and I ran the check anyway), 486 kept from 95 repositories, near-duplicates merged into 86 families, and split by family, never by function, into test 315 and dev 171. The split seed is in the repository.

**Equal budget everywhere.** 2,048 new tokens, at most 8 test functions, greedy decoding, thinking off. The 8-test cap is enforced after generation by keeping the first eight, identically for every arm, so nobody wins by writing more tests. The same normalisation applies to every arm's output: an unclosed code fence is trimmed back to the last parseable statement, a missing `import pytest` is injected, and both are counted and reported per arm.

**Paired statistics.** Every arm scores the same 315 functions, so every comparison is within-function: a paired bootstrap confidence interval on the difference, and exact McNemar on validity. I wrote down what n = 315 can resolve before I had a result to be disappointed by: about six points of validity. Anything smaller would be reported as unresolved, whichever direction it pointed.

## What I measured

A ten-case pilot first. For each of ten hand-written functions I wrote a strong suite and a weak one and checked that the harness ranks strong above weak on every case. It did. The pilot is also where the surviving mutants got hand-labelled as equivalent or not, which is how the equivalence filter and the labelling rule got into the design.

Then baselines on the 315 test functions for three candidate base models.

| model, condition | validity | mutation score on valid suites |
|---|---|---|
| Qwen3.5-4B, 4-bit, zero-shot | 0.400 | 0.869 |
| Qwen3.5-4B, 4-bit, few-shot | 0.384 | 0.889 |
| Qwen3.5-9B, 4-bit, zero-shot | 0.378 | 0.864 |
| Qwen3.5-9B, 4-bit, few-shot | 0.340 | 0.871 |
| Qwen2.5-Coder-7B, 4-bit, zero-shot | 0.254 | 0.882 |
| Qwen2.5-Coder-7B, 4-bit, few-shot | 0.352 | 0.908 |

Validity is the share of suites that pass on the correct function as written. Mutation score is only over those.

## The result

Read the two columns as two different findings.

When a suite is valid, it already kills 86 to 91 percent of the mutants, and every model and condition lands in that band. Bug-finding, in the sense the metric measures it, was never the problem.

Validity is the problem. Six suites in ten fail on the correct function, and when I read the failures they are almost all the same failure: an `assert f(x) == <value>` where the value is wrong. The model chose a reasonable input, wrote the call correctly, and predicted the output incorrectly. In the 4B's zero-shot arm, 176 of 315 suites are invalid, and nearly all of them for that reason.

The 9B was not better than the 4B. Paired on the same functions the difference was not resolvable, and the 4B is less than half the size and trains in a fraction of the time on my machine, so the 4B became the target. I want to say plainly that this was a null result before a single fine-tune ran, and it already told me something: within this family, size was not buying value prediction.

One more number from that weekend that I would keep. I sampled four suites per function at temperature 0.7 and kept the first that passed on the reference. On the first 100 functions that reached 0.54 validity. Four independent draws at 0.40 would give 0.87 if failures were independent, and they are not: the functions the model gets wrong, it gets wrong four times. Failures are per function, not per sample.

## What this does not say

It does not say the model writes bad tests. The valid suites are good by this metric. It says the model cannot reliably tell me what its own tests should expect, which is a different and more specific complaint.

It does not say anything about tests for classes, stateful code, or a repository. One function, one module, one number.

## Threats to validity

The mutants are mine. A small custom set of AST operators, chosen before any model was run, but still a definition of "bug" that I wrote. A suite that kills all of them has been measured against my definition, nobody else's. The equivalence filter is bytecode fingerprinting, which catches the trivial cases and no others, and the hand-labelling of the rest was done only on the pilot.

"Post-cutoff" is a margin, not a proof. The cutoff is a date on a commit, and models are trained on snapshots whose dates are approximate. The decontamination against MBPP is the check I can run; the check I cannot run is against the base model's training set.

The 4-bit weights scored 0.400 where the bf16 weights of the same model later scored 0.438 on the same split. I did not know that yet when I chose the base, and every later comparison holds the weights format fixed because of it.

## How to check this instead of trusting me

```
git clone https://github.com/Catafal/llm-test-generation
make setup && make test && make pilot
```

The harness has its own tests. Every run writes a manifest with the git sha, the model, the budget and the aggregate table, and the manifests are committed. The held-out pool ships with its decontamination report and a notice listing every source repository and licence.

## What I take from this

I went in expecting to measure how well a model finds bugs, and found that the thing standing between the model and the metric was arithmetic. Predict the output of a nine-line function for the input you just chose. That is what six in ten failures were, on the largest model I could run, with or without examples in the prompt.

That reframed what the fine-tune had to do. Not "write better tests". Write correct expected values. Whether a few hundred examples can teach a 4B model to do that is the question the next entries answer, and I did not guess the answer correctly.

← [The Harness](index.md)
