# The recipe was never the problem

Note · 14 September 2026 · reasoned, with one measurement · 7 min read

**Summary.** Five iterations comparing prompting and fine-tuning on one task. I spent three of them trying to install a capability in a 4B model with a few hundred of its own examples, and one changing the task so the capability was not needed. The gain came from a style, which a prompt delivers without training; the harness did more than any fine-tune; and pre-registration saved me more often than it vindicated me.

The project started as a work sample. The question a hiring manager would ask, I thought, was "can you fine-tune a small model and prove it got better?" The honest answer, after two weeks, is yes, and it was not worth doing. Ranked by what moved the metric, on the same 315 functions:

1. The harness. Filling expected values by execution: 28 validity points, no training, every arm.
2. A paragraph of system prompt: +0.076 over zero-shot, no training, the fastest arm.
3. Twelve thousand teacher examples: +0.062, the same style, slower, and −0.014 against the prompt with an interval including zero.
4. The model's own outputs, four ways: nothing that resolved.

The fine-tune is not wrong. It is redundant, and the proof took four failures and a control to construct, and those are the parts I would show first.

## The one number per iteration

| iteration | signal | examples | source | headline, vs base zero-shot | resolved? |
|---|---|---|---|---|---|
| 1 | SFT on oracle-corrected suites | 388 | own | validity −0.041 [−0.098, +0.013] | no |
| 2 | DPO on own pass/fail pairs | 497 | own | validity +0.003 [−0.048, +0.051] | no |
| 3 | SFT on execution-trace scratchpads | 358 | own | validity +0.054 [−0.010, +0.114], kills −0.070 | no |
| 4 | DPO on own grounded pairs, harness fills values | 645 | own | grounded score +0.024 [−0.013, +0.064] | no |
| 5a | SFT, harness fills values | 4,000 | GPT-4o via KodCode | grounded score +0.059 [+0.013, +0.105] | yes |
| 5b | same | 12,000 | same | grounded score +0.062 [+0.015, +0.106] | yes, and +0.003 over 5a |
| control | no training, base prompted for the teacher's style | 0 | a system prompt | grounded score +0.076 [+0.034, +0.118]; adapters −0.014 / −0.017 vs it | yes |

Rows 1 to 4 are one experiment repeated with a different signal each time. None of the four intervals excludes zero and none could have caught an effect under about five points, so they are unresolved rather than null; what closed each one was the mechanism measured on the generations, and the per-assert accuracy of expected values, the thing every one of them was aimed at, never moved. Row 5 is a different experiment: same recipe, different author of the data. The control row is what row 5 was worth: the style the data taught, asked for directly, with nothing trained.

![Where the fine-tune finally moved](../charts/results.png)

## Four things I would say to someone starting this

**Build the judge before the student, and let the judge have a probe.** The harness, the post-cutoff pool and the paired statistics cost a weekend and they are the reason every null in the table is a result rather than a shrug. The held-out mutation category cost nothing and gave me one sentence, "no operator drift", that I could say five times with a number behind it.

**Read the generations.** Every mechanism in this series came from reading what the model wrote, not from the aggregate. The literal-assert share in 002, the chance-level held-out pair accuracy in 003, the per-test false-failure rate with and without a derivation in 003, the counts of what the harness had to rewrite in 004 and 005. An aggregate null tells you to stop. A mechanism tells you what to change.

**Run the control a stranger would ask for before you publish.** Three reviewers read a draft that claimed six points for a fine-tune and asked the same question: did you prompt the base for the style the fine-tune learned? I had not. One generation run later the fine-tune was matched by a paragraph of system prompt. It was the cheapest run in the series and the one that changed the headline.

**When the model cannot do the thing, ask whether the thing needs doing.** The harness filling expected values by execution was worth 28 validity points, which is more than any fine-tune in the series and more than the two shipped adapters combined. It was available from the first day. I did not take it until three iterations had failed, because I had framed the task as the model's job, and the reframing was a product decision I kept treating as a training one.

## The choice I got wrong on purpose

In entry 002 I chose self-distillation over a teacher because a teacher would have made the result uninteresting. I stand by the reasoning and not the outcome. The reasoning gave me four clean nulls with mechanisms, and the literature I read afterwards says those nulls were predictable: a model this size cannot verify its own outputs well enough to learn from them. Had I read that literature first I would have skipped to entry 005 and had one positive result and no understanding of why the alternatives fail. I do not think the two weeks were wasted, but I would not spend them again the same way, and the difference between those two sentences is the point of writing this note.

## Three times the process caught me

The dev-60 winner's curse in 002: ten checkpoints, sixty functions, the best one seven functions ahead of the base on dev and four points behind on test. I had written "dev at this size is too small" as a risk and used it anyway. Every later iteration selected on all 171.

The dev gap in 003: 0.468 against 0.29, the largest in the series, on a checkpoint that lost kills on test. The dev curve was measuring brevity. I would have shipped that model on the dev number alone.

The pre-registered bar in 004: +0.024, twice, on two arms, with a lower bound that did not clear zero. It would have been easy to run a third arm, or to prefer the secondary metric where the p-value was smaller. The rule I had written down said stop, and I stopped, and the next entry is what made the difference instead.

## When to fine-tune, and when not

The series reduces to a rule, and I would rather state it than leave it implied by a table. There were two gaps between what the base model wrote and what the metric wanted, and they behaved differently under every treatment.

The first was a capability gap. The model could not compute the expected value of a function call. Four iterations aimed training at that gap, from 388 to 645 of the model's own examples, and unaided validity went from 0.438 to 0.467 across the whole series without resolving. One change to the harness, filling the value by execution, moved the same number by 28 points on the same day. If the model lacks a capability the task needs, and a tool has it, the tool should have the job. Training a 4B model to acquire it from its own outputs was the most expensive way to learn that.

The second was a style gap. The base wrote long suites with membership asserts; the metric rewarded short suites of exact-value asserts. Four thousand and twelve thousand teacher examples closed that gap for +0.059 and +0.062. A paragraph of system prompt closed it for +0.076. If the gap is style or format, prompt first, and fine-tune only when the prompt fails to reach it.

There is one argument left for the adapters, and I measured it because it is the argument I would have made in an interview: the adapter bakes the prompt in, so it should be cheaper per call. It is not, on this hardware.

| arm | prompt tokens | completion tokens | seconds per suite |
|---|---|---|---|
| base, zero-shot | 380 | 758 | 9.0 |
| 12k adapter, plain prompt | 380 | 457 | 8.7 |
| base, style prompt (control) | 738 | 262 | 4.0 |
| 12k adapter, style prompt | 738 | 246 | 4.3 |

The prompt costs 358 tokens of prefill and saves about 200 tokens of generation, and prefill is cheap where generation is not. The control is the fastest arm in the series as well as the best. So for this task, on this model, the production answer is the harness and the prompt, with no training, and the adapters are on the Hub as a record of what the training did rather than as a recommendation.

## What is left in the harness

The obvious next question is whether the harness can go further, since it did the most. I bucketed every test that still fails on the correct function after the fill, for the control and for the base, by reading the pytest message and the assert shape of the failing test. `make failures RUN=runs/<grounded run>` reproduces the table.

| | control | base, zero-shot |
|---|---|---|
| grounded-invalid suites | 44 | 89 |
| failing tests in them | 90 | 239 |
| the function itself raised on the model's inputs | 53 | 103 |
| `pytest.raises` on an input that did not raise | 0 | 22 |
| an assert shape the fill does not touch (`in`, `is`, `approx`, non-literal `==`, ordering) | 34 | 96 |
| filled literal still failing (mocks, nondeterminism) | 2 | 7 |
| suites where every test fails | 8 | 7 |

Three levers come out of it, and they are not the same kind of thing.

**Fill more shapes.** Membership, identity and approximate asserts could be rewritten the way equality is. That is about a third of the remaining failures, spread over roughly twenty of the control's suites, and it would help the base more than the control because the base writes more of those shapes. Worth doing, bounded, and it narrows the gap between arms rather than widening it.

**Prune failing tests instead of failing the suite.** Validity is per suite, so one bad test in five zeroes the function. Dropping the failing tests and scoring what survives would keep 36 of the control's 44 invalid suites and 82 of the base's 89, which puts both arms at about 0.975 grounded validity. That is the largest lever by far, and it is also why I have not pulled it: it deletes the thing the metric measures. Under pruning every arm is valid and the score collapses to kills per surviving test, which is a different question with a different answer. It is a defensible variant to pre-register and run on every arm; it is not a fix to the primary metric.

**Choose inputs.** More than half of what remains is the function raising on the inputs the model chose: a missing key, a wrong type, a mock of an attribute the module does not have. The fill repairs outputs, not inputs, and on the suites that are valid a fifth of the mutants still survive for the same reason. A harness that searched for inputs against the live mutants would close that, and at that point the harness is the test generator and the model is decoration. That is search-based test generation, an established field, and a different project.

## What I am not claiming

Not that a 4B model learned to predict the outputs of code. Unaided validity moved from 0.438 to 0.467 across the whole series and never resolved. Everything the adapters gained, they gained under a harness that executes the function for them.

Not that scale is the lever. Twelve thousand examples did what four thousand did, and a prompt did what both did.

Not that the adapters are useless. They match the prompt without needing it, which is a small convenience, and they are on the Hub with this row on their card.

Not that the harness is a free lunch. It snapshots the reference, so a wrong function gets a suite that enshrines the wrong behaviour, it fills only literal-equality asserts, and it cannot repair an input the function rejects. The 28 points come with that caveat attached, and the model card says so.

The next thing worth measuring is the one stage of the ladder I never reached: the harness as a reward for online reinforcement learning, on the dense base where rollouts are fast enough. It is the one angle with a published small-model result behind it that I did not try, and it would test whether the model can learn to choose inputs from the harness directly rather than from a teacher's examples.

← [The Harness](index.md)
