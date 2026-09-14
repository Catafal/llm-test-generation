# The Harness

A 4B model, fine-tuned to write pytest suites that expose bugs, measured against the same model prompted, on 315 functions it could not have seen. Five iterations. Four of them did not move the metric, and each was measured down to its mechanism before the next one started. None of the four was powered to rule out a small effect; what closed them was the mechanism, not the interval. The lever, when it finally moved, was not the recipe, and a control run after peer review showed it was not the data either: the base model prompted for the style the adapters had learned scored higher than the adapters. In order of what moved the metric: the harness, then a paragraph of prompt, then twelve thousand teacher examples that added nothing on top of it, then four rounds of self-training that added nothing at all.

![Where the fine-tune finally moved](../charts/results.png)

Charts are generated from the run artifacts by `make charts` (`docs/charts/`, interactive HTML plus PNG). The model card is `docs/model-card.md`, the demo is `docs/demo.md`, and every number traces to `docs/results/weekend-2.md`.

## Experiments

**[001](001-the-metric-came-first.md) · The metric came first.** Before any training I built the thing that would judge it: a sandbox that runs a generated suite against the correct function and against dozens of deliberately broken copies, a pool of real-world functions written after the model's training cutoff, and paired statistics. The pilot said the metric could tell a good suite from a bad one. The baselines said validity, not bug-finding, was the headroom. 2026 · measured · 8 min read

**[002](002-i-gave-the-model-the-right-answers-and-it-got-worse.md) · I gave the model the right answers and it got worse.** The first fine-tune trained on the model's own suites with the wrong expected values corrected by execution. Validity fell 4 points. The mechanism was in the data: I had taught it to state exact values with confidence, and at inference it still could not compute them. 2026 · measured · 9 min read

**[003](003-preferences-traces-and-the-one-skill-i-could-not-install.md) · Preferences, traces, and the one skill I could not install.** Two more signals on the same bottleneck. Preference pairs from the model's own passes and failures: a clean null. Execution traces written above each assert as a scratchpad: a validity gain that turned out to be brevity, and derivations that were fluent and invented. Three signals, two bases, one number that never moved. 2026 · measured · 8 min read

**[004](004-so-i-stopped-asking-it-to-predict-outputs.md) · So I stopped asking it to predict outputs.** If the model cannot compute expected values, let the harness compute them, in every arm, base included. Filling values by execution was worth 28 validity points on its own. A fine-tune on top of it, on the model's own pairs, was worth +0.024 with an interval that included zero. Pre-registered, and it failed the bar it was registered against. 2026 · measured · 7 min read

**[005](005-four-thousand-examples-from-a-stronger-teacher.md) · Four thousand examples from a stronger teacher.** Three research passes said the recipe was fine and the data was the problem: every working small-model recipe uses a stronger teacher at a scale I had refused on purpose. Four thousand execution-verified examples from a public GPT-4o dataset, curated through my own harness, cleared the pre-registered bar. Twelve thousand repeated it and added nothing. Then the base model, prompted for the same style with no training, beat both. 2026 · measured · 10 min read

## Notes

[**The recipe was never the problem.**](the-recipe-was-never-the-problem.md) Five iterations comparing prompting and fine-tuning on one task. I spent three of them trying to install a capability in a 4B model with a few hundred of its own examples, and one changing the task so the capability was not needed. The gain came from a style a prompt delivers without training, the harness did more than any fine-tune, and pre-registration saved me more often than it vindicated me. Ends with the rule the series leaves behind, when to fine-tune and when not, and a measurement of what the harness still cannot repair. 2026 · 7 min read
