# Is the 4B ceiling capacity or data? Evidence review

**Question:** four LoRA fine-tuning iterations on Qwen3.5-4B/Qwen3-4B, 400-650 self-generated execution-verified examples, no external teacher, all null/unresolved (best +0.024 [-0.013, +0.064] on mutation score, n=315 held-out functions). Base model's dominant failure mode is predicting expected output *values* (validity 0.44 → 0.72 when the harness fills values by execution). Is this "4B capacity ceiling" or "too little / wrong data"?

Short answer up front: the published evidence points mostly toward **capacity/data-regime, not a hard 4B ceiling**, but the specific failure mode (value prediction) looks partly capacity-bound and partly a training-signal problem, and the study is underpowered to distinguish a true +0.02-0.03 effect from noise at n=315. The most defensible next move is teacher distillation, not more self-generated data at the current scale.

---

## 1. Fine-tuning small models (≤8B) for unit-test generation and adjacent tasks

| Work | Base size | # training examples | Data source | Metric | Reported gain over same-base prompted | Test n |
|---|---|---|---|---|---|---|
| **UTGen** (Prasad et al. 2025, arXiv:2502.01619) | Qwen2.5 3B and 7B (SFT) | Not disclosed in abstract; built by perturbing existing code-gen datasets (bugs injected) + CoT rationales — described as large-scale bootstrapped set, not hand-authored | Self-bootstrapped from existing code-gen corpora (not a stronger-teacher distillation of *test generation* specifically, though built via an LLM pipeline) | Composite metric requiring both bug-revealing input AND correct expected output | +7.59 pts on the combined metric vs. best LLM baseline; downstream pass@1 gains of +3.17% (HumanEvalFix) and +12.35% (harder MBPP+ debugging split) when used as a feedback signal for Qwen2.5-32B | Not disclosed in accessible abstract |
| **Large-scale empirical study on FT LLMs for unit testing** (arXiv:2412.16620) | 37 models, "various architectures and sizes," >3,000 A100-hrs | Not extracted (PDF unparsable via fetch) | Mixed | Multiple (syntax correctness, coverage, etc., per abstract) | Reports "large decoder-only models achieve best results" — i.e., a size-dependent finding, direction consistent with capacity mattering | Not extracted |
| **PEFT for Unit Test Generation empirical study** (Storhaug & Li, arXiv:2411.02462) | 13 models, LoRA / (IA)³ / prompt-tuning vs full FT | Not extracted | Presumably existing test corpora (Java) | CodeBLEU, pass@1, instruction/branch coverage, **mutation score** | LoRA ≈ full fine-tuning in several settings; tuned models sometimes emit *fewer executable* tests (calls to nonexistent methods, type errors) even while raw coverage improves — i.e., same "predicts plausible-looking but wrong values/calls" failure mode you observed | Not extracted |
| **ChatUniTest** (arXiv:2305.04764) | Code Llama fine-tuned variants (7B class) for Java | Not disclosed | Existing test corpora + generation-validation-repair loop | Line/branch coverage vs EvoSuite/TestSpark | Beats EvoSuite/AthenaTest on coverage, but relies on an execution-feedback *repair* loop rather than raw generation — i.e., most of the gain comes from post-hoc execution filtering, not from the fine-tune improving raw generation quality | — |
| **TestGen-LLM (Meta)** (Alshahwan et al., FSE 2024) | Not a from-scratch fine-tune; production LLM + "Assured LLMBSE" filtering pipeline | N/A (no training set — pure inference + filter) | N/A | Build rate / pass rate / coverage improvement, deployed at Meta on Instagram Reels/Stories | 75% built, 57% passed reliably, 25% improved coverage — of *generated* tests, most were filtered out by execution before being kept | N/A |
| **CoverUp** (arXiv:2403.16218) | GPT-4-class (not small) | N/A — iterative coverage-guided prompting, no fine-tuning | N/A | Line/branch coverage | Large coverage gains, but achieved via iterative execution-in-the-loop prompting/repair, not weight updates | — |

**Pattern across this table:** every unit-test-generation system that gets a real, reproducible gain at small-to-mid model scale does so by putting **execution in the loop at inference time** (repair loops, coverage-guided iteration, verifier filtering) rather than by fine-tuning alone on a few hundred self-generated examples. UTGen is the closest analogue to your setup (SFT for test generation, explicit "gets expected outputs wrong" framing) and it used a *large, systematically bootstrapped* dataset (perturbation-based, likely thousands of examples, not hundreds) plus explicit chain-of-thought supervision targeting the value-prediction failure — exactly your failure mode — and needed the 7B size to show the largest downstream effect. This is circumstantial but consistent with "your data volume and 4B size are both below the regime where this literature sees effects," not proof of either alone.

Sources: [UTGen (2502.01619)](https://arxiv.org/abs/2502.01619), [Learning to Generate Unit Tests for Automated Debugging (OpenReview)](https://openreview.net/pdf?id=yeVBHPLXxi), [Large-scale empirical study on FT for unit testing (2412.16620)](https://arxiv.org/pdf/2412.16620), [PEFT for Unit Test Generation (2411.02462)](https://arxiv.org/pdf/2411.02462), [ChatUniTest (2305.04764)](https://arxiv.org/abs/2305.04764), [TestGen-LLM writeup](https://www.qodo.ai/blog/we-created-the-first-open-source-implementation-of-metas-testgen-llm/), [CoverUp (2403.16218)](https://arxiv.org/pdf/2403.16218).

---

## 2. Scaling curves for self-generated / rejection-sampled data

The single most relevant paper found is **"Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models"** (arXiv:2412.02674, ICLR 2025). It formalizes self-improvement as bounded by the model's own **generation-verification (GV) gap** and reports:

- The GV-gap **scales monotonically with pretraining FLOPs** (roughly linear in log-FLOPs for stable verifiers).
- **Qwen2-0.5B, 1.5B, and 7B, and Llama-2-7B all show a non-positive GV-gap** across nearly all verification methods tested — meaning these models, despite non-trivial raw generation accuracy, **cannot self-improve via self-generated + self/execution-filtered data**, because their own verification signal isn't reliable enough to separate good self-generated samples from bad ones.
- Qwen2-72B, by contrast, shows a large positive gap (e.g., ~17-200% relative gain on Sudoku-style tasks).
- Authors' interpretation: self-improvement requires a **minimum level of instruction-following/reasoning capability** developed in pretraining; below that threshold, self-training on your own outputs doesn't move the needle regardless of how much self-generated data you throw at it.

This maps directly onto your result: a 4B model, four iterations, all self-generated, all null. It is independent, converging evidence for a **capacity-linked ceiling on self-improvement specifically** — not a ceiling on "can a 4B model ever be improved" (teacher-distillation evidence in §3 says no), but on "can a 4B model improve itself with its own execution-verified samples."

**ReST-EM** (arXiv:2312.06585, Google DeepMind) is the positive counter-case but it was only tested on **PaLM-2 model family** (not disclosed at small scale in the abstract; PaLM-2 variants used in the paper are mid-to-large) and explicitly reports gains that **"scale favorably with model size"** — the paper's headline claim is that self-training benefits *increase* with model size, i.e., the opposite regime from a 4B model.

**STaR** (arXiv:2203.14465) has a directly load-bearing negative result: bootstrapping fails below a capability floor — **GPT-2 could not bootstrap even on arithmetic**, because "few-shot performance must be above chance" for the first STaR iteration to generate any usable signal. This is the closest thing in the literature to a formal statement of "too small to self-improve," and it is a size/capability statement, not a data-volume one — more self-generated data from a model below the floor does not help, because the floor is about whether *any* generated sample is a useful teaching signal.

Net read on §2: the self-improvement literature converges on a specific, falsifiable claim — self-improvement gains require the base model to already clear a reasoning/verification floor, and that floor is empirically above 7B on some tasks (Mind the Gap's Qwen2-7B failure) and clearly above 0.5-1.5B on others. **A 4B model doing execution-verified self-generation is in the range where the literature has directly observed null self-improvement in comparable or larger models.** This is evidence for "capacity-bound," specifically for the *self-generated-only* condition — it does not by itself say a 4B model is uneditable by any means (see §3).

Sources: [Mind the Gap (2412.02674)](https://arxiv.org/pdf/2412.02674), [ReST-EM (2312.06585)](https://arxiv.org/pdf/2312.06585), [STaR (2203.14465)](https://arxiv.org/abs/2203.14465).

---

## 3. Teacher distillation for small code models — smallest dataset with a real gain

| Work | Student size | Data size | Teacher | Gain | Notes |
|---|---|---|---|---|---|
| **Magicoder / OSS-Instruct** (arXiv:2312.02120) | CodeLlama-Python-**7B** | **75K** synthetic instructions from real open-source code snippets, teacher-generated (ChatGPT-class) | GPT-3.5/GPT-4-class | MagicoderS-CL-7B reaches **66.5 pass@1 on HumanEval+**, beating ChatGPT (65.9) — a large jump from CodeLlama-Python-7B's un-tuned baseline (roughly mid-30s to low-40s pass@1 depending on version) | 75K teacher-distilled examples on a 7B model produced one of the largest reported small-model code gains in the literature; far larger than any self-generated dataset in this space |
| **WizardCoder / Evol-Instruct** (arXiv:2306.08568) | StarCoder-15B (also CodeLlama variants) | Evol-Instruct-expanded teacher data (exact count not confirmed here, historically ~78K seed→expanded) | GPT-3.5/GPT-4-class evolution of Code Alpaca seeds | Reported to surpass Claude/Bard-era models on HumanEval/HumanEval+ | Same family of result: tens of thousands of teacher-curated/evolved examples, not hundreds |
| **SWE-Gym** (arXiv:2412.21139) | Not disclosed at small scale in abstract | **2,438** real-world Python task instances (train), self-generated agent trajectories filtered by execution + separately trained verifier | Execution-verified, not a stronger-LLM teacher | Up to **+19 pts absolute** resolve rate on SWE-bench Verified/Lite; +32.0/+26.0% with verifier | Closer to your paradigm (execution-verified, not teacher-distilled) but the *task count* (2,438) is 4-6x your largest iteration, and it fine-tunes agents/trajectories, not single-shot generation |

**Smallest teacher-distilled dataset with a statistically-meaningful, reproducible gain on a held-out code metric found in this search: ~75K examples (Magicoder/OSS-Instruct), on a 7B model.** No paper was found reporting a real gain from a teacher-distilled set in the low hundreds to low thousands range at 1-7B scale — the smallest credible positive result in adjacent execution-verified (non-teacher) settings was SWE-Gym's 2,438 task instances, still ~4-6x your data volume, and even then most of the measured lift came from stacking a *separately trained verifier* on top, not from SFT alone.

This is the strongest single piece of evidence that **your 400-650-example scale is far outside the regime where any comparable published work has ever found a real effect**, independent of the model-size question. The field's positive results cluster at 10³-10⁵ examples; yours is at the extreme low end of what's been tried, and it's tried with the harder condition (self-generated, no teacher) rather than the easier one (teacher-distilled) that dominates the positive-result literature.

Sources: [Magicoder (2312.02120)](https://arxiv.org/abs/2312.02120), [WizardCoder (2306.08568)](https://arxiv.org/abs/2306.08568), [SWE-Gym (2412.21139)](https://arxiv.org/abs/2412.21139).

---

## 4. Statistical power at n=315 paired

With n=315 paired observations (per-function outcome, arm vs. baseline on the same function), the achievable power depends on the outcome's variance and the correlation between paired arms, but ballparking with a binary/near-binary mutation-score-style outcome:

- For a paired proportion difference (McNemar-style) at n=315, detecting a **true effect of +0.05** at 80% power and α=0.05 typically requires the discordant-pair rate to be favorable; a rough two-proportion (unpaired-equivalent) calculation needs **n≈310-620 per arm** to detect a 5-point absolute swing around a 40-70% base rate at 80% power — i.e., n=315 sits right at the edge of being adequately powered for a ~5pt effect, and is clearly **underpowered for anything in the 2-3pt range**, which is where your observed +0.024 sits.
- Paired designs (same function scored twice) roughly halve the required n relative to unpaired for the same detectable effect, if the correlation between conditions is high (it typically is here, since it's the same function/mutants) — this pulls the "detectable at n=315" floor down toward ~3-4pts, but that is still above your observed +0.024.
- The CI you report, **[-0.013, +0.064]**, spans zero and is ~7.7 points wide — consistent with a study that is exactly powered to detect ~5-8pt effects and therefore cannot distinguish a true +0.02-0.03 effect from noise. This is not evidence the effect is fake; it's evidence the study **cannot tell** real-but-small from zero.
- For comparison, papers with strong, reproducible test-generation results tend to either (a) use much larger held-out sets (thousands of functions, e.g. HumanEval+/MBPP+ derivatives, TestEval-style corpora often in the low thousands) or (b) report much larger effect sizes (Magicoder's ~10-30pt HumanEval swings, SWE-Gym's 19pt absolute), both of which sidestep the power problem rather than solve it statistically.

Bottom line: **+0.024 is plausibly a real, small, positive effect that n=315 simply cannot resolve** — but it is equally plausibly zero or a small negative. The data cannot adjudicate; only a larger n or a larger true effect can.

(No single canonical "sample size for test-generation LLM eval" paper was pinned down with a URL in this pass — this section is a power-calculation argument from your reported CI, cross-checked against the observation that positive-result papers in §1 and §3 use n in the thousands or effect sizes an order of magnitude larger.)

---

## 5. Verdict — most evidence-backed path to a measurable (≥+0.05) gain on a Mac, 48GB, no cloud

| Path | What the evidence says | Probability of reaching ≥+0.05 | Confidence in that estimate |
|---|---|---|---|
| **(a) 10-50x more self-generated data (same recipe, more training functions)** | Directly contradicted by Mind the Gap (7B and smaller show non-positive GV-gap; more self-generated volume doesn't fix a broken verification signal) and by STaR's capability-floor argument. Your own within-study evidence (4 iterations, 400→650 examples, all null) is itself a mini dose-response curve showing no volume-response inside the range you tried. More of the *same* self-generated recipe is the least-supported option. | **~10-15%** | Medium — grounded in Mind the Gap + your own null dose-response, but "50x" is a big enough jump that a nonlinear threshold effect can't be fully ruled out. |
| **(b) Teacher distillation from a local 30B-class coder (4-bit) on the Mac** | Best-supported path. Every unambiguous positive result in §1 and §3 (UTGen, Magicoder, WizardCoder, SWE-Gym-style verifier data) either uses a stronger model/teacher signal or execution-in-the-loop repair at *much* larger scale than you've tried. Distillation directly targets your diagnosed failure mode (value prediction) by handing the 4B model gold expected-output reasoning traces instead of asking it to invent them. 48GB unified memory comfortably runs a 30B 4-bit coder (e.g., Qwen2.5-Coder-32B-Instruct-Q4) for offline generation, even if slow. | **~45-55%** | Medium-high — this is where essentially all the field's positive evidence lives, but no paper was found running *this exact* recipe (4-bit local 30B teacher → 4B LoRA student on unit-test generation), so the estimate is an analogy, not a direct replication. |
| **(c) RL with verifiable rewards (execution pass/fail or mutation score as reward)** | Mixed. RL-with-verifiable-rewards has strong results at ≥7-8B in the broader literature, but Mind the Gap's core mechanism (GV-gap) applies to RL-style self-filtering too — if the model's own verification/generation gap is non-positive at 4B, RL against a *self*-computed reward faces the same wall as rejection sampling. However, your reward here (execution pass/mutation score) is **externally computed, not self-verified** — this sidesteps the GV-gap argument, which specifically concerns *self*-verification. That makes RL-with-external-reward more promising than plain self-generated SFT, but exploration at 4B with only 315-function-scale reward signal risks the same power/data-volume problem as (a), just reshaped as an RL sample-efficiency problem. | **~20-30%** | Low-medium — theoretically the best-motivated non-distillation option, but no small-model (≤4B) unit-test RLVR paper was found to anchor the estimate; it's a mechanism argument, not an observed result. |
| **(d) None — 4B capability bounds it regardless of data strategy** | Not well supported as an absolute claim. UTGen shows 3B and 7B models *can* be pushed on this exact failure mode with the right (larger, teacher/perturbation-bootstrapped) data. The "ceiling" evidence in §2 is specific to *self-improvement*, not to fine-tuning per se — Magicoder-style teacher distillation moves 7B models a large amount. There's no direct evidence a 4B model is unmovable on this task by any method; the evidence says self-generated-only, at hundreds-of-examples scale, is the wrong regime for a 4B model, not that 4B itself is the wall. | **~10-15%** residual (i.e., probability that even (b) and (c) also fail) | Medium — this is the "none of the above works" catch-all, kept nonzero because no paper tests *this exact* task/model/scale combination. |

**Recommendation, ranked by evidence:** (b) teacher distillation from a local 4-bit 30B-class coder is the best-supported next experiment — it is the only option with direct precedent for producing gains of the magnitude you need (double digits of pass@1/coverage points in the literature, so ≥+0.05 on your metric is a modest ask by comparison), it directly targets the diagnosed value-prediction failure, and it is feasible within your Mac/48GB/no-cloud constraint. (a) is the least supported — it repeats a recipe your own four iterations already showed doesn't respond to more volume, and it's contradicted by the self-improvement-capacity literature for models in your size class. (c) is worth a smaller side bet if (b) is inconclusive, because the external-reward framing is the one lever that formally escapes the GV-gap argument that dooms (a).

---

### All sources cited
- [UTGen — arXiv:2502.01619](https://arxiv.org/abs/2502.01619)
- [Learning to Generate Unit Tests for Automated Debugging — OpenReview](https://openreview.net/pdf?id=yeVBHPLXxi)
- [A Large-scale Empirical Study on Fine-tuning LLMs for Unit Testing — arXiv:2412.16620](https://arxiv.org/pdf/2412.16620)
- [Parameter-Efficient Fine-Tuning of LLMs for Unit Test Generation — arXiv:2411.02462](https://arxiv.org/pdf/2411.02462)
- [ChatUniTest — arXiv:2305.04764](https://arxiv.org/abs/2305.04764)
- [TestGen-LLM at Meta — Qodo writeup](https://www.qodo.ai/blog/we-created-the-first-open-source-implementation-of-metas-testgen-llm/)
- [CoverUp — arXiv:2403.16218](https://arxiv.org/pdf/2403.16218)
- [Mind the Gap: Examining the Self-Improvement Capabilities of LLMs — arXiv:2412.02674](https://arxiv.org/pdf/2412.02674)
- [Beyond Human Data: Scaling Self-Training (ReST-EM) — arXiv:2312.06585](https://arxiv.org/pdf/2312.06585)
- [STaR: Bootstrapping Reasoning With Reasoning — arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- [Magicoder: Empowering Code Generation with OSS-Instruct — arXiv:2312.02120](https://arxiv.org/abs/2312.02120)
- [WizardCoder — arXiv:2306.08568](https://arxiv.org/abs/2306.08568)
- [SWE-Gym — arXiv:2412.21139](https://arxiv.org/abs/2412.21139)

### Caveats on this research pass
- Several PDFs (2412.16620, 2411.02462) could not be parsed beyond their abstracts by the fetch tool used, so table cells marked "not extracted" reflect a tooling limit, not an absence of data in the paper — worth a manual re-check before treating those rows as final.
- The web search budget for this session was exhausted partway through, so items explicitly requested but not directly verified with a fresh search include: CodeDPO, PLUM, exact WizardCoder/Evol-Instruct example counts, Nemotron/Genetic-Instruct code data specifics, TestEval test-set size, V-STaR, Self-Taught Evaluator, Llama 3's rejection-sampling scale, "Scaling Laws for Synthetic Data," and "The Lessons of Developing Process Reward Models." The verdict in §5 does not depend on these, but they should be checked before treating this as a complete literature sweep.
- No paper was found that runs the *exact* configuration in question (4B student, ≤1K self-generated examples, unit-test generation, paired n≈300 held-out functions) — every quantitative comparison above is an analogy across task, scale, or model family, not a replication.
