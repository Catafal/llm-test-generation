# Source report 03 — Prior art on fine-tuning small LLMs for unit-test generation

Research agent report (Claude Sonnet), 2026-09-06. Unedited except for this header.
Claims marked unverified by the agent remain unverified.

---

## 1. Papers/projects 2023–2026

| Work | Base model/size | Training signal | Metric | Result vs. prompting |
|---|---|---|---|---|
| **TestGen-LLM / ACH** (Meta, [2501.12862](https://arxiv.org/html/2501.12862v1)) | **Llama 3.1 70B**, agentic pipeline, no fine-tuning | Prompted per-mutant "kill this mutant" agent loop over 10,795 Android/Kotlin classes; 31,677 candidate mutants → 9,095 buildable → 4,660 non-equivalent → 571 accepted tests | Mutant kill rate, engineer acceptance, coverage-add rate | ACH (mutation-guided) kill rate **15%** vs. original coverage-only TestGen-LLM **2.4%**; 73% engineer acceptance; 49% of accepted tests caught faults with *no* coverage gain (mutation > coverage as target) |
| **MuTAP** ([2308.16557](https://arxiv.org/abs/2308.16557)) | LLM + augmented prompt (not fine-tuned) | Iterative re-prompting with surviving mutants | Mutation score on HumanEval, Refactory | 94% mutation score vs. Pynguin's 66%; 94.9% faulty-submission detection vs. 67.5% |
| **Mutation-Guided Unit Test Generation with a LLM** ([2506.02954](https://arxiv.org/abs/2506.02954)) | GPT-4, prompt-only | Mutation operators injected into prompt on Defects4J | Mutation score | ~15-20% gain over EvoSuite; beats plain prompting |
| **UTGen / UTDebug** ([2502.01619](https://arxiv.org/abs/2502.01619)) | **Qwen2.5-7B and Qwen2.5-32B** (SFT) | SFT jointly on error-revealing test *inputs* + correct expected *outputs* | "attack rate + correct output"; downstream pass@1 on HumanEvalFix / MBPP+ | Beats other LLM baselines by **7.59%**; UTGen-32B tests raise debugging pass@1 **+3.17%** (HumanEvalFix) and **+12.35%** (MBPP+ hard); UTGen-7B as reward judge beats a dedicated 8B reward model by 4.43% — **clearest small-model (7B) fine-tune-beats-prompting result** |
| **Go-UT-Bench** ([2511.10868](https://arxiv.org/pdf/2511.10868)) | **DeepSeek-Coder-1.3B-Instruct** (LoRA), **Llama-3.2-3B-Instruct** (full FT), DeepSeek-Coder-V2-Lite | SFT on code↔test pairs mined from Go repos | Pairwise win-rate judged by GPT-4o-mini (not execution) | 14.2% → **81.9%** (DS-Coder-V2-Lite LoRA); Llama-3.2-3B 22.0% → **76.7%**. Authors flag as judge-only, not execution-verified — weak evidence |
| **CodaMosa** ([ICSE 2023](https://ieeexplore.ieee.org/document/10172800/)) | Codex, hybrid with SBST/Pynguin | LLM called when SBST coverage plateaus | Coverage over 486 benchmarks | Higher coverage on 173-279 benchmarks vs 10-4 regressions |
| **CoverUp** ([2403.16218](https://arxiv.org/abs/2403.16218)) | GPT-4-class, prompt-only, coverage loop | Coverage-guided re-prompting | Line+branch coverage | 80% median vs CodaMosa's 47% |
| **ChatUniTest** ([2305.04764](https://arxiv.org/abs/2305.04764)) | GPT-3.5/4-class | Focal-context prompting + generate-validate-repair | Line coverage (Java) | Highest coverage in half of projects |
| **TestPilot** ([2302.06527](https://arxiv.org/abs/2302.06527)) | Codex-class | Doc-mined examples + failure re-prompt | Coverage (npm) | 70.2%/52.8% vs Nessie's 51.3%/25.6% |
| **TestEval** ([2406.04531](https://arxiv.org/abs/2406.04531)) | 16-17 models, benchmark | Targeted line/branch/path coverage, 210 LeetCode programs | Coverage | All models struggle on targeted path coverage |
| **TestGenEval** ([2410.00752](https://arxiv.org/html/2410.00752v1)) | 7B→405B, prompt-only | Real-world repo functions | Pass@1, coverage, **mutation score** | See section 2 |
| **ACE** ([2605.16299](https://arxiv.org/pdf/2605.16299)) | Not verified | Execution-boolean-table filtering for SFT + KTO | Execution-based | Not independently confirmed |

## 2. Small coder models (1.5B–9B), prompting only — TestGenEval numbers

Temperature 0.2, real-world repo test generation:

| Model | Pass@1 | Coverage | Mutation score |
|---|---|---|---|
| CodeLlama 7B | 4.1% | 1.2% | 0.5% |
| Llama 3.1 8B | 30.5% | 14.1% | 6.8% |
| Gemma 9B | 42.1% | 20.2% | 9.0% |
| GPT-4o (reference) | 64.0% | 35.2% | 18.8% |

Best small model (Gemma 9B) sits at roughly half GPT-4o's mutation score; CodeLlama 7B is nearly non-functional. No TestGenEval number exists for Qwen2.5-Coder-1.5B/3B or DeepSeek-Coder-1.3B/6.7B — that gap is unfilled.

## 3. Contamination

- **DeepSeek-Coder**: stated HumanEval/MBPP filtering, but LeetCode-recency-correlated performance drop is read as contamination evidence.
- **Qwen2.5-Coder**: 10-gram decontamination claimed; independent OpenCompass run on the 3B Instruct found HumanEval 45.12 / MBPP 30.20 vs reported 84.1 / 73.6 ([GitHub issue](https://github.com/QwenLM/Qwen3-Coder/issues/420)) — harness mismatch or inflation, unresolved.
- **Decontaminated alternatives**: EvalPlus HumanEval+/MBPP+ (stronger oracle, same problems); LiveCodeBench (date-segmented, the only rigorous control); BigCodeBench (from-scratch tasks, n-gram checked). For a held-out pool the LiveCodeBench pattern is right: pick a cutoff date, use functions after it.

## 4. Python mutation testing tooling

- **mutmut** — simplest CLI, narrower operators, intermittent maintenance, "inconclusive" in one comparison.
- **cosmic-ray** — most actively maintained; ~25.7% competent-mutant score in one comparison.
- **mutpy** — older, limited maintenance.
- **Poodle** — newer; 50.9% competent-mutant rate in one study (single source).
- **Equivalent mutants**: none of the tools detect them automatically. Meta's ACH filters ~49% of buildable mutants as likely-equivalent via LLM "believed killable" heuristics before generation. Custom AST selection layers on top of standard mutation are the norm in the papers.

Sources: [IEEE comparison](https://ieeexplore.ieee.org/document/10818231/), [ACM comparison](https://dl.acm.org/doi/10.1145/3701625.3701659)

---

## (a) Lessons for a two-weekend 1.5B–3B fine-tune project

1. **Fine-tune on the (input, expected-output) pair, not just the test body.** UTGen: small/mid models fail more on predicting correct assertion *values* than on test *shape*.
2. **Filter training data through execution, not compile-success.** ACH: only 29% of generated mutants built, 51% of those non-equivalent. Keep only tests that run, pass on the reference, and kill at least one mutant.
3. **Use mutation score, not coverage.** Coverage and mutation score diverge sharply; spot-check equivalent mutants manually.
4. **Pick base size deliberately; expect a bounded gain, not a rescue.** CodeLlama 7B is nearly non-functional under prompting; a 1.5B may be so weak that fine-tuning teaches format compliance rather than reasoning. 3B or 7B is the safer floor given what Go-UT-Bench and UTGen actually tested.
5. **Decontaminate the eval pool with a date cutoff, not just a benchmark name.** Build the held-out pool the LiveCodeBench way.

## (b) Which base sizes give a visible gain

- **7B is the smallest size with a directly measured, execution-verified fine-tune gain** (UTGen, Qwen2.5-7B).
- **1.3B–3B evidence is judge-based only** (Go-UT-Bench), large relative gains, unvalidated by execution.
- Headroom clearly exists at 7-9B (published); plausible but unproven at 1.5-3B.

## (c) Claims the agent could not verify

- No TestGenEval-style prompting numbers for Qwen2.5-Coder-1.5B/3B or DeepSeek-Coder-1.3B/6.7B.
- ACE framework base model and numbers.
- Qwen2.5-Coder-3B independent vs reported HumanEval/MBPP gap cause.
- MUTGEN precise numeric deltas.
- Poodle's edge replicating outside one study.

Full source list: [2501.12862](https://arxiv.org/html/2501.12862v1), [2506.02954](https://arxiv.org/abs/2506.02954), [2308.16557](https://arxiv.org/abs/2308.16557), [2502.01619](https://arxiv.org/abs/2502.01619), [2511.10868](https://arxiv.org/pdf/2511.10868), [CodaMosa](https://ieeexplore.ieee.org/document/10172800/), [2403.16218](https://arxiv.org/abs/2403.16218), [2305.04764](https://arxiv.org/abs/2305.04764), [2302.06527](https://arxiv.org/abs/2302.06527), [2406.04531](https://arxiv.org/abs/2406.04531), [2410.00752](https://arxiv.org/html/2410.00752v1), [2409.12186](https://arxiv.org/pdf/2409.12186), [Qwen3-Coder issue #420](https://github.com/QwenLM/Qwen3-Coder/issues/420), [2403.07974](https://arxiv.org/abs/2403.07974), [2406.15877](https://arxiv.org/html/2406.15877v4), [EvalPlus](https://evalplus.github.io/).
