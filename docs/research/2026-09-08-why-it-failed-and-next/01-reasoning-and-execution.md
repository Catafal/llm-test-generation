# Reviewer 1 — can a ~4B model learn to predict what code returns? (Sonnet, web-backed)

## The core problem in one line

The validity plateau at ~0.40 with wrong-expected-value failures is the textbook "execution reasoning" failure, well documented at this scale with thinking off. SFT on corrected literals trained the model to be more confident about literals without teaching it to compute them: the overconfidence failure mode papers report.

## 1. CRUXEval and the CoT lift: real but scale-gated

CRUXEval (Gu et al. 2024, ICML): 800 short Python functions, input/output prediction. GPT-4 + CoT reaches 75% (input) / 81% (output). CoT helps Code Llama 34B and GPT-4 on both tasks, GPT-3.5 only on output prediction, and Code Llama 13B on neither. Output prediction benefits more from CoT than input prediction (step-by-step simulation, not search), but only once the model can execute the steps reliably. CodeMind (2024): execution-reasoning accuracy degrades sharply with program complexity. CRUXEval-X: execution reasoning is shallow and pattern-bound. "Demystifying Errors in LLM Reasoning Traces: code execution simulation" (arXiv 2512.00215, 2025) categorises wrong-value and overconfidence failures.

## 2. Training on execution traces: quality and alignment matter more than volume

- Scratchpads (Nye et al. 2021): emitting intermediate states line by line dramatically beats direct answer prediction for program execution, few-shot and fine-tuned.
- NExT (Ni et al. 2024, ICML): PaLM 2 trained to reason over execution traces rendered as inline comments, for repair; +26.1 points MBPP fix rate, +10.3 HumanEval; rationales bootstrapped by rejection sampling (STaR family, grounded in real traces).
- CodeExecutor (Liu et al. 2023): pretraining on mutation-augmented line-by-line traces; later work found trace objectives not always better than plain SFT. Traces help when their format aligns with the eval task.

Takeaway: our targets gave the model the answer but not the execution path that produces it; closer to the "no-op or hurts" case than to Scratchpads/NExT.

## 3. Self-taught reasoners: rejection sampling works only with a rationale in the loop

- STaR (Zelikman 2022): generate rationale, keep if the answer checks out, rationalise backward on failures; matched a model ~30x larger on CommonsenseQA. The fix is never "state the literal", it is "produce a chain that yields the literal".
- ReST-EM (Singh et al. 2023): generate-filter-finetune over a few rounds; scales with model size; ≤8B curves not well documented.
- V-STaR (Hosseini et al. 2024, COLM): additionally trains a verifier by DPO on correct and incorrect self-generated solutions, used at inference to rank candidates; +4 to +17 points over STaR-style baselines on code and math with LLaMA-2 7B/13B. The most transferable idea: the wrong-expected-value suites we discard are the verifier's training signal.
- Quiet-STaR (2024): pretraining-scale, not actionable here.

## 4. Reasoning distillation into small models

- DeepSeek-R1-Distill-Qwen-7B: MATH-500 92.8, Codeforces ~1189; 1.5B: AIME24 28.7. Distillation reaches 1.5B, but with 800K traces.
- s1 (2025) and LIMO (2025): ~1k curated hard traces move a 32B model; no result at 4B; curation quality is the lever.
- SFT on non-reasoning targets degrading later thinking: only adjacent evidence (forgetting scaling laws; replay as mitigation). Not a direct citation.

## 5. Thinking budget

- s1 budget forcing: AIME24 50.0 → 53.3 by extending thinking.
- Budget studies on Qwen3-class: accuracy plateaus past an entropy-predicted average (~2.8k tokens on MATH-500) and degrades sharply below it. A floor exists; more is not always better.
- No model-size-specific budget curve exists for code output prediction under 14B; needs its own ablation.

## 6. Ranked recipes for this project

1. **Trace-augmented targets** (train the path that produces the assert, not only the literal). Strongest evidence (Scratchpads, NExT). Blocked on Qwen3.5 by the 1024-token training cap; needs a dense base (Qwen3-4B, Qwen2.5-Coder-7B).
2. **Verifier-augmented rejection sampling (V-STaR style)**: train a discriminator on passing vs failing self-samples, rank candidates at inference. +4–17 points in the source paper at 7–13B. Compatible with the token cap and with Qwen3.5.
3. **Thinking on at inference with an equal or larger budget, no training**: CRUXEval says the CoT payoff threshold is above 13B, so expect little at 4–8B; a cheap ablation, not the primary recipe; do not use a tiny budget (below-floor is worse than off).

Flagged for direct reading: arXiv 2604.03253 "Self-Execution Simulation Improves Coding Models" (2026) and 2512.00215.

## Sources
- CRUXEval (Gu et al. 2024) https://arxiv.org/pdf/2401.03065 ; https://crux-eval.github.io/
- CodeMind https://arxiv.org/html/2402.09664v6
- Demystifying Errors in LLM Reasoning Traces https://arxiv.org/pdf/2512.00215
- NExT (Ni et al. 2024) https://arxiv.org/abs/2404.14662
- Scratchpads (Nye et al. 2021) https://arxiv.org/abs/2112.00114
- CodeExecutor (Liu et al. 2023) https://arxiv.org/abs/2305.05383
- STaR (Zelikman 2022) https://openreview.net/pdf?id=_3ELRdg2sgI
- ReST-EM (Singh et al. 2023) https://arxiv.org/abs/2312.06585
- V-STaR (Hosseini et al. 2024) https://arxiv.org/abs/2402.06457
- Quiet-STaR https://arxiv.org/html/2403.09629v1
- DeepSeek-R1 https://arxiv.org/html/2501.12948v1
- s1 https://arxiv.org/pdf/2501.19393 ; LIMO https://arxiv.org/pdf/2502.03387
- Scaling laws for forgetting https://arxiv.org/html/2401.05605v1

## Addendum (read directly): "Self-Execution Simulation Improves Coding Models" (arXiv 2604.03253, 2026)

- Base models: Qwen2.5-Base 3B and 7B, CWM-base; comparisons with Qwen3-32B.
- Data (NLEX): line-by-line execution traces with variable states from ~30M
  public Python functions + 35k competitive-programming solutions, rewritten
  into natural-language execution explanations by Qwen3-32B; ~80M
  descriptions for general functions, 115k for competitive programming.
- RL: binary terminal reward (+1 correct output prediction, −1 otherwise);
  4k steps for Qwen; multi-task with output prediction weighted 0.8.
- **CRUXEval-O output prediction pass@1: Qwen2.5-7B 48.5 → 75.5; Qwen2.5-3B
  37.5 → 68.0** with the trace-explanation SFT data. Gains hold at 3B (LCB-IO
  57.1 → 66.4), with larger "simulation gaps" for smaller models.
- Self-verification (best@k with simulated execution): +2 to +8 points.
- Implication for us: the skill is learnable at 3–7B when the target is an
  execution *explanation* grounded in real traces, not a bare literal. Their
  data scale is ~5 orders of magnitude beyond ours, so the open question is
  the low-data regime: a few hundred to a few thousand traced functions with
  LoRA. That is a measurable experiment, and the dense Qwen2.5 base is the
  one they used.
