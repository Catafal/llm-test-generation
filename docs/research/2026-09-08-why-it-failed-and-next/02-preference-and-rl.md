# Reviewer 2 — training signals beyond SFT: execution-feedback preference/RL, and MLX feasibility (Sonnet, web-backed)

## 1. DPO-family methods with execution-derived pairs

- **PLUM** (Zhang et al., arXiv:2406.06887, 2024): auto-generated tests, sample candidates, label pass/fail pairs, DPO; no reward model. +4.8% average pass rate on HumanEval/MBPP, up to +11.8% on LiveCodeBench, on models already SFT-saturated. Matches our 0.40 → 0.40 plateau: pairs add signal that positives-only SFT exhausted.
- **CodeDPO** (arXiv:2410.05605): self-generation + self-verification loop builds pairs, then DPO. **DSTC** (arXiv:2411.13611), **IterPref / Focused-DPO** (arXiv:2503.02783): concentrate the DPO gradient on the erroneous span found by debugging, directly relevant to a wrong literal in one assert. Pair counts in these papers: thousands, not hundreds; our pool: 550 fns × 8 samples ≈ 1,000–2,500 usable pairs after ties.
- **RLEF** (Gehring et al., ICML 2025, arXiv:2410.02089): PPO-style RL with real test feedback in context; 8B and 70B; an order of magnitude fewer samples for the same pass rate. Lesson transfers without full RL: feed the failure back, reward on the outcome.
- **RLTF** (2023), **CodeRL** (2022), **PPOCoder**: graded rewards (CodeRL: 1.0 pass / −0.3 test fail / −0.6 runtime error / −1.0 compile fail); RLTF adds per-line penalty attribution. Backbones 770M–2.7B: an existence proof that RL from execution works at small scale.

## 2. GRPO / RLVR on small models

DeepScaleR-1.5B (2025): GRPO with a verifiable math reward, 1.5B past o1-preview on AIME, single node. SimpleRL-Zoo (arXiv:2503.18892): binary correctness rewards plus difficulty-matched data are what make small-model RLVR work. JustRL (arXiv:2512.16649): plain GRPO competitive at 1.5B. No paper found with a binary *execution* reward at 3–4B in a few hundred steps, but the mechanism is identical; our K=8 samples per function are a usable GRPO group (no value model needed).

## 3. Negative-example alternatives to plain SFT

- **NAT, "Learning from Failure"** (NAACL 2025): performance rises monotonically as negative trajectories are mixed into fine-tuning versus positives only. Direct match to our failure mode.
- Unlikelihood / **CRINGE loss** (arXiv:2211.05826): push down the specific wrong tokens (the wrong literal) while pushing up the correct completion; a loss change, no pair infrastructure.
- **SCoRe** (arXiv:2409.12917): multi-turn self-correction RL, +9.1% HumanEval; needs a first RL phase; too heavy here.
- **LEMA** (arXiv:2310.20689): SFT on mistake + explanation + fix; our corrector is free (execute the reference).

## 4. Test-generation-specific signals

- **UTGen** (Prasad et al., arXiv:2502.01619, 2025): the most on-point prior work. A model can generate error-revealing inputs *or* predict expected outputs; training on either alone does not fix the other. Their fix is preference/RL training on output-prediction correctness: +7.6% on the combined metric, +3.2 to +12.4 downstream debugging pass@1. Their ablation shows SFT on generated tests underperforms training with a correctness-discriminating signal: external corroboration of our result.
- **TOGA** (ICSE 2022): oracle correctness is a distinct sub-problem needing its own supervision.
- **TestGen-LLM** (Meta, 2024): coverage improvement, not assert correctness; not informative for validity.
- **Vikram et al. 2023**: LLMs default to trivial properties without prompting toward invariants; keep the mutation gate.

## 5. MLX tooling (checked Sept 2026)

`mlx-lm` core ships no DPO/GRPO. **`mlx-lm-lora`** (Goekdeniz-Guelmez) is active (441 commits, releases through mid-2026): SFT, DPO, CPO, ORPO, GRPO, GSPO, Dr.GRPO, DAPO, Online DPO, XPO, RLHF-Reinforce-KL, PPO. It explicitly autodetects hybrid recurrent models (Qwen3.5/Next, Mamba). Memory guidance brackets a 4B on 48 GB (1–3B: batch 4, 16 layers; 7B: batch 2, 8 layers, 8-bit). Default max sequence 2048, adjustable. No public MLX run of DPO/GRPO on a 4B code model for test generation: budget a hyperparameter search.

## 6. Ranked recipes

- **A. DPO on self-sampled pass/fail pairs (PLUM-style).** Re-labels data we already have. Evidence: PLUM, CodeDPO, UTGen. Runs today with `mlx-lm-lora dpo`. Guard: only candidates that pass *and* clear the mutation gate may be "chosen", so the policy cannot drift to trivial asserts.
- **B. GRPO with a graded execution reward (CodeRL-style).** Group = the 8 samples; reward = runs + passes + kill rate, length-normalised; Dr.GRPO to counter length collapse. Medium-high evidence at 1.5B; novel at this reward shape.
- **C. Unlikelihood term on the wrong literal inside the existing SFT.** Cheapest; most surgical; lowest ceiling.

Recommended order: C as a same-day diagnostic, then A, then B.

## Sources
PLUM https://arxiv.org/abs/2406.06887 ; CodeDPO https://arxiv.org/html/2410.05605v2 ; DSTC https://arxiv.org/pdf/2411.13611 ; IterPref https://arxiv.org/html/2503.02783v1 ; RLEF https://arxiv.org/abs/2410.02089 ; RLTF https://arxiv.org/abs/2307.04349 ; CodeRL https://github.com/salesforce/CodeRL ; PPOCoder https://github.com/reddy-lab-code-research/PPOCoder ; DeepScaleR https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview ; SimpleRL-Zoo https://arxiv.org/pdf/2503.18892 ; JustRL https://arxiv.org/pdf/2512.16649 ; NAT https://github.com/Reason-Wang/NAT ; CRINGE https://arxiv.org/pdf/2211.05826 ; SCoRe https://arxiv.org/abs/2409.12917 ; LEMA https://arxiv.org/abs/2310.20689 ; UTGen https://arxiv.org/abs/2502.01619 ; TOGA https://arxiv.org/abs/2109.09262 ; TestGen-LLM https://arxiv.org/abs/2402.09171 ; Vikram https://arxiv.org/abs/2307.04346 ; mlx-lm-lora https://github.com/Goekdeniz-Guelmez/mlx-lm-lora ; GRPO PR https://github.com/ml-explore/mlx-examples/pull/1233 ; DPO PR https://github.com/ml-explore/mlx-examples/pull/1209
