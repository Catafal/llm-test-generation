# Reviewer 3 — the test-oracle problem and base-model choice (Sonnet, web-backed)

## A. Can the output format sidestep mental execution while still killing mutants?

1. **Property-based / metamorphic tests** assert a relation, not a value. Vikram et al. (arXiv:2307.04346, 2023): GPT-4 produced a valid-and-sound property-based test in 2.4 samples on average, correct for 21% of extractable properties; soundness detection precision 100% / recall 97%. The hard part shifts from "get the number right" to "find enough properties" (a coverage problem). 2025–26 metamorphic work (LLMorph, MR-Coupler, survey arXiv:2605.13898) is directional, without single-function mutation-score head-to-heads.
2. **Differential / pseudo-oracle re-implementation**: TOGA (2022) and TOGLL (arXiv:2405.03786): TOGLL 3.8x more correct assertion oracles than TOGA, ~5% non-compiling. Konstantinou et al. (arXiv:2410.21136, 2024): LLM oracles tend to encode the *actual* (buggy) behaviour rather than the intended one; oracle accuracy below 50% on buggy code. Model-written reference implementations risk tautology. Mokav (arXiv:2406.10375) needs two programs.
3. **Execution-grounded / golden-master tests**: harness records the value (Pynguin's default assertion strategy, keeping assertions killed by a mutant). Validity 100% by construction; locks in current behaviour; cannot express intent. A fallback generator, not an answer if catching real bugs is the point.
4. **Weak oracles** (exceptions, types, shapes, `pytest.approx`): provably weaker per check, near-zero validity cost; no pytest-specific mutation-score comparison exists (gap). Jahangirova's oracle-assessment work (UCL) is the tool to measure it.

Synthesis for validity × mutation score: exception/type/shape checks first; property/invariant assertions second; exact values only where execution can verify them; avoid model-authored reference implementations unless independently verified.

## B. Base model for training feasibility on mlx-lm

The Qwen3.5-4B problems are filed, open bugs specific to the hybrid Gated-DeltaNet path: mlx-lm#1185 (descriptor leak), mlx#3539 (OOM above short sequences), mlx-lm#1480.

| Model | Params | Thinking switch | Licence | Code numbers found | Architecture |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct/Thinking-2507 | 4B | yes (separate checkpoints) | Apache 2.0 | Thinking: LiveCodeBench 55.2; HumanEval+/CRUXEval not found | dense |
| Qwen3-8B | 8B | yes | Apache 2.0 | LiveCodeBench v5 22.8 (tech report) | dense |
| **Qwen2.5-Coder-7B-Instruct** | 7B | no | Apache 2.0 | **HumanEval+ 84.1, CRUXEval-I 56.5 / -O 56.0** | dense |
| Qwen2.5-Coder-3B-Instruct | 3B | no | Apache 2.0 | below 7B | dense |
| Gemma 3 4B | 4B | no | Apache 2.0 | ~71 HumanEval (secondary) | dense |
| Phi-4-mini | 3.8B | no | MIT | code numbers not surfaced | dense |
| SmolLM3-3B | 3B | limited | Apache 2.0 | below Qwen2.5-Coder | dense |
| Llama 3.x 3B/8B | | no | Llama licence | not surfaced | dense |

mlx-lm LoRA throughput on dense 8B-class models: ~285–296 tok/s reported on M-series (proxy).

**Recommendation:** Qwen2.5-Coder-7B-Instruct for fastest iteration (dense, Apache 2.0, strongest measured code numbers incl. CRUXEval ~56). Lost vs Qwen3.5-4B: the thinking switch and general breadth. If the thinking switch matters (chain-of-thought before asserts, plausibly relevant to the wrong-value bottleneck), Qwen3-4B-Instruct/Thinking-2507 is the closer dense match; its HumanEval+/CRUXEval numbers need a direct benchmark.

## Sources
Vikram https://arxiv.org/abs/2307.04346 ; TOGA https://arxiv.org/pdf/2109.09262 ; TOGLL https://arxiv.org/pdf/2405.03786 ; Konstantinou et al. https://arxiv.org/abs/2410.21136 ; Mokav https://arxiv.org/html/2406.10375 ; ChatUniTest https://arxiv.org/abs/2305.04764 ; Jahangirova thesis https://discovery.ucl.ac.uk/1493269/1/main.pdf ; weak/strong mutation https://rahul.gopinath.org/post/2015/07/18/weak-and-strong-mutation/ ; Pynguin https://arxiv.org/abs/2202.05218 ; MT survey https://arxiv.org/html/2605.13898v1 ; MR-Coupler https://arxiv.org/pdf/2604.10126 ; Qwen3 report https://arxiv.org/pdf/2505.09388 ; Qwen3-4B-Instruct-2507 https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 ; Qwen2.5-Coder report https://arxiv.org/pdf/2409.12186 ; Phi-4-mini https://arxiv.org/pdf/2503.01743 ; mlx-lm#1185 https://github.com/ml-explore/mlx-lm/issues/1185 ; mlx#3539 https://github.com/ml-explore/mlx/issues/3539 ; mlx-lm#1480 https://github.com/ml-explore/mlx-lm/issues/1480
