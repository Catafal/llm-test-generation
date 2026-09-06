# Source report 01 — Open-weight coder models 0.5B–9B for pytest generation

Research agent report (Claude Sonnet), 2026-09-06. Unedited except for this header.
Claims marked unverified by the agent remain unverified. The agent's ranking
optimises accuracy-per-size; the synthesis document reweights for headroom
over prompting, which is what this project measures.

---

### Qwen2.5-Coder family (Alibaba, released Sept 2024)
All sizes: 32,768 context, Apache 2.0 license **except the 3B, which is under the restrictive "Qwen Research" (non-commercial) license** — everything else (0.5B/1.5B/7B/14B/32B) is Apache 2.0 and allows commercial fine-tuning/derivatives.
Training explicitly decontaminated against HumanEval/MBPP (and GSM8K/MATH) via 10-gram overlap removal — so scores are less likely to be memorization, but the model has still seen many *similar* test-generation-style problems.
Instruct variants exist for all sizes. MLX-community conversions exist for 0.5B/1.5B/3B/7B (4-bit and bf16), confirmed on Hugging Face.

| Size | HumanEval (instruct) | MBPP (instruct) | License |
|---|---|---|---|
| 0.5B | not found in searchable sources | not found | Apache 2.0 |
| 1.5B | ~43.3% | ~50.0% | Apache 2.0 |
| 3B | community-reported ~45.1% (opencompass, differs from official) | not found | **Qwen Research (non-commercial)** |
| 7B | 88.4% | 83.5% | Apache 2.0 |
[Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186) · [Qwen2.5-Coder blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/) · [HF collection](https://huggingface.co/collections/Qwen/qwen25-coder) · [MLX 7B](https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-4bit)

Note: the jump from 1.5B (~43%) to 7B (~88%) is large; an official 3B HumanEval number could not be verified (a GitHub issue reports a community re-run of 45.12%, suggesting eval-harness sensitivity — unverified).

### Qwen3-Coder — no small dense models
As of the search date, QwenLM's official Qwen3-Coder lineup is 480B-A35B (MoE), 30B-A3B (MoE, 3B active), and Qwen3-Coder-Next (built on Qwen3-Next-80B-A3B). **No 0.5–9B dense Qwen3-Coder variant found** — ruled out for this task except possibly 30B-A3B if MoE-with-3B-active is acceptable (likely too large for 48GB with LoRA training). [Qwen3-Coder GitHub](https://github.com/QwenLM/Qwen3-Coder)

### DeepSeek-Coder-V2-Lite
16B total / 2.4B active params (MoE), 128K context, permissive commercial-friendly license. HumanEval-Python 81.1%, HumanEval+MBPP+ multilingual avg 65.6%. Total weight footprint (16B) makes LoRA fine-tuning on 48GB tight but plausible with 4-bit base; MoE is less common in MLX tooling — MLX-community availability not confirmed. [HF Instruct card](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)

### CodeGemma (Google, built on Gemma, not Gemma 3)
2B and 7B sizes, both base + instruct-tuned (7B). Gemma license (gated; commercial use generally allowed under Gemma ToS but not Apache/MIT). Exact HumanEval/MBPP scores not retrieved. No Gemma 3 code-specific release found. [Model card](https://huggingface.co/google/codegemma-7b)

### Phi-4-mini-instruct (Microsoft, Feb 2025)
3.8B params, MIT license, 128K context. Evaluated on HumanEval/HumanEval+/MBPP/MBPP+ per model card but exact numbers not exposed in the fetched text. No MLX-community conversion confirmed. [Model card](https://huggingface.co/microsoft/Phi-4-mini-instruct)

### Mistral Codestral / Ministral
Codestral is 22B — out of range. **Ministral 3 family (Dec 2025)** at 3B/8B/14B dense, Apache 2.0, edge-targeted. HumanEval/MBPP numbers not verified. No MLX conversion confirmed.

### StarCoder2 (BigCode/ServiceNow/Hugging Face)
3B/7B/15B, 16K context. BigCode OpenRAIL-M license (behavioural restrictions). HumanEval-Python: 3B = 65.9%, 7B = 73.2%. Base models not officially instruction-tuned (community `TechxGenus` instruct fine-tunes exist). No MLX conversion confirmed.

### IBM Granite Code / Granite 4.x
Granite 8B-code-base: Apache 2.0, HumanEval-Python 43.9% / MBPP 42.2% / MBPP+ 49.6% — lower than Qwen2.5-Coder at comparable size. A 2026 aggregator claim that "Granite 4.1 8B" leads HumanEval is unverified.

### HuggingFace SmolLM2 / SmolLM3
SmolLM2: 135M/360M/1.7B; SmolLM3: 3B, Apache 2.0, fully open. Not code-specialised; exact HumanEval+/MBPP+ numbers not retrieved. Likely weaker than Qwen2.5-Coder at this size for pytest generation.

### Other 2025-2026 finds
- **Yi-Coder** (01.AI): 1.5B and 9B, Apache 2.0, 128K context. Yi-Coder-9B reportedly 85.4% HumanEval — secondary source, unverified.
- **VibeThinker-3B** (WeiboAI): MIT-licensed fine-tune of Qwen2.5-Coder-3B; relicensing cleanliness unverified.
- Llama 3.2 1B/3B: Llama Community License, not code-specialised; weak choice regardless of licence.

### Ranked shortlist (agent's own, accuracy-per-size)

1. **Qwen2.5-Coder-7B-Instruct** — best verified accuracy-per-size (88.4% HumanEval), Apache 2.0, MLX 4-bit conversion exists, fits 48GB for LoRA.
2. **Qwen2.5-Coder-1.5B-Instruct** — if speed matters more than peak accuracy; Apache 2.0, MLX conversion exists, ~43%/50% baseline.
3. **Phi-4-mini-instruct (3.8B)** — MIT, 128K context; exact numbers and MLX conversion unconfirmed.
4. **DeepSeek-Coder-V2-Lite-Instruct (16B/2.4B active MoE)** — strong scores, least proven MLX fit.
5. **Yi-Coder-9B** — intriguing claim, unverified.

Avoid: Qwen2.5-Coder-3B (non-commercial licence), Codestral (22B), Qwen3-Coder (no small dense variant), CodeGemma (gated, unverified scores), Llama 3.2 1B/3B, StarCoder2 (OpenRAIL-M, lower scores).

### Claims the agent could not verify
- Qwen2.5-Coder-0.5B-Instruct exact HumanEval/MBPP numbers
- Qwen2.5-Coder-3B-Instruct official HumanEval/MBPP numbers
- Phi-4-mini-instruct exact HumanEval+/MBPP+ pass@1 and MLX conversion
- Ministral 3 (3B/8B) HumanEval/MBPP scores
- "Granite 4.1 8B" HumanEval claim
- Yi-Coder-9B 85.4% HumanEval
- VibeThinker-3B MIT relicensing
- MLX-community conversions for DeepSeek-Coder-V2-Lite, CodeGemma, StarCoder2, Granite Code, Ministral 3

Suggested follow-up: Qwen2.5-Coder technical report Table 16 (arxiv 2409.12186) and the Phi-4-mini model card benchmark table.
