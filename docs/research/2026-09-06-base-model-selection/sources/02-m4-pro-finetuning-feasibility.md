# Source report 02 — LoRA/QLoRA feasibility on Apple M4 Pro 48 GB

Research agent report (Claude Sonnet), 2026-09-06. Unedited except for this header.
Claims marked unverified by the agent remain unverified.

---

## Mac M4 Pro (48GB unified memory) — LoRA/QLoRA fine-tuning feasibility, Sept 2026

### 1. mlx-lm (Apple MLX) fine-tuning capabilities

The official tool is `mlx_lm.lora`, documented at [mlx-lm/mlx_lm/LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

- Supported fine-tuning types: `lora` (default), `dora`, and `full` (full weight tuning).
- **QLoRA is automatic and built-in**: "If `--model` points to a quantized model, then the training will use QLoRA, otherwise it will use regular LoRA." No separate setup — quantization is native to MLX. A 4-bit 7B model cuts weight memory ~3.5x vs full precision, per [KDnuggets' MLX fine-tuning writeup](https://www.kdnuggets.com/fine-tuning-language-models-on-apple-silicon-with-mlx), bringing a 7B QLoRA fine-tune into roughly 8GB of working memory (unverified exact figure, sourced from a secondary blog, not the primary docs).
- Supported architectures (per the official LORA.md): Mistral, Llama, Phi2, Mixtral, Qwen2, Gemma, OLMo, MiniCPM, InternLM2. Note: this list predates newer Qwen3/Qwen3-VL releases — I could not verify whether Qwen3-VL or other 2026-era architectures are supported; check the repo directly before committing to a model family.
- Chat/instruct data format: JSONL with system/user/assistant turns, using HuggingFace [chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating) — multi-turn and tool-use formats supported.
- Default batch size 4, adjustable via `--batch-size`; `--grad-accumulation-steps` available for larger effective batch without proportional memory cost.
- Fusing/export: `mlx_lm.fuse` merges adapters into the base model, supports upload to HF Hub, and GGUF export — but GGUF export is limited to Mistral, Mixtral, and Llama architectures in fp16 only.
- A separate community project, [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora), extends training to more techniques (DPO, GRPO, etc.) beyond the official tool's SFT/LoRA/DoRA scope.

### 2. Memory and throughput — real numbers

Solid numbers are scarce for M4 Pro specifically; most public benchmarks are M1/M2/M3 Max/Ultra. Treat the M4 Pro estimate as an extrapolation, not a citation.

| Chip | Model | Task | Throughput | Memory | Source |
|---|---|---|---|---|---|
| M1 Max 32GB | Mistral-7B | LoRA train (official example) | ~250 tok/s | — | [LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) |
| M2 Max 32GB | Mistral-7B | LoRA train, 5,000 examples | ~90 min total | ~7GB peak | [Towards Data Science](https://towardsdatascience.com/lora-fine-tuning-on-your-apple-silicon-macbook-432c7dab614a/) |
| M2 Ultra | Mistral-7B | LoRA train | ~475 tok/s | — | [dev.to writeup](https://dev.to/wellallytech/local-ai-therapy-fine-tuning-mistral-7b-on-apple-silicon-with-mlx-lora-m3-max-performance-2kj3) |
| M3 Max | ~7B (unspecified) | inference | 30–50 tok/s | — | same dev.to source |
| M4 Max | text models, unspecified size | inference | up to ~525 tok/s peak; general range 38–62 tok/s | — | [digitalapplied MLX 2026 guide](https://www.digitalapplied.com/blog/apple-mlx-framework-local-ai-developers-2026-guide) |

**M4 Pro memory bandwidth is 273 GB/s** ([Apple's M4 Pro announcement](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)), vs 546 GB/s on the top M4 Max — roughly half. Since MLX training/generation is largely memory-bandwidth-bound, a reasonable extrapolation from the M2 Max (M2 Max is 400GB/s vs M4 Pro's 273GB/s, so M4 Pro is somewhat slower than M2 Max despite newer silicon) is that 7B LoRA training on M4 Pro likely lands **below** the M2 Max's ~250-475 tok/s figures, plausibly in the 150–300 tok/s range for training, and 7B QLoRA training should comfortably fit within 48GB (M2 Max hit only ~7GB peak at 32GB total). No single benchmark run specifically on M4 Pro (20-core GPU variant) for training was found — this is the biggest verification gap.

Reported general finding: MLX training throughput is far below discrete GPU throughput — one source cites 1–3 tok/s training vs 30–50 tok/s on an H100 in some workload characterization, though this conflicts with the 250-475 tok/s Mistral-7B numbers above, suggesting these are measuring different things (possibly full fine-tune vs LoRA, or different sequence lengths/batch sizes). Flagged as **unverified/contradictory**.

### 3. Alternatives on Mac

- **PyTorch MPS + HF PEFT/TRL**: functional but has real stability issues. A documented "silent NaN contiguity bug" causes `addcmul_`/`addcdiv_` ops to silently zero out or NaN tensors on non-contiguous memory ([kokoro-coreml MPS guide](https://github.com/mattmireles/kokoro-coreml/blob/main/README/Guides/apple-silicon/pytorch-mps.md)). bfloat16 on MPS can be up to 10x slower than float16 due to unoptimized kernels — use fp16, not bf16. PyTorch nightly builds carry MPS fixes that stable releases lack. Verdict: usable for experimentation, not recommended as the primary path — MLX is more mature for this hardware.
- **Unsloth on Apple Silicon**: native Unsloth does **not** have full MPS support as of 2026; Apple/MLX support is described as "in the works" per Unsloth's own docs. Two unofficial bridge projects exist: [unsloth-mlx](https://pypi.org/project/unsloth-mlx/0.3.5/) and its rename [mlx-tune](https://github.com/ARahim3/mlx-tune), which wrap native MLX with an Unsloth-compatible API (SFT/DPO/GRPO, GGUF export). Neither is the official Unsloth project — evaluate maturity/maintenance before depending on them.
- **llama.cpp finetune**: LoRA fine-tuning support exists and works on CPU with quantized GGUF files. Lowest-resource / most portable option but likely the slowest for actual training throughput — no throughput numbers found for M4 Pro specifically.

### 4. Cloud fallback pricing (Sept 2026, approximate — verify live before budgeting)

| Provider | GPU | Price |
|---|---|---|
| Modal | A10G | ~$1.10/hr (serverless, $0.000306/sec) |
| Modal | A100 40GB | ~$3.73/hr |
| RunPod | A100 80GB | $1.39/hr |
| RunPod | general on-demand range | $0.24/hr–$1.44/hr avg across 36+ GPU types; L4-specific rate not found |
| Lambda Labs | A10 24GB | $0.60/hr |
| Lambda Labs | A10G | ~$0.75/hr (one source, unverified) |
| Lambda Labs | A100 40GB | $1.99/hr |

**Cost estimate for 500–1000 examples × 3 epochs on a 7B model**: a 7B LoRA fine-tune of this size typically completes in well under an hour on an A10G or A100. Roughly **$0.50–$2 total** on Modal/Lambda A10G, or **$1–$4** on an A100. Agent's own extrapolation, not a cited benchmark.

### 5. Inference for evaluation

- mlx-lm supports batch generation for multiple prompts (see [mlx_parallm](https://github.com/willccbb/mlx_parallm) and MLX-LM's own batch generation example), with continuous/batched KV caching. One report cites 1300+ tok/s aggregate throughput for gemma-2b on an M3 Max 128GB with batching — not comparable to a 48GB M4 Pro.
- **Native batching inside `mlx_lm.server` is still an open feature request** ([GitHub issue #499](https://github.com/ml-explore/mlx-lm/issues/499)).
- Single-stream generation: M3 Max ~30-50 tok/s (likely 7-8B class); M4 Max 38-62 tok/s general range. No M4 Pro-specific inference numbers found; given 273GB/s vs M4 Max's 546GB/s bandwidth, expect roughly half the M4 Max per-stream throughput.
- Generating hundreds of suites per condition via batching is achievable but needs `mlx_parallm` or custom batching beyond the stock CLI.

### Recommendation

- **Comfortable**: 0.5B–3B LoRA/QLoRA fine-tuning. Fast iterations, memory is a non-issue.
- **Marginal**: 7B–8B LoRA/QLoRA. Fits comfortably in 48GB; throughput on M4 Pro is a genuine unknown, likely tens of minutes to a couple hours for a few thousand examples × few epochs. Full fine-tuning of 7-8B not recommended locally.
- **Not worth attempting locally**: anything beyond 8-9B full fine-tune, or any size needing many training runs per day — cloud A10G/A100 rental is under $2-4 per run for the stated dataset size.

### Claims the agent could not verify
1. No fine-tuning throughput or peak-memory benchmark found for an M4 Pro specifically.
2. The "1-3 tok/s training vs 30-50 tok/s H100" figure contradicts the 250-475 tok/s Mistral-7B LoRA numbers.
3. Whether mlx-lm's supported architecture list includes Qwen3 or other 2026-era families.
4. Lambda Labs A10G rate (~$0.75/hr) from a single secondary source.
5. RunPod's L4-specific on-demand hourly rate.
6. The $0.50-$4 cost estimate is an extrapolation, not a benchmark.
