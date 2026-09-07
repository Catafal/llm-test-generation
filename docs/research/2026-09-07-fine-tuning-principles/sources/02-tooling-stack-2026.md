# Source report 02 — Fine-tuning stack for Qwen3.5-9B, September 2026

Research agent report (Claude Sonnet), 2026-09-07. Unedited except for this header.
Claims marked unverified by the agent remain unverified.

---

## 1. Unsloth

- **Model coverage**: Unsloth officially supports the full Qwen3.5 family (0.8B–122B-A10B) including LoRA, QLoRA, full fine-tuning, vision, and RL fine-tuning. ([Qwen3.5 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune))
- **QLoRA caveat**: Unsloth's own docs explicitly say "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense, due to higher than normal quantization differences." Use LoRA (bf16/16-bit) or full fine-tune instead.
- **VRAM (bf16 LoRA)**: 9B ≈ 22GB. Speed/memory claim: 1.5x faster, 50% less VRAM vs FA2 setups.
- **Gotchas**: requires transformers v5; Qwen3.5 uses custom Mamba/Triton kernels which can be slow on older GPUs (e.g., T4); GGUF export works, vLLM export needs vLLM ≥0.17.0.
- **Apple Silicon — NOT production-ready**: maintainer statement (Feb 2025, [discussion #1732](https://github.com/unslothai/unsloth/discussions/1732)): a community PR awaiting review. Not first-party.
- **Unofficial Mac wrappers**: "unsloth-mlx" renamed to **mlx-tune** (ARahim3) to disclaim affiliation; wraps native MLX with an Unsloth-compatible API. Qwen3.5 *vision* fine-tuning reported broken ([issue #7](https://github.com/ARahim3/mlx-tune/issues/7)); text-only LoRA more mature.
- **Bottom line**: Unsloth is GPU-only in practice for Qwen3.5 today.

## 2. mlx-lm (`mlx_lm.lora`)

- **Qwen3.5 support timeline**: [issue #1136](https://github.com/ml-explore/mlx-lm/issues/1136) (April 2026) "Model type qwen3_5 not supported" was open as filed. By September 2026 mlx-community publishes Qwen3.5 MLX weights and there is active kernel work (shared fused gated-delta kernel; fixing float32 dtype for gated delta SSM ops). A separate ml-explore/ollama issue reports `gated_delta_step` writing recurrent state in bf16 instead of fp32, corrupting Qwen3.5/3.6 output — a live numerical-precision risk on any MLX Gated DeltaNet path.
- **LoRA fusion correctness risk**: [issue #1058](https://github.com/ml-explore/mlx-lm/issues/1058) reports a Qwen3.5-4B LoRA adapter, merged and converted to MLX, diverging from the HF/Transformers version after 20–30 tokens.
- **LoRA config**: YAML via `-c/--config`; `lora_parameters.keys` sets target modules (defaults `self_attn.q_proj/k_proj/v_proj/o_proj`, `mlp.gate_proj/up_proj/down_proj`). Gated DeltaNet layers use different names; exact keys mlx-lm exposes not confirmed.
- **Completion-only loss**: `--mask-prompt` for chat/tools-format datasets.
- **Adapter fusion**: `mlx_lm.fuse`; GGUF export limited to Mistral/Mixtral/Llama fp16 — Qwen3.5 GGUF via mlx-lm not supported.
- **LORA.md static family list** does not include Qwen3.5; verify against installed version.
- **Alternative — mlx-lm-lora** ([Goekdeniz-Guelmez](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)): community superset, 12 training algorithms, QAT, and per release notes expanded Qwen3.5 support with a training notebook. Likely more current for Qwen3.5 on MLX; verify GatedDeltaNet naming and vision-weight handling first.

## 3. HF TRL + PEFT (SFTTrainer) — GPU reference stack

- Qwen3.5 is a "known model family" for `assistant_only_loss` auto-templating, contingent on `{% generation %}` tags in the chat template. TRL test fixtures migrated to Qwen3.5 Think/NoThink variants.
- Default on prompt-completion datasets is completion-only loss; `completion_only_loss=False` reverts. [Issue #5324](https://github.com/huggingface/trl/issues/5324): a custom formatting function that glues prompt+completion breaks masking — keep them as separate fields.
- **MPS**: no confirmation TRL/PEFT runs stably on MPS for Qwen3.5; bitsandbytes 4-bit does not support MPS at all, so QLoRA via TRL is CUDA-only. Do not rely on TRL for the Mac leg.

## 4. Axolotl, torchtune, LLaMA-Factory

- **Axolotl**: YAML, strong for multi-GPU/FSDP; Qwen3.5 status unclear.
- **torchtune**: Qwen3 support May 2025; Qwen3.5 hybrid support unverified.
- **LLaMA-Factory**: 2026 release notes claim "primary support for Qwen3.5/Qwen3.6/Gemma4" — clearest stated support of the three; Gated DeltaNet LoRA target handling unverified.

## 5. Reproducibility & experiment tracking

- Config-as-code (YAML/JSON committed), fixed seed in config, run manifest with base-model revision, dataset hash, pinned library versions, W&B URLs or committed JSONL logs.
- Tag each adapter with training date + dataset hash + base revision in `adapter_config.json` or a sidecar README.
- W&B most common; plain JSONL + README table is an acceptable minimal substitute.

## 6. Cloud cost/time for a 9B run (~1,000 examples)

- 2026 pricing: RunPod H100 PCIe ≈ $1.99/hr, A100 80GB ≈ $1.19–1.39/hr, RTX 4090 Community ≈ $0.34/hr; Lambda A100 40GB from $1.99/hr; Modal per-second.
- A 7B–13B LoRA on ~1k examples typically **$3–$20** and an afternoon on a 24GB-class card. Renting H100/A100-80GB for this is overpaying.
- Workflow: dataset in via volume/S3 or git clone; adapter (hundreds of MB) back via rsync or push to a private HF repo.

## Recommendation

**(a) Cloud-first for the real training run, local for iteration.** Qwen3.5's MLX support is recent (Feb–Mar 2026) with active kernel bugs (state dtype), a documented LoRA-fusion divergence bug (#1058), and Unsloth's Mac path unmerged.
- Prototype data pipeline and sanity-check load/generate on the M4 Pro with mlx-lm-lora or `mlx_lm.lora`, bf16 LoRA (not QLoRA).
- Real run on a RunPod A100/4090 with **TRL + PEFT SFTTrainer**, bf16 LoRA, skipping QLoRA per Unsloth's Qwen3.5 quantisation warning. ~$5–20, a few hours on a 24GB card.
- Alternative: Unsloth on the same GPU, 1.5x faster, easy GGUF export.

**(b) Risks per stack**
- *Unsloth*: Mac unofficial; QLoRA discouraged on Qwen3.5; transformers v5 and vLLM ≥0.17 required.
- *mlx-lm / mlx-lm-lora*: Qwen3.5 support new with real correctness bugs (GatedDeltaNet state dtype; LoRA-fusion divergence). Verify against installed changelog; vision fine-tuning broken in at least one wrapper.
- *TRL/PEFT*: solid on GPU; QLoRA needs bitsandbytes (no MPS); pre-formatting prompt+completion silently breaks masking.
- *Axolotl/torchtune/LLaMA-Factory*: only LLaMA-Factory states Qwen3.5 support explicitly.

**(c) Could not verify**
- Exact `lora_parameters.keys` names for Qwen3.5 Gated DeltaNet layers in mlx-lm.
- Whether mlx-lm-lora handles vision weights or text backbone only.
- TRL/PEFT stability on MPS for Qwen3.5.
- Axolotl/torchtune Qwen3.5-specific support.
- Whether the gated-delta bf16/fp32 bug affects mlx-lm's training path or only the Ollama runner.

Sources: [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune), [Unsloth MLX discussion #1732](https://github.com/unslothai/unsloth/discussions/1732), [mlx-tune](https://github.com/ARahim3/mlx-tune), [mlx-lm LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md), [mlx-lm #1136](https://github.com/ml-explore/mlx-lm/issues/1136), [mlx-lm #1058](https://github.com/ml-explore/mlx-lm/issues/1058), [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora), [TRL sft_trainer.md](https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md), [TRL #5324](https://github.com/huggingface/trl/issues/5324), [RunPod pricing 2026](https://www.buildmvpfast.com/api-costs/gpu), [Qwen3.5 architecture writeup](https://huggingface.co/blog/mlabonne/qwen35).
