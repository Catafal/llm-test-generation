# Why the weekend-2 fine-tune failed, and what has evidence of working (2026-09-08)

Three web-backed research passes after the null result in
`docs/results/weekend-2.md`. Sources: `01-reasoning-and-execution.md`
(can a small model learn to predict what code returns), `02-preference-and-rl.md`
(signals beyond SFT; MLX tooling), `03-oracles-and-base-model.md`
(changing the output format; base-model choice).

## The diagnosis the literature agrees with

The measured failure is *mental execution*: the model mispredicts what a
function returns. This is a named, benchmarked weakness (CRUXEval 2024;
CodeMind 2024) that chain-of-thought lifts only above ~13B for base code
models, and that SFT on positive examples does not fix (UTGen 2025 ablation:
SFT on generated tests underperforms training with a correctness-discriminating
signal). Our targets, correct literals the model could not have produced,
taught confidence without derivation; the Scratchpads/NExT line predicts
exactly that: traces help when the *path* to the value is in the target, not
just the value.

## What has measured evidence, at or near our scale

| Lever | Evidence | Scale shown | Fits our stack? |
|---|---|---|---|
| Execution-explanation SFT targets (natural-language line-by-line traces before the answer) | Self-Execution Simulation (2026): Qwen2.5-3B CRUXEval-O 37.5 → 68.0; NExT (2024) +10–26 pts repair; Scratchpads (2021) | 3B, 7B | Needs long targets → dense base (Qwen3.5 caps training at 1024 tokens) |
| Preference learning on pass/fail pairs (DPO, PLUM-style) | PLUM (2024) +4.8 avg, +11.8 LiveCodeBench on SFT-saturated models; CodeDPO; UTGen (2025) | 1–7B | Yes: `mlx-lm-lora dpo`, hybrid models autodetected; pairs already exist (4,400 samples with pass/fail + kills) |
| Negative examples in SFT (NAT), unlikelihood on the wrong literal (CRINGE) | NAT (NAACL 2025): monotonic gain with more negatives | agents, dialogue | Yes: a loss change in the existing trainer |
| GRPO with a graded execution reward | DeepScaleR / SimpleRL-Zoo / JustRL at 1.5B; CodeRL/RLTF at <3B | 1.5–3B | `mlx-lm-lora grpo`; novel reward shape; length-collapse risk |
| Verifier-augmented rejection sampling (V-STaR) | +4–17 pts at 7–13B | 7B+ | Yes, but it is a second model at inference (budget contract changes) |
| Property / invariant / exception assertions instead of exact values | Vikram (2023): valid-and-sound property test in 2.4 samples; Konstantinou (2024): LLM oracles encode actual behaviour <50% correct | GPT-4 class | Yes; mutation score of weak oracles unmeasured for pytest (gap) |
| Thinking on at inference, larger equal budget | CRUXEval: no CoT gain at 13B for Code Llama; Qwen3-class thinking models are trained for it; s1: a budget floor exists | mixed | Yes; dev-60 probe running (`baselines-dev-think4096`) |
| Dense base for training feasibility | mlx-lm#1185, mlx#3539 are Qwen3.5-specific; Qwen2.5-Coder-7B HumanEval+ 84.1, CRUXEval-O 56.0; Qwen3-4B-2507 dense with thinking checkpoints | | Unblocks long targets and normal LoRA speed |

## Recommended programme (for D026+)

1. **Switch the training base to a dense model** for recipe iteration:
   Qwen3-4B-Instruct-2507 (same size class, thinking variant available) or
   Qwen2.5-Coder-7B-Instruct (strongest measured execution reasoning). The
   hybrid stack cost most of weekend two and caps targets at 1024 tokens; the
   claim is relative to prompting of the same model, so the base can change.
2. **Change the training signal, not the volume.** First DPO on the pairs we
   already have (chosen = passes and clears the mutation gate; rejected = the
   same function's false-failure suite), then, if the dense base allows long
   targets, execution-explanation SFT (trace before assert) with STaR-style
   rejection sampling, thinking on, under an equal larger budget.
3. **Shape the output toward oracles the model can get right**: in the
   prompt and the targets, prefer exception/type/shape/property assertions and
   exact values only for short, simple outputs; report the assertion mix and
   the mutation score of each oracle class (new measurement, a real gap).
4. **Selection on all 171 dev functions**, never 60; one epoch; rank 8.
5. Keep everything else frozen: test split, budget, normalisation, probe.

## What we would stop doing

More positives-only SFT on Qwen3.5-4B; rank/LR/DoRA tuning as the primary
lever; oracle-filled literals as targets without the derivation.

## Measured after the reports: thinking on at inference (dev-60, budget 4096)

Qwen3.5-4B bf16, same 60 dev functions, same prompt: **validity 0.333 with
thinking on vs 0.32 off** (re-scored offline after fixing the think-block
stripper: the template opens `<think>` in the prompt, so only the closing tag
appears in the output; 55/60 outputs reasoned, ~2.5k characters each; 5 hit
the 4,096 budget); mutation score of valid suites 0.813 vs 0.850; mean 1,454
completion tokens vs 717; 17.5 s per function vs 8.7. Run
`baselines-dev-think4096-20260908T155256Z` (its manifest carries the first,
mis-stripped scoring: 0.34). One extra
valid suite for double the tokens: at 4B, reasoning tokens do not fix
expected-value prediction, as CRUXEval predicted for this size. Consequence
for the ranking: inference-time thinking and the STaR-with-thinking loop drop
out; preference learning on pass/fail pairs and the oracle-shape change move
up; execution-explanation SFT stays as the heavy bet (it trains a different
thing than thinking at inference, and its evidence is at 3B).

## Measured: LoRA speed on a dense base (Qwen3-4B-Instruct-2507 bf16)

Same overfit-8 check, same wrapper, **all 36 layers**, sequence 2048, batch 1:
**110–139 tok/s, peak memory 10.5 GB**, val loss 0.515 → 0.048 in 40 steps.
Against Qwen3.5-4B's training path (16 layers, sequence capped at 1024):
~37 tok/s at 25 GB. So the dense base is ~3.5x faster with all layers
adapted, at 40% of the memory, with twice the sequence length; batch 4 and
DPO's two sequences per example fit comfortably. Training feasibility is no
longer a constraint on the recipe.
