# Recipe audit: is a fundamental missing, or is the task the wall?

Audits `configs/lora-4b.yaml`, `configs/dpo-4b.yaml`, `configs/lora-q3-4b.yaml`,
`testgen/train/{train,dpo,filter,pairs}.py`, `testgen/generate/prompts.py`,
`testgen/baselines.py`, `docs/results/weekend-2.md`, and the installed
`mlx-lm==0.31.3` / `mlx-lm-lora==3.1.2` source in `.venv`. Question asked:
"is there some LLM fundamental we are missing?" Short answer: no single
hyperparameter explains the four nulls; the recipe is close to textbook on
every axis that was checked against source, and the one real anomaly found
(DPO's unmasked prompt tokens) is provably inert for the loss actually used.
The nulls are best explained by a data/task mismatch that gets worse, not
better, with more correct training.

## 1. LoRA capacity and placement

**What was done.** SFT (`lora-4b.yaml`): rank 16, `scale: 2.0`, dropout
0.05, on the last 16 of 32 layers of the Qwen3.5-4B hybrid (D024, forced by
a Gated DeltaNet training-kernel limitation), keys covering every linear
projection in those layers — `self_attn.{q,k,v,o}_proj`,
`linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`,
`mlp.{gate,up,down}_proj`. DPO (`dpo-4b.yaml`): rank 8, same scale
convention, same 16 layers. The dense-base run (`lora-q3-4b.yaml`,
iteration 3) used rank 8 but **all 36 layers** (`num_layers: -1`), attention
+ MLP only (no linear-attention path to cover).

**Verified from source, not docs.** `mlx_lm/tuner/lora.py`'s `LoRALinear`
computes `y + scale * (lora_b.T @ lora_a.T) @ x` — `scale` is a *direct*
multiplier on the adapter's output, not `alpha` divided at call time. So
the yaml comment ("`scale = alpha/rank`, mlx-lm multiplies the adapter
output by scale directly") is correct: rank 16/scale 2.0 = alpha 32 under
the `alpha = 2×rank` convention; rank 8/scale 2.0 = alpha 16, same
convention. This matches the parametrization
[Thinking Machines' "LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/)
recommends specifically because it keeps the optimal learning rate
"approximately independent of rank" — a real risk (getting `alpha`
semantics backwards, which silently changes effective LR by the rank
ratio) was checked and is not present here.

**Best practice vs. what was done.** LoRA Without Regret's two load-bearing
claims: (1) LoRA "performs better when applied to all weight matrices,
especially MLP... attention-only LoRA underperforms" — this project already
does all-linear, so that lever was pulled correctly from the first run. (2)
LoRA underperforms full fine-tuning specifically when "the dataset exceeds
LoRA capacity," i.e. it's a data-size vs. capacity relationship, not a hard
rank floor. At rank 16 on 388 examples this project is nowhere near a
capacity ceiling — [Biderman et al., "LoRA Learns Less and Forgets Less"](https://arxiv.org/abs/2405.09673)
report full fine-tuning learns updates with effective rank "10-100x greater"
than typical LoRA settings, but that gap matters at the tens-of-thousands-
of-examples scale they study, not at n=388-609. **Verdict: rank/placement
is not a plausible explanation for the null.** The one real placement
compromise — LoRA on the last 16 of 32 layers of the hybrid model instead
of all 32 — is a legitimate capacity restriction (half the depth is frozen
and can never toun a gradient), but iteration 3's all-36-layer run on the
dense base changed the base model *and* the target format simultaneously
(D024's own limitations note), so it cannot isolate the depth variable; it
is suggestive, not evidence, and it is also the run with the worst
mutation-score regression (−0.070), which argues against "more depth would
have fixed it" being the missing lever.

## 2. Learning rate, schedule, batch size, iters accounting

**What was done.** SFT: LR 1e-4, cosine decay, warmup 10, 3 epochs of 388
examples at micro-batch 1 / accumulation 8 = 1200 micro-iterations = 150
optimizer steps; the `lr_schedule.arguments` decay-steps value (150) is set
in optimizer-step units. DPO: LR 5e-6, beta 0.1, no warm-up documented in
the yaml snippet, 2 epochs / 117 optimizer steps.

**Verified from source.** `mlx_lm/tuner/trainer.py`'s `train()` calls
`optimizer.update(model, grad)` only when `it % grad_accum_steps == 0`; mlx
optimizers increment their internal step counter exactly once per `update`
call. So the schedule's decay-step argument does count optimizer steps, as
the yaml comment claims, and 1200/8=150 lines up exactly — no
off-by-accumulation-factor bug (a common mistake with HF-style trainers,
where schedulers sometimes step per micro-batch). This one checks out
cleanly.

**LR magnitude vs. best practice.** Thinking Machines' central empirical
claim is "the optimal learning rate for FullFT is lower by a factor of 10
than for high-rank LoRAs," across SL and RL. A typical full-FT LR for a
4B-class model in this data regime is O(1e-5); 1e-4 for LoRA SFT here sits
almost exactly at that 10x multiplier — this is not an underpowered LR. For
DPO, TRL's own default is 1e-6 for full-FT DPO and its documentation
explicitly recommends "≈1e-5" when training adapters
([TRL DPO docs](https://huggingface.co/docs/trl/main/en/dpo_trainer)) — 5e-6
used here sits between those two, i.e. plausibly a little conservative but
not obviously wrong, and the yaml's own justification (Rafailov et al. used
1e-6 full-FT, "LoRA tolerates ~5e-6") is a defensible middle ground rather
than a fundamentals miss. **Verdict: LR/schedule/iters accounting is sound;
this is not the explanation.**

## 3. Loss masking

**What was done and verified.** `mask_prompt: true` in `lora-4b.yaml` feeds
`mlx_lm.tuner.datasets.ChatDataset`, which computes an `offset` as
`len(apply_chat_template(messages[:-1], add_generation_prompt=True))` and
returns `(tokens, offset)`; `default_loss` in `trainer.py` masks every
position below that offset. This is completion-only loss, correctly done —
equivalent to what Unsloth's `train_on_responses_only` does, and to the
standard SFT-chat recipe. **No issue here.**

**DPO is a genuinely interesting finding, but not the culprit.**
`mlx_lm_lora.trainer.datasets.DPODataset` tokenizes the *full* rendered
sequence (system + prompt + assistant reply) for both `chosen` and
`rejected`, and the DPO trainer's `chosen_masks`/`rejected_masks` in
`dpo_trainer.py` are pure padding masks — 1.0 over the whole sequence up to
its length, not just the completion. Read naively this looks like a
prompt-masking bug: the loss appears to include log-probs of the shared
system+prompt tokens. It is algebraically inert, though, for the
`loss_type="sigmoid"` config actually used (`dpo_cpo_loss_type: sigmoid` in
`dpo-4b.yaml`): `compute_score` sums per-token log-probs without dividing by
token count for anything but `"ipo"`, and because `chosen` and `rejected`
share an identical prompt prefix under strictly causal attention, the
prompt-token log-probs are bit-identical in both forward passes. The DPO
loss depends only on `(chosen_score − rejected_score) − (ref terms)`, so
the shared prompt term cancels exactly in that subtraction, in both the
forward value and its gradient. **This would matter if the loss type were
ever switched to `"ipo"`** (which normalizes by token count including the
prompt, so a longer shared prompt would shrink both chosen and rejected
per-token scores non-identically once truncation differs) — worth fixing as
a latent footgun for future runs, but it is not why iteration 2's DPO
run went nowhere.

## 4. Data quality: training on your own oracle-corrected outputs

This is the part of the recipe most worth interrogating, and the project's
own numbers make the mechanism legible. From `data/train/sft/yield.json`:
4400 candidates from 550 functions (K=8, temp 0.7), of which 1247 (28%)
were valid *before* any oracle correction and 2165 (49%) valid *after* the
harness rewrote wrong literal expected values (`testgen/train/oracle.py`).
Of 33,546 assert-value sites the harness checked, 23,689 (83%) were
**already correct** — the model's own self-samples get most literals right
unaided — and 4,157 (12.4%) were silently replaced. `filter.py`'s
`best_per_function` then keeps, per function, the highest-mutation-score
valid-*after-fill* candidate; the assistant turn written to `train.jsonl`
is the corrected suite (`to_chat` uses `k["suite"]`, the post-oracle text,
not `suite_unaided`).

Read one kept example (`encode_ber_length`, a bit-manipulation function):
the target suite is well-formed, non-trivial, has 8 boundary-crossing tests
and exact byte-literal assertions — good pytest, by any style rubric. The
problem is invisible at the level of a single example and only shows up in
aggregate: `kept_assertion_styles` shows `literal_eq: 2978` dominating
(`membership: 814`, everything else negligible), and the write-up reports
this share *rising* after SFT (0.724 → 0.771) while validity *fell* (0.438
→ 0.397). The target distribution teaches "assert exact literal values with
confidence"; roughly 1 in 8 of those literals in the model's own unaided
attempts are wrong, and SFT on the corrected version does not teach the
model to compute the right value — it teaches it to *state* a value more
often, without changing its arithmetic. That is the opposite of what
[ReST-EM](https://arxiv.org/abs/2312.06585) and STaR-style self-training
rely on: their filtering criterion (a binary pass/fail the model itself
achieved) selects samples the model already produced correctly and
reinforces exactly that skill, which is why ReST-EM "significantly
surpasses fine-tuning on human data" in the regimes it was tested. Here the
filter is pass/fail *after an oracle intervention the model never
performed*, so the reinforced skill (confident assertion of an exact value)
and the tested skill (computing that value at inference, unaided) are not
the same skill — the training signal actively rewards a behavior (assert
literal values) that is decoupled from, and at this model scale
antagonistic to, the capability being measured (predict the value
correctly). This is consistent with, and a plausible sufficient
explanation for, the SFT validity *drop*, independent of any
architecture/hyperparameter choice.

## 5. DPO specifics

**What was done.** beta 0.1 (TRL's own default), LR 5e-6, reference model =
frozen copy of the base (mlx-lm-lora's default when no reference path is
given), no SFT warm-start stage on the chosen responses before DPO. Pairs
(`pairs.py`, D030 grounded variant): of 645 pairs, 536 rejected are "invalid
after fill" (gross failures — bad inputs, wrong arity), only 108 are the
hardest-negative type (oracle-rescued, i.e. "good test, wrong literal" — a
near-miss on values specifically). So the *majority* of the preference
signal contrasts a good suite against a badly broken one, not a good suite
against an otherwise-identical suite with one wrong number — the fine-
grained "value accuracy" contrast the null is actually about is a minority
of the training signal.

**Failure mode observed matches the literature.** The write-up reports
DPO training accuracy reaching 0.9 while accuracy on 27 held-out pairs
stayed at chance (0.44–0.61) throughout training — textbook DPO
memorization without generalization. TRL and the DPO literature generally
recommend an SFT stage on the chosen responses before DPO precisely to
avoid this (the original [DPO paper](https://huggingface.co/papers/2305.18290)'s
own recipe fine-tunes on preferred completions first); RPO/"DPO+NLL"
variants add an explicit NLL term on the chosen response to the DPO loss
for the same reason, and IPO was designed to bound the reward margin
because unconstrained DPO can drive the chosen/rejected gap unboundedly
without improving held-out behavior. This project skipped the warm-start
by design (same base as the null SFT run, deliberately isolating "just
change the signal") — a reasonable ablation choice for isolating variables,
but it does forgo a lever ([TRL DPO docs](https://huggingface.co/docs/trl/main/en/dpo_trainer)
list `loss_type="sft"` combination, i.e. RPO-style, as a supported
multi-loss option) that specifically targets the chance-on-held-out
symptom actually observed. **This is a real, plausible, and cheap-to-try
lever** — but the pairs themselves being coarse (mostly gross-failure
contrasts, not literal near-misses) suggests IPO/SimPO/ORPO would not help
much either: those mainly fix length bias and margin instability, neither
of which is the reported mechanism (assertion-style mix reverting to base
levels, false-failure rate identical to base). The signal that's missing is
not a better DPO variant; it's pairs that isolate "same inputs, same
structure, one has the right number" at a scale bigger than 108 examples.

## 6. Evaluation

Greedy, thinking off, 2048-token budget, ≤8 tests, held out by family, n=315
test / n=60-171 dev. Greedy decoding is the right choice for measuring a
deployed, reproducible tool and does not plausibly explain a systematic
value-prediction gap — temperature only adds variance, it doesn't grant the
model arithmetic it lacks. The dev-selection problem is real and the
project already caught it: bootstrap CIs on n=315 have half-width ≈0.04-0.05
(consistent with the reported −0.098 to +0.013 spans), and the write-up's
own "Dev curve and its lesson" section documents a winner's-curse checkpoint
pick on dev-60 that did not transfer to test. This is a correctly diagnosed
noise problem, not a hidden one, and the project's remedy (pre-registered
CI-excludes-zero bar, iteration 4) is the right fix, not a missing
fundamental.

## 7. Base-model choice

Qwen3.5-4B is a hybrid Gated DeltaNet architecture whose training path
required a custom kernel-freezing wrapper just to make LoRA trainable on
this hardware (D024) — an unusual choice that cost significant engineering
overhead (T5's `train.py`/`dpo.py` wrappers) and constrained sequence length
to 1024 and depth to 16 layers, for reasons orthogonal to the model's
actual capability at the task. Iteration 3 swapped to dense
`Qwen3-4B-Instruct-2507`, removing the architecture confound but not the
capability one, since neither model was pretrained with heavy code-execution
supervision. Neither Qwen2.5-Coder-3B/7B (code-pretrained, extensively
documented to have materially higher output-prediction accuracy on
execution benchmarks) nor any base above the CRUXEval-O ≥70 bar the project
itself names as the target
([CRUXEval leaderboard](https://crux-eval.github.io/leaderboard.html))
was tried. This is the single largest untested lever in the whole
project, and it is the one the write-up already flags as "outside this
project's budget." Nothing in the recipe audit contradicts that framing —
if anything it reinforces it: every recipe axis checked out reasonably,
which raises the prior that the bottleneck is what the base model already
knows about execution, not how it was subsequently adapted.

## 8. Bugs and mismatches

Checked and **found clean**: chat-template / fence format consistency
(`filter.py`'s `to_chat` emits `` ```python\n{suite}``` ``, exactly what
`extract_suite()` in `prompts.py` parses at inference — confirmed by
in-code cross-reference, "FT10"); `thinking` flag consistently off in both
`baselines.py` generation and training; `grad_checkpoint(model.layers[0])`
in `mlx_lm/tuner/trainer.py` patches `type(layer).__call__` — verified in
`mlx_lm/models/qwen3_5.py` that all 32 layers share one `DecoderLayer`
class (with an internal `is_linear` flag choosing attention vs. DeltaNet),
so this checkpoints every layer, not just the first layer's type — not the
bug it could have been on a genuinely heterogeneous layer stack. The one
confirmed anomaly is the DPO prompt-masking gap in §3, and it is provably
inert for the loss type actually run.

## Ranked list: what to try differently, and expected effect

1. **Distill from a code-pretrained or execution-strong base (Qwen2.5-Coder,
   or any model with CRUXEval-O ≥ 70) instead of adapting Qwen3.5/Qwen3-4B-
   Instruct.** Highest expected effect size of anything on this list —
   plausibly the only lever that touches the actual bottleneck (arithmetic/
   execution prediction), per the project's own diagnosis and the general
   finding that output-prediction is a base-model, pretraining-time
   property. Not validated in this project; recorded as the "untried
   lever."
2. **Local teacher distillation at 10k+ examples**, already named in the
   write-up as the one lever consistent with the field's published
   successes (ReST-EM/STaR-scale gains use tens of thousands of examples,
   not 388-645). Medium-high expected effect on validity if paired with
   lever 1; low expected effect *without* a stronger teacher, since
   distilling the same model's own corrected outputs is what iteration 1
   already did and it made things worse.
3. **DPO with an SFT warm-start on chosen (or RPO/DPO+NLL) and pairs
   deliberately restricted to same-input, near-miss-literal contrasts**
   (only 108 of 645 pairs were this type). Would plausibly fix the
   held-out-chance-while-training-fits-0.9 symptom — a real, diagnosed
   failure mode with literature support — but the ceiling is capped by
   point 4: if the model structurally cannot compute the value, no amount
   of preference-signal quality teaches it to. Expect a small, not
   qualitative, improvement.
4. **All-32-layer LoRA on the hybrid base** (removing the D024 depth
   restriction), if the kernel/memory problem can be solved. Plausible but
   unconfirmed; the one run that removed the layer restriction also swapped
   base model and target format, so it's untested in isolation, and that
   run had the worst mutation-score regression of the four iterations —
   weak evidence against, not for.
5. **Everything else audited (LoRA rank/alpha/scale semantics, LR
   magnitude, schedule/iters accounting, loss masking, checkpointing,
   greedy decoding, chat-template consistency) is correct or defensible as
   implemented.** Expected effect of "fixing" any of these: approximately
   zero, because none of them is broken.

**Honest statement.** None of the above would plausibly turn the measured
+0.024 into a resolved CI-excludes-zero gain at the current data scale
(n=315, half-width ≈0.04). The +0.024 point estimate recurring identically
across two independent training arms (re-scored SFT and confirmatory
grounded DPO) is itself evidence of a small real effect, not noise, but
"small real effect, unresolved at n=315" is the correct description whether
or not any recipe change is made — resolving it needs either ~1,500 held-out
functions (the project's own estimate) or an effect several times larger,
which only a base-model change (lever 1) or an order-of-magnitude data-scale
change (lever 2) is likely to produce. The recipe itself is not the
missing fundamental; the fundamental the four iterations correctly located
is that a 4B model — hybrid or dense, adapted by LoRA or DPO, on a few
hundred self-generated examples — does not have the execution-prediction
capability the task needs, and no amount of correctly-executed fine-tuning
recipe teaches a capability the base checkpoint does not have the compute
budget here to acquire.

## Sources

- [Thinking Machines — "LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/)
- [Biderman et al., "LoRA Learns Less and Forgets Less" (arXiv 2405.09673)](https://arxiv.org/abs/2405.09673)
- [Rafailov et al., "Direct Preference Optimization" (arXiv 2305.18290)](https://huggingface.co/papers/2305.18290)
- [TRL DPOTrainer documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [Singh et al., "Beyond Human Data: ReST-EM" (arXiv 2312.06585)](https://arxiv.org/abs/2312.06585)
- [CRUXEval leaderboard](https://crux-eval.github.io/leaderboard.html)
- Installed source: `mlx-lm==0.31.3` (`mlx_lm/tuner/lora.py`, `mlx_lm/tuner/trainer.py`,
  `mlx_lm/tuner/datasets.py`, `mlx_lm/models/qwen3_5.py`) and
  `mlx-lm-lora==3.1.2` (`mlx_lm_lora/trainer/dpo_trainer.py`,
  `mlx_lm_lora/trainer/datasets.py`), read directly from
  `.venv/lib/python3.12/site-packages/` in this repo.
- In-repo: `configs/lora-4b.yaml`, `configs/dpo-4b.yaml`,
  `configs/lora-q3-4b.yaml`, `testgen/train/train.py`, `testgen/train/dpo.py`,
  `testgen/train/filter.py`, `testgen/train/pairs.py`,
  `testgen/generate/prompts.py`, `testgen/baselines.py`,
  `docs/results/weekend-2.md`, `data/train/sft/yield.json`,
  `data/train/dpo-grounded/pairs.json`, `data/train/sft/train.jsonl`,
  `data/train/dpo-grounded/train.jsonl`.
