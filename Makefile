# Three commands that matter. Everything else is a script under testgen/.
.PHONY: setup lint test eval pilot baselines

setup:            ## create .venv with dev + models groups (the default working set)
	uv sync --group dev --group models

sync-models:      ## switch venv to the models group (mlx-lm; transformers 5)
	uv sync --group dev --group models

sync-decontam:    ## switch venv to the decontam group (jina embeddings; transformers 4)
	uv sync --group dev --group decontam

lint:             ## ruff check + format check
	uv run ruff check . && uv run ruff format --check .

test:             ## harness unit tests
	uv run pytest -q

pilot:            ## run the 10-case pilot: harness must rank strong > weak on every case
	uv run python -m testgen.pilot

pilot-survivors:  ## same, listing surviving mutants per case (for equivalence labelling)
	uv run python -m testgen.pilot --survivors

split:            ## family-wise test/dev split of data/heldout/pool.jsonl (D020)
	uv run python -m testgen.data.split

derisk:           ## D023 de-risk: 4B self-samples, rejection vs oracle-fill on 40 dev fns
	uv run --group models python -m testgen.train.derisk --n $(or $(N),40) --k $(or $(K),4) --batch $(or $(BATCH),16)

propose:          ## T2: 4B self-samples K per training fn; K=8 BATCH=16 [RESUME=runs/propose-*]
	caffeinate -i uv run --group models python -m testgen.train.propose --k $(or $(K),8) --batch $(or $(BATCH),16) $(if $(RESUME),--resume $(RESUME),)

filter:           ## T3: oracle-fill + execution filter -> data/train/sft/; RUN=runs/propose-*
	uv run --group models python -m testgen.train.filter --run $(RUN)

train-single:     ## T5: one-process LoRA (compile disabled) -> models/adapters/$(RUN) [CONFIG=configs/lora-4b.yaml]
	caffeinate -i env HF_HUB_OFFLINE=1 uv run --group models python -m testgen.train.train -c $(or $(CONFIG),configs/lora-4b.yaml) --adapter-path models/adapters/$(or $(RUN),lora-4b)

train-dense:      ## D028: SFT on the dense base; RUN= DATA=data/train/trace/inline ITERS=
	caffeinate -i env HF_HUB_OFFLINE=1 uv run --group models python -m testgen.train.train -c configs/lora-q3-4b.yaml --keep-compile --adapter-path models/adapters/$(or $(RUN),trace-q3-4b) $(if $(DATA),--data $(DATA),) $(if $(ITERS),--iters $(ITERS),)

train:            ## T5 fallback: segmented LoRA (mlx-lm#1185) -> models/adapters/$(RUN); RUN=lora-4b-<tag> [START=k]
	uv run --group models python -m testgen.train.segments --run models/adapters/$(or $(RUN),lora-4b) $(if $(START),--start $(START),)

devcurve:         ## T5: harness score of every checkpoint on 60 dev fns; RUN=models/adapters/<run> [GROUNDED=1 EVERY=2]
	caffeinate -i env HF_HUB_OFFLINE=1 uv run --group models python -m testgen.train.devcurve --run $(RUN) --limit $(or $(LIMIT),60) --model $(or $(MODEL),4b-bf16) $(if $(GROUNDED),--grounded,) $(if $(EVERY),--every $(EVERY),)

curate-ext:       ## D032 stage 1: KodCode through the harness; STAGE=a|b|c [LIMIT= KEEP= OUT= RESUME=1]
	caffeinate -i uv run python -m testgen.train.curate_ext --stage $(or $(STAGE),a) $(if $(LIMIT),--limit $(LIMIT),) $(if $(KEEP),--keep $(KEEP),) $(if $(OUT),--out $(OUT),) $(if $(RESUME),--resume,)

pairs:            ## D026: preference pairs from data/train/sft/scored.jsonl -> data/train/dpo/ [GROUNDED=1 -> dpo-grounded/]
	HF_HUB_OFFLINE=1 uv run --group models python -m testgen.train.pairs $(if $(GROUNDED),--grounded,)

dpo:              ## D026: DPO with mlx-lm-lora -> models/adapters/$(RUN); ITERS= (one epoch = pairs)
	caffeinate -i env HF_HUB_OFFLINE=1 uv run --group models python -m testgen.train.dpo -c configs/dpo-4b.yaml --adapter-path models/adapters/$(or $(RUN),dpo-4b) $(if $(ITERS),--iters $(ITERS),) $(if $(DATA),--data $(DATA),)

baselines:        ## zero-shot + few-shot for all candidate models; POOL=pilot|test|dev MODELS=9b,4b,coder7b [LIMIT= ADAPTER= TAG= CONDITIONS=]
	caffeinate -i uv run --group models python -m testgen.baselines --pool $(or $(POOL),pilot) --models $(or $(MODELS),9b,4b,coder7b) $(if $(LIMIT),--limit $(LIMIT),) $(if $(ADAPTER),--adapter $(ADAPTER),) $(if $(TAG),--tag $(TAG),) $(if $(CONDITIONS),--conditions $(CONDITIONS),) $(if $(THINKING),--thinking,)

harvest:          ## harvest post-cutoff pure functions from GitHub into data/heldout/candidates.jsonl
	uv run python -m testgen.data.harvest --repos $(or $(REPOS),50)

harvest-train:    ## D022 overflow harvest into data/train/candidates.jsonl; REPOS=400
	uv run python -m testgen.data.harvest --overflow --repos $(or $(REPOS),400)

decontaminate-train: ## D022: data/train/pool.jsonl decontaminated vs both held-out splits
	uv run --group dev --group decontam python -m testgen.data.decontaminate --dir data/train --against heldout

decontaminate:    ## build data/heldout/pool.jsonl + report (needs `make sync-decontam` first)
	uv run --group dev --group decontam python -m testgen.data.decontaminate

models-list:      ## models on disk under models/hf/ with sizes
	uv run python -m testgen.models list

models-pull:      ## download models into the repo: KEYS="9b 4b coder7b embed"
	uv run python -m testgen.models pull $(or $(KEYS),9b 4b coder7b)

models-rm:        ## delete models: KEYS="4b" or ALL=1 to free everything
	uv run python -m testgen.models rm $(if $(ALL),--all,$(KEYS))

clean-harvest:    ## delete cloned repos under .cache/harvest (safe once the pool is frozen)
	rm -rf .cache/harvest

figure:           ## results.png + devcurves.png under docs/figures from run artifacts
	uv run python -m testgen.figure

adapters-push:    ## publish the shipped adapters to the HF Hub (needs `uv run hf auth login`); USER=
	uv run --group models python -m testgen.publish push $(if $(USER),--user $(USER),)

adapters-pull:    ## download the shipped adapters into models/adapters/ (fresh clone); USER=
	uv run --group models python -m testgen.publish pull $(if $(USER),--user $(USER),)

demo:             ## 90-s demo: base vs fine-tune on one held-out fn; ID=<fn id> [FROM_RUNS=1] or PICK=1 to list
	env HF_HUB_OFFLINE=1 uv run --group models python -m testgen.demo $(if $(PICK),--pick,) $(if $(ID),--id "$(ID)",) $(if $(FROM_RUNS),--from-runs,)

eval:             ## T6: fine-tuned 4B zero-shot on the test split; ADAPTER=models/adapters/<run>/ckpt-NNNNNNN [GROUNDED=1]
	caffeinate -i env HF_HUB_OFFLINE=1 uv run --group models python -m testgen.baselines --pool test --models $(or $(MODEL),4b-bf16) --conditions zero --adapter $(ADAPTER) --tag finetune $(if $(GROUNDED),--grounded,)
