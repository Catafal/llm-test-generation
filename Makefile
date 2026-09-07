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

baselines:        ## zero-shot + few-shot for all candidate models; POOL=pilot|test|dev MODELS=9b,4b,coder7b
	uv run python -m testgen.baselines --pool $(or $(POOL),pilot) --models $(or $(MODELS),9b,4b,coder7b)

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

eval:             ## score a run against the held-out pool
	@echo "not implemented yet (weekend 1, T7)"; exit 1
