# Three commands that matter. Everything else is a script under testgen/.
.PHONY: setup lint test eval pilot baselines

setup:            ## create .venv and install all deps (incl. dev)
	uv sync --all-groups

lint:             ## ruff check + format check
	uv run ruff check . && uv run ruff format --check .

test:             ## harness unit tests
	uv run pytest -q

pilot:            ## run the 10-case pilot: harness must rank strong > weak on every case
	uv run python -m testgen.pilot

pilot-survivors:  ## same, listing surviving mutants per case (for equivalence labelling)
	uv run python -m testgen.pilot --survivors

baselines:        ## zero-shot + few-shot for all pilot models, writes runs/<id>/manifest.json
	@echo "not implemented yet (weekend 1, T7)"; exit 1

harvest:          ## harvest post-cutoff pure functions from GitHub into data/heldout/candidates.jsonl
	uv run python -m testgen.data.harvest --repos $(or $(REPOS),50)

eval:             ## score a run against the held-out pool
	@echo "not implemented yet (weekend 1, T7)"; exit 1
