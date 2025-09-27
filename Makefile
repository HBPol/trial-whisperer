.RECIPEPREFIX := >
.PHONY: run seed format test index

run:
>uvicorn app.main:app --reload

seed:
>bash scripts/seed_smallset.sh


format:
>bash scripts/format.sh

test: export TRIALS_DATA_PATH ?= .data/test_processed/trials.jsonl
test:
>pytest -q

index:
>python -m scripts.index
