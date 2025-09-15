.RECIPEPREFIX := >
.PHONY: run seed format test index

run:
>uvicorn app.main:app --reload

seed:
>bash scripts/seed_smallset.sh


format:
>bash scripts/format.sh

test:
>pytest -q

index:
>python -m scripts.index
