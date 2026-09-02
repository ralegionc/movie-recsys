.PHONY: install data features train bench api clean

install:
	pip install -r requirements.txt

# Fast iteration on ml-latest-small (default). For the full run:
#   RECSYS_DATASET=full make data features train bench
data:
	python -m src.data.prepare

features:
	python -m src.models.content

train:
	python -m src.train_two_tower

bench:
	python -m src.benchmark

api:
	uvicorn src.api.main:app --reload --port 8000

pipeline: data features train bench

clean:
	rm -rf data/processed/* artifacts/*
