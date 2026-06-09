.PHONY: install pipeline train serve test docker-build

install:
	pip install -e ".[dev]"
	python -m spacy download en_core_web_sm

pipeline:
	python -m ml.data_pipeline

train: pipeline
	python ml/train.py

serve:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

docker-build:
	docker compose build
