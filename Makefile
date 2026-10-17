.PHONY: setup generate etl validate test up down logs

setup:
	pip install -r requirements.txt

generate:
	python -m src.generate_dataset

etl:
	python -m src.data_pipeline

validate:
	python -m src.validate

test:
	pytest -q --cov=src --cov-report=term-missing

up:
	docker compose up --build -d

down:
	docker compose down -v

logs:
	docker compose logs -f
