.PHONY: help install train run test lint clean docker-up docker-down

help:
	@echo "Campaign ROI Estimation — Available Commands"
	@echo "============================================"
	@echo "install       Install dependencies"
	@echo "data          Generate synthetic training data"
	@echo "train         Train ML model"
	@echo "run           Start API server (development)"
	@echo "test          Run test suite with coverage"
	@echo "lint          Run linting and type checking"
	@echo "docker-up     Start all services with Docker"
	@echo "docker-down   Stop Docker services"

install:
	pip install -r backend/requirements-dev.txt

data:
	python scripts/generate_sample_data.py

train: data
	python scripts/train_model.py --data data/raw/campaign_data.csv

run:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	cd backend && pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

lint:
	ruff check backend/app/
	mypy backend/app/ --ignore-missing-imports

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf backend/.pytest_cache backend/htmlcov
