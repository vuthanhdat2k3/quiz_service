# Makefile for Quiz Generation Service

.PHONY: install test lint run clean

# Install dependencies
install:
	pip install -r requirements.prod.txt
	pip install -r requirements.txt
	pip install black isort flake8 pytest httpx

# Run tests
test:
	pytest tests/unit

# Run linters
lint:
	black .
	isort .
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Run application locally
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Clean pycache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
