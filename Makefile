# Makefile for AI Evolution Environment

.PHONY: help install test run clean lint format docker

# Default target
help:
	@echo "AI Animal Evolution Environment"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  test       - Run test suite"
	@echo "  test-cov   - Run tests with coverage"
	@echo "  run        - Run simulation CLI"
	@echo "  ui         - Launch Streamlit UI"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code with black"
	@echo "  docker     - Build Docker image"
	@echo "  clean      - Clean generated files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black flake8 mypy pytest-cov

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=ai_evo --cov-report=html --cov-report=term

test-determinism:
	pytest tests/test_determinism.py -v

test-performance:
	pytest tests/test_performance.py -v

# Running
run:
	python main.py --verbose --steps 1000

run-interactive:
	python main.py --interactive --verbose

ui:
	streamlit run ui/streamlit_app.py

# Code quality
lint:
	flake8 ai_evo/ tests/ main.py
	mypy ai_evo/ --ignore-missing-imports

format:
	black ai_evo/ tests/ main.py ui/

format-check:
	black --check ai_evo/ tests/ main.py ui/

# Docker
docker:
	docker build -t ai-evolution-environment .

docker-run:
	docker run -p 8501:8501 ai-evolution-environment

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ dist/ build/

# Data and results
clean-data:
	rm -f *.json *.txt *.csv

# Development helpers
dev-setup: install-dev
	pre-commit install  # If using pre-commit hooks

profile:
	python main.py --profile --steps 500

benchmark:
	python -m pytest tests/test_performance.py::TestPerformance::test_simulation_step_performance -v
