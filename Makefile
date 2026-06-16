# Adaptive Offers Platform — command shortcuts.
# Cross-platform primary interface is the `adaptive-offers` CLI; this Makefile
# wraps it for convenience (works in CI, Linux, macOS, Docker, Git-Bash on Win).

.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install dev data synth train evaluate pipeline serve dashboard test lint format docker-build docker-up clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (runtime only).
	$(PY) -m pip install -e .

dev: ## Install with dev + BI extras.
	$(PY) -m pip install -e ".[dev,bi]"

data: ## Stage 1 — build processed base (no leakage).
	adaptive-offers data build

synth: ## Stage 2 — generate synthetic enrichment.
	adaptive-offers synth generate

train: ## Stage 3 — simulate & train bandit policies.
	adaptive-offers train

evaluate: ## Stage 4 — offline evaluation + golden set + fairness.
	adaptive-offers evaluate

pipeline: ## Stages 1-4 end-to-end in one command.
	adaptive-offers pipeline

serve: ## Stage 5 — run the FastAPI decision service.
	adaptive-offers serve

dashboard: ## Run the Streamlit BI dashboard.
	streamlit run dashboard/app.py

test: ## Run the test suite (unit + integration).
	$(PY) -m pytest

lint: ## Lint + type-check.
	$(PY) -m ruff check src tests
	$(PY) -m mypy src

format: ## Auto-format with ruff.
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

docker-build: ## Build the container image.
	docker build -t adaptive-offers:local .

docker-up: ## Start the full stack (api + mlflow + dashboard).
	docker compose up --build

clean: ## Remove caches and generated artifacts.
	rm -rf .pytest_cache .ruff_cache .mypy_cache artifacts mlruns
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
