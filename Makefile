.PHONY: help install install-dev test lint format eval benchmark clean baseline gate

PYTHON := python3
PIP := $(PYTHON) -m pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	$(PIP) install -e .

install-dev:  ## Install with dev dependencies
	$(PIP) install -e ".[dev]"

install-all:  ## Install with all optional backends
	$(PIP) install -e ".[all,dev]"

test:  ## Run the test suite
	$(PYTHON) -m pytest -v

test-cov:  ## Run tests with coverage
	$(PYTHON) -m pytest --cov=evalforge --cov-report=term-missing --cov-report=html

lint:  ## Run ruff + mypy
	$(PYTHON) -m ruff check evalforge tests
	$(PYTHON) -m mypy evalforge

format:  ## Auto-format with ruff
	$(PYTHON) -m ruff check --fix evalforge tests
	$(PYTHON) -m ruff format evalforge tests

eval:  ## Run the full benchmark suite (the headline command from README)
	@mkdir -p benchmarks/results
	$(PYTHON) benchmarks/run_all.py --output benchmarks/results/current.json

baseline:  ## Promote current.json to baseline.json (for regression gate)
	@mkdir -p benchmarks/results
	@if [ ! -f benchmarks/results/current.json ]; then \
		echo "No current.json — run 'make eval' first."; exit 1; \
	fi
	cp benchmarks/results/current.json benchmarks/results/baseline.json
	@echo "Promoted current.json → baseline.json"

gate:  ## Run regression gate against baseline (fails on regression)
	$(PYTHON) -m evalforge.regression_gate \
		--baseline benchmarks/results/baseline.json \
		--current  benchmarks/results/current.json \
		--threshold 0.03

example:  ## Run the small demo example
	$(PYTHON) examples/basic_rag_eval.py

clean:  ## Clean caches, build artifacts
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf benchmarks/results
