# makefile for memory-agent-security research framework
# usage: make <target>

.DEFAULT_GOAL := help
SHELL := /bin/bash
PYTHON := python
PYTEST := $(PYTHON) -m pytest
SRC := src
TESTS := src/tests tests

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------

.PHONY: install
install: ## install core dependencies
	$(PYTHON) -m pip install -e ".[ml,llm]"

.PHONY: install-dev
install-dev: ## install all dependencies including dev tools
	$(PYTHON) -m pip install -e ".[all]"
	pre-commit install

.PHONY: install-hooks
install-hooks: ## install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

.PHONY: format
format: ## auto-format code with black + isort
	$(PYTHON) -m black $(SRC)
	$(PYTHON) -m isort $(SRC)

.PHONY: format-check
format-check: ## check formatting without modifying files
	$(PYTHON) -m black --check --diff $(SRC)
	$(PYTHON) -m isort --check-only --diff $(SRC)

# ---------------------------------------------------------------------------
# linting
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## run all linters (flake8 + ruff)
	$(PYTHON) -m flake8 $(SRC)
	$(PYTHON) -m ruff check $(SRC)

.PHONY: lint-fix
lint-fix: ## run ruff with auto-fix
	$(PYTHON) -m ruff check --fix $(SRC)

.PHONY: flake8
flake8: ## run flake8 only
	$(PYTHON) -m flake8 $(SRC)

.PHONY: ruff
ruff: ## run ruff only
	$(PYTHON) -m ruff check $(SRC)

# ---------------------------------------------------------------------------
# type checking
# ---------------------------------------------------------------------------

.PHONY: typecheck
typecheck: ## run mypy type checker
	$(PYTHON) -m mypy $(SRC) --ignore-missing-imports

# ---------------------------------------------------------------------------
# security
# ---------------------------------------------------------------------------

.PHONY: security
security: ## run bandit security scanner
	$(PYTHON) -m bandit -r $(SRC) -c pyproject.toml --quiet

.PHONY: safety
safety: ## check dependencies for known vulnerabilities
	$(PYTHON) -m safety check

# ---------------------------------------------------------------------------
# testing
# ---------------------------------------------------------------------------

.PHONY: test
test: ## run main test suite (src/tests/)
	PYTHONPATH=$(SRC) $(PYTEST) src/tests/test_memory_security.py \
		--override-ini="addopts=" -q --tb=short

.PHONY: test-all
test-all: ## run all tests (src/tests/ + tests/)
	PYTHONPATH=$(SRC) $(PYTEST) src/tests/ tests/ \
		--override-ini="addopts=" -q --tb=short

.PHONY: test-fast
test-fast: ## run tests excluding slow markers
	PYTHONPATH=$(SRC) $(PYTEST) src/tests/ tests/ \
		--override-ini="addopts=" -q --tb=short -m "not slow"

.PHONY: test-smoke
test-smoke: ## run smoke tests only
	PYTHONPATH=$(SRC) $(PYTHON) smoke_test.py

.PHONY: test-cov
test-cov: ## run tests with coverage report
	PYTHONPATH=$(SRC) $(PYTEST) src/tests/ tests/ \
		--override-ini="addopts=" -q --tb=short \
		--cov=$(SRC) --cov-report=html:reports/coverage \
		--cov-report=term-missing --cov-fail-under=80

.PHONY: test-verbose
test-verbose: ## run tests with full output
	PYTHONPATH=$(SRC) $(PYTEST) src/tests/ tests/ \
		--override-ini="addopts=" -v --tb=long

# ---------------------------------------------------------------------------
# pre-commit
# ---------------------------------------------------------------------------

.PHONY: pre-commit
pre-commit: ## run all pre-commit hooks on all files
	pre-commit run --all-files

.PHONY: pre-commit-manual
pre-commit-manual: ## run manual-stage hooks (mypy, bandit)
	pre-commit run --all-files --hook-stage manual

# ---------------------------------------------------------------------------
# ci (composite target)
# ---------------------------------------------------------------------------

.PHONY: ci
ci: format-check lint typecheck test-all ## run full ci pipeline locally

# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

.PHONY: pipeline
pipeline: ## run end-to-end evaluation pipeline (quick mode)
	PYTHONPATH=$(SRC) $(PYTHON) -m scripts.run_pipeline --mode quick

.PHONY: pipeline-full
pipeline-full: ## run end-to-end evaluation pipeline (full mode)
	PYTHONPATH=$(SRC) $(PYTHON) -m scripts.run_pipeline --mode full

# ---------------------------------------------------------------------------
# paper
# ---------------------------------------------------------------------------

.PHONY: paper
paper: ## compile latex paper
	cd docs/paper && pdflatex -interaction=nonstopmode main.tex && \
		bibtex main && pdflatex -interaction=nonstopmode main.tex && \
		pdflatex -interaction=nonstopmode main.tex

.PHONY: figures
figures: ## generate all paper figures
	PYTHONPATH=$(SRC) $(PYTHON) -m scripts.generate_paper_results

.PHONY: tables
tables: ## generate all paper tables
	PYTHONPATH=$(SRC) $(PYTHON) -m scripts.generate_tables

# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ reports/coverage/
	rm -rf pipeline_output/

.PHONY: clean-all
clean-all: clean ## remove everything including model caches
	rm -rf wandb/ outputs/ logs/
