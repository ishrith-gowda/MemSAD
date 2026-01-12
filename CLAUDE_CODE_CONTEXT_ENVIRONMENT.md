# Claude Code Context: Environment Setup & Infrastructure
# BAIR Memory Agent Security Research - Dr. Xuandong Zhao's Group

---

## Document Metadata

```yaml
document_type: claude_code_context
project: memory-agent-security
module: environment_setup
version: 1.0.0
last_updated: 2026-01-10
research_group: UC Berkeley AI Research (BAIR)
advisor: Dr. Xuandong Zhao
timeline: January 2026 - June 2026
commitment: 15-20 hours/week
```

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Requirements](#system-requirements)
3. [Development Environment Setup](#development-environment-setup)
4. [Python Environment Management](#python-environment-management)
5. [Core Dependencies Installation](#core-dependencies-installation)
6. [Memory System Dependencies](#memory-system-dependencies)
7. [Attack Framework Dependencies](#attack-framework-dependencies)
8. [Watermarking Framework Dependencies](#watermarking-framework-dependencies)
9. [Database and Storage Setup](#database-and-storage-setup)
10. [API Configuration](#api-configuration)
11. [Berkeley Savio HPC Setup](#berkeley-savio-hpc-setup)
12. [Cloud Infrastructure Setup](#cloud-infrastructure-setup)
13. [Docker Configuration](#docker-configuration)
14. [Git and Version Control](#git-and-version-control)
15. [Experiment Tracking Setup](#experiment-tracking-setup)
16. [Testing Infrastructure](#testing-infrastructure)
17. [Continuous Integration](#continuous-integration)
18. [Security and Secrets Management](#security-and-secrets-management)
19. [Troubleshooting Guide](#troubleshooting-guide)
20. [Quick Reference Commands](#quick-reference-commands)

---

## Executive Summary

This document provides exhaustive guidance for setting up the complete development environment for the Memory Agent Security research project. The project investigates adversarial robustness of memory-augmented LLM agents (Mem0, A-MEM, MemGPT/Letta) and develops a provenance-aware defense framework using Dr. Xuandong Zhao's watermarking techniques.

### Project Overview

**Research Objectives:**
- Characterize memory poisoning attacks on production memory systems
- Reproduce and extend AgentPoison, MINJA, and InjecMEM attack methodologies
- Develop watermarking-based provenance verification for memory entries
- Achieve publication at NeurIPS 2026 or ACM CCS 2026

**Technical Stack:**
- Python 3.10+ (primary language)
- PyTorch 2.0+ (deep learning framework)
- Vector databases (Qdrant, Chroma, FAISS)
- Graph databases (Neo4j for relationship modeling)
- LLM APIs (OpenAI GPT-4, Anthropic Claude)
- HuggingFace Transformers
- Weights & Biases (experiment tracking)

### Timeline Context

| Phase | Weeks | Focus |
|-------|-------|-------|
| Foundation | 1-4 | Environment setup, baseline deployment |
| Attack Characterization | 5-10 | Reproduce attacks, run evaluations |
| Defense Development | 11-18 | Implement watermarking defenses |
| Analysis & Writing | 19-24 | Paper preparation |
| Buffer & Submission | 25-26 | Final revisions, submission |

---

## System Requirements

### Hardware Requirements

#### Local Development Machine

```yaml
minimum_requirements:
  cpu: 8 cores (Intel i7/AMD Ryzen 7 or better)
  ram: 32GB DDR4
  storage: 500GB SSD (NVMe preferred)
  gpu: NVIDIA RTX 3080 10GB (for local model inference)
  network: Stable internet for API calls

recommended_requirements:
  cpu: 16 cores (Intel i9/AMD Ryzen 9)
  ram: 64GB DDR4/DDR5
  storage: 1TB NVMe SSD + 2TB HDD for datasets
  gpu: NVIDIA RTX 4090 24GB or A100 40GB
  network: High-speed fiber connection
```

#### GPU Memory Requirements by Model

| Model | Minimum VRAM | Recommended VRAM | Quantization |
|-------|--------------|------------------|--------------|
| Llama-2-7B | 14GB | 16GB | FP16 |
| Llama-2-7B-4bit | 6GB | 8GB | GPTQ/AWQ |
| Llama-2-13B | 26GB | 32GB | FP16 |
| Llama-2-13B-4bit | 10GB | 12GB | GPTQ/AWQ |
| Llama-2-70B | 140GB | 160GB | FP16 |
| Llama-2-70B-4bit | 40GB | 48GB | GPTQ/AWQ |
| Llama-3-8B | 16GB | 20GB | FP16 |
| Llama-3-70B | 140GB | 160GB | FP16 |
| Mistral-7B | 14GB | 16GB | FP16 |
| GPT-4 API | N/A | N/A | API-based |
| Claude-3.5 API | N/A | N/A | API-based |

### Operating System Compatibility

```yaml
fully_supported:
  - Ubuntu 22.04 LTS (primary development)
  - Ubuntu 24.04 LTS
  - Debian 12 (Bookworm)
  - Rocky Linux 9 (Savio HPC)
  - macOS 13+ (Apple Silicon with limitations)

partially_supported:
  - Windows 11 with WSL2
  - Fedora 38+
  - Arch Linux (rolling)

not_recommended:
  - Windows native (compatibility issues)
  - macOS Intel (deprecated GPU support)
  - Ubuntu versions < 22.04
```

### Network Requirements

```yaml
api_endpoints_required:
  - api.openai.com (port 443)
  - api.anthropic.com (port 443)
  - huggingface.co (port 443)
  - api.wandb.ai (port 443)
  - pypi.org (port 443)
  - github.com (port 443)

optional_endpoints:
  - cloud.qdrant.io (if using Qdrant Cloud)
  - aura.neo4j.com (if using Neo4j Aura)
  - api.pinecone.io (if using Pinecone)

bandwidth_recommendations:
  minimum: 50 Mbps download, 10 Mbps upload
  recommended: 100+ Mbps symmetric
  note: "Large model downloads can be 10-50GB each"
```

---

## Development Environment Setup

### Directory Structure Creation

Execute the following to create the complete project structure:

```bash
#!/bin/bash
# Project structure initialization script
# Run from desired project location

PROJECT_NAME="memory-agent-security"
PROJECT_ROOT="${HOME}/research/${PROJECT_NAME}"

# Create root directory
mkdir -p "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

# Create main directory structure
mkdir -p {src,tests,notebooks,scripts,configs,data,models,reports,docs}

# Source code subdirectories
mkdir -p src/{attacks,defenses,memory_systems,evaluation,utils,watermark}
mkdir -p src/attacks/{agentpoison,minja,injecmem,custom,base}
mkdir -p src/defenses/{watermark,detection,filtering,base}
mkdir -p src/memory_systems/{mem0,amem,memgpt,wrappers,base}
mkdir -p src/evaluation/{benchmarks,metrics,runners,base}
mkdir -p src/watermark/{unigram,pf_decoder,detection,base}
mkdir -p src/utils/{logging,config,io,visualization}

# Data subdirectories
mkdir -p data/{raw,processed,external,cache,embeddings}
mkdir -p data/raw/{longmemeval,locomo,custom}
mkdir -p data/processed/{attacks,defenses,benchmarks}
mkdir -p data/external/{datasets,models,checkpoints}

# Model storage
mkdir -p models/{pretrained,finetuned,watermarked,checkpoints}
mkdir -p models/pretrained/{llama,mistral,embeddings}
mkdir -p models/checkpoints/{attacks,defenses}

# Configuration files
mkdir -p configs/{attacks,defenses,experiments,models,infrastructure}

# Reports and outputs
mkdir -p reports/{figures,tables,logs,wandb}
mkdir -p reports/figures/{attacks,defenses,benchmarks}

# Documentation
mkdir -p docs/{api,guides,papers,notes}

# Test directories
mkdir -p tests/{unit,integration,e2e,fixtures}
mkdir -p tests/unit/{attacks,defenses,memory_systems,watermark}

# Notebook organization
mkdir -p notebooks/{exploration,analysis,visualization,experiments}

# Scripts
mkdir -p scripts/{setup,experiments,evaluation,preprocessing}

# Create placeholder __init__.py files
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;

echo "Project structure created at ${PROJECT_ROOT}"
tree -L 3 "${PROJECT_ROOT}"
```

### Essential Configuration Files

#### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "memory-agent-security"
version = "0.1.0"
description = "Adversarial Robustness of Memory-Augmented LLM Agents"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Research Team", email = "research@berkeley.edu"}
]
keywords = [
    "llm",
    "memory",
    "security",
    "adversarial",
    "watermarking",
    "agents"
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    # Core ML
    "torch>=2.0.0",
    "transformers>=4.36.0",
    "accelerate>=0.25.0",
    "bitsandbytes>=0.41.0",
    "safetensors>=0.4.0",
    
    # Memory Systems
    "mem0ai>=0.1.0",
    "chromadb>=0.4.0",
    "qdrant-client>=1.7.0",
    "faiss-cpu>=1.7.4",
    "neo4j>=5.0.0",
    
    # LLM APIs
    "openai>=1.0.0",
    "anthropic>=0.18.0",
    "tiktoken>=0.5.0",
    
    # Embeddings
    "sentence-transformers>=2.2.0",
    
    # Experiment Tracking
    "wandb>=0.16.0",
    "mlflow>=2.9.0",
    
    # Data Processing
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    
    # Configuration
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.0",
    
    # Utilities
    "tqdm>=4.66.0",
    "rich>=13.0.0",
    "typer>=0.9.0",
    "loguru>=0.7.0",
    
    # Visualization
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
    
    # Testing
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "hypothesis>=6.92.0",
]

[project.optional-dependencies]
dev = [
    "black>=23.12.0",
    "isort>=5.13.0",
    "ruff>=0.1.8",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
    "ipython>=8.18.0",
    "jupyter>=1.0.0",
    "jupyterlab>=4.0.0",
]
gpu = [
    "torch>=2.0.0+cu118",
    "faiss-gpu>=1.7.4",
    "flash-attn>=2.3.0",
    "vllm>=0.2.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.24.0",
]

[project.urls]
Repository = "https://github.com/username/memory-agent-security"
Documentation = "https://memory-agent-security.readthedocs.io"

[project.scripts]
mas-attack = "src.attacks.cli:main"
mas-defend = "src.defenses.cli:main"
mas-eval = "src.evaluation.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.setuptools.package-data]
"*" = ["*.yaml", "*.yml", "*.json"]

[tool.black]
line-length = 100
target-version = ['py310', 'py311']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
  | data
  | models
)/
'''

[tool.isort]
profile = "black"
line_length = 100
known_first_party = ["src"]
skip = [".venv", "data", "models"]

[tool.ruff]
line-length = 100
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # Pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]
exclude = [
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".pytype",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "venv",
    "data",
    "models",
]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
exclude = [
    "data",
    "models",
    "notebooks",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "-ra",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
    "api: marks tests that require API calls",
    "integration: marks integration tests",
]
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::UserWarning",
]

[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

#### Makefile

```makefile
# Makefile for Memory Agent Security Research Project
# Usage: make <target>

.PHONY: help install install-dev install-gpu setup clean test lint format
.PHONY: attack defend eval notebook docker savio

# Default shell
SHELL := /bin/bash

# Python interpreter
PYTHON := python3
PIP := pip3

# Virtual environment
VENV := .venv
VENV_BIN := $(VENV)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

# Project directories
SRC_DIR := src
TEST_DIR := tests
DATA_DIR := data
CONFIG_DIR := configs
NOTEBOOK_DIR := notebooks
SCRIPTS_DIR := scripts

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

help: ## Show this help message
	@echo "$(BLUE)Memory Agent Security Research - Makefile Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------

venv: ## Create virtual environment
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo "$(GREEN)Virtual environment created at $(VENV)$(NC)"

install: venv ## Install base dependencies
	@echo "$(GREEN)Installing base dependencies...$(NC)"
	$(VENV_PIP) install -e .
	@echo "$(GREEN)Base installation complete$(NC)"

install-dev: venv ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(VENV_PIP) install -e ".[dev]"
	$(VENV_BIN)/pre-commit install
	@echo "$(GREEN)Development installation complete$(NC)"

install-gpu: venv ## Install GPU dependencies (requires CUDA)
	@echo "$(GREEN)Installing GPU dependencies...$(NC)"
	$(VENV_PIP) install -e ".[gpu]"
	@echo "$(GREEN)GPU installation complete$(NC)"

install-all: venv ## Install all dependencies
	@echo "$(GREEN)Installing all dependencies...$(NC)"
	$(VENV_PIP) install -e ".[dev,gpu,docs]"
	$(VENV_BIN)/pre-commit install
	@echo "$(GREEN)Full installation complete$(NC)"

setup: install-dev ## Full setup including data download
	@echo "$(GREEN)Running full setup...$(NC)"
	$(MAKE) setup-dirs
	$(MAKE) setup-configs
	$(MAKE) download-data
	@echo "$(GREEN)Setup complete$(NC)"

setup-dirs: ## Create necessary directories
	@echo "$(GREEN)Creating directory structure...$(NC)"
	mkdir -p $(DATA_DIR)/{raw,processed,external,cache,embeddings}
	mkdir -p models/{pretrained,finetuned,checkpoints}
	mkdir -p reports/{figures,tables,logs}
	mkdir -p .cache/{huggingface,wandb}

setup-configs: ## Initialize configuration files
	@echo "$(GREEN)Setting up configuration files...$(NC)"
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@if [ ! -f $(CONFIG_DIR)/local.yaml ]; then \
		cp $(CONFIG_DIR)/default.yaml $(CONFIG_DIR)/local.yaml; \
	fi

#------------------------------------------------------------------------------
# Data Management
#------------------------------------------------------------------------------

download-data: ## Download required datasets
	@echo "$(GREEN)Downloading datasets...$(NC)"
	$(VENV_PYTHON) $(SCRIPTS_DIR)/download_datasets.py
	@echo "$(GREEN)Dataset download complete$(NC)"

download-longmemeval: ## Download LongMemEval dataset
	@echo "$(GREEN)Downloading LongMemEval...$(NC)"
	$(VENV_PYTHON) -c "from huggingface_hub import snapshot_download; \
		snapshot_download('xiaowu0162/longmemeval-cleaned', \
		local_dir='$(DATA_DIR)/external/longmemeval')"

download-locomo: ## Download LoCoMo dataset
	@echo "$(GREEN)Downloading LoCoMo...$(NC)"
	@if [ ! -d "$(DATA_DIR)/external/locomo" ]; then \
		git clone https://github.com/snap-research/locomo.git \
			$(DATA_DIR)/external/locomo; \
	fi

download-models: ## Download pretrained models
	@echo "$(GREEN)Downloading pretrained models...$(NC)"
	$(VENV_PYTHON) $(SCRIPTS_DIR)/download_models.py

#------------------------------------------------------------------------------
# Attack Commands
#------------------------------------------------------------------------------

attack: ## Run attack experiments (use ATTACK=<name> CONFIG=<path>)
	@echo "$(GREEN)Running attack: $(ATTACK)$(NC)"
	$(VENV_PYTHON) -m src.attacks.cli run \
		--attack $(ATTACK) \
		--config $(CONFIG)

attack-agentpoison: ## Run AgentPoison attack
	@echo "$(GREEN)Running AgentPoison attack...$(NC)"
	$(VENV_PYTHON) -m src.attacks.agentpoison.run \
		--config $(CONFIG_DIR)/attacks/agentpoison.yaml

attack-minja: ## Run MINJA attack
	@echo "$(GREEN)Running MINJA attack...$(NC)"
	$(VENV_PYTHON) -m src.attacks.minja.run \
		--config $(CONFIG_DIR)/attacks/minja.yaml

attack-injecmem: ## Run InjecMEM attack
	@echo "$(GREEN)Running InjecMEM attack...$(NC)"
	$(VENV_PYTHON) -m src.attacks.injecmem.run \
		--config $(CONFIG_DIR)/attacks/injecmem.yaml

attack-all: ## Run all attack experiments
	@echo "$(GREEN)Running all attacks...$(NC)"
	$(MAKE) attack-agentpoison
	$(MAKE) attack-minja
	$(MAKE) attack-injecmem

#------------------------------------------------------------------------------
# Defense Commands
#------------------------------------------------------------------------------

defend: ## Run defense experiments (use DEFENSE=<name> CONFIG=<path>)
	@echo "$(GREEN)Running defense: $(DEFENSE)$(NC)"
	$(VENV_PYTHON) -m src.defenses.cli run \
		--defense $(DEFENSE) \
		--config $(CONFIG)

defend-watermark: ## Run watermarking defense
	@echo "$(GREEN)Running watermarking defense...$(NC)"
	$(VENV_PYTHON) -m src.defenses.watermark.run \
		--config $(CONFIG_DIR)/defenses/watermark.yaml

defend-detection: ## Run anomaly detection defense
	@echo "$(GREEN)Running detection defense...$(NC)"
	$(VENV_PYTHON) -m src.defenses.detection.run \
		--config $(CONFIG_DIR)/defenses/detection.yaml

defend-all: ## Run all defense experiments
	@echo "$(GREEN)Running all defenses...$(NC)"
	$(MAKE) defend-watermark
	$(MAKE) defend-detection

#------------------------------------------------------------------------------
# Evaluation Commands
#------------------------------------------------------------------------------

eval: ## Run evaluation (use BENCHMARK=<name> CONFIG=<path>)
	@echo "$(GREEN)Running evaluation: $(BENCHMARK)$(NC)"
	$(VENV_PYTHON) -m src.evaluation.cli run \
		--benchmark $(BENCHMARK) \
		--config $(CONFIG)

eval-longmemeval: ## Run LongMemEval benchmark
	@echo "$(GREEN)Running LongMemEval benchmark...$(NC)"
	$(VENV_PYTHON) -m src.evaluation.benchmarks.longmemeval \
		--config $(CONFIG_DIR)/evaluation/longmemeval.yaml

eval-locomo: ## Run LoCoMo benchmark
	@echo "$(GREEN)Running LoCoMo benchmark...$(NC)"
	$(VENV_PYTHON) -m src.evaluation.benchmarks.locomo \
		--config $(CONFIG_DIR)/evaluation/locomo.yaml

eval-asb: ## Run Agent Security Bench
	@echo "$(GREEN)Running ASB benchmark...$(NC)"
	$(VENV_PYTHON) -m src.evaluation.benchmarks.asb \
		--config $(CONFIG_DIR)/evaluation/asb.yaml

eval-all: ## Run all benchmarks
	@echo "$(GREEN)Running all benchmarks...$(NC)"
	$(MAKE) eval-longmemeval
	$(MAKE) eval-locomo
	$(MAKE) eval-asb

#------------------------------------------------------------------------------
# Memory System Commands
#------------------------------------------------------------------------------

mem0-start: ## Start Mem0 with local vector store
	@echo "$(GREEN)Starting Mem0...$(NC)"
	$(VENV_PYTHON) -m src.memory_systems.mem0.start \
		--config $(CONFIG_DIR)/memory/mem0.yaml

memgpt-start: ## Start MemGPT/Letta agent
	@echo "$(GREEN)Starting MemGPT...$(NC)"
	$(VENV_PYTHON) -m src.memory_systems.memgpt.start \
		--config $(CONFIG_DIR)/memory/memgpt.yaml

amem-start: ## Start A-MEM system
	@echo "$(GREEN)Starting A-MEM...$(NC)"
	$(VENV_PYTHON) -m src.memory_systems.amem.start \
		--config $(CONFIG_DIR)/memory/amem.yaml

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(TEST_DIR) -v

test-unit: ## Run unit tests
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(TEST_DIR)/unit -v

test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(TEST_DIR)/integration -v -m integration

test-e2e: ## Run end-to-end tests
	@echo "$(GREEN)Running e2e tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(TEST_DIR)/e2e -v

test-cov: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(VENV_PYTHON) -m pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html

test-fast: ## Run fast tests only (no slow/GPU/API)
	@echo "$(GREEN)Running fast tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(TEST_DIR) -v -m "not slow and not gpu and not api"

#------------------------------------------------------------------------------
# Code Quality
#------------------------------------------------------------------------------

lint: ## Run linters
	@echo "$(GREEN)Running linters...$(NC)"
	$(VENV_BIN)/ruff check $(SRC_DIR) $(TEST_DIR)
	$(VENV_BIN)/mypy $(SRC_DIR)

lint-fix: ## Fix linting issues automatically
	@echo "$(GREEN)Fixing linting issues...$(NC)"
	$(VENV_BIN)/ruff check $(SRC_DIR) $(TEST_DIR) --fix

format: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	$(VENV_BIN)/black $(SRC_DIR) $(TEST_DIR)
	$(VENV_BIN)/isort $(SRC_DIR) $(TEST_DIR)

format-check: ## Check code formatting
	@echo "$(GREEN)Checking code formatting...$(NC)"
	$(VENV_BIN)/black --check $(SRC_DIR) $(TEST_DIR)
	$(VENV_BIN)/isort --check-only $(SRC_DIR) $(TEST_DIR)

pre-commit: ## Run pre-commit hooks
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	$(VENV_BIN)/pre-commit run --all-files

#------------------------------------------------------------------------------
# Documentation
#------------------------------------------------------------------------------

docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	$(VENV_BIN)/mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation...$(NC)"
	$(VENV_BIN)/mkdocs serve

#------------------------------------------------------------------------------
# Notebooks
#------------------------------------------------------------------------------

notebook: ## Start Jupyter notebook server
	@echo "$(GREEN)Starting Jupyter notebook...$(NC)"
	$(VENV_BIN)/jupyter notebook --notebook-dir=$(NOTEBOOK_DIR)

lab: ## Start JupyterLab
	@echo "$(GREEN)Starting JupyterLab...$(NC)"
	$(VENV_BIN)/jupyter lab --notebook-dir=$(NOTEBOOK_DIR)

#------------------------------------------------------------------------------
# Docker
#------------------------------------------------------------------------------

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t memory-agent-security:latest .

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run -it --gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/configs:/app/configs \
		--env-file .env \
		memory-agent-security:latest

docker-shell: ## Open shell in Docker container
	docker run -it --gpus all \
		-v $(PWD):/app \
		--env-file .env \
		memory-agent-security:latest /bin/bash

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

#------------------------------------------------------------------------------
# Savio HPC
#------------------------------------------------------------------------------

savio-upload: ## Upload code to Savio
	@echo "$(GREEN)Uploading to Savio...$(NC)"
	rsync -avz --exclude '.venv' --exclude 'data' --exclude 'models' \
		--exclude '.git' --exclude '__pycache__' \
		. $(SAVIO_USER)@dtn.brc.berkeley.edu:~/memory-agent-security/

savio-download-results: ## Download results from Savio
	@echo "$(GREEN)Downloading results from Savio...$(NC)"
	rsync -avz $(SAVIO_USER)@dtn.brc.berkeley.edu:~/memory-agent-security/reports/ \
		./reports/savio/

savio-submit: ## Submit job to Savio (use JOB=<job_script>)
	@echo "$(GREEN)Submitting job to Savio...$(NC)"
	ssh $(SAVIO_USER)@hpc.brc.berkeley.edu "cd memory-agent-security && sbatch $(JOB)"

savio-status: ## Check Savio job status
	@echo "$(GREEN)Checking Savio job status...$(NC)"
	ssh $(SAVIO_USER)@hpc.brc.berkeley.edu "squeue -u $(SAVIO_USER)"

#------------------------------------------------------------------------------
# Weights & Biases
#------------------------------------------------------------------------------

wandb-login: ## Login to Weights & Biases
	@echo "$(GREEN)Logging in to W&B...$(NC)"
	$(VENV_BIN)/wandb login

wandb-sync: ## Sync offline runs to W&B
	@echo "$(GREEN)Syncing W&B runs...$(NC)"
	$(VENV_BIN)/wandb sync reports/wandb/

#------------------------------------------------------------------------------
# Cleaning
#------------------------------------------------------------------------------

clean: ## Clean build artifacts
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete

clean-all: clean ## Clean everything including venv and data
	@echo "$(RED)Cleaning everything...$(NC)"
	rm -rf $(VENV)
	rm -rf data/cache/
	rm -rf data/processed/
	rm -rf .cache/

clean-models: ## Clean downloaded models
	@echo "$(YELLOW)Cleaning models...$(NC)"
	rm -rf models/pretrained/
	rm -rf models/checkpoints/

#------------------------------------------------------------------------------
# Utility
#------------------------------------------------------------------------------

tree: ## Show project tree
	@tree -I '__pycache__|*.pyc|.git|.venv|node_modules|data|models' -L 3

size: ## Show directory sizes
	@du -sh */ | sort -h

gpu-info: ## Show GPU information
	@nvidia-smi

check-env: ## Check environment setup
	@echo "$(GREEN)Checking environment...$(NC)"
	@echo "Python: $$($(VENV_PYTHON) --version)"
	@echo "Pip: $$($(VENV_PIP) --version)"
	@echo "PyTorch: $$($(VENV_PYTHON) -c 'import torch; print(torch.__version__)')"
	@echo "CUDA Available: $$($(VENV_PYTHON) -c 'import torch; print(torch.cuda.is_available())')"
	@echo "Transformers: $$($(VENV_PYTHON) -c 'import transformers; print(transformers.__version__)')"
```

---

## Python Environment Management

### Conda Environment Setup

For users preferring Conda (especially on Savio HPC):

```yaml
# environment.yml
name: memory-agent-security
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  # Python version
  - python=3.10
  
  # Core ML packages
  - pytorch=2.1.0
  - pytorch-cuda=11.8
  - torchvision
  - torchaudio
  
  # Scientific computing
  - numpy>=1.24
  - scipy>=1.11
  - pandas>=2.0
  - scikit-learn>=1.3
  
  # Visualization
  - matplotlib>=3.8
  - seaborn>=0.13
  
  # Database drivers
  - psycopg2
  - redis-py
  
  # Development
  - jupyter
  - jupyterlab
  - ipython
  - black
  - isort
  
  # pip-only packages
  - pip:
    - transformers>=4.36.0
    - accelerate>=0.25.0
    - bitsandbytes>=0.41.0
    - safetensors>=0.4.0
    - mem0ai>=0.1.0
    - chromadb>=0.4.0
    - qdrant-client>=1.7.0
    - faiss-cpu>=1.7.4
    - neo4j>=5.0.0
    - openai>=1.0.0
    - anthropic>=0.18.0
    - tiktoken>=0.5.0
    - sentence-transformers>=2.2.0
    - wandb>=0.16.0
    - hydra-core>=1.3.0
    - omegaconf>=2.3.0
    - python-dotenv>=1.0.0
    - tqdm>=4.66.0
    - rich>=13.0.0
    - typer>=0.9.0
    - loguru>=0.7.0
    - pytest>=7.4.0
    - pytest-asyncio>=0.21.0
    - ruff>=0.1.8
    - mypy>=1.8.0
    - pre-commit>=3.6.0
```

#### Conda Environment Commands

```bash
#!/bin/bash
# Conda environment management commands

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate memory-agent-security

# Update environment
conda env update -f environment.yml --prune

# Export environment (for reproducibility)
conda env export --no-builds > environment.lock.yml

# Remove environment
conda env remove -n memory-agent-security

# List all environments
conda env list

# Clone environment for experiments
conda create --name mas-experiment --clone memory-agent-security
```

### Virtual Environment with venv

```bash
#!/bin/bash
# Virtual environment setup script

# Set Python version (requires Python 3.10+)
PYTHON_VERSION="python3.10"

# Create virtual environment
${PYTHON_VERSION} -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install project in editable mode with all dependencies
pip install -e ".[dev,gpu,docs]"

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import mem0; print(f'Mem0 version: {mem0.__version__}')"

# Install pre-commit hooks
pre-commit install

echo "Environment setup complete!"
```

### Poetry Configuration (Alternative)

```toml
# pyproject.toml (Poetry version)
[tool.poetry]
name = "memory-agent-security"
version = "0.1.0"
description = "Adversarial Robustness of Memory-Augmented LLM Agents"
authors = ["Research Team <research@berkeley.edu>"]
license = "MIT"
readme = "README.md"
packages = [{include = "src"}]

[tool.poetry.dependencies]
python = "^3.10"
torch = "^2.1.0"
transformers = "^4.36.0"
accelerate = "^0.25.0"
bitsandbytes = "^0.41.0"
mem0ai = "^0.1.0"
chromadb = "^0.4.0"
qdrant-client = "^1.7.0"
faiss-cpu = "^1.7.4"
neo4j = "^5.0.0"
openai = "^1.0.0"
anthropic = "^0.18.0"
wandb = "^0.16.0"
hydra-core = "^1.3.0"
numpy = "^1.24.0"
pandas = "^2.0.0"
scipy = "^1.11.0"
scikit-learn = "^1.3.0"
tqdm = "^4.66.0"
rich = "^13.0.0"
loguru = "^0.7.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"
black = "^23.12.0"
isort = "^5.13.0"
ruff = "^0.1.8"
mypy = "^1.8.0"
pre-commit = "^3.6.0"
jupyter = "^1.0.0"

[tool.poetry.group.gpu.dependencies]
flash-attn = "^2.3.0"
vllm = "^0.2.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

---

## Core Dependencies Installation

### PyTorch Installation

```bash
#!/bin/bash
# PyTorch installation script with CUDA support

# Check NVIDIA driver version
nvidia-smi

# Determine CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "Detected CUDA version: ${CUDA_VERSION}"

# Install PyTorch based on CUDA version
if [[ "${CUDA_VERSION}" == "12."* ]]; then
    # CUDA 12.x
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "${CUDA_VERSION}" == "11.8"* ]]; then
    # CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "${CUDA_VERSION}" == "11.7"* ]]; then
    # CUDA 11.7
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
else
    # CPU only fallback
    echo "Warning: Using CPU-only PyTorch"
    pip install torch torchvision torchaudio
fi

# Verify installation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### HuggingFace Transformers Setup

```bash
#!/bin/bash
# HuggingFace ecosystem installation

# Install transformers and related packages
pip install transformers>=4.36.0
pip install accelerate>=0.25.0
pip install bitsandbytes>=0.41.0
pip install safetensors>=0.4.0
pip install datasets>=2.15.0
pip install evaluate>=0.4.0
pip install peft>=0.7.0

# Install sentence-transformers for embeddings
pip install sentence-transformers>=2.2.0

# Configure HuggingFace cache directory
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}"

# Login to HuggingFace (required for gated models like Llama)
# huggingface-cli login

# Verify installation
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

print('Transformers installation verified')
print('SentenceTransformers installation verified')
"
```

### Quantization Libraries

```bash
#!/bin/bash
# Quantization libraries for efficient inference

# BitsAndBytes (4-bit/8-bit quantization)
pip install bitsandbytes>=0.41.0

# AutoGPTQ for GPTQ quantization
pip install auto-gptq>=0.7.0 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# AutoAWQ for AWQ quantization
pip install autoawq>=0.1.8

# GGUF/llama.cpp support
pip install llama-cpp-python>=0.2.0

# vLLM for high-throughput inference
pip install vllm>=0.2.0

# Verify quantization support
python -c "
import bitsandbytes as bnb
print(f'BitsAndBytes version: {bnb.__version__}')

try:
    from auto_gptq import AutoGPTQForCausalLM
    print('AutoGPTQ available')
except ImportError:
    print('AutoGPTQ not available')

try:
    from awq import AutoAWQForCausalLM
    print('AutoAWQ available')
except ImportError:
    print('AutoAWQ not available')
"
```

---

## Memory System Dependencies

### Mem0 Installation and Configuration

```bash
#!/bin/bash
# Mem0 memory system installation

# Install Mem0
pip install mem0ai>=0.1.0

# Install optional vector store backends
pip install qdrant-client>=1.7.0    # Qdrant (recommended)
pip install chromadb>=0.4.0          # ChromaDB
pip install faiss-cpu>=1.7.4         # FAISS
pip install pinecone-client>=2.2.0   # Pinecone
pip install weaviate-client>=4.0.0   # Weaviate
pip install pgvector>=0.2.0          # PostgreSQL with pgvector

# Install optional graph store backends
pip install neo4j>=5.0.0             # Neo4j
pip install memgraph>=0.1.0          # Memgraph

# Install embedding providers
pip install openai>=1.0.0            # OpenAI embeddings
pip install cohere>=4.0.0            # Cohere embeddings
pip install voyageai>=0.1.0          # Voyage AI embeddings

# Verify Mem0 installation
python -c "
from mem0 import Memory
print('Mem0 installation verified')
print(f'Available vector stores: qdrant, chroma, faiss, pinecone, weaviate, pgvector')
print(f'Available graph stores: neo4j, memgraph')
"
```

#### Mem0 Configuration Template

```python
# configs/memory/mem0_config.py
"""
Mem0 configuration templates for different deployment scenarios.
"""

# Development configuration (local, lightweight)
DEV_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memory_agent_dev",
            "path": "./data/cache/chroma",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "embedding_dims": 1536,
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2000,
        },
    },
    "version": "v1.1",
}

# Production configuration (Qdrant + Neo4j)
PROD_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "memory_agent_prod",
            "embedding_model_dims": 1536,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "embedding_dims": 1536,
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 4000,
        },
    },
    "version": "v1.1",
}

# Research configuration (local embeddings, minimal API calls)
RESEARCH_CONFIG = {
    "vector_store": {
        "provider": "faiss",
        "config": {
            "collection_name": "memory_agent_research",
            "path": "./data/cache/faiss",
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dims": 384,
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 2000,
        },
    },
    "version": "v1.1",
}

# High-security configuration (for watermarking experiments)
WATERMARK_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "memory_agent_watermark",
            "embedding_model_dims": 1536,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "embedding_dims": 1536,
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.0,  # Deterministic for reproducibility
            "max_tokens": 4000,
        },
    },
    "custom_prompts": {
        "extract_memory": """
        Extract key information from the conversation.
        For each memory, include a provenance tag: [PROVENANCE:{source}:{timestamp}]
        """,
    },
    "version": "v1.1",
}
```

### A-MEM Installation

```bash
#!/bin/bash
# A-MEM (Agentic Memory) installation

# Clone the repository
git clone https://github.com/agiresearch/A-mem.git
cd A-mem

# Install dependencies
pip install -r requirements.txt

# Install as package
pip install -e .

# Verify installation
python -c "
from agentic_memory.memory_system import AgenticMemorySystem
print('A-MEM installation verified')
"

cd ..
```

#### A-MEM Configuration

```python
# configs/memory/amem_config.py
"""
A-MEM configuration for Zettelkasten-style memory system.
"""

AMEM_CONFIG = {
    # Embedding model
    "model_name": "all-MiniLM-L6-v2",
    
    # LLM backend
    "llm_backend": "openai",
    "llm_model": "gpt-4o-mini",
    
    # Memory parameters
    "similarity_threshold": 0.7,
    "max_links": 5,
    "evolution_enabled": True,
    
    # Storage
    "chroma_persist_directory": "./data/cache/amem_chroma",
    
    # Keyword extraction
    "max_keywords": 10,
    "keyword_model": "yake",
}

AMEM_RESEARCH_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "llm_backend": "openai",
    "llm_model": "gpt-4o",
    "similarity_threshold": 0.6,
    "max_links": 10,
    "evolution_enabled": True,
    "chroma_persist_directory": "./data/cache/amem_research",
    "max_keywords": 15,
}
```

### MemGPT/Letta Installation

```bash
#!/bin/bash
# MemGPT/Letta installation

# Install from PyPI
pip install letta>=0.5.0

# Or install from source for latest features
git clone https://github.com/letta-ai/letta.git
cd letta
pip install -e .
cd ..

# Install CLI tools
pip install letta-cli

# Initialize Letta
letta quickstart

# Verify installation
python -c "
from letta_client import Letta
print('Letta/MemGPT installation verified')
"
```

#### MemGPT/Letta Configuration

```python
# configs/memory/memgpt_config.py
"""
MemGPT/Letta configuration for hierarchical memory management.
"""

import os

LETTA_CONFIG = {
    # API configuration
    "api_key": os.getenv("LETTA_API_KEY"),
    "base_url": os.getenv("LETTA_BASE_URL", "http://localhost:8283"),
    
    # Agent configuration
    "agent": {
        "model": "openai/gpt-4.1",
        "embedding_model": "openai/text-embedding-3-small",
        "context_window": 128000,
    },
    
    # Memory blocks
    "memory_blocks": [
        {
            "label": "human",
            "value": "Name: User\nPreferences: None specified",
            "limit": 2000,
        },
        {
            "label": "persona",
            "value": "I am a helpful AI assistant with long-term memory capabilities.",
            "limit": 2000,
        },
    ],
    
    # Archival memory settings
    "archival": {
        "embedding_dim": 1536,
        "chunk_size": 1000,
        "overlap": 200,
    },
}

LETTA_RESEARCH_CONFIG = {
    "api_key": os.getenv("LETTA_API_KEY"),
    "base_url": "http://localhost:8283",
    "agent": {
        "model": "openai/gpt-4o",
        "embedding_model": "openai/text-embedding-3-small",
        "context_window": 128000,
    },
    "memory_blocks": [
        {
            "label": "human",
            "value": "",
            "limit": 5000,
        },
        {
            "label": "persona",
            "value": "Research assistant for memory security experiments.",
            "limit": 2000,
        },
        {
            "label": "experiment",
            "value": "",
            "limit": 3000,
        },
    ],
}
```

---

## Attack Framework Dependencies

### AgentPoison Installation

```bash
#!/bin/bash
# AgentPoison attack framework installation

# Clone repository
git clone https://github.com/AI-secure/AgentPoison.git
cd AgentPoison

# Create conda environment
conda env create -f environment.yml
conda activate agentpoison

# Or install with pip
pip install -r requirements.txt

# Download required models
python -c "
from transformers import DPRContextEncoder, DPRQuestionEncoder
DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
print('DPR models downloaded')
"

# Verify installation
python -c "
import algo.trigger_optimization as trigger
print('AgentPoison installation verified')
"

cd ..
```

#### AgentPoison Requirements

```txt
# AgentPoison requirements.txt
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
numpy>=1.24.0
scipy>=1.11.0
tqdm>=4.65.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
wandb>=0.15.0
hydra-core>=1.3.0
omegaconf>=2.3.0
openai>=1.0.0
langchain>=0.1.0
chromadb>=0.4.0
tiktoken>=0.5.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
einops>=0.7.0
xformers>=0.0.23
```

### Agent Security Bench (ASB) Installation

```bash
#!/bin/bash
# Agent Security Bench installation

# Clone repository
git clone https://github.com/agiresearch/ASB.git
cd ASB

# Install dependencies
pip install -r requirements.txt

# Download benchmark data
python scripts/download_data.py

# Verify installation
python -c "
import scripts.agent_attack as attack
print('ASB installation verified')
"

cd ..
```

### MINJA Framework Setup

```bash
#!/bin/bash
# MINJA attack framework setup
# Note: Official code may not be released yet as of January 2026

# Create placeholder structure
mkdir -p src/attacks/minja
cat > src/attacks/minja/__init__.py << 'EOF'
"""
MINJA: Memory Injection Attacks via Query-Only Interaction
Paper: arXiv:2503.03704 (NeurIPS 2025)

This module implements the MINJA attack methodology:
1. Bridging steps generation
2. Indication prompt crafting
3. Progressive shortening strategy
4. Query-only attack execution
"""

from .attack import MINJAAttack
from .bridging import BridgingStepGenerator
from .indication import IndicationPromptCrafter
from .shortening import ProgressiveShorteningStrategy

__all__ = [
    "MINJAAttack",
    "BridgingStepGenerator", 
    "IndicationPromptCrafter",
    "ProgressiveShorteningStrategy",
]
EOF

# Create base attack class
cat > src/attacks/minja/attack.py << 'EOF'
"""
MINJA Attack Implementation
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MINJAConfig:
    """Configuration for MINJA attack."""
    target_action: str
    victim_query: str
    max_shortening_iterations: int = 10
    bridging_model: str = "gpt-4o"
    similarity_threshold: float = 0.7


class MINJAAttack:
    """
    MINJA: Query-only memory injection attack.
    
    Key insight: Any regular user can inject malicious memories
    through carefully crafted queries that cause the agent to
    generate and store attacker-controlled reasoning.
    """
    
    def __init__(self, config: MINJAConfig):
        self.config = config
        self.bridging_generator = None
        self.indication_crafter = None
        self.shortening_strategy = None
        
    def execute(self, memory_system: Any) -> Dict[str, Any]:
        """Execute the MINJA attack."""
        raise NotImplementedError("MINJA attack implementation pending")
        
    def generate_bridging_steps(self, victim_query: str) -> List[str]:
        """Generate bridging steps linking victim query to malicious reasoning."""
        raise NotImplementedError()
        
    def craft_indication_prompt(self, bridging_steps: List[str]) -> str:
        """Craft indication prompt to guide autonomous bridging."""
        raise NotImplementedError()
        
    def apply_progressive_shortening(self, prompt: str) -> str:
        """Progressively shorten indication prompt."""
        raise NotImplementedError()
EOF

echo "MINJA framework structure created"
```

---

## Watermarking Framework Dependencies

### Unigram-Watermark Installation

```bash
#!/bin/bash
# Unigram-Watermark installation (Dr. Zhao's ICLR 2024 paper)

# Clone repository
git clone https://github.com/XuandongZhao/Unigram-Watermark.git
cd Unigram-Watermark

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "
from watermark import UnigramWatermark
print('Unigram-Watermark installation verified')
"

cd ..
```

### PF-Decoding Installation

```bash
#!/bin/bash
# Permute-and-Flip decoder installation (Dr. Zhao's ICLR 2025 paper)

# Clone repository
git clone https://github.com/XuandongZhao/pf-decoding.git
cd pf-decoding

# Install dependencies
pip install -r requirements.txt

# Verify installation
python run.py --help

cd ..
```

### MarkLLM Toolkit Installation

```bash
#!/bin/bash
# MarkLLM comprehensive watermarking toolkit

# Clone repository
git clone https://github.com/THU-BPM/MarkLLM.git
cd MarkLLM

# Install dependencies
pip install -r requirements.txt

# Install as package
pip install -e .

# Verify installation
python -c "
from watermark.kgw.kgw import KGW
from watermark.pf.pf import PF
print('MarkLLM installation verified')
"

cd ..
```

### Watermarking Framework Configuration

```python
# configs/watermark/watermark_config.py
"""
Watermarking configuration for memory provenance verification.
"""

# Unigram Watermark configuration
UNIGRAM_CONFIG = {
    "gamma": 0.25,  # Green list proportion (0.25-0.5)
    "delta": 2.0,   # Bias strength (1.0-3.0)
    "z_threshold": 4.0,  # Detection threshold
    "min_tokens": 50,    # Minimum tokens for reliable detection
    "secret_key": None,  # Will be generated if None
    "seeding_scheme": "simple_1",
}

# Permute-and-Flip configuration
PF_CONFIG = {
    "ngram": 8,
    "temperature": 0.9,
    "top_p": 1.0,
    "max_gen_len": 256,
    "secret_key": None,
}

# Multi-bit watermark for metadata encoding
MULTIBIT_CONFIG = {
    "bits_per_entry": 8,  # Bits for encoding metadata
    "encoding": {
        "source_bits": 4,     # Source identification
        "timestamp_bits": 4,  # Timestamp encoding
    },
    "redundancy": 3,  # Repetition for robustness
}

# Defense framework configuration
DEFENSE_CONFIG = {
    # Entry-level watermarking
    "entry_watermark": {
        "method": "unigram",
        "config": UNIGRAM_CONFIG,
    },
    
    # Quality-critical entries
    "quality_watermark": {
        "method": "pf",
        "config": PF_CONFIG,
    },
    
    # Detection parameters
    "detection": {
        "z_threshold": 4.0,
        "min_tokens": 100,
        "confidence_threshold": 0.95,
    },
    
    # Provenance tracking
    "provenance": {
        "track_source": True,
        "track_timestamp": True,
        "track_modifications": True,
    },
}
```

---

## Database and Storage Setup

### Qdrant Vector Database

```bash
#!/bin/bash
# Qdrant vector database setup

# Option 1: Docker (recommended)
docker pull qdrant/qdrant:latest
docker run -d --name qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -v $(pwd)/data/qdrant:/qdrant/storage \
    qdrant/qdrant:latest

# Option 2: Binary installation
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz | tar xz
./qdrant --config-path config/config.yaml

# Verify connection
python -c "
from qdrant_client import QdrantClient
client = QdrantClient('localhost', port=6333)
print(f'Qdrant collections: {client.get_collections()}')
"
```

#### Qdrant Configuration

```yaml
# config/qdrant/config.yaml
storage:
  storage_path: ./data/qdrant/storage
  snapshots_path: ./data/qdrant/snapshots
  
service:
  http_port: 6333
  grpc_port: 6334
  enable_tls: false
  
cluster:
  enabled: false
  
telemetry_disabled: true

log_level: INFO
```

### ChromaDB Setup

```bash
#!/bin/bash
# ChromaDB setup

# Install ChromaDB
pip install chromadb>=0.4.0

# Create persistent directory
mkdir -p data/cache/chromadb

# Verify installation
python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/cache/chromadb')
print(f'ChromaDB initialized with {client.list_collections()} collections')
"
```

### Neo4j Graph Database

```bash
#!/bin/bash
# Neo4j graph database setup

# Option 1: Docker
docker pull neo4j:5.15-community
docker run -d --name neo4j \
    -p 7474:7474 \
    -p 7687:7687 \
    -v $(pwd)/data/neo4j/data:/data \
    -v $(pwd)/data/neo4j/logs:/logs \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:5.15-community

# Option 2: Desktop installation
# Download from https://neo4j.com/download/

# Verify connection
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
with driver.session() as session:
    result = session.run('RETURN 1 as n')
    print(f'Neo4j connection verified: {result.single()[\"n\"]}')
driver.close()
"
```

### Redis Cache (Optional)

```bash
#!/bin/bash
# Redis setup for caching

# Docker
docker run -d --name redis \
    -p 6379:6379 \
    -v $(pwd)/data/redis:/data \
    redis:7-alpine redis-server --appendonly yes

# Verify connection
python -c "
import redis
r = redis.Redis(host='localhost', port=6379)
print(f'Redis ping: {r.ping()}')
"
```

### Docker Compose for All Databases

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: mas-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant:/qdrant/storage
    environment:
      - QDRANT__LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  neo4j:
    image: neo4j:5.15-community
    container_name: mas-neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - ./data/neo4j/data:/data
      - ./data/neo4j/logs:/logs
      - ./data/neo4j/plugins:/plugins
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: mas-redis
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Ollama for local LLM inference
  ollama:
    image: ollama/ollama:latest
    container_name: mas-ollama
    ports:
      - "11434:11434"
    volumes:
      - ./data/ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  qdrant_data:
  neo4j_data:
  redis_data:
  ollama_data:

networks:
  default:
    name: mas-network
```

---

## API Configuration

### Environment Variables

```bash
# .env.example
# Copy to .env and fill in your values

# =============================================================================
# LLM API Keys
# =============================================================================
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
COHERE_API_KEY=your-cohere-api-key-here
VOYAGE_API_KEY=your-voyage-api-key-here

# =============================================================================
# Memory System API Keys
# =============================================================================
LETTA_API_KEY=your-letta-api-key-here
LETTA_BASE_URL=http://localhost:8283

# =============================================================================
# Vector Database Configuration
# =============================================================================
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=  # Leave empty for local deployment

PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-west1-gcp

WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=

# =============================================================================
# Graph Database Configuration
# =============================================================================
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# =============================================================================
# Experiment Tracking
# =============================================================================
WANDB_API_KEY=your-wandb-api-key-here
WANDB_PROJECT=memory-agent-security
WANDB_ENTITY=your-username

MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=memory-agent-security

# =============================================================================
# HuggingFace Configuration
# =============================================================================
HF_TOKEN=hf_your-huggingface-token-here
HF_HOME=/path/to/.cache/huggingface
TRANSFORMERS_CACHE=/path/to/.cache/huggingface/hub

# =============================================================================
# Compute Configuration
# =============================================================================
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# =============================================================================
# Project Configuration
# =============================================================================
PROJECT_ROOT=/path/to/memory-agent-security
DATA_DIR=${PROJECT_ROOT}/data
MODEL_DIR=${PROJECT_ROOT}/models
CONFIG_DIR=${PROJECT_ROOT}/configs
LOG_LEVEL=INFO

# =============================================================================
# Savio HPC Configuration (Berkeley)
# =============================================================================
SAVIO_USER=your-savio-username
SAVIO_ACCOUNT=fc_your-account
SCRATCH_DIR=/global/scratch/users/${SAVIO_USER}
```

### API Client Configuration

```python
# src/utils/api_clients.py
"""
Centralized API client configuration.
"""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@lru_cache()
def get_openai_client():
    """Get OpenAI API client."""
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@lru_cache()
def get_anthropic_client():
    """Get Anthropic API client."""
    from anthropic import Anthropic
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@lru_cache()
def get_qdrant_client():
    """Get Qdrant vector database client."""
    from qdrant_client import QdrantClient
    
    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        return QdrantClient(
            url=f"https://{os.getenv('QDRANT_HOST')}:{os.getenv('QDRANT_PORT')}",
            api_key=api_key,
        )
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )


@lru_cache()
def get_neo4j_driver():
    """Get Neo4j graph database driver."""
    from neo4j import GraphDatabase
    
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password"),
        ),
    )


def get_mem0_client(config: Optional[dict] = None):
    """Get Mem0 memory client with specified configuration."""
    from mem0 import Memory
    
    if config is None:
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", 6333)),
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                },
            },
        }
    
    return Memory(config)


class APIRateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 90000,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._request_times = []
        self._token_counts = []
    
    def wait_if_needed(self, estimated_tokens: int = 0):
        """Wait if rate limit would be exceeded."""
        import time
        from collections import deque
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        self._request_times = [t for t in self._request_times if t > minute_ago]
        self._token_counts = [
            (t, c) for t, c in self._token_counts if t > minute_ago
        ]
        
        # Check request limit
        if len(self._request_times) >= self.requests_per_minute:
            sleep_time = self._request_times[0] - minute_ago
            time.sleep(sleep_time)
        
        # Check token limit
        total_tokens = sum(c for _, c in self._token_counts)
        if total_tokens + estimated_tokens > self.tokens_per_minute:
            sleep_time = self._token_counts[0][0] - minute_ago
            time.sleep(sleep_time)
        
        # Record this request
        self._request_times.append(current_time)
        if estimated_tokens > 0:
            self._token_counts.append((current_time, estimated_tokens))
```

---

## Berkeley Savio HPC Setup

### Savio Account Setup

```bash
#!/bin/bash
# Savio HPC initial setup

# SSH to Savio
ssh ${SAVIO_USER}@hpc.brc.berkeley.edu

# Check account allocations
sacctmgr show associations user=$USER

# Check available partitions
sinfo -s

# Create project directory structure
cd /global/scratch/users/${USER}
mkdir -p memory-agent-security/{data,models,results,logs}

# Set up environment module system
module avail python
module avail cuda
module avail pytorch

# Load required modules
module load python/3.10
module load cuda/11.8
module load pytorch/2.0.1

# Verify GPU access
srun --partition=savio3_gpu --gres=gpu:A40:1 --time=00:05:00 \
    nvidia-smi
```

### Savio Job Scripts

#### Interactive GPU Session

```bash
#!/bin/bash
# scripts/savio/interactive_gpu.sh

# Request interactive GPU session
srun \
    --partition=savio3_gpu \
    --account=fc_${SAVIO_ACCOUNT} \
    --gres=gpu:A40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=04:00:00 \
    --pty bash
```

#### Batch Job Template

```bash
#!/bin/bash
# scripts/savio/batch_template.sh
#SBATCH --job-name=mas-experiment
#SBATCH --partition=savio3_gpu
#SBATCH --account=fc_${SAVIO_ACCOUNT}
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@berkeley.edu

# Load modules
module load python/3.10
module load cuda/11.8
module load pytorch/2.0.1

# Activate conda environment
source ~/.bashrc
conda activate memory-agent-security

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline  # Sync later

# Run experiment
cd /global/scratch/users/${USER}/memory-agent-security
python -m src.attacks.agentpoison.run \
    --config configs/attacks/agentpoison.yaml \
    --output-dir results/agentpoison_${SLURM_JOB_ID}

# Sync W&B results
wandb sync results/agentpoison_${SLURM_JOB_ID}/wandb/
```

#### Multi-GPU Job

```bash
#!/bin/bash
# scripts/savio/multi_gpu.sh
#SBATCH --job-name=mas-multi-gpu
#SBATCH --partition=savio3_gpu
#SBATCH --account=fc_${SAVIO_ACCOUNT}
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

module load python/3.10
module load cuda/11.8
module load pytorch/2.0.1

source ~/.bashrc
conda activate memory-agent-security

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

cd /global/scratch/users/${USER}/memory-agent-security

# Distributed training with torchrun
torchrun \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/defenses/watermark/train.py \
    --config configs/defenses/watermark_train.yaml
```

#### Array Job for Experiments

```bash
#!/bin/bash
# scripts/savio/array_experiments.sh
#SBATCH --job-name=mas-sweep
#SBATCH --partition=savio3_gpu
#SBATCH --account=fc_${SAVIO_ACCOUNT}
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%5  # 10 jobs, max 5 concurrent
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

module load python/3.10
module load cuda/11.8

source ~/.bashrc
conda activate memory-agent-security

cd /global/scratch/users/${USER}/memory-agent-security

# Define experiment parameters
SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python -m src.evaluation.run_benchmark \
    --config configs/evaluation/sweep.yaml \
    --seed $SEED \
    --output-dir results/sweep_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
```

### Savio Data Transfer

```bash
#!/bin/bash
# scripts/savio/sync_data.sh

SAVIO_USER="your_username"
LOCAL_DIR="./data"
SAVIO_DIR="/global/scratch/users/${SAVIO_USER}/memory-agent-security/data"

# Upload data to Savio
upload_data() {
    rsync -avzP \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        ${LOCAL_DIR}/ \
        ${SAVIO_USER}@dtn.brc.berkeley.edu:${SAVIO_DIR}/
}

# Download results from Savio
download_results() {
    rsync -avzP \
        ${SAVIO_USER}@dtn.brc.berkeley.edu:/global/scratch/users/${SAVIO_USER}/memory-agent-security/results/ \
        ./results/savio/
}

# Check data transfer node quota
check_quota() {
    ssh ${SAVIO_USER}@hpc.brc.berkeley.edu "lfs quota -u ${SAVIO_USER} /global/scratch"
}

case "$1" in
    upload) upload_data ;;
    download) download_results ;;
    quota) check_quota ;;
    *) echo "Usage: $0 {upload|download|quota}" ;;
esac
```

---

## Cloud Infrastructure Setup

### RunPod Configuration

```python
# scripts/cloud/runpod_setup.py
"""
RunPod serverless GPU configuration.
"""

import runpod
import os

# Configure RunPod API
runpod.api_key = os.getenv("RUNPOD_API_KEY")

# Define pod template
POD_TEMPLATE = {
    "name": "memory-agent-security",
    "image_name": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    "gpu_type_id": "NVIDIA A100 40GB",  # or "NVIDIA RTX A4000" for cheaper
    "cloud_type": "SECURE",
    "volume_in_gb": 100,
    "container_disk_in_gb": 50,
    "min_vcpu_count": 8,
    "min_memory_in_gb": 32,
    "docker_args": "",
    "ports": "8888/http,6006/http",  # Jupyter, TensorBoard
    "volume_mount_path": "/workspace",
    "env": {
        "JUPYTER_PASSWORD": "research",
        "PUBLIC_KEY": os.getenv("SSH_PUBLIC_KEY", ""),
    },
}

def create_pod():
    """Create a new RunPod instance."""
    pod = runpod.create_pod(**POD_TEMPLATE)
    print(f"Pod created: {pod['id']}")
    return pod

def terminate_pod(pod_id: str):
    """Terminate a RunPod instance."""
    runpod.terminate_pod(pod_id)
    print(f"Pod {pod_id} terminated")
```

### Google Colab Pro Setup

```python
# notebooks/colab_setup.ipynb
# Run this cell first in Google Colab

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Check GPU
!nvidia-smi

# Clone repository
!git clone https://github.com/username/memory-agent-security.git
%cd memory-agent-security

# Install dependencies
!pip install -q -r requirements.txt

# Set environment variables
import os
os.environ['OPENAI_API_KEY'] = 'your-key-here'  # Use Colab secrets instead

# Enable Colab secrets (recommended)
from google.colab import userdata
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Lambda Labs Setup

```bash
#!/bin/bash
# Lambda Labs instance setup

# SSH to Lambda instance
ssh ubuntu@your-lambda-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install additional dependencies
sudo apt install -y git tmux htop nvtop

# Clone repository
git clone https://github.com/username/memory-agent-security.git
cd memory-agent-security

# Create conda environment
conda create -n mas python=3.10 -y
conda activate mas

# Install PyTorch with CUDA
pip install torch torchvision torchaudio

# Install project dependencies
pip install -e ".[dev,gpu]"

# Set up persistent storage
mkdir -p ~/data ~/models ~/results
ln -s ~/data ./data
ln -s ~/models ./models
ln -s ~/results ./results

# Configure tmux for persistent sessions
cat > ~/.tmux.conf << 'EOF'
set -g mouse on
set -g history-limit 50000
set -g default-terminal "screen-256color"
bind | split-window -h
bind - split-window -v
EOF

echo "Lambda Labs setup complete!"
```

---

## Docker Configuration

### Dockerfile

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    tmux \
    htop \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create non-root user
RUN useradd -m -s /bin/bash researcher
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Install PyTorch with CUDA
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy project files
COPY . .

# Install project
RUN pip install -e .

# Change ownership
RUN chown -R researcher:researcher /app

# Switch to non-root user
USER researcher

# Expose ports
EXPOSE 8888 6006 8000

# Default command
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
```

### Docker Compose Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: mas-dev
    volumes:
      - .:/app
      - ./data:/app/data
      - ./models:/app/models
      - ~/.cache/huggingface:/home/researcher/.cache/huggingface
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - WANDB_API_KEY=${WANDB_API_KEY}
      - HF_TOKEN=${HF_TOKEN}
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard
      - "8000:8000"  # FastAPI
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    depends_on:
      - qdrant
      - neo4j
      - redis
    networks:
      - mas-network
    command: >
      bash -c "
        pip install -e . &&
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
      "

  qdrant:
    image: qdrant/qdrant:latest
    container_name: mas-qdrant
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
    networks:
      - mas-network

  neo4j:
    image: neo4j:5.15-community
    container_name: mas-neo4j
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"
    networks:
      - mas-network

  redis:
    image: redis:7-alpine
    container_name: mas-redis
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - mas-network

volumes:
  qdrant_data:
  neo4j_data:
  redis_data:

networks:
  mas-network:
    driver: bridge
```

---

## Git and Version Control

### .gitignore

```gitignore
# .gitignore

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# Environments
.env
.env.local
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Ruff
.ruff_cache/

# Data directories (large files)
data/raw/
data/processed/
data/external/
data/cache/
data/embeddings/

# Model directories (very large)
models/pretrained/
models/finetuned/
models/checkpoints/

# Results (generate fresh)
results/
reports/logs/
reports/wandb/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Secrets (NEVER commit)
*.pem
*.key
secrets/
.secrets/

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp

# Large binary files
*.bin
*.pt
*.pth
*.safetensors
*.gguf
*.ggml
*.h5
*.hdf5

# Compressed files (usually data)
*.zip
*.tar.gz
*.tar
*.7z
*.rar

# Database files
*.db
*.sqlite
*.sqlite3
```

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: check-case-conflict
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: mixed-line-ending
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: ["--fix", "--exit-non-zero-on-fix"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-requests
          - types-PyYAML
        args: ["--ignore-missing-imports"]

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest tests/unit -v --tb=short -q
        language: system
        pass_filenames: false
        always_run: true
        stages: [push]
```

---

## Experiment Tracking Setup

### Weights & Biases Configuration

```python
# src/utils/wandb_config.py
"""
Weights & Biases experiment tracking configuration.
"""

import os
from typing import Dict, Any, Optional
import wandb


def init_wandb(
    project: str = "memory-agent-security",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    mode: str = "online",
) -> wandb.run:
    """
    Initialize Weights & Biases run.
    
    Args:
        project: W&B project name
        name: Run name (auto-generated if None)
        config: Hyperparameters and configuration
        tags: List of tags for filtering
        group: Group name for related runs
        job_type: Type of job (e.g., "attack", "defense", "eval")
        mode: "online", "offline", or "disabled"
    
    Returns:
        W&B run object
    """
    run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        group=group,
        job_type=job_type,
        mode=mode,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
        ),
    )
    
    # Log system info
    if config:
        wandb.config.update(config, allow_val_change=True)
    
    return run


def log_attack_results(
    results: Dict[str, Any],
    attack_name: str,
    step: Optional[int] = None,
):
    """Log attack experiment results."""
    metrics = {
        f"attack/{attack_name}/ASR_r": results.get("asr_r"),
        f"attack/{attack_name}/ASR_a": results.get("asr_a"),
        f"attack/{attack_name}/ASR_t": results.get("asr_t"),
        f"attack/{attack_name}/ISR": results.get("isr"),
        f"attack/{attack_name}/benign_ACC": results.get("benign_acc"),
        f"attack/{attack_name}/poison_rate": results.get("poison_rate"),
    }
    wandb.log({k: v for k, v in metrics.items() if v is not None}, step=step)


def log_defense_results(
    results: Dict[str, Any],
    defense_name: str,
    step: Optional[int] = None,
):
    """Log defense experiment results."""
    metrics = {
        f"defense/{defense_name}/TPR": results.get("tpr"),
        f"defense/{defense_name}/FPR": results.get("fpr"),
        f"defense/{defense_name}/DACC": results.get("dacc"),
        f"defense/{defense_name}/ASR_d": results.get("asr_d"),
        f"defense/{defense_name}/NRP": results.get("nrp"),
        f"defense/{defense_name}/z_score": results.get("z_score"),
    }
    wandb.log({k: v for k, v in metrics.items() if v is not None}, step=step)


def log_benchmark_results(
    results: Dict[str, Any],
    benchmark_name: str,
    step: Optional[int] = None,
):
    """Log benchmark evaluation results."""
    metrics = {
        f"benchmark/{benchmark_name}/accuracy": results.get("accuracy"),
        f"benchmark/{benchmark_name}/f1": results.get("f1"),
        f"benchmark/{benchmark_name}/rouge_l": results.get("rouge_l"),
        f"benchmark/{benchmark_name}/llm_judge": results.get("llm_judge"),
    }
    wandb.log({k: v for k, v in metrics.items() if v is not None}, step=step)


# W&B Sweep configuration template
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "defense/watermark/TPR", "goal": "maximize"},
    "parameters": {
        "gamma": {"min": 0.15, "max": 0.5},
        "delta": {"min": 1.0, "max": 3.0},
        "z_threshold": {"min": 3.0, "max": 5.0},
        "min_tokens": {"values": [50, 75, 100, 150]},
    },
}
```

### MLflow Configuration (Alternative)

```python
# src/utils/mlflow_config.py
"""
MLflow experiment tracking configuration.
"""

import os
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient


def init_mlflow(
    experiment_name: str = "memory-agent-security",
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> str:
    """
    Initialize MLflow tracking.
    
    Returns:
        Experiment ID
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        )
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_params(params: Dict[str, Any]):
    """Log parameters to MLflow."""
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics to MLflow."""
    for key, value in metrics.items():
        if value is not None:
            mlflow.log_metric(key, value, step=step)


def log_artifacts(artifact_paths: list):
    """Log artifact files to MLflow."""
    for path in artifact_paths:
        mlflow.log_artifact(path)
```

---

## Testing Infrastructure

### Pytest Configuration

```python
# tests/conftest.py
"""
Pytest fixtures and configuration.
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import torch


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_conversation():
    """Sample conversation for testing."""
    return [
        {"role": "user", "content": "My name is Alice and I work at TechCorp."},
        {"role": "assistant", "content": "Nice to meet you, Alice! How can I help?"},
        {"role": "user", "content": "I'm working on a machine learning project."},
    ]


@pytest.fixture
def sample_memories():
    """Sample memories for testing."""
    return [
        {
            "id": "mem_001",
            "content": "User's name is Alice",
            "user_id": "user_123",
            "metadata": {"source": "conversation", "timestamp": "2026-01-01"},
        },
        {
            "id": "mem_002",
            "content": "User works at TechCorp",
            "user_id": "user_123",
            "metadata": {"source": "conversation", "timestamp": "2026-01-01"},
        },
        {
            "id": "mem_003",
            "content": "User is working on ML project",
            "user_id": "user_123",
            "metadata": {"source": "conversation", "timestamp": "2026-01-01"},
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(10, 384).astype(np.float32)


@pytest.fixture
def sample_attack_config():
    """Sample attack configuration."""
    return {
        "attack_type": "agentpoison",
        "target_action": "reveal_secret",
        "poison_rate": 0.001,
        "num_triggers": 10,
        "trigger_length": 5,
    }


@pytest.fixture
def sample_defense_config():
    """Sample defense configuration."""
    return {
        "defense_type": "watermark",
        "method": "unigram",
        "gamma": 0.25,
        "delta": 2.0,
        "z_threshold": 4.0,
        "min_tokens": 50,
    }


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test response"))]
    )
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=np.random.randn(1536).tolist())]
    )
    return client


@pytest.fixture
def mock_mem0_client():
    """Mock Mem0 client."""
    client = MagicMock()
    client.add.return_value = {"id": "mem_test_001"}
    client.search.return_value = [
        {"id": "mem_001", "content": "Test memory", "score": 0.95}
    ]
    client.get_all.return_value = []
    return client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    from qdrant_client.models import ScoredPoint
    
    client = MagicMock()
    client.search.return_value = [
        ScoredPoint(id="1", score=0.95, payload={"content": "Test"}),
    ]
    return client


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Create temporary data directory structure."""
    (temp_dir / "raw").mkdir()
    (temp_dir / "processed").mkdir()
    (temp_dir / "cache").mkdir()
    return temp_dir


@pytest.fixture
def temp_model_dir(temp_dir: Path) -> Path:
    """Create temporary model directory structure."""
    (temp_dir / "checkpoints").mkdir()
    return temp_dir


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get available device (GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device for testing."""
    return torch.device("cpu")


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment without API keys."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# =============================================================================
# Skip Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "api: marks tests that require API calls")
    config.addinivalue_line("markers", "integration: marks integration tests")


# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available resources."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Skip API tests unless explicitly enabled
    if not os.getenv("RUN_API_TESTS"):
        skip_api = pytest.mark.skip(reason="API tests disabled")
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api)
```

### Sample Test Files

```python
# tests/unit/test_watermark.py
"""
Unit tests for watermarking functionality.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestUnigramWatermark:
    """Tests for Unigram watermark implementation."""
    
    def test_watermark_initialization(self, sample_defense_config):
        """Test watermark initialization with config."""
        # Test implementation
        pass
    
    def test_green_list_generation(self):
        """Test deterministic green list generation."""
        pass
    
    def test_watermark_embedding(self):
        """Test watermark embedding in text."""
        pass
    
    def test_watermark_detection(self):
        """Test watermark detection accuracy."""
        pass
    
    def test_z_score_calculation(self):
        """Test z-score calculation for detection."""
        pass
    
    @pytest.mark.parametrize("gamma", [0.15, 0.25, 0.5])
    def test_gamma_parameter_effect(self, gamma):
        """Test effect of gamma parameter on detection."""
        pass
    
    @pytest.mark.parametrize("delta", [1.0, 2.0, 3.0])
    def test_delta_parameter_effect(self, delta):
        """Test effect of delta parameter on detection."""
        pass


class TestPFDecoder:
    """Tests for Permute-and-Flip decoder."""
    
    def test_pf_initialization(self):
        """Test PF decoder initialization."""
        pass
    
    def test_distortion_free_property(self):
        """Test that PF decoder is distortion-free."""
        pass
    
    def test_watermark_detection(self):
        """Test PF watermark detection."""
        pass


# tests/unit/test_attacks.py
"""
Unit tests for attack implementations.
"""

import pytest


class TestAgentPoison:
    """Tests for AgentPoison attack."""
    
    def test_trigger_optimization(self, sample_attack_config):
        """Test trigger optimization algorithm."""
        pass
    
    def test_embedding_space_mapping(self):
        """Test mapping to unique embedding region."""
        pass
    
    def test_poison_injection(self, mock_mem0_client):
        """Test poison injection into memory."""
        pass
    
    def test_asr_calculation(self):
        """Test attack success rate calculation."""
        pass


class TestMINJA:
    """Tests for MINJA query-only attack."""
    
    def test_bridging_step_generation(self):
        """Test bridging step generation."""
        pass
    
    def test_indication_prompt_crafting(self):
        """Test indication prompt creation."""
        pass
    
    def test_progressive_shortening(self):
        """Test progressive shortening strategy."""
        pass
    
    def test_query_only_constraint(self):
        """Test that attack only uses query interface."""
        pass
```

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  PYTHON_VERSION: "3.10"
  POETRY_VERSION: "1.7.1"

jobs:
  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install ruff black isort mypy
      
      - name: Run ruff
        run: ruff check src tests
      
      - name: Run black
        run: black --check src tests
      
      - name: Run isort
        run: isort --check-only src tests
      
      - name: Run mypy
        run: mypy src --ignore-missing-imports

  test:
    name: Test
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Cache pip
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests
        run: |
          pytest tests/unit -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  integration-test:
    name: Integration Test
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push'
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run integration tests
        env:
          QDRANT_HOST: localhost
          QDRANT_PORT: 6333
        run: |
          pytest tests/integration -v -m integration
```

---

## Security and Secrets Management

### Secrets Configuration

```python
# src/utils/secrets.py
"""
Secure secrets management.
"""

import os
from typing import Optional
from functools import lru_cache


class SecretsManager:
    """Manage API keys and secrets securely."""
    
    REQUIRED_SECRETS = [
        "OPENAI_API_KEY",
    ]
    
    OPTIONAL_SECRETS = [
        "ANTHROPIC_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN",
        "QDRANT_API_KEY",
        "NEO4J_PASSWORD",
    ]
    
    @classmethod
    def validate_required(cls) -> bool:
        """Validate that all required secrets are set."""
        missing = []
        for secret in cls.REQUIRED_SECRETS:
            if not os.getenv(secret):
                missing.append(secret)
        
        if missing:
            raise EnvironmentError(
                f"Missing required secrets: {', '.join(missing)}\n"
                f"Please set them in your .env file or environment."
            )
        return True
    
    @classmethod
    @lru_cache()
    def get(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value."""
        return os.getenv(key, default)
    
    @classmethod
    def mask(cls, value: str, visible_chars: int = 4) -> str:
        """Mask a secret value for logging."""
        if len(value) <= visible_chars * 2:
            return "*" * len(value)
        return value[:visible_chars] + "*" * (len(value) - visible_chars * 2) + value[-visible_chars:]


def validate_secrets():
    """Validate secrets at startup."""
    return SecretsManager.validate_required()
```

---

## Troubleshooting Guide

### Common Issues and Solutions

```markdown
# Troubleshooting Guide

## CUDA / GPU Issues

### Issue: CUDA out of memory
```bash
# Solution 1: Reduce batch size
export BATCH_SIZE=8  # Default might be 32

# Solution 2: Enable gradient checkpointing
# In code: model.gradient_checkpointing_enable()

# Solution 3: Use mixed precision
# In code: with torch.cuda.amp.autocast():

# Solution 4: Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Issue: CUDA version mismatch
```bash
# Check versions
nvidia-smi  # Driver CUDA version
nvcc --version  # Toolkit CUDA version
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA

# Solution: Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## API Issues

### Issue: OpenAI rate limit exceeded
```python
# Solution: Implement exponential backoff
import time
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def call_openai_with_retry(prompt):
    return client.chat.completions.create(...)
```

### Issue: API key not found
```bash
# Check if .env is loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"

# Verify .env file location
ls -la .env

# Check for typos in variable names
grep OPENAI .env
```

## Database Issues

### Issue: Qdrant connection refused
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Start Qdrant
docker start qdrant
# or
docker-compose up -d qdrant

# Check logs
docker logs qdrant
```

### Issue: Neo4j authentication failed
```bash
# Reset Neo4j password
docker exec -it neo4j neo4j-admin set-initial-password newpassword

# Update .env
NEO4J_PASSWORD=newpassword
```

## Memory System Issues

### Issue: Mem0 extraction fails
```python
# Debug extraction
from mem0 import Memory
import logging
logging.basicConfig(level=logging.DEBUG)

memory = Memory(config)
result = memory.add(messages, user_id="test")
print(result)  # Check for errors
```

### Issue: ChromaDB persistence error
```bash
# Reset ChromaDB
rm -rf data/cache/chromadb
mkdir -p data/cache/chromadb

# Recreate client
python -c "import chromadb; c = chromadb.PersistentClient('./data/cache/chromadb'); print(c.list_collections())"
```

## HPC (Savio) Issues

### Issue: Job stuck in queue
```bash
# Check queue status
squeue -u $USER

# Check partition availability
sinfo -p savio3_gpu

# Try different partition
sbatch --partition=savio2_gpu job.sh
```

### Issue: Module not found on Savio
```bash
# Search for module
module spider python

# Load specific version
module load python/3.10

# Add to ~/.bashrc for persistence
echo "module load python/3.10" >> ~/.bashrc
```
```

---

## Quick Reference Commands

```bash
# =============================================================================
# Quick Reference Commands
# =============================================================================

# --- Environment Setup ---
make install-dev          # Install all dev dependencies
make setup               # Full project setup
source .venv/bin/activate # Activate virtual environment

# --- Running Experiments ---
make attack-agentpoison  # Run AgentPoison attack
make defend-watermark    # Run watermark defense
make eval-longmemeval    # Run LongMemEval benchmark

# --- Testing ---
make test               # Run all tests
make test-fast          # Run fast tests only (no GPU/API)
make test-cov           # Run tests with coverage

# --- Code Quality ---
make lint               # Run all linters
make format             # Format code
make pre-commit         # Run pre-commit hooks

# --- Database Services ---
docker-compose up -d    # Start all databases
docker-compose down     # Stop all databases
docker-compose logs -f  # View logs

# --- Savio HPC ---
make savio-upload       # Upload code to Savio
make savio-submit JOB=scripts/savio/batch.sh  # Submit job
make savio-status       # Check job status

# --- Experiment Tracking ---
make wandb-login        # Login to W&B
wandb sync reports/wandb/  # Sync offline runs

# --- Utilities ---
make tree               # Show project structure
make gpu-info           # Show GPU information
make check-env          # Verify environment setup
make clean              # Clean build artifacts
make clean-all          # Clean everything including venv
```

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-10 | Initial comprehensive environment setup document |

---

## Contact and Support

**Research Group:** UC Berkeley AI Research (BAIR)  
**Advisor:** Dr. Xuandong Zhao  
**Project:** Memory Agent Security  

For technical issues with this setup guide, please file an issue in the project repository.

---

*This document is part of the Memory Agent Security research project. For the complete research context, see the main research documentation.*