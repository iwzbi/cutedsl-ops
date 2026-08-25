.PHONY: help install sync style style-unsafe quality run-gemm run-flash run-moe test

PYTHON := .venv/bin/python
OPS := gemm flash_attn megamoe

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install a CUDA torch build (edit the index URL for your CUDA version)
	uv pip install torch --index-url https://download.pytorch.org/whl/cu124

sync: ## Create venv + install dependencies (uv)
	uv sync

style: ## Auto-fix lint + format
	ruff check . --fix
	ruff format .

style-unsafe: ## Auto-fix with unsafe fixes
	ruff check . --fix --unsafe-fixes
	ruff format .

quality: ## Check lint + format without changing files
	ruff check .
	ruff format --check .

run-gemm: ## Run the GEMM operator harness
	$(PYTHON) ops/gemm/run_gemm.py

run-flash: ## Run the FlashAttention operator harness
	$(PYTHON) ops/flash_attn/run_flash_attn.py

run-moe: ## Run the MegaMoE operator harness
	$(PYTHON) ops/megamoe/run_megamoe.py

run-all: $(addprefix run-,$(OPS)) ## Run every operator harness

test: quality ## Lint gate used as the test target (kernels need a GPU)
