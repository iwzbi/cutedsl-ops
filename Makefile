.PHONY: help sync style quality bench-gemm bench-gemm-ncu bench-flash bench-moe test

PYTHON := .venv/bin/python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

sync: ## Create venv + install dependencies (uv)
	uv sync

style: ## Auto-fix lint + format
	ruff check . --fix && ruff format .

quality: ## Check lint + format without changing files
	ruff check . && ruff format --check .

bench-gemm: ## Benchmark GEMM (4096³ fp16, timing + theoretical analysis)
	$(PYTHON) ops/gemm/run_gemm.py 4096 4096 4096

bench-gemm-ncu: ## Benchmark GEMM with ncu profiling (hardware-level analysis)
	$(PYTHON) ops/gemm/run_gemm.py 4096 4096 4096 --ncu

bench-flash: ## Benchmark FlashAttention
	$(PYTHON) ops/flash_attn/run_flash_attn.py

bench-moe: ## Benchmark MegaMoE
	$(PYTHON) ops/megamoe/run_megamoe.py

test: quality ## Lint gate (kernels need a GPU)
