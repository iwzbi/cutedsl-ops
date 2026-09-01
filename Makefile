.PHONY: help sync style quality bench-gemm bench-gemm-ncu bench-gemm-ncu-raw bench-gemm-ncu-gui bench-flash bench-flash-prefill bench-flash-decode bench-flash-prefill-ncu bench-flash-decode-ncu bench-moe test

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

bench-gemm-ncu: ## Benchmark GEMM with ncu profiling (parsed report)
	$(PYTHON) ops/gemm/run_gemm.py 4096 4096 4096 --ncu

bench-gemm-ncu-raw: ## Benchmark GEMM with raw ncu output (full original report)
	$(PYTHON) ops/gemm/run_gemm.py 4096 4096 4096 --ncu-raw

bench-gemm-ncu-gui: ## Generate ncu .ncu-rep + serve for laptop download
	$(PYTHON) ops/gemm/run_gemm.py 4096 4096 4096 --ncu-gui
	@curl -s http://localhost:8899/ > /dev/null 2>&1 || (cd . && $(PYTHON) -m http.server 8899 --bind 0.0.0.0 > /dev/null 2>&1 &)
	@echo ""
	@echo "=== Download from your laptop ==="
	@echo "  curl -o ~/workspace/log/ncu/gemm_profile.ncu-rep http://11.167.35.90:8899/gemm_profile.ncu-rep"
	@echo "=== Or open in browser ==="
	@echo "  http://11.167.35.90:8899/gemm_profile.ncu-rep"

bench-flash: ## Benchmark FlashAttention (lesson 0 warm-up)
	$(PYTHON) ops/flash_attn/run_flash_attn.py

bench-flash-prefill: ## Benchmark FA prefill exercises (ex.1,2,4) — correctness + TFLOPS
	$(PYTHON) ops/flash_attn/run_prefill.py --bench

bench-flash-decode: ## Benchmark FA decode exercises (ex.3,5) — correctness + latency µs
	$(PYTHON) ops/flash_attn/run_decode.py --bench

bench-flash-prefill-ncu: ## ncu profiling of FA prefill (ex.1)
	$(PYTHON) ops/flash_attn/run_prefill.py --ex 1 --ncu

bench-flash-decode-ncu: ## ncu profiling of FA decode (ex.3)
	$(PYTHON) ops/flash_attn/run_decode.py --ex 3 --ncu

bench-moe: ## Benchmark MegaMoE
	$(PYTHON) ops/megamoe/run_megamoe.py

test: quality ## Lint gate (kernels need a GPU)
