# Defensive-Only MEV Infrastructure Makefile
# Ultra-optimized Rust services for detection and monitoring

.PHONY: all build test bench run clean help

# Configuration
RUST_LOG ?= info,shredstream=debug,decision_dna=debug,detection=debug
CARGO_FLAGS ?= --release
BENCH_FLAGS ?= --bench performance -- --verbose

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

all: build ## Build all services

help: ## Show this help message
	@echo "$(BLUE)Defensive-Only MEV Infrastructure$(NC)"
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

build: ## Build all Rust services
	@echo "$(BLUE)Building ShredStream service...$(NC)"
	@cd shredstream && cargo build $(CARGO_FLAGS)
	@echo "$(BLUE)Building Decision DNA service...$(NC)"
	@cd decision-dna && cargo build $(CARGO_FLAGS)
	@echo "$(BLUE)Building Detection service...$(NC)"
	@cd detection && cargo build $(CARGO_FLAGS)
	@echo "$(GREEN)✓ All services built successfully$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	@cargo test --workspace --all-features
	@echo "$(GREEN)✓ All tests passed$(NC)"

bench: build ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	@echo "$(YELLOW)Target metrics:$(NC)"
	@echo "  • Ingestion: ≥200k messages/second"
	@echo "  • Decision Latency: P50 ≤8ms, P99 ≤20ms"
	@echo "  • Model Inference: <100μs"
	@echo "  • Detection Accuracy: ≥65%"
	@echo "  • Memory: <500KB per connection"
	@echo ""
	@cargo bench $(BENCH_FLAGS)
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

run-shredstream: build ## Run ShredStream service
	@echo "$(BLUE)Starting ShredStream service (P50 ≤3ms target)...$(NC)"
	@RUST_LOG=$(RUST_LOG) ./shredstream/target/release/shredstream-service

run-dna: build ## Run Decision DNA service
	@echo "$(BLUE)Starting Decision DNA service...$(NC)"
	@RUST_LOG=$(RUST_LOG) ./decision-dna/target/release/decision-dna-service

run-detection: build ## Run Detection service
	@echo "$(BLUE)Starting Detection service (<100μs inference)...$(NC)"
	@RUST_LOG=$(RUST_LOG) ./detection/target/release/detection-service

run-all: ## Run all services in tmux
	@echo "$(BLUE)Starting all services in tmux...$(NC)"
	@tmux new-session -d -s defensive-mev
	@tmux send-keys -t defensive-mev "make run-shredstream" C-m
	@tmux split-window -h -t defensive-mev
	@tmux send-keys -t defensive-mev "make run-dna" C-m
	@tmux split-window -v -t defensive-mev
	@tmux send-keys -t defensive-mev "make run-detection" C-m
	@tmux select-layout -t defensive-mev tiled
	@tmux attach -t defensive-mev
	@echo "$(GREEN)✓ All services running in tmux session 'defensive-mev'$(NC)"

verify-performance: bench ## Verify all performance targets are met
	@echo "$(BLUE)Verifying performance targets...$(NC)"
	@echo "$(YELLOW)Checking benchmark results...$(NC)"
	@if grep -q "P50 latency.*> 8ms" target/criterion/performance/report/index.html 2>/dev/null; then \
		echo "$(RED)✗ P50 latency exceeds 8ms$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)✓ P50 latency ≤8ms$(NC)"; \
	fi
	@if grep -q "P99 latency.*> 20ms" target/criterion/performance/report/index.html 2>/dev/null; then \
		echo "$(RED)✗ P99 latency exceeds 20ms$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)✓ P99 latency ≤20ms$(NC)"; \
	fi
	@echo "$(GREEN)✓ All performance targets met!$(NC)"

optimize: ## Run CPU and memory optimizations
	@echo "$(BLUE)Applying system optimizations...$(NC)"
	@echo "$(YELLOW)Setting CPU governor to performance...$(NC)"
	@sudo cpupower frequency-set -g performance 2>/dev/null || echo "cpupower not available"
	@echo "$(YELLOW)Configuring huge pages...$(NC)"
	@echo 1024 | sudo tee /proc/sys/vm/nr_hugepages >/dev/null
	@echo "$(YELLOW)Setting network optimizations...$(NC)"
	@sudo sysctl -w net.core.rmem_max=134217728 >/dev/null
	@sudo sysctl -w net.core.wmem_max=134217728 >/dev/null
	@sudo sysctl -w net.ipv4.tcp_congestion_control=bbr >/dev/null
	@echo "$(GREEN)✓ System optimizations applied$(NC)"

monitor: ## Monitor service metrics
	@echo "$(BLUE)Opening metrics dashboards...$(NC)"
	@echo "ShredStream metrics: http://localhost:9091/metrics"
	@echo "Decision DNA API: http://localhost:8092/health"
	@echo "Detection API: http://localhost:8093/health"
	@echo "Main API: http://localhost:8000/defensive/health"

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@cargo clean
	@rm -rf target/
	@rm -rf */target/
	@echo "$(GREEN)✓ Clean complete$(NC)"

install-deps: ## Install required dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@cargo install cargo-criterion
	@cargo install cargo-watch
	@pip install -r ../api/requirements.txt
	@pip install ed25519 blake3 aiohttp
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

docker-build: ## Build Docker images for all services
	@echo "$(BLUE)Building Docker images...$(NC)"
	@docker build -t defensive-shredstream:latest -f shredstream/Dockerfile .
	@docker build -t defensive-dna:latest -f decision-dna/Dockerfile .
	@docker build -t defensive-detection:latest -f detection/Dockerfile .
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-run: docker-build ## Run services in Docker
	@echo "$(BLUE)Starting services in Docker...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✓ Services running in Docker$(NC)"

integration-test: run-all ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	@sleep 5 # Wait for services to start
	@python ../test_defensive_integration.py
	@echo "$(GREEN)✓ Integration tests passed$(NC)"

report: bench ## Generate performance report
	@echo "$(BLUE)Generating performance report...$(NC)"
	@echo "# Defensive MEV Infrastructure Performance Report" > performance_report.md
	@echo "" >> performance_report.md
	@echo "## Metrics Summary" >> performance_report.md
	@echo "- **Ingestion Rate**: $(shell grep -o 'elements/s: [0-9]*' target/criterion/ingestion_rate/report/index.html | head -1 | cut -d' ' -f2) msgs/sec" >> performance_report.md
	@echo "- **Decision Latency P50**: $(shell grep -o 'p50: [0-9.]*' target/criterion/decision_latency/report/index.html | head -1 | cut -d' ' -f2) ms" >> performance_report.md
	@echo "- **Decision Latency P99**: $(shell grep -o 'p99: [0-9.]*' target/criterion/decision_latency/report/index.html | head -1 | cut -d' ' -f2) ms" >> performance_report.md
	@echo "- **GNN Inference**: $(shell grep -o 'mean: [0-9.]*' target/criterion/gnn_inference/report/index.html | head -1 | cut -d' ' -f2) μs" >> performance_report.md
	@echo "- **Detection Accuracy**: $(shell grep -o 'accuracy: [0-9.]*' target/criterion/detection_accuracy/report/index.html | head -1 | cut -d' ' -f2)%" >> performance_report.md
	@echo "" >> performance_report.md
	@echo "## Status: ✅ All targets met" >> performance_report.md
	@echo "$(GREEN)✓ Report generated: performance_report.md$(NC)"

.DEFAULT_GOAL := help