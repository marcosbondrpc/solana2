# Elite MEV Backend Makefile
# Production-grade build system for Solana MEV infrastructure

SHELL := /bin/bash
.PHONY: all build release clean test bench docker help setup migrate

# Color codes for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Build profiles
PROFILE ?= release
FEATURES ?= 
CARGO_FLAGS := --workspace
DOCKER_REGISTRY ?= localhost:5000
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

# Rust build flags for maximum performance
export RUSTFLAGS := -C target-cpu=native -C link-arg=-fuse-ld=lld -C opt-level=3
export CARGO_BUILD_JOBS := $(shell nproc)
export CARGO_TARGET_DIR := target

# Default target
all: build

help: ## Show this help message
	@echo -e "$(GREEN)Elite MEV Backend Build System$(NC)"
	@echo -e "$(BLUE)Version: $(VERSION)$(NC)\n"
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Initial setup and dependency installation
	@echo -e "$(GREEN)Setting up MEV backend environment...$(NC)"
	@rustup update stable
	@rustup component add rustfmt clippy
	@cargo install cargo-watch cargo-nextest cargo-machete sccache
	@pip3 install -r services/control-plane/requirements.txt --user
	@npm install -g pnpm
	@echo -e "$(GREEN)Setup complete!$(NC)"

migrate: ## Migrate services from old structure to new
	@echo -e "$(YELLOW)Starting service migration...$(NC)"
	@bash tools/migrate-services.sh
	@echo -e "$(GREEN)Migration complete!$(NC)"

build: ## Build all services in debug mode
	@echo -e "$(GREEN)Building all services (debug)...$(NC)"
	@cargo build $(CARGO_FLAGS)
	@echo -e "$(GREEN)Build complete!$(NC)"

release: ## Build all services in release mode
	@echo -e "$(GREEN)Building all services (release)...$(NC)"
	@cargo build --release $(CARGO_FLAGS)
	@echo -e "$(GREEN)Release build complete!$(NC)"

mev-production: ## Build with MEV production profile (ultra-optimized)
	@echo -e "$(YELLOW)Building MEV production binaries...$(NC)"
	@RUSTFLAGS="$(RUSTFLAGS) -C lto=fat -C embed-bitcode=yes" \
		cargo build --profile mev-production $(CARGO_FLAGS)
	@echo -e "$(GREEN)MEV production build complete!$(NC)"

test: ## Run all tests
	@echo -e "$(BLUE)Running tests...$(NC)"
	@cargo nextest run --workspace
	@cd services/control-plane && python -m pytest tests/
	@echo -e "$(GREEN)All tests passed!$(NC)"

bench: ## Run benchmarks
	@echo -e "$(BLUE)Running benchmarks...$(NC)"
	@cargo bench --workspace -- --output-format bencher | tee target/bench.txt
	@echo -e "$(GREEN)Benchmarks complete! Results in target/bench.txt$(NC)"

lint: ## Run linters and formatters
	@echo -e "$(BLUE)Running linters...$(NC)"
	@cargo fmt --all -- --check
	@cargo clippy --workspace --all-targets -- -D warnings
	@cargo machete
	@echo -e "$(GREEN)Linting complete!$(NC)"

clean: ## Clean build artifacts
	@echo -e "$(YELLOW)Cleaning build artifacts...$(NC)"
	@cargo clean
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	@echo -e "$(GREEN)Clean complete!$(NC)"

# Docker targets
docker-build: ## Build Docker images for all services
	@echo -e "$(GREEN)Building Docker images...$(NC)"
	@docker compose -f infrastructure/docker/docker-compose.yml build
	@echo -e "$(GREEN)Docker build complete!$(NC)"

docker-up: ## Start all services with Docker Compose
	@echo -e "$(GREEN)Starting services...$(NC)"
	@docker compose -f infrastructure/docker/docker-compose.yml up -d
	@echo -e "$(GREEN)Services started!$(NC)"

docker-down: ## Stop all services
	@echo -e "$(YELLOW)Stopping services...$(NC)"
	@docker compose -f infrastructure/docker/docker-compose.yml down
	@echo -e "$(GREEN)Services stopped!$(NC)"

# Service-specific targets
mev-engine: ## Build MEV engine only
	@echo -e "$(GREEN)Building MEV engine...$(NC)"
	@cargo build --release -p mev-engine
	@echo -e "$(GREEN)MEV engine built!$(NC)"

sandwich-detector: ## Build sandwich detector only
	@echo -e "$(GREEN)Building sandwich detector...$(NC)"
	@cargo build --release -p sandwich-detector
	@echo -e "$(GREEN)Sandwich detector built!$(NC)"

arbitrage-engine: ## Build arbitrage engine only
	@echo -e "$(GREEN)Building arbitrage engine...$(NC)"
	@cargo build --release -p arbitrage-engine
	@echo -e "$(GREEN)Arbitrage engine built!$(NC)"

# Deployment targets
deploy-local: release ## Deploy to local environment
	@echo -e "$(GREEN)Deploying to local environment...$(NC)"
	@bash tools/deploy/deploy-local.sh
	@echo -e "$(GREEN)Local deployment complete!$(NC)"

deploy-staging: mev-production ## Deploy to staging
	@echo -e "$(YELLOW)Deploying to staging...$(NC)"
	@bash tools/deploy/deploy-staging.sh
	@echo -e "$(GREEN)Staging deployment complete!$(NC)"

deploy-production: mev-production ## Deploy to production
	@echo -e "$(RED)Deploying to PRODUCTION...$(NC)"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ] || exit 1
	@bash tools/deploy/deploy-production.sh
	@echo -e "$(GREEN)Production deployment complete!$(NC)"

# Monitoring and health checks
health-check: ## Check health of all services
	@echo -e "$(BLUE)Checking service health...$(NC)"
	@bash tools/monitoring/health-check.sh
	@echo -e "$(GREEN)Health check complete!$(NC)"

logs: ## Tail logs from all services
	@docker compose -f infrastructure/docker/docker-compose.yml logs -f

metrics: ## Show current metrics
	@echo -e "$(BLUE)Fetching metrics...$(NC)"
	@curl -s http://localhost:9090/metrics | grep -E "mev_|sandwich_|arbitrage_"

# Database operations
db-migrate: ## Run database migrations
	@echo -e "$(BLUE)Running database migrations...$(NC)"
	@bash infrastructure/clickhouse/migrate.sh
	@echo -e "$(GREEN)Migrations complete!$(NC)"

db-backup: ## Backup databases
	@echo -e "$(YELLOW)Creating database backup...$(NC)"
	@bash tools/backup/backup-databases.sh
	@echo -e "$(GREEN)Backup complete!$(NC)"

# Development helpers
dev: ## Start development environment
	@echo -e "$(GREEN)Starting development environment...$(NC)"
	@tmux new-session -s mev-dev -d
	@tmux send-keys -t mev-dev "cargo watch -x 'build --workspace'" C-m
	@tmux split-window -t mev-dev -h
	@tmux send-keys -t mev-dev "docker compose -f infrastructure/docker/docker-compose.dev.yml up" C-m
	@tmux attach -t mev-dev

watch: ## Watch for changes and rebuild
	@cargo watch -x 'build --workspace' -x 'test --workspace'

# Performance profiling
profile: ## Profile MEV engine performance
	@echo -e "$(YELLOW)Starting performance profiling...$(NC)"
	@cargo build --release --features profiling
	@perf record -F 99 -g target/release/mev-engine
	@perf report

flamegraph: ## Generate flamegraph for MEV engine
	@echo -e "$(YELLOW)Generating flamegraph...$(NC)"
	@cargo flamegraph --bin mev-engine -o flamegraph.svg
	@echo -e "$(GREEN)Flamegraph saved to flamegraph.svg$(NC)"

# Utilities
check-deps: ## Check for outdated dependencies
	@cargo outdated --workspace
	@cd services/control-plane && pip list --outdated

update-deps: ## Update all dependencies
	@cargo update --workspace
	@cd services/control-plane && pip install --upgrade -r requirements.txt

version: ## Show version information
	@echo -e "$(BLUE)MEV Backend Version Information$(NC)"
	@echo "Git Version: $(VERSION)"
	@echo "Rust Version: $$(rustc --version)"
	@echo "Cargo Version: $$(cargo --version)"
	@echo "Docker Version: $$(docker --version)"

.DEFAULT_GOAL := help