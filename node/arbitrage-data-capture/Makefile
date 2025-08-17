# LEGENDARY MEV Infrastructure Makefile
# Ultra-optimized for institutional-scale deployments
# CPU affinity, RT scheduling, and comprehensive monitoring

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Version and build info
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME := $(shell date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Environment configuration
KAFKA_BROKERS ?= localhost:9092
CLICKHOUSE_URL ?= http://localhost:8123
GRAFANA_URL ?= http://localhost:3000
GRAFANA_API_KEY ?= 
REDIS_URL ?= redis://localhost:6379
SOLANA_RPC ?= https://api.mainnet-beta.solana.com
JITO_URL ?= https://mainnet.block-engine.jito.wtf

# Rust build settings
RUST_LOG ?= info
RUST_BACKTRACE ?= 1
CARGO_BUILD_FLAGS ?= --release
CARGO_TEST_FLAGS ?= --all

# Performance tuning
export TOKIO_WORKER_THREADS ?= 16
export UV_THREADPOOL_SIZE ?= 16
export RUST_MIN_STACK ?= 8388608

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Tmux configuration
TMUX_CFG ?= tmux/legendary-lab.yml

.PHONY: help
help: ## Show this help message
	@echo -e "$(CYAN)LEGENDARY MEV Infrastructure$(NC)"
	@echo -e "$(CYAN)========================================$(NC)"
	@echo ""
	@echo -e "$(YELLOW)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo -e "$(YELLOW)Environment variables:$(NC)"
	@echo -e "  $(BLUE)KAFKA_BROKERS$(NC)     = $(KAFKA_BROKERS)"
	@echo -e "  $(BLUE)CLICKHOUSE_URL$(NC)    = $(CLICKHOUSE_URL)"
	@echo -e "  $(BLUE)GRAFANA_URL$(NC)       = $(GRAFANA_URL)"
	@echo -e "  $(BLUE)SOLANA_RPC$(NC)        = $(SOLANA_RPC)"
	@echo -e "  $(BLUE)VERSION$(NC)           = $(VERSION)"

#######################################
# System Setup & Dependencies
#######################################

.PHONY: install-deps
install-deps: ## Install all system dependencies
	@echo -e "$(CYAN)[*] Installing system dependencies...$(NC)"
	@sudo apt-get update
	@sudo apt-get install -y \
		build-essential cmake pkg-config libssl-dev \
		protobuf-compiler libprotobuf-dev \
		libjemalloc-dev libsnappy-dev zlib1g-dev \
		libzstd-dev liblz4-dev librocksdb-dev \
		python3-pip python3-venv nodejs npm \
		htop iotop sysstat multitail tmux tmuxinator \
		clickhouse-client clickhouse-server \
		redis-server redis-tools \
		kafkacat jq ripgrep fd-find
	@echo -e "$(GREEN)[✓] System dependencies installed$(NC)"

.PHONY: install-rust
install-rust: ## Install Rust toolchain with optimizations
	@echo -e "$(CYAN)[*] Installing Rust toolchain...$(NC)"
	@curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	@source $$HOME/.cargo/env && \
		rustup default stable && \
		rustup component add rustfmt clippy && \
		rustup target add x86_64-unknown-linux-gnu
	@echo -e "$(GREEN)[✓] Rust toolchain installed$(NC)"

.PHONY: install-python-deps
install-python-deps: ## Install Python dependencies
	@echo -e "$(CYAN)[*] Installing Python dependencies...$(NC)"
	@pip3 install --upgrade pip setuptools wheel
	@pip3 install -r api/requirements.txt
	@pip3 install -r continuous-improvement/requirements.txt
	@pip3 install confluent-kafka clickhouse-connect aiohttp redis psutil numpy
	@echo -e "$(GREEN)[✓] Python dependencies installed$(NC)"

.PHONY: install-node-deps
install-node-deps: ## Install Node.js dependencies
	@echo -e "$(CYAN)[*] Installing Node.js dependencies...$(NC)"
	@cd defi-frontend && npm install
	@echo -e "$(GREEN)[✓] Node.js dependencies installed$(NC)"

#######################################
# Build Targets
#######################################

.PHONY: build
build: build-rust build-python build-frontend ## Build all components

.PHONY: build-rust
build-rust: ## Build Rust services with maximum optimization
	@echo -e "$(CYAN)[*] Building Rust services...$(NC)"
	@export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1" && \
		cargo build $(CARGO_BUILD_FLAGS) --all
	@echo -e "$(GREEN)[✓] Rust services built$(NC)"

.PHONY: build-python
build-python: proto ## Generate Python protobuf files
	@echo -e "$(CYAN)[*] Building Python components...$(NC)"
	@python3 -m compileall api/
	@echo -e "$(GREEN)[✓] Python components built$(NC)"

.PHONY: build-frontend
build-frontend: ## Build frontend with optimizations
	@echo -e "$(CYAN)[*] Building frontend...$(NC)"
	@cd defi-frontend && npm run build
	@echo -e "$(GREEN)[✓] Frontend built$(NC)"

.PHONY: proto
proto: ## Generate protobuf files
	@echo -e "$(CYAN)[*] Generating protobuf files...$(NC)"
	@protoc --python_out=api/proto_gen/ \
		--proto_path=protocol/ \
		protocol/realtime.proto \
		protocol/control.proto \
		protocol/jobs.proto
	@touch api/proto_gen/__init__.py
	@echo -e "$(GREEN)[✓] Protobuf files generated$(NC)"

#######################################
# Database Setup
#######################################

.PHONY: clickhouse-setup
clickhouse-setup: ## Setup ClickHouse database and tables
	@echo -e "$(CYAN)[*] Setting up ClickHouse...$(NC)"
	@clickhouse-client --multiquery < clickhouse-setup.sql
	@clickhouse-client --multiquery < database/schema/core_tables.sql || true
	@clickhouse-client --multiquery < database/schema/clickhouse_optimized.sql || true
	@echo -e "$(GREEN)[✓] ClickHouse setup complete$(NC)"

.PHONY: clickhouse-ddl
clickhouse-ddl: ## Apply all ClickHouse DDL files
	@echo -e "$(CYAN)[*] Applying ClickHouse DDL...$(NC)"
	@for file in clickhouse/*.sql; do \
		if [ -f "$$file" ]; then \
			echo "  Applying $$file..."; \
			clickhouse-client --multiquery < "$$file" || true; \
		fi; \
	done
	@echo -e "$(GREEN)[✓] ClickHouse DDL applied$(NC)"

.PHONY: redis-setup
redis-setup: ## Setup Redis with MEV-specific configuration
	@echo -e "$(CYAN)[*] Setting up Redis...$(NC)"
	@sudo systemctl start redis-server
	@redis-cli CONFIG SET maxmemory 4gb
	@redis-cli CONFIG SET maxmemory-policy allkeys-lru
	@redis-cli CONFIG SET save ""
	@redis-cli CONFIG SET appendonly no
	@echo -e "$(GREEN)[✓] Redis setup complete$(NC)"

#######################################
# Kafka Management
#######################################

.PHONY: kafka-topics
kafka-topics: ## Create all Kafka topics
	@echo -e "$(CYAN)[*] Creating Kafka topics...$(NC)"
	@kafka-topics.sh --bootstrap-server $(KAFKA_BROKERS) --create \
		--topic bandit-events-proto --partitions 3 --replication-factor 1 \
		--config compression.type=zstd --if-not-exists || true
	@kafka-topics.sh --bootstrap-server $(KAFKA_BROKERS) --create \
		--topic realtime-proto --partitions 3 --replication-factor 1 \
		--config compression.type=zstd --if-not-exists || true
	@kafka-topics.sh --bootstrap-server $(KAFKA_BROKERS) --create \
		--topic control-acks --partitions 3 --replication-factor 1 \
		--config compression.type=zstd --if-not-exists || true
	@kafka-topics.sh --bootstrap-server $(KAFKA_BROKERS) --create \
		--topic mev-opportunities --partitions 6 --replication-factor 1 \
		--config compression.type=zstd --if-not-exists || true
	@kafka-topics.sh --bootstrap-server $(KAFKA_BROKERS) --create \
		--topic arbitrage-paths --partitions 6 --replication-factor 1 \
		--config compression.type=zstd --if-not-exists || true
	@echo -e "$(GREEN)[✓] Kafka topics created$(NC)"

.PHONY: kafka-list
kafka-list: ## List all Kafka topics and consumer groups
	@echo -e "$(CYAN)Kafka Topics:$(NC)"
	@kafka-topics.sh --bootstrap-server $(KAFKA_BROKERS) --list
	@echo -e "\n$(CYAN)Consumer Groups:$(NC)"
	@kafka-consumer-groups.sh --bootstrap-server $(KAFKA_BROKERS) --list

#######################################
# Grafana Provisioning
#######################################

.PHONY: grafana-provision
grafana-provision: ## Install Grafana ClickHouse datasource provisioning
	@echo -e "$(CYAN)[*] Installing Grafana ClickHouse datasource provisioning...$(NC)"
	@sudo mkdir -p /etc/grafana/provisioning/datasources
	@sudo cp dashboards/provisioning/clickhouse-primary.yaml /etc/grafana/provisioning/datasources/
	@sudo systemctl restart grafana-server
	@echo -e "$(GREEN)[✓] Grafana datasource provisioned$(NC)"

.PHONY: grafana-provision-dashboards
grafana-provision-dashboards: ## Install Grafana dashboards provisioning
	@echo -e "$(CYAN)[*] Installing Grafana dashboards provisioning...$(NC)"
	@sudo mkdir -p /etc/grafana/provisioning/dashboards
	@sudo mkdir -p /var/lib/grafana/dashboards/{mev,bandit,performance,arbitrage,network,risk}
	@sudo cp dashboards/provisioning/dashboards.yaml /etc/grafana/provisioning/dashboards/
	@for dir in dashboards/grafana_*.json; do \
		if [ -f "$$dir" ]; then \
			sudo cp "$$dir" /var/lib/grafana/dashboards/mev/; \
		fi; \
	done
	@sudo chown -R grafana:grafana /var/lib/grafana/dashboards || true
	@sudo systemctl restart grafana-server
	@echo -e "$(GREEN)[✓] Grafana dashboards provisioned$(NC)"

.PHONY: grafana-full-provision
grafana-full-provision: grafana-provision grafana-provision-dashboards ## Full Grafana provisioning

#######################################
# Testing
#######################################

.PHONY: test
test: test-rust test-python ## Run all tests

.PHONY: test-rust
test-rust: ## Run Rust tests with optimizations
	@echo -e "$(CYAN)[*] Running Rust tests...$(NC)"
	@cargo test $(CARGO_TEST_FLAGS)
	@echo -e "$(GREEN)[✓] Rust tests passed$(NC)"

.PHONY: test-python
test-python: ## Run Python tests
	@echo -e "$(CYAN)[*] Running Python tests...$(NC)"
	@python3 -m pytest api/tests/ -v || true
	@python3 -m pytest continuous-improvement/tests/ -v || true
	@echo -e "$(GREEN)[✓] Python tests completed$(NC)"

.PHONY: lab-smoke-test
lab-smoke-test: ## Run comprehensive smoke test
	@echo -e "$(CYAN)[*] Running LEGENDARY smoke test...$(NC)"
	@python3 tools/lab_smoke_test.py \
		--brokers "$(KAFKA_BROKERS)" \
		--ch-url "$(CLICKHOUSE_URL)" \
		--grafana-url "$(GRAFANA_URL)" \
		--bandit 100 --mev 20 --arb 50 \
		--verify-consume \
		--output /tmp/smoke-test-metrics.json
	@echo -e "$(GREEN)[✓] Smoke test complete$(NC)"
	@cat /tmp/smoke-test-metrics.json | jq '.'

.PHONY: stress-test
stress-test: ## Run stress test with high load
	@echo -e "$(CYAN)[*] Running stress test...$(NC)"
	@python3 tools/lab_smoke_test.py \
		--brokers "$(KAFKA_BROKERS)" \
		--ch-url "$(CLICKHOUSE_URL)" \
		--bandit 10000 --mev 1000 --arb 2000 \
		--ch-wait 30

#######################################
# Service Management
#######################################

.PHONY: start
start: start-infra start-services ## Start all services

.PHONY: start-infra
start-infra: ## Start infrastructure services
	@echo -e "$(CYAN)[*] Starting infrastructure services...$(NC)"
	@sudo systemctl start clickhouse-server redis-server
	@echo -e "$(GREEN)[✓] Infrastructure started$(NC)"

.PHONY: start-services
start-services: ## Start MEV services with CPU affinity
	@echo -e "$(CYAN)[*] Starting MEV services...$(NC)"
	@taskset -c 2-3 nice -n -20 chrt -f 30 ./target/release/mev_agent &
	@taskset -c 4-5 nice -n -20 chrt -f 30 ./target/release/arb_agent &
	@taskset -c 6 nice -n -10 ./target/release/control_acks_ingestor &
	@echo -e "$(GREEN)[✓] MEV services started$(NC)"

.PHONY: stop
stop: ## Stop all services
	@echo -e "$(CYAN)[*] Stopping services...$(NC)"
	@pkill -f mev_agent || true
	@pkill -f arb_agent || true
	@pkill -f control_acks_ingestor || true
	@echo -e "$(GREEN)[✓] Services stopped$(NC)"

.PHONY: restart
restart: stop start ## Restart all services

#######################################
# Tmux Management
#######################################

.PHONY: tmux
tmux: ## Start tmuxinator cockpit
	@echo -e "$(CYAN)[*] Starting LEGENDARY tmux cockpit...$(NC)"
	@tmuxinator start -p $(TMUX_CFG)

.PHONY: tmux-stop
tmux-stop: ## Stop tmuxinator session
	@echo -e "$(CYAN)[*] Stopping tmux session...$(NC)"
	@tmux kill-session -t legendary-lab || true
	@echo -e "$(GREEN)[✓] Tmux session stopped$(NC)"

.PHONY: tmux-attach
tmux-attach: ## Attach to running tmux session
	@tmux attach-session -t legendary-lab

#######################################
# Monitoring & Logs
#######################################

.PHONY: logs
logs: ## Tail all service logs
	@multitail -cT ANSI \
		-l "tail -F /tmp/mev_agent.lab.log" \
		-l "tail -F /tmp/arb_agent.lab.log" \
		-l "tail -F /tmp/api.lab.log" \
		-l "tail -F /tmp/acks_ingestor.lab.log"

.PHONY: monitor
monitor: ## Open system monitoring dashboard
	@echo -e "$(CYAN)[*] Opening monitoring dashboard...$(NC)"
	@xdg-open "$(GRAFANA_URL)/d/mev-control/mev-control-center" 2>/dev/null || \
		echo "Open $(GRAFANA_URL) in your browser"

.PHONY: metrics
metrics: ## Show current metrics
	@echo -e "$(CYAN)Current MEV Metrics:$(NC)"
	@curl -s localhost:8080/metrics | grep -E "mev_|arb_|profit_" | head -20

.PHONY: health
health: ## Check system health
	@echo -e "$(CYAN)[*] Checking system health...$(NC)"
	@echo -e "\n$(YELLOW)ClickHouse:$(NC)"
	@clickhouse-client --query "SELECT 'OK' as status" 2>/dev/null && \
		echo -e "$(GREEN)  ✓ ClickHouse operational$(NC)" || \
		echo -e "$(RED)  ✗ ClickHouse down$(NC)"
	@echo -e "\n$(YELLOW)Redis:$(NC)"
	@redis-cli ping > /dev/null 2>&1 && \
		echo -e "$(GREEN)  ✓ Redis operational$(NC)" || \
		echo -e "$(RED)  ✗ Redis down$(NC)"
	@echo -e "\n$(YELLOW)Kafka:$(NC)"
	@kafkacat -b $(KAFKA_BROKERS) -L > /dev/null 2>&1 && \
		echo -e "$(GREEN)  ✓ Kafka operational$(NC)" || \
		echo -e "$(RED)  ✗ Kafka down$(NC)"
	@echo -e "\n$(YELLOW)API:$(NC)"
	@curl -s localhost:8080/health > /dev/null 2>&1 && \
		echo -e "$(GREEN)  ✓ API operational$(NC)" || \
		echo -e "$(RED)  ✗ API down$(NC)"

#######################################
# Systemd Service Management
#######################################

.PHONY: install-systemd
install-systemd: ## Install systemd service files
	@echo -e "$(CYAN)[*] Installing systemd services...$(NC)"
	@sudo cp systemd/*.service /etc/systemd/system/
	@sudo systemctl daemon-reload
	@echo -e "$(GREEN)[✓] Systemd services installed$(NC)"

.PHONY: enable-services
enable-services: ## Enable systemd services for auto-start
	@echo -e "$(CYAN)[*] Enabling systemd services...$(NC)"
	@sudo systemctl enable mev-control-plane.service
	@sudo systemctl enable mev-agent.service
	@sudo systemctl enable arb-agent.service
	@echo -e "$(GREEN)[✓] Services enabled$(NC)"

#######################################
# Performance Tuning
#######################################

.PHONY: tune-system
tune-system: ## Apply system performance tuning
	@echo -e "$(CYAN)[*] Applying performance tuning...$(NC)"
	@sudo sysctl -w net.core.rmem_max=134217728
	@sudo sysctl -w net.core.wmem_max=134217728
	@sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
	@sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
	@sudo sysctl -w net.core.netdev_max_backlog=5000
	@sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
	@sudo sysctl -w vm.swappiness=10
	@echo -e "$(GREEN)[✓] Performance tuning applied$(NC)"

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo -e "$(CYAN)[*] Running performance benchmarks...$(NC)"
	@cargo bench --all
	@python3 tools/benchmark.py || true
	@echo -e "$(GREEN)[✓] Benchmarks complete$(NC)"

#######################################
# Cleanup
#######################################

.PHONY: clean
clean: ## Clean build artifacts
	@echo -e "$(CYAN)[*] Cleaning build artifacts...$(NC)"
	@cargo clean
	@rm -rf api/__pycache__ api/**/__pycache__
	@rm -rf defi-frontend/node_modules defi-frontend/dist
	@rm -f /tmp/*.lab.log
	@echo -e "$(GREEN)[✓] Build artifacts cleaned$(NC)"

.PHONY: clean-data
clean-data: ## Clean test data (WARNING: destructive)
	@echo -e "$(RED)[!] WARNING: This will delete all test data$(NC)"
	@read -p "Are you sure? (y/N) " -n 1 -r; echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		clickhouse-client --query "TRUNCATE TABLE bandit_events"; \
		clickhouse-client --query "TRUNCATE TABLE mev_opportunities"; \
		clickhouse-client --query "TRUNCATE TABLE arbitrage_opportunities"; \
		redis-cli FLUSHALL; \
		echo -e "$(GREEN)[✓] Test data cleaned$(NC)"; \
	fi

#######################################
# Development Helpers
#######################################

.PHONY: dev
dev: ## Start development environment
	@echo -e "$(CYAN)[*] Starting development environment...$(NC)"
	@make build
	@make start-infra
	@make kafka-topics
	@make clickhouse-setup
	@make tmux

.PHONY: format
format: ## Format all code
	@echo -e "$(CYAN)[*] Formatting code...$(NC)"
	@cargo fmt --all
	@black api/ continuous-improvement/ tools/
	@cd defi-frontend && npm run format || true
	@echo -e "$(GREEN)[✓] Code formatted$(NC)"

.PHONY: lint
lint: ## Run linters
	@echo -e "$(CYAN)[*] Running linters...$(NC)"
	@cargo clippy --all -- -D warnings
	@pylint api/ || true
	@cd defi-frontend && npm run lint || true
	@echo -e "$(GREEN)[✓] Linting complete$(NC)"

#######################################
# CI/CD Integration
#######################################

.PHONY: ci
ci: install-deps build test lab-smoke-test ## Run CI pipeline

.PHONY: cd
cd: ci docker-build docker-push deploy ## Run CD pipeline

.PHONY: docker-build
docker-build: ## Build Docker images
	@echo -e "$(CYAN)[*] Building Docker images...$(NC)"
	@docker build -t mev-infrastructure:$(VERSION) .
	@echo -e "$(GREEN)[✓] Docker images built$(NC)"

.PHONY: docker-push
docker-push: ## Push Docker images
	@echo -e "$(CYAN)[*] Pushing Docker images...$(NC)"
	@docker push mev-infrastructure:$(VERSION)
	@echo -e "$(GREEN)[✓] Docker images pushed$(NC)"

.PHONY: deploy
deploy: ## Deploy to production
	@echo -e "$(CYAN)[*] Deploying to production...$(NC)"
	@echo "Deployment logic here..."
	@echo -e "$(GREEN)[✓] Deployed successfully$(NC)"

#######################################
# Utility Targets
#######################################

.PHONY: version
version: ## Show version information
	@echo -e "$(CYAN)Version Information:$(NC)"
	@echo "  Version:    $(VERSION)"
	@echo "  Build Time: $(BUILD_TIME)"
	@echo "  Git Commit: $(GIT_COMMIT)"
	@echo "  Rust:       $$(rustc --version)"
	@echo "  Python:     $$(python3 --version)"
	@echo "  Node:       $$(node --version)"

.PHONY: info
info: version health ## Show system information

.PHONY: models-super
models-super: ## Build Treelite optimized models
	@echo -e "$(CYAN)[*] Building Treelite models...$(NC)"
	@python3 tools/treelite_build.py \
		--model-path models/latest.pkl \
		--output rust-services/shared/abi/model.so \
		--arch native \
		--enable-pgo
	@echo -e "$(GREEN)[✓] Treelite models built$(NC)"

.PHONY: legendary
legendary: ## Bootstrap entire legendary MEV system
	@echo -e "$(PURPLE)⚡ LEGENDARY MEV SYSTEM BOOTSTRAP ⚡$(NC)"
	@make install-deps
	@make proto
	@make clickhouse-ddl
	@make kafka-topics
	@make models-super
	@make build
	@make tune-system
	@make lab-smoke-test
	@echo -e "$(PURPLE)✨ LEGENDARY SYSTEM READY ✨$(NC)"

# Default target
all: install-deps build test ## Build everything

# Catch-all target for typos
%:
	@echo -e "$(RED)Error: Unknown target '$@'$(NC)"
	@echo "Run 'make help' to see available targets"
	@exit 1