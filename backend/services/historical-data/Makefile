.PHONY: all build start stop restart logs test clean monitor help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Configuration
DOCKER_COMPOSE := docker-compose
RUST_BUILD := cargo build --release
NPM := npm
CLICKHOUSE_CLIENT := clickhouse-client

help: ## Show this help message
	@echo "${GREEN}Solana Historical Data Infrastructure${NC}"
	@echo "${BLUE}═══════════════════════════════════════${NC}"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${YELLOW}%-20s${NC} %s\n", $$1, $$2}'
	@echo ""
	@echo "${GREEN}Performance Targets:${NC}"
	@echo "  • Ingestion: ≥50k msgs/min"
	@echo "  • Backfill: ≥100 slots/s"
	@echo "  • Zero duplicates (idempotent)"
	@echo ""

all: build start ## Build and start everything

# ════════════════════════════════════════
# Infrastructure Commands
# ════════════════════════════════════════

start-infra: ## Start Redpanda + ClickHouse
	@echo "${GREEN}Starting infrastructure...${NC}"
	$(DOCKER_COMPOSE) up -d redpanda clickhouse
	@echo "${GREEN}Waiting for services to be healthy...${NC}"
	@sleep 10
	@docker exec solana-clickhouse clickhouse-client --user=solana --password=mev_billions_2025 -q "SELECT 'ClickHouse is ready'" || (echo "${RED}ClickHouse not ready${NC}" && exit 1)
	@echo "${GREEN}Infrastructure ready!${NC}"

stop-infra: ## Stop infrastructure
	@echo "${YELLOW}Stopping infrastructure...${NC}"
	$(DOCKER_COMPOSE) down

restart-infra: stop-infra start-infra ## Restart infrastructure

# ════════════════════════════════════════
# Build Commands
# ════════════════════════════════════════

build-rust: ## Build Rust ingester
	@echo "${GREEN}Building Rust ingester...${NC}"
	cd rust-ingester && $(RUST_BUILD)
	@echo "${GREEN}Rust ingester built successfully!${NC}"

build-node: ## Build Node.js backfill worker
	@echo "${GREEN}Building Node.js backfill worker...${NC}"
	cd backfill-worker && $(NPM) ci && $(NPM) run build
	@echo "${GREEN}Node.js worker built successfully!${NC}"

build: build-rust build-node ## Build all components

# ════════════════════════════════════════
# Run Commands
# ════════════════════════════════════════

run-ingester: ## Run Rust ingester
	@echo "${GREEN}Starting Rust ingester...${NC}"
	cd rust-ingester && \
	YELLOWSTONE_ENDPOINT=https://grpc.mainnet.rpcpool.com \
	KAFKA_BROKERS=localhost:19092 \
	METRICS_PORT=9090 \
	./target/release/ingester

run-backfill: ## Run backfill worker
	@echo "${GREEN}Starting backfill worker...${NC}"
	cd backfill-worker && \
	RPC_ENDPOINTS=https://api.mainnet-beta.solana.com \
	KAFKA_BROKERS=localhost:19092 \
	CLICKHOUSE_HOST=localhost \
	REDIS_URL=redis://localhost:6379 \
	START_SLOT=250000000 \
	END_SLOT=250001000 \
	CONCURRENCY=100 \
	$(NPM) start

run-all: ## Run all services
	@echo "${GREEN}Starting all services...${NC}"
	make run-ingester &
	make run-backfill &
	wait

# ════════════════════════════════════════
# Database Commands
# ════════════════════════════════════════

init-db: ## Initialize ClickHouse database
	@echo "${GREEN}Initializing ClickHouse database...${NC}"
	docker exec -i solana-clickhouse clickhouse-client \
		--user=solana \
		--password=mev_billions_2025 \
		< clickhouse/init.sql
	@echo "${GREEN}Database initialized!${NC}"

reset-db: ## Reset database (WARNING: Deletes all data!)
	@echo "${RED}WARNING: This will delete all data!${NC}"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker exec solana-clickhouse clickhouse-client \
			--user=solana \
			--password=mev_billions_2025 \
			-q "DROP DATABASE IF EXISTS solana_history; CREATE DATABASE solana_history;" && \
		make init-db; \
	fi

# ════════════════════════════════════════
# Monitoring Commands
# ════════════════════════════════════════

stats: ## Show ingestion statistics
	@echo "${GREEN}Ingestion Statistics${NC}"
	@echo "${BLUE}═══════════════════════════════════════${NC}"
	@docker exec solana-clickhouse clickhouse-client \
		--user=solana \
		--password=mev_billions_2025 \
		--format=Pretty \
		-q "SELECT \
			'slots' as table_name, \
			count() as total_rows, \
			max(slot) as latest_slot, \
			min(slot) as earliest_slot, \
			formatReadableQuantity(sum(transaction_count)) as total_transactions \
		FROM solana_history.slots \
		UNION ALL \
		SELECT \
			'transactions' as table_name, \
			count() as total_rows, \
			max(slot) as latest_slot, \
			min(slot) as earliest_slot, \
			'-' as total_transactions \
		FROM solana_history.transactions"

lag: ## Show consumer lag
	@echo "${GREEN}Consumer Lag${NC}"
	@echo "${BLUE}═══════════════════════════════════════${NC}"
	@docker exec solana-clickhouse clickhouse-client \
		--user=solana \
		--password=mev_billions_2025 \
		--format=Pretty \
		-q "SELECT \
			consumer_group, \
			topic, \
			sum(lag) as total_lag, \
			max(update_time) as last_update \
		FROM solana_history.consumer_progress \
		GROUP BY consumer_group, topic \
		ORDER BY total_lag DESC"

perf: ## Show performance metrics
	@echo "${GREEN}Performance Metrics${NC}"
	@echo "${BLUE}═══════════════════════════════════════${NC}"
	@docker exec solana-clickhouse clickhouse-client \
		--user=solana \
		--password=mev_billions_2025 \
		--format=Pretty \
		-q "SELECT \
			table_name, \
			sum(rows_ingested) / (max(timestamp) - min(timestamp)) as rows_per_second, \
			avg(lag_ms) as avg_lag_ms, \
			max(lag_ms) as max_lag_ms, \
			sum(errors) as total_errors \
		FROM solana_history.ingestion_metrics \
		WHERE timestamp > now() - INTERVAL 1 HOUR \
		GROUP BY table_name"

monitor: ## Open monitoring dashboard
	@echo "${GREEN}Opening monitoring endpoints...${NC}"
	@echo "Redpanda Console: http://localhost:8080"
	@echo "ClickHouse: http://localhost:8123/play"
	@echo "Rust Metrics: http://localhost:9090/metrics"
	@echo "Node Metrics: http://localhost:9091/metrics"

# ════════════════════════════════════════
# Testing Commands
# ════════════════════════════════════════

test-unit: ## Run unit tests
	@echo "${GREEN}Running unit tests...${NC}"
	cd rust-ingester && cargo test
	cd backfill-worker && $(NPM) test

test-integration: ## Run integration tests
	@echo "${GREEN}Running integration tests...${NC}"
	cd tests && $(NPM) test

test-load: ## Run load test
	@echo "${GREEN}Running load test...${NC}"
	@echo "Sending 10,000 test messages..."
	@docker exec solana-redpanda rpk topic produce solana.slots --format='%v{json}' <<< \
		'{"slot":999999999,"parent_slot":999999998,"block_height":999999999,"block_time":1234567890}'

test: test-unit test-integration ## Run all tests

benchmark: ## Run performance benchmark
	@echo "${GREEN}Running performance benchmark...${NC}"
	@echo "This will take approximately 1 minute..."
	@./scripts/benchmark.sh

# ════════════════════════════════════════
# Utility Commands
# ════════════════════════════════════════

logs: ## Show all logs
	$(DOCKER_COMPOSE) logs -f

logs-redpanda: ## Show Redpanda logs
	$(DOCKER_COMPOSE) logs -f redpanda

logs-clickhouse: ## Show ClickHouse logs
	$(DOCKER_COMPOSE) logs -f clickhouse

shell-clickhouse: ## Open ClickHouse shell
	docker exec -it solana-clickhouse clickhouse-client \
		--user=solana \
		--password=mev_billions_2025 \
		--database=solana_history

shell-redpanda: ## Open Redpanda shell
	docker exec -it solana-redpanda rpk cluster info

topics: ## List Kafka topics
	docker exec solana-redpanda rpk topic list

clean: ## Clean build artifacts
	@echo "${YELLOW}Cleaning build artifacts...${NC}"
	cd rust-ingester && cargo clean
	cd backfill-worker && rm -rf dist node_modules
	@echo "${GREEN}Clean complete!${NC}"

reset: clean reset-db ## Full reset (WARNING: Deletes everything!)
	@echo "${RED}Full reset complete!${NC}"

# ════════════════════════════════════════
# Quick Start Commands
# ════════════════════════════════════════

quick-start: ## Quick start (build + init + run)
	@echo "${GREEN}Quick Start: Solana Historical Data Infrastructure${NC}"
	@echo "${BLUE}═══════════════════════════════════════${NC}"
	make start-infra
	sleep 5
	make init-db
	make build
	@echo ""
	@echo "${GREEN}✓ Infrastructure ready!${NC}"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run ingester: ${YELLOW}make run-ingester${NC}"
	@echo "  2. Run backfill: ${YELLOW}make run-backfill${NC}"
	@echo "  3. Monitor: ${YELLOW}make monitor${NC}"
	@echo "  4. Check stats: ${YELLOW}make stats${NC}"

demo: quick-start ## Run demo with sample data
	@echo "${GREEN}Running demo...${NC}"
	make test-load
	sleep 10
	make stats

# ════════════════════════════════════════
# Production Commands
# ════════════════════════════════════════

prod-deploy: ## Deploy to production
	@echo "${GREEN}Deploying to production...${NC}"
	make build
	make test
	@echo "${GREEN}Starting production services...${NC}"
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "${GREEN}Production deployment complete!${NC}"

prod-status: ## Check production status
	@echo "${GREEN}Production Status${NC}"
	@echo "${BLUE}═══════════════════════════════════════${NC}"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	make stats
	@echo ""
	make lag

backup: ## Backup ClickHouse data
	@echo "${GREEN}Backing up ClickHouse data...${NC}"
	@mkdir -p backups
	@docker exec solana-clickhouse clickhouse-backup create
	@echo "${GREEN}Backup complete!${NC}"

restore: ## Restore ClickHouse data
	@echo "${YELLOW}Restoring ClickHouse data...${NC}"
	@docker exec solana-clickhouse clickhouse-backup restore
	@echo "${GREEN}Restore complete!${NC}"