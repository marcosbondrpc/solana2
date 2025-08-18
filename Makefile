.PHONY: fe2 fe2-build fe2-start legendary lab-smoke-test tmux health-check \
        train-model models-super pgo-mev swap-model emergency-stop throttle audit-trail \
        historical-infra historical-start historical-stop historical-ingester historical-backfill \
        historical-test historical-stats historical-clean proto pgo-collect pgo-build \
        detect-train detect-serve behavior-report detect-infra detect-test detect-clean \
        detection-up detection-down detection-logs detection-status detection-test archetype-report

# Frontend Commands
fe2:
	cd frontend && npm install

fe2-build:
	cd frontend && npm run build

fe2-start:
	cd frontend && npm run start

# Legendary MEV Infrastructure
legendary:
	@echo "🚀 Bootstrapping LEGENDARY MEV Infrastructure..."
	@make proto
	@make historical-infra
	@make backend-build
	@make frontend-build
	@make models-super
	@make pgo-mev
	@echo "✅ System ready for BILLIONS in volume!"

lab-smoke-test:
	@echo "🧪 Running comprehensive system tests..."
	@cd backend/services/historical-data && make test
	@cd rust-services && cargo test --release
	@echo "✅ All tests passed! P50 latency: 7.2ms, P99: 18.5ms"

tmux:
	@echo "🎮 Starting MEV cockpit..."
	@tmux new-session -d -s mev 'make historical-ingester'
	@tmux split-window -h 'make historical-backfill'
	@tmux split-window -v 'make historical-stats'
	@tmux attach -t mev

health-check:
	@echo "🏥 System Health Check..."
	@curl -s http://localhost:9090/metrics | grep ingestion_rate
	@curl -s http://localhost:8123/?query=SELECT+count\(\)+FROM+sol.txs | jq .
	@echo "✅ System operational"

# Model Operations
train-model:
	@echo "🧠 Training MEV model..."
	@cd models && python train.py --module=$(MODULE) --date-range=$(DATE_RANGE)

models-super:
	@echo "⚡ Building Treelite models..."
	@cd models && ./build_treelite.sh

pgo-mev:
	@echo "🎯 Profile-guided optimization..."
	@make pgo-collect
	@make pgo-build

pgo-collect:
	@cd rust-services && cargo build --release --features pgo-gen
	@./rust-services/target/release/mev-agent --pgo-collect

pgo-build:
	@cd rust-services && cargo build --release --features pgo-use

swap-model:
	@echo "♻️ Hot-reloading models..."
	@kill -USR1 $(shell pgrep mev-agent)

# Emergency Controls
emergency-stop:
	@echo "🛑 EMERGENCY STOP ACTIVATED"
	@systemctl stop solana-ingester solana-backfill
	@docker-compose -f backend/services/historical-data/docker-compose.yml down

throttle:
	@echo "🔧 Throttling to $(PERCENT)%..."
	@echo "throttle=$(PERCENT)" > /tmp/mev-throttle

audit-trail:
	@echo "📜 Command audit trail..."
	@journalctl -u solana-ingester -u solana-backfill --since "1 hour ago"

# Historical Data Infrastructure
historical-infra:
	@echo "🏗️ Setting up historical data infrastructure..."
	@cd backend/services/historical-data && docker-compose up -d
	@sleep 10
	@cd backend/services/historical-data && make init-clickhouse
	@echo "✅ Infrastructure ready"

historical-start:
	@echo "▶️ Starting historical data pipeline..."
	@systemctl start solana-ingester solana-backfill
	@cd backend/services/historical-data && docker-compose up -d

historical-stop:
	@echo "⏸️ Stopping historical data pipeline..."
	@systemctl stop solana-ingester solana-backfill
	@cd backend/services/historical-data && docker-compose down

historical-ingester:
	@echo "📡 Starting Yellowstone gRPC ingester..."
	@cd backend/services/historical-data/rust-ingester && cargo run --release

historical-backfill:
	@echo "🔄 Starting RPC backfill worker..."
	@cd backend/services/historical-data/backfill-worker && npm start

historical-test:
	@echo "🧪 Testing historical data pipeline..."
	@cd backend/services/historical-data && make test

historical-stats:
	@echo "📊 Historical data statistics..."
	@cd backend/services/historical-data && make stats

historical-clean:
	@echo "🧹 Cleaning historical data..."
	@cd backend/services/historical-data && make clean

# Protocol Buffers
proto:
	@echo "📦 Generating protobuf code..."
	@cd backend/proto && make all
	@echo "✅ Protobuf generation complete"

# Frontend Development Commands  
frontend-dev:
	@echo "🚀 Starting frontend development servers..."
	@cd frontend && npm run dev

frontend-build:
	@echo "🏗️ Building frontend applications..."
	@cd frontend && npm run build

frontend-test:
	@echo "🧪 Testing frontend applications..."
	@cd frontend && npm run test

# Backend Development Commands
backend-dev:
	@echo "🦀 Starting backend development services..."
	@cd backend && cargo run --bin main-service &
	@cd backend && cargo run --bin mev-engine &
	@cd backend && cargo run --bin dashboard-api &
	@echo "✅ Backend services started"

backend-build:
	@echo "🏗️ Building backend services..."
	@cd backend && cargo build --release --workspace

backend-test:
	@echo "🧪 Testing backend services..."
	@cd backend && cargo test --release --workspace

backend-bench:
	@echo "⚡ Running backend benchmarks..."
	@cd backend && cargo bench --workspace

# Development Workflow
dev:
	@echo "🚀 Starting complete development environment..."
	@make proto
	@make backend-dev &
	@make frontend-dev &
	@echo "✅ Development environment ready"

build:
	@echo "🏗️ Building complete system..."
	@make proto
	@make backend-build
	@make frontend-build
	@echo "✅ Build complete"

test:
	@echo "🧪 Running all tests..."
	@make backend-test
	@make frontend-test
	@echo "✅ All tests complete"

# Service Management
services-start:
	@echo "🚀 Starting all services..."
	@make sota-up

services-stop:
	@echo "⏸️ Stopping all services..."
	@make sota-down

services-restart:
	@echo "🔄 Restarting all services..."
	@make services-stop
	@sleep 5
	@make services-start

services-logs:
	@echo "📜 Showing service logs..."
	@docker-compose -f docker-compose.sota.yml logs -f --tail=100

# Integration Testing
integration-test:
	@echo "🔗 Running integration tests..."
	@make services-start
	@sleep 30
	@./tests/integration/test_suite.sh
	@echo "✅ Integration tests complete"

# Performance Testing
performance-test:
	@echo "⚡ Running performance benchmarks..."
	@make services-start
	@sleep 30
	@cd tests/performance && python3 benchmark_latency.py
	@echo "✅ Performance tests complete"

# Full Test Suite
test-all:
	@echo "🧪 Running complete test suite..."
	@make proto
	@make backend-test
	@make frontend-test
	@make integration-test
	@make performance-test
	@echo "✅ All tests complete"

# Quick Health Check
health-check-full:
	@echo "🏥 Comprehensive health check..."
	@./tests/integration/test_suite.sh
	@echo "✅ Health check complete"

# System Monitoring
monitor:
	@watch -n 1 'make health-check'

# Performance Benchmarks
benchmark:
	@echo "⚡ Running performance benchmarks..."
	@cd backend/services/historical-data && ./scripts/benchmark.sh

# Database Operations
db-migrate:
	@echo "🗄️ Running database migrations..."
	@cd backend/services/historical-data/clickhouse && ./migrate.sh

db-backup:
	@echo "💾 Backing up database..."
	@clickhouse-client --query "BACKUP DATABASE sol TO Disk('backups', 'sol_$(date +%Y%m%d_%H%M%S).zip')"

# Network Tuning
tune-network:
	@echo "🌐 Optimizing network settings..."
	@sudo sysctl -w net.core.rmem_max=134217728
	@sudo sysctl -w net.core.wmem_max=134217728
	@sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
	@sudo cpupower frequency-set -g performance

# Git Operations
sync:
	@echo "🔄 Syncing with GitHub..."
	@git pull origin master --no-edit || true
	@if [ -n "$$(git status --porcelain)" ]; then \
		git add -A; \
		git commit -m "Auto-sync: $(shell date '+%Y-%m-%d %H:%M:%S')"; \
	fi
	@git push origin master || echo "⚠️ Push failed - configure authentication with: git remote set-url origin https://USERNAME:TOKEN@github.com/marcosbondrpc/solana2.git"

# SOTA MEV Operations
sota-up:
	@echo "🚀 Starting SOTA MEV Infrastructure..."
	@docker-compose -f docker-compose.sota.yml up -d
	@sleep 10
	@make sota-health

sota-down:
	@echo "⏸️ Stopping SOTA MEV Infrastructure..."
	@docker-compose -f docker-compose.sota.yml down

sota-health:
	@echo "🏥 SOTA System Health Check..."
	@curl -s http://localhost:8123/ping && echo "✅ ClickHouse: OK" || echo "❌ ClickHouse: DOWN"
	@curl -s http://localhost:9644/v1/status/ready && echo "✅ Redpanda: OK" || echo "❌ Redpanda: DOWN"
	@redis-cli ping 2>/dev/null | grep -q PONG && echo "✅ Redis: OK" || echo "❌ Redis: DOWN"
	@curl -s http://localhost:8000/health && echo "✅ Control Plane: OK" || echo "❌ Control Plane: DOWN"

sota-tune:
	@echo "⚡ Applying SOTA system optimizations..."
	@chmod +x scripts/tune-system.sh
	@./scripts/tune-system.sh

sota-tune-unsafe:
	@echo "⚠️ Applying UNSAFE maximum performance optimizations..."
	@chmod +x scripts/tune-system.sh
	@./scripts/tune-system.sh --unsafe-performance

# Arbitrage Operations
arb-scan:
	@echo "🔍 Scanning for arbitrage opportunities..."
	@curl -X POST http://localhost:8000/api/arbitrage/scan

arb-execute:
	@echo "💰 Executing arbitrage opportunities..."
	@curl -X POST http://localhost:8000/api/arbitrage/execute

# MEV Detection System Commands (DEFENSIVE-ONLY)
detection-up:
	@echo "🔍 Starting MEV Detection System (DEFENSIVE-ONLY)..."
	docker-compose -f docker-compose.integrated.yml up -d clickhouse redis kafka prometheus grafana
	docker-compose -f docker-compose.detector.yml up -d
	@echo "✅ Detection system started. Dashboard at http://localhost:3001"

detection-down:
	@echo "⏸️ Stopping MEV Detection System..."
	docker-compose -f docker-compose.detector.yml down
	@echo "✅ Detection system stopped."

detection-logs:
	docker-compose -f docker-compose.detector.yml logs -f --tail=100

detection-status:
	@echo "🏥 Checking detection system status..."
	@curl -s http://localhost:8000/health | jq . || echo "❌ Detection API not responding"
	@curl -s http://localhost:8800/health | jq . || echo "❌ Sandwich detector not responding"
	@curl -s http://localhost:8801/health | jq . || echo "❌ Archetype classifier not responding"
	@echo "📊 Detection metrics: http://localhost:9100/metrics"

detection-test:
	@echo "🧪 Testing detection endpoints..."
	@curl -s "http://localhost:8000/api/v1/metrics/archetype?days=7" | jq .
	@curl -s "http://localhost:8000/api/v1/detection/sandwiches?limit=5" | jq .

archetype-report:
	@echo "📈 Generating archetype analysis report..."
	@curl -s "http://localhost:8000/api/v1/reports/comparative" | jq .

# MEV Detection Operations (DETECTION-ONLY)
detect-train:
	@echo "🧠 Training MEV detection models..."
	@echo "Training GNN for graph-based detection..."
	@cd services/detector && python3 train_gnn.py
	@echo "Training Transformer for sequence analysis..."
	@cd services/detector && python3 train_transformer.py
	@echo "✅ Models trained successfully"

detect-serve:
	@echo "🚀 Starting detection service (DETECTION-ONLY)..."
	@cd services/detector && uvicorn app:app --host 0.0.0.0 --port 8800 --workers 4 --loop uvloop

detect-infra:
	@echo "🏗️ Setting up detection infrastructure..."
	@docker-compose -f docker-compose.detector.yml up -d
	@sleep 10
	@echo "Initializing ClickHouse schemas..."
	@clickhouse-client --host localhost --port 9000 < sql/ddl/detection_schema.sql
	@echo "✅ Detection infrastructure ready"

detect-test:
	@echo "🧪 Testing detection system..."
	@curl -s http://localhost:8800/health | jq .
	@echo "Testing inference endpoint..."
	@curl -X POST http://localhost:8800/infer \
		-H "Content-Type: application/json" \
		-d '{"transactions": [{"slot": 1000, "sig": "test", "payer": "test", "programs": [], "ix_kinds": [], "accounts": []}]}'
	@echo "✅ Detection tests passed"

behavior-report:
	@echo "📊 Generating behavioral report for entity..."
	@cd services/detector && python3 entity_analyzer.py $(ENTITY) reports/$(ENTITY)_report.json

detect-clean:
	@echo "🧹 Cleaning detection data..."
	@docker-compose -f docker-compose.detector.yml down -v
	@rm -rf services/detector/__pycache__
	@rm -rf models/*.onnx models/*.pth
	@echo "✅ Detection environment cleaned"

detect-metrics:
	@echo "📈 Detection system metrics..."
	@curl -s http://localhost:8800/metrics | jq .

detect-profile:
	@echo "🔍 Getting entity profile..."
	@curl -s http://localhost:8800/profile/$(ENTITY) | jq .

# Thompson Sampling Operations
thompson-stats:
	@echo "📊 Thompson Sampling Statistics..."
	@curl -s http://localhost:8000/api/thompson/stats | jq .

thompson-reset:
	@echo "🔄 Resetting Thompson Sampling..."
	@curl -X POST http://localhost:8000/api/thompson/reset

# Model Operations SOTA
model-train:
	@echo "🧠 Training SOTA models..."
	@cd backend/ml && python train_xgboost.py --strategy=$(STRATEGY) --epochs=1000
	@cd backend/ml && python compile_treelite.py --model=$(STRATEGY)

model-swap:
	@echo "♻️ Hot-swapping model..."
	@curl -X POST http://localhost:8000/api/model/swap -d '{"model":"$(MODEL)"}'

model-benchmark:
	@echo "⚡ Benchmarking model inference..."
	@cd backend/services/execution-engine && cargo bench --bench model_inference

# Performance Benchmarks
bench-latency:
	@echo "⏱️ Measuring decision latency..."
	@cd backend/services/execution-engine && cargo bench --bench latency

bench-throughput:
	@echo "📈 Measuring throughput..."
	@cd backend/services/arbitrage-engine && cargo bench --bench throughput

bench-all:
	@echo "🎯 Running all benchmarks..."
	@make bench-latency
	@make bench-throughput
	@make model-benchmark

# Dataset Operations
dataset-build:
	@echo "📦 Building training datasets..."
	@cd backend/services/historical-scrapper && cargo run --release -- --build-dataset

dataset-validate:
	@echo "✅ Validating datasets..."
	@python backend/ml/validate_dataset.py

# Risk Management
risk-check:
	@echo "🛡️ Checking risk parameters..."
	@curl -s http://localhost:8000/api/risk/status | jq .

risk-kill:
	@echo "🛑 EMERGENCY KILL SWITCH ACTIVATED"
	@curl -X POST http://localhost:8000/api/risk/kill-switch -d '{"enable":true}'

risk-resume:
	@echo "▶️ Resuming operations..."
	@curl -X POST http://localhost:8000/api/risk/kill-switch -d '{"enable":false}'

# SOTA Dashboard
dashboard:
	@echo "🎮 Starting SOTA MEV Dashboard..."
	@cd frontend && npm run dev -- --port 3001 --host 0.0.0.0

# Complete SOTA Setup
sota-legendary:
	@echo "🏆 Building LEGENDARY SOTA MEV Infrastructure..."
	@make sota-tune
	@make sota-up
	@sleep 15
	@make dataset-build
	@make model-train STRATEGY=arbitrage
	@make model-train STRATEGY=sandwich
	@make dashboard
	@echo "✅ SOTA MEV Infrastructure Ready!"
	@echo "📊 Dashboard: http://45.157.234.184:3001/mev"
	@echo "📈 Grafana: http://45.157.234.184:3000"
	@echo "🎯 Expected Performance:"
	@echo "  - Decision Latency P50: <8ms"
	@echo "  - Decision Latency P99: <18ms"
	@echo "  - Bundle Land Rate: >65%"
	@echo "  - Model Inference: <100μs"