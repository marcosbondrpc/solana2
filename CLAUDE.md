# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

**Legendary Solana MEV Infrastructure** - A protobuf-first, ultra-low-latency MEV system handling billions in volume with sub-10ms decision latency.

### Critical Performance Requirements
- **Decision Latency**: ≤8ms P50, ≤20ms P99
- **Bundle Land Rate**: ≥65% contested
- **Model Inference**: ≤100μs P99
- **ClickHouse Ingestion**: ≥235k rows/s

## Development Commands

### Essential Operations
```bash
make legendary          # Bootstrap entire system
make lab-smoke-test     # Run all tests (must pass before any deployment)
make tmux              # Start MEV cockpit (monitoring interface)
make health-check      # Verify system health
make sync              # Sync with GitHub repository
```

### Frontend Development (Monorepo Structure)
```bash
# Workspace management
npm run dev                    # Start all frontend apps in dev mode
npm run build                  # Build all frontend apps
npm run test                   # Run tests across workspace
npm run lint                   # Lint all packages
npm run type-check             # Type check all packages

# Individual app development
npm run dashboard:dev          # Start dashboard app (port 3001)
npm run operator:dev           # Start operator app (port 3002)

# Direct app commands
cd frontend/apps/dashboard
npm run dev                    # Dashboard development server
npm run build                  # Build dashboard for production
npm run test                   # Run dashboard tests

# Packages development
cd frontend/packages/ui        # Shared UI components library
cd frontend/packages/charts    # Chart components library
cd frontend/packages/websocket # WebSocket client library
```

### Model & ML Operations
```bash
make train-model MODULE=mev DATE_RANGE=7d  # Train models
make models-super                          # Build Treelite models
make pgo-mev                              # Profile-guided optimization
make swap-model                           # Hot-reload models
```

### Historical Data Infrastructure
```bash
make historical-infra      # Setup ClickHouse + Redpanda infrastructure
make historical-start      # Start historical data pipeline
make historical-stop       # Stop historical data pipeline
make historical-ingester   # Run Yellowstone gRPC ingester
make historical-backfill   # Run RPC backfill worker
make historical-test       # Test infrastructure
make historical-stats      # View ingestion statistics
cd backend/services/historical-data && make quick-start  # Quick setup
```

### Testing Requirements
Before ANY code changes:
1. Run `make lab-smoke-test` - MUST pass all tests
2. Frontend: `npm run typecheck` and `npm run test` in frontend/
3. Backend: `cargo test --release` in arbitrage-data-capture/rust-services/
4. Detection services: `python -m pytest tests/` in services/detector/

### Emergency Controls
```bash
make emergency-stop        # Kill all trading immediately
make throttle PERCENT=10   # Reduce to 10% capacity
make audit-trail          # View command history
```

### Detection System Commands (DEFENSIVE-ONLY)
```bash
make detection-up          # Start MEV detection system
make detection-down        # Stop detection system
make detection-status      # Check system health
make detection-test        # Test detection endpoints
make archetype-report      # Generate analysis report
make detect-train          # Train detection models
make detect-serve          # Start detection service
make behavior-report ENTITY=address  # Generate entity behavior report
```

## Architecture & Code Structure

### Core Components

1. **Ingestion Layer** (`arbitrage-data-capture/`)
   - WebSocket/WebTransport with protobuf
   - Zero-copy message processing
   - Located in: `api/wt_gateway.py`, `api/realtime.py`

2. **Decision Engine** (`rust-services/`)
   - Treelite-compiled XGBoost models
   - Thompson Sampling in `shared/src/bandit_budgeted.rs`
   - Decision DNA tracking in `shared/src/decision_dna.rs`

3. **Execution Layer** (`backend/services/`)
   - Jito bundle submission: `jito-engine/`
   - Hedged sending: `rust-services/shared/src/hedged_sender.rs`
   - Path selection: `jito-engine/src/path_selector.rs`

4. **Storage** (`clickhouse/`)
   - Schema definitions in DDL files
   - Protobuf Kafka integration: `11_kafka_proto.sql`

5. **Frontend Monorepo** (`frontend/`)
   - **Apps**: Individual applications with specific purposes
     - `apps/dashboard/`: Main MEV dashboard with real-time data
     - `apps/operator/`: Operator control interface
   - **Packages**: Shared libraries and utilities
     - `packages/ui/`: Radix-based design system components
     - `packages/charts/`: Specialized MEV visualization components  
     - `packages/websocket/`: Ultra-low-latency WebSocket client
     - `packages/protobuf/`: Protocol buffer definitions
   - **Architecture**: Turborepo-managed monorepo with optimal build caching
   - **Performance**: Sub-10ms render times with virtual scrolling and Web Workers

### Protocol Buffers
All communication uses protobuf. Schemas in `protocol/`:
- `realtime.proto` - Market data & opportunities
- `control.proto` - Signed commands with Ed25519
- `jobs.proto` - Background job management

### Key Technologies
- **Languages**: Rust (performance-critical), Python (ML/API), TypeScript (frontend)
- **Databases**: ClickHouse (time-series), Redis (caching)
- **Messaging**: Kafka/Redpanda with protobuf
- **Models**: XGBoost compiled to Treelite with PGO

## Safety & Security

### Cryptographic Requirements
- All control commands MUST be Ed25519 signed
- 2-of-3 multisig for critical operations
- Decision DNA fingerprints tracked for all decisions
- ACK hash-chain for audit trail

### Testing Checklist
Before deployment, verify:
1. ✅ P50 latency ≤ 8ms
2. ✅ P99 latency ≤ 20ms  
3. ✅ Bundle land rate ≥ 65%
4. ✅ ClickHouse ingestion ≥ 200k/s
5. ✅ Kill-switches functional
6. ✅ Decision DNA tracking active
7. ✅ ACK chain intact

## Network Access
- **Frontend**: http://45.157.234.184:3001 or http://localhost:3001 (dev)
- **Backend API**: http://45.157.234.184:8000 or http://localhost:8000 (dev)
- **API Docs**: http://45.157.234.184:8000/docs or http://localhost:8000/docs (dev)
- **Detection API**: http://localhost:8800/docs (dev)
- **Detection Dashboard**: http://localhost:4001 (dev)

## GitHub Integration
Repository: https://github.com/marcosbondrpc/solana2
- Auto-pull every minute (restarts on changes)
- Auto-push every 5 minutes
- Use `make sync` for manual sync

## Performance Optimization

### CPU Tuning
```bash
sudo cpupower frequency-set -g performance
taskset -c 0-7 ./mev-agent  # Pin to cores
```

### Network Tuning  
```bash
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
```

### Model Optimization
```bash
make pgo-collect  # Collect profile
make pgo-build    # Build with profile
```

## Critical Rules
- **NEVER** disable Decision DNA tracking
- **NEVER** bypass cryptographic signing
- **ALWAYS** run `make lab-smoke-test` before deployment
- **ALWAYS** verify SLOs are met
- **ALWAYS** keep kill-switches enabled

## Common Issues

| Issue | Solution |
|-------|----------|
| Port 3000 conflict | Frontend uses 3001 |
| Missing protobuf | Run `make proto` |
| Model not loading | Check `make models-super` |
| Low land rate | Adjust Thompson Sampling priors |
| High latency | Check DSCP marking |
| ClickHouse connection | Check `make historical-infra` |
| Rust build fails | Run `cargo clean` then rebuild |
| Frontend build errors | Delete node_modules, run `npm install` |
| Detection services down | Run `make detection-up` |

## Feature Flags
```bash
export TRANSPORT_MODE=proto      # Use protobuf
export HEDGED_SEND=on           # Enable hedged sending
export KILL_SWITCH=on           # Enable auto-throttle
export DNA_TRACKING=on          # Track decision lineage
```

**Remember**: Built for billions. Optimized for microseconds. Every decision tracked. Every command signed.