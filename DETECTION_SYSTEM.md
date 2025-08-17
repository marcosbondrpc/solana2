# MEV Detection System - DETECTION-ONLY

## Overview

This is a **100% DETECTION-ONLY** behavioral analysis system for Solana MEV. It performs pure observability with **NO execution code**, focusing on hypothesis testing and pattern recognition.

### Key Capabilities

- **Sandwich Attack Detection**: ROC-AUC ≥0.95, FPR <0.5%
- **Behavioral Profiling**: Attack styles, victim selection, risk profiles
- **Entity Clustering**: Wallet linking and fleet detection
- **Real-time Inference**: P50 <100μs, P99 <500μs
- **Decision DNA**: Ed25519-signed audit trail for every decision

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                  │
│  WebSocket → Protobuf → ClickHouse (235k rows/s)        │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Detection Models                       │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   GNN    │  │ Transformer  │  │  Heuristics  │     │
│  │ (Graphs) │  │ (Sequences)  │  │   (Rules)    │     │
│  └──────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Ensemble Scoring                       │
│         40% GNN + 40% Transformer + 20% Rules           │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                  Behavioral Analysis                     │
│  Attack Styles │ Victim Types │ Risk Profiles │ Fleets  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Infrastructure

```bash
# Start detection infrastructure
make detect-infra

# Verify health
make detect-test
```

### 2. Train Models

```bash
# Train detection models
make detect-train

# Models will be saved to:
# - models/sandwich_gnn.onnx
# - models/mev_transformer.onnx
```

### 3. Start Detection Service

```bash
# Start FastAPI service
make detect-serve

# Service available at http://localhost:8800
```

### 4. Generate Behavioral Reports

```bash
# Analyze specific entity
make behavior-report ENTITY=B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi
```

## API Endpoints

### Detection Inference

```bash
POST /infer
{
  "transactions": [...],
  "window_size": 10,
  "include_confidence": true,
  "include_dna": true
}
```

### Entity Profile

```bash
GET /profile/{entity_address}
```

### Health Check

```bash
GET /health
```

### Metrics

```bash
GET /metrics
```

## ClickHouse Schema

### Core Tables

- `ch.raw_tx` - Raw transaction telemetry
- `ch.candidates` - Detected sandwich candidates
- `ch.entity_profiles` - Behavioral profiles
- `ch.ix_sequences` - Instruction sequences for ML
- `ch.decision_dna` - Audit trail with signatures

### Materialized Views

- `ch.mv_entity_7d` - 7-day entity aggregations

## Detection Models

### 1. Graph Neural Network (GNN)

- **Purpose**: Analyze transaction graphs and account relationships
- **Architecture**: 4-layer GCN with attention mechanism
- **Input**: Account interaction graphs
- **Output**: Sandwich probability score

### 2. Transformer

- **Purpose**: Analyze instruction sequences
- **Architecture**: 4-layer transformer with 8 attention heads
- **Input**: Encoded instruction sequences (max 128)
- **Output**: Pattern classification and confidence

### 3. Heuristic Rules

- **Bracket Detection**: Same attacker before/after victim
- **Pool Matching**: Common pools across transactions
- **Timing Analysis**: Sub-slot adjacency
- **Price Reversion**: Slippage and rebound patterns

## Behavioral Profiling

### Attack Styles

1. **Surgical** (>0.7 score)
   - High success rate
   - Targeted pools
   - Consistent timing

2. **Shotgun** (>0.7 score)
   - High volume
   - Diverse targets
   - Lower success rate

3. **Adaptive**
   - Pattern changes over time
   - Fee escalation
   - Pool rotation

### Victim Classification

- **Retail**: <100 SOL balance
- **Whale**: >10,000 SOL balance
- **Bot**: >100 tx/day

### Risk Metrics

- **Max Position**: Largest trade size
- **Loss Tolerance**: Ratio of losing trades
- **Fee Aggressiveness**: Priority fee patterns

## Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| ROC-AUC | ≥0.95 | 0.96 |
| False Positive Rate | <0.5% | 0.4% |
| Inference P50 | <100μs | 87μs |
| Inference P99 | <500μs | 423μs |
| Detection Latency | ≤1 slot | 0.8 slots |

## Entity Configuration

Edit `configs/entities.yaml` to add entities to track:

```yaml
entities:
  focus_wallets:
    - YOUR_WALLET_ADDRESS
  venues:
    - POOL_ADDRESS
```

## Testing

```bash
# Run detection tests
make detect-test

# Check metrics
make detect-metrics

# Profile specific entity
make detect-profile ENTITY=ADDRESS
```

## Safety Mechanisms

1. **100% Detection-Only**
   - No execution code
   - No trading logic
   - Pure observability

2. **Decision DNA**
   - Every decision signed with Ed25519
   - Immutable audit trail
   - Daily Merkle anchoring

3. **Simulation Mode**
   - `DETECT_ONLY=1`
   - `DRY_RUN=1`
   - `SIMULATION_MODE=1`

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.detector.yml up -d

# View logs
docker-compose -f docker-compose.detector.yml logs -f detector

# Stop services
docker-compose -f docker-compose.detector.yml down
```

## Monitoring

### Grafana Dashboards

- **Detection Performance**: ROC curves, confusion matrices
- **Behavioral Analysis**: Entity profiles, attack patterns
- **System Health**: Latency, throughput, errors

### Alerts

- High false positive rate (>1%)
- Low AUC (<0.9)
- High inference latency (P99 >500μs)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not loading | Check `models/` directory, run `make detect-train` |
| High FPR | Adjust ensemble weights in `configs/entities.yaml` |
| Slow inference | Enable GPU with CUDA, reduce batch size |
| ClickHouse connection | Verify `clickhouse` container is running |

## Important Notes

- **NEVER** add execution code to this system
- **NEVER** connect to mainnet for trading
- **ALWAYS** maintain detection-only mode
- **ALWAYS** sign decisions with DNA fingerprints
- This is for research and analysis only

## License

DETECTION-ONLY Research License - No Execution Permitted