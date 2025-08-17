# Elite MEV Detection System

## Overview

State-of-the-art **DETECTION-ONLY** MEV behavioral analysis system for Solana. This system provides sophisticated observability and inference capabilities with **zero execution code** - purely focused on pattern recognition and entity profiling.

## Performance Targets

- **Accuracy**: ROC-AUC ≥0.95, False Positive Rate <0.5%
- **Latency**: P50 ≤1 slot, P95 ≤2 slots
- **Throughput**: 200k+ events/second
- **Detection Coverage**: Sandwich, Arbitrage, JIT, Liquidation patterns

## Architecture Components

### 1. ClickHouse Schema (`/clickhouse/schemas/`)
- **Raw Transactions**: Slot-aligned telemetry with 30-day retention
- **Sandwich Candidates**: Detected patterns with confidence scores
- **Model Scores**: Multi-model inference results
- **Entity Profiles**: Behavioral aggregates and clustering
- **Decision DNA**: Cryptographic audit trail with Merkle anchoring

### 2. Detection Models (`/services/detector/models.py`)

#### Rule-Based Detector
- Bracket heuristic for sandwich detection
- Pattern matching with known DEX programs
- Sub-millisecond inference

#### Statistical Anomaly Detector
- Z-score based outlier detection
- Adaptive window statistics
- Real-time anomaly scoring

#### Graph Neural Network (GNN)
- Transaction flow analysis
- 3-layer GCN architecture
- Global pooling for graph-level predictions

#### Transformer
- Instruction sequence modeling
- 6-layer encoder with attention
- Positional encoding for temporal patterns

#### Hybrid Model
- GNN + Transformer fusion
- Adversarial training for robustness
- Ensemble voting with confidence weighting

### 3. Entity Analysis (`/services/detector/entity_analyzer.py`)

#### Behavioral Profiling
- Attack style classification (surgical/shotgun/mixed)
- Victim selection patterns
- Risk appetite scoring
- Fee posture analysis

#### Temporal Analysis
- Active hour detection
- Uptime percentage calculation
- Burst pattern classification
- Transaction rate metrics

#### Clustering & Coordination
- DBSCAN clustering on behavioral embeddings
- Coordinated actor identification
- Similarity scoring with cosine distance

### 4. Integration Bridge (`/integration-bridge.js`)

#### Transaction Processing
- WebSocket connection to Solana
- Real-time feature extraction
- Priority address monitoring
- Batch processing pipeline

#### ClickHouse Ingestion
- Buffered writes with 100ms flush
- Table-specific routing
- Automatic retry on failure
- Protobuf support

#### Decision DNA System
- Ed25519 signature generation
- Merkle tree construction
- Immutable audit trail
- Periodic anchoring

### 5. FastAPI Service (`/services/detector/app.py`)

#### Endpoints
- `POST /detect` - Single transaction detection
- `POST /detect/batch` - Batch processing
- `POST /detect/priority` - Priority address fast path
- `GET /entity/{address}` - Behavioral profile
- `GET /clusters` - Entity clustering
- `GET /stats` - System statistics
- `GET /merkle` - Current Merkle root
- `GET /metrics` - Prometheus metrics

## Monitored Addresses

```
B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi - High-volume arbitrageur
6gAnjderE13TGGFeqdPVQ438jp2FPVeyXAszxKu9y338 - Sophisticated sandwicher
CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C - Raydium CPMM pool
E6YoRP3adE5XYneSseLee15wJshDxCsmyD2WtLvAmfLi - Flash loan expert
pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA - PumpSwap specialist
```

## Behavioral Spectrum Metrics

### Attack Styles
- **Surgical**: Focused, consistent attacks on specific pools
- **Shotgun**: Wide-ranging, high-volume attacks
- **Mixed**: Combination of strategies

### Risk Appetite
- Calculated from failure rate, fee aggression, and strategy diversity
- Score range: 0.0 (conservative) to 1.0 (aggressive)

### Fee Postures
- **Aggressive**: >0.001 SOL priority fees (95th percentile)
- **Moderate**: >0.0001 SOL average fees
- **Conservative**: Lower fee brackets

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install @solana/web3.js clickhouse websockets axios protobufjs

# Start ClickHouse
docker-compose up -d clickhouse

# Initialize database schema
clickhouse-client < clickhouse/schemas/mev_detection_schema.sql

# Start detection service
python services/detector/app.py

# Start integration bridge
node integration-bridge.js
```

## Configuration

### Environment Variables
```bash
SOLANA_RPC=https://api.mainnet-beta.solana.com
SOLANA_WS=wss://api.mainnet-beta.solana.com
CLICKHOUSE_URL=http://localhost:8123
DETECTOR_API=http://localhost:8000
REDIS_URL=redis://localhost:6390
```

### Performance Tuning
```bash
# CPU optimization
sudo cpupower frequency-set -g performance
taskset -c 0-7 python app.py

# Network tuning
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
```

## Monitoring

### Grafana Dashboards
- MEV Detection Overview
- Entity Behavioral Profiles
- Model Performance Metrics
- System Health & Latency

### Prometheus Metrics
- `mev_detections_total` - Total detections by type
- `detection_latency_ms` - Inference latency histogram
- `model_accuracy` - Current model accuracy
- `entity_profiles_total` - Total profiles generated

## API Examples

### Detect MEV
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {...},
    "features": {
      "signature": "...",
      "slot": 123456,
      "program_ids": ["..."],
      "account_keys": ["..."]
    }
  }'
```

### Get Entity Profile
```bash
curl http://localhost:8000/entity/B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi
```

### Get Clusters
```bash
curl http://localhost:8000/clusters
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Load testing
locust -f tests/load_test.py --host http://localhost:8000
```

## Safety & Constraints

- **DETECTION ONLY** - No execution, trading, or bundle assembly
- **SIMULATION MODE** - Analysis and observation only
- All decisions have cryptographic DNA for audit
- Immutable decision chain with Merkle anchoring
- Feature hash tracking for reproducibility

## Model Training

```bash
# Prepare dataset
python scripts/prepare_dataset.py

# Train GNN model
python scripts/train_gnn.py --epochs 100 --lr 0.001

# Train Transformer
python scripts/train_transformer.py --epochs 50 --batch-size 32

# Export to ONNX
python scripts/export_onnx.py --model hybrid --output models/ensemble.onnx
```

## Troubleshooting

### High Latency
- Check ClickHouse ingestion rate
- Verify model loading in GPU/CPU mode
- Review batch size settings

### Low Detection Rate
- Verify DEX program IDs are current
- Check confidence thresholds
- Review entity transaction history

### Memory Issues
- Adjust batch sizes
- Configure Redis caching
- Implement sliding window for decision chain

## Architecture Diagram

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Solana RPC/WS  │────▶│  Integration │────▶│  ClickHouse │
└─────────────────┘     │    Bridge    │     └─────────────┘
                        └──────┬───────┘              │
                               │                       │
                               ▼                       ▼
                        ┌──────────────┐       ┌─────────────┐
                        │   FastAPI    │◀─────▶│    Redis    │
                        │   Service    │       └─────────────┘
                        └──────┬───────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │ Rule-Based  │ │     GNN     │ │ Transformer │
        │  Detector   │ │   Detector  │ │  Detector   │
        └─────────────┘ └─────────────┘ └─────────────┘
                        │              │
                        ▼              ▼
                ┌─────────────┐ ┌─────────────┐
                │   Hybrid    │ │    ONNX     │
                │   Model     │ │   Server    │
                └─────────────┘ └─────────────┘
```

## License

Proprietary - Elite MEV Detection System

## Contact

Lead Developer: @marcosbondrpc
GitHub: https://github.com/marcosbondrpc/solana2

---

**Built for detection. Optimized for microseconds. Ready for billions.**