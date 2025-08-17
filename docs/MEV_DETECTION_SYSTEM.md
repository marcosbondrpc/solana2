# MEV Detection and Analysis System

## Overview

This is a **DEFENSIVE-ONLY** MEV detection and analysis infrastructure for the Solana blockchain. The system provides comprehensive monitoring, detection, and classification of MEV activity without any execution capabilities.

### Key Principles
- **Observation Only**: No bundle creation, no trading, no execution
- **High Performance**: Sub-8ms P50 detection latency, 235k+ rows/s ingestion
- **Comprehensive Analysis**: Multi-pattern detection with archetype classification
- **Privacy Preserving**: Decision DNA tracking with cryptographic signatures

## Architecture Components

### 1. ShredStream Ingestion Service (`services/shred-ingest/`)
- Subscribes to Jito ShredStream gRPC for earliest transaction visibility
- Achieves sub-10ms ingestion latency
- Forwards raw entries to ClickHouse with microsecond timestamps
- Handles 235k+ rows/second throughput
- Zero-copy message processing with ring buffers

**Key Features:**
- Lock-free ring buffer architecture
- DSCP marking for network prioritization
- CPU affinity for dedicated core allocation
- Batch accumulation with timeout-based flushing

### 2. ClickHouse Schema (`sql/ddl/detection_schema.sql`)
The detection system uses optimized tables for high-throughput ingestion:

- **raw_tx**: Transaction telemetry with slot-aligned data
- **candidates**: Sandwich attack candidates with evidence scoring
- **entity_profiles**: Behavioral profiles and archetype classification
- **ix_sequences**: Instruction patterns for ML models
- **decision_dna**: Cryptographic audit trail

**Performance Optimizations:**
- DoubleDelta and Gorilla compression codecs
- Bloom filter indexes for entity lookups
- TTL policies for automatic data retention
- Materialized views for real-time aggregation

### 3. Sandwich Detection Engine (`services/detector/sandwich_detector.rs`)
Implements sophisticated pattern detection algorithms:

**Detection Methods:**
- **Slot-local bracket heuristic**: [A swap] → [V swap] → [A swap] patterns
- **Multi-address detection**: Identifies obfuscated coordinated attacks
- **Slip-rebound analysis**: Price impact and reversion patterns
- **Bundle adjacency metrics**: Timing and proximity analysis

**Evidence Scoring:**
- Bracket pattern: 0.85 confidence
- Slip-rebound: 0.75 confidence
- Combined evidence: 0.95 confidence
- Weak signals: 0.60 confidence

### 4. Archetype Classification System (`services/infer/archetype_classifier.py`)

Classifies MEV entities into behavioral archetypes based on measurable indices:

#### Empire Archetype
- **Characteristics**: High-volume, 24/7 operations, sophisticated infrastructure
- **Metrics**:
  - Daily transactions > 1000
  - Landing rate > 65%
  - P99 latency < 20ms
  - Broad market coverage (50+ unique pools)
- **Index Calculation**: Volume (40%) + Infrastructure (30%) + Timing (20%) + Coverage (10%)

#### Warlord Archetype
- **Characteristics**: Specialized presence, program-specific, tactical operations
- **Metrics**:
  - High pool specialization (>70%)
  - Tactical precision >60%
  - Moderate consistency (40-70%)
  - Limited wallet rotation (<30%)
- **Index Calculation**: Specialization (40%) + Precision (30%) + Consistency (20%) + Rotation (10%)

#### Guerrilla Archetype
- **Characteristics**: Opportunistic, niche pools, wallet rotation
- **Metrics**:
  - High rotation rate (>50%)
  - Low consistency (<30%)
  - Burst activity patterns
  - Lower sophistication scores
- **Index Calculation**: Rotation (40%) + Inconsistency (30%) + Opportunism (20%) + Simplicity (10%)

### 5. Fleet Detection (`services/infer/fleet_detector.py`)
Uses DBSCAN clustering to identify coordinated wallet fleets:

- **Feature Extraction**: Timing patterns, fee structures, target pools
- **Clustering Algorithm**: DBSCAN with adaptive epsilon
- **Fleet Indicators**:
  - Similar timing patterns (±10ms average)
  - Fee structure correlation (±20%)
  - Pool overlap >50%
  - Coordinated wallet creation/abandonment

### 6. Read-Only API Endpoints (`integration-bridge.js`)

All endpoints are GET-only for observation:

#### `/api/v1/metrics/archetype`
Returns archetype distribution metrics with confidence intervals.

#### `/api/v1/metrics/adjacency`
Provides bundle adjacency and timing metrics.

#### `/api/v1/metrics/economic-impact`
Calculates total victim losses and attacker profits.

#### `/api/v1/detection/sandwiches`
Lists recent sandwich detections with confidence scores.

#### `/api/v1/entities/:address`
Returns detailed behavioral profile for an entity.

#### `/api/v1/reports/comparative`
Generates 7/30/90-day comparative analysis with p-values.

## Detection Methodology

### Sandwich Detection Algorithm
```
1. Slot-local analysis within ±3 slots
2. Pool-specific transaction grouping
3. Bracket pattern identification
4. Price impact calculation
5. Evidence accumulation
6. Ensemble scoring
7. Decision DNA generation
```

### Behavioral Profiling Pipeline
```
1. Aggregate 30-day transaction history
2. Calculate timing distributions (P50, P95, P99)
3. Analyze pool diversity and specialization
4. Detect wallet rotation patterns
5. Compute archetype indices
6. Assign primary classification
7. Identify linked wallets/fleets
```

## Performance Metrics

### Target SLOs
- **Detection Latency**: P50 ≤8ms, P99 ≤20ms
- **Ingestion Rate**: ≥235k rows/second
- **False Positive Rate**: <1%
- **Model AUC**: >0.9
- **API Response Time**: P99 <100ms

### Monitoring
- Prometheus metrics at ports 9100-9105
- Grafana dashboards at port 3002
- Real-time WebSocket feed at ws://localhost:4000/ws

## Configuration

### Entity Tracking (`configs/entities.yaml`)
```yaml
entities:
  focus_wallets:
    - B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi  # Known high-volume
    - E6YoRP3adE5XYneSseLee15wJshDxCsmyD2WtLvAmfLi  # Sophisticated operator
    
  venues:
    - 675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8  # Raydium V4
    - whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc  # Whirlpool
```

### Detection Thresholds
```yaml
thresholds:
  sandwich:
    min_confidence: 0.75
    max_slot_distance: 3
    min_price_impact: 0.001  # 0.1%
  
  archetype:
    empire_threshold: 0.7
    warlord_threshold: 0.7
    guerrilla_threshold: 0.7
```

## Deployment

### Docker Compose
```bash
# Start detection infrastructure
docker-compose -f docker-compose.integrated.yml up -d
docker-compose -f docker-compose.detector.yml up -d

# View logs
docker-compose logs -f sandwich-detector
docker-compose logs -f archetype-classifier

# Check metrics
curl http://localhost:9100/metrics  # ShredStream metrics
curl http://localhost:8000/api/v1/metrics/archetype  # Archetype distribution
```

### Manual Testing
```bash
# Test sandwich detection
curl -X GET "http://localhost:8000/api/v1/detection/sandwiches?limit=10&min_confidence=0.8"

# Get entity profile
curl -X GET "http://localhost:8000/api/v1/entities/B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi"

# Fetch comparative reports
curl -X GET "http://localhost:8000/api/v1/reports/comparative"
```

## Security Considerations

### Decision DNA
Every detection generates a cryptographic fingerprint:
```
DNA = SHA256(signature || slot || instruction_sequence || timestamp)
```

This provides:
- Immutable audit trail
- Detection lineage tracking
- Reproducibility verification
- Tamper-evident logging

### Access Control
- All endpoints are read-only (GET methods only)
- No execution capabilities
- No private key handling
- No transaction submission

### Data Privacy
- Entity addresses are partially redacted in logs
- Victim information is anonymized
- No storage of transaction payloads
- Automatic data retention policies (90-365 days)

## Economic Impact Analysis

The system tracks and reports on:
- **Total Victim Losses**: Aggregate SOL lost to MEV
- **Extraction Efficiency**: Profit/Loss ratios
- **Fee Burn**: Priority fees paid by attackers
- **Victim Demographics**: Retail vs Whale targeting
- **Attack Styles**: Surgical vs Shotgun approaches

Monthly reports show:
- Top extractors by profit
- Most targeted pools
- Victim loss distribution
- Archetype evolution trends

## Limitations

This is a **DEFENSIVE-ONLY** system with the following constraints:
- No bundle creation or submission
- No trading or arbitrage execution
- No front-running or back-running
- No liquidation execution
- Observation and analysis only

## Future Enhancements

Potential improvements (maintaining defensive stance):
1. Real-time alerting for victim protection
2. ML model improvements with transformer architectures
3. Cross-chain MEV detection
4. Advanced obfuscation pattern recognition
5. Predictive victim identification
6. Economic impact forecasting

## Compliance

This system is designed for:
- Academic research
- Market surveillance
- Risk assessment
- Regulatory reporting
- Security auditing

It does NOT enable:
- MEV extraction
- Market manipulation
- Unfair trading advantages
- Protocol exploitation

## Support

For questions about the detection system:
- Review logs: `docker-compose logs [service-name]`
- Check metrics: Prometheus at port 9090
- View dashboards: Grafana at port 3002
- API documentation: http://localhost:8000/docs

Remember: This is defensive security infrastructure for understanding MEV patterns, not for executing MEV strategies.