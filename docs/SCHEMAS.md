# Schema Documentation

## Overview

This document describes the data schemas and protocols used in the MEV monitoring infrastructure.

## Protobuf Messages

All real-time communication uses Protocol Buffers for optimal performance. Messages are defined in `/schemas/ws_messages.proto`.

### Core Message Types

#### Envelope
Wrapper for all WebSocket messages:
- `seq`: Sequence number for ordering
- `topic`: Message topic (node.health, arb.alert, etc.)
- `payload`: Encoded message payload
- `ts_ns`: Source timestamp in nanoseconds for latency tracking
- `node_id`: Source node identifier

#### NodeHealth
System health monitoring:
- `node_id`: Unique node identifier
- `slot`: Current slot number
- `tps`: Transactions per second
- `health`: Status (healthy/degraded/critical)
- `peers`: Number of connected peers
- `avg_block_time_ms`: Average block time
- `cpu_usage`: CPU utilization percentage
- `mem_usage`: Memory utilization percentage
- `ingestion_rate`: Messages per second

#### ArbitrageAlert (Defensive Only)
Detection of arbitrage opportunities:
- `tx_signature`: Transaction signature
- `slot`: Slot number
- `roi_pct`: Return on investment percentage
- `est_profit`: Estimated profit
- `legs`: Number of legs in arbitrage
- `dex_route`: DEX routing path
- `tokens`: Tokens involved
- `confidence`: Detection confidence score

#### SandwichAlert (Defensive Only)
Sandwich attack detection:
- `victim_tx`: Victim transaction
- `front_tx`: Front-running transaction
- `back_tx`: Back-running transaction
- `slot`: Slot number
- `victim_loss`: Estimated victim loss
- `attacker_profit`: Attacker profit
- `token_pair`: Token pair involved
- `dex`: DEX where attack occurred

## JSON Schemas

REST API request/response schemas are defined in `/schemas/api_schemas.json`.

### Key Schemas

#### ExportRequest
Dataset export configuration:
```json
{
  "dataset": "blocks|transactions|dex_fills|arbitrage_events",
  "format": "parquet|arrow|csv",
  "timeRange": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-02T00:00:00Z"
  },
  "filters": {},
  "columns": ["slot", "signature", "profit"]
}
```

#### ClickHouseQuery
Read-only SQL query:
```json
{
  "sql": "SELECT * FROM blocks WHERE slot > {slot:UInt64}",
  "params": {"slot": 12345},
  "format": "json",
  "timeout_ms": 5000
}
```

#### TrainingRequest
ML model training job:
```json
{
  "model_type": "arbitrage_detector",
  "dataset": "s3://bucket/dataset.parquet",
  "parameters": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "validation_split": 0.2
  }
}
```

## Database Schemas

ClickHouse table schemas are defined in `/infra/clickhouse/ddl.sql`.

### Core Tables

#### blocks
- Primary key: (slot, block_time)
- Partitioned by: month
- TTL: 90 days to cold storage
- Indexes: slot, block_time

#### transactions
- Primary key: (slot, signature)
- Partitioned by: month
- TTL: 60 days to cold storage
- Indexes: signature (bloom filter), slot

#### arbitrage_events
- Primary key: (slot, tx_signature)
- Partitioned by: month
- TTL: 180 days
- Indexes: slot, est_profit

#### audit_events
- Primary key: (timestamp, user_id)
- Partitioned by: month
- No TTL (permanent retention)
- Includes hash chain for integrity

### Materialized Views

#### tps_by_minute
Aggregates transactions per second by minute.

#### arbitrage_stats_hourly
Hourly arbitrage statistics including count, average ROI, and total profit.

## Type Generation

### Frontend TypeScript Types
Generate from protobuf:
```bash
cd frontend
npm run proto:gen
```

Types are output to `/frontend/src/types/generated/`.

### Python Pydantic Models
Models are defined in `/api/models/schemas.py` and automatically validated by FastAPI.

## Validation Rules

### Security Constraints
- All queries are parameterized to prevent SQL injection
- Table/column names are whitelisted
- Maximum query size: 10KB
- Query timeout: 30 seconds
- Export size limit: 10GB

### Performance Constraints
- Maximum batch size: 256 messages
- Batching window: 15ms
- WebSocket frame size: 1MB
- Compression: gzip for HTTP, zstd for storage

## Schema Evolution

### Backward Compatibility
- New fields must be optional
- Enum values can be added but not removed
- Field numbers in protobuf must never be reused

### Migration Process
1. Add new schema version
2. Deploy readers that handle both versions
3. Migrate writers to new version
4. Backfill historical data if needed
5. Remove old version support after grace period

## Best Practices

### Protobuf
- Use fixed-size integers when possible
- Avoid deeply nested messages
- Keep message size under 1MB
- Use field numbers 1-15 for frequently used fields

### JSON
- Use ISO 8601 for timestamps
- Validate with JSON Schema
- Compress large payloads
- Use camelCase for field names

### Database
- Use appropriate data types (UInt64 for IDs)
- Create indexes for frequently queried columns
- Partition by time for efficient TTL
- Use materialized views for aggregations