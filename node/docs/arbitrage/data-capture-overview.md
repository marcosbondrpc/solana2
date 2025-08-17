# 🚀 Solana Arbitrage Data Capture System

## Ultra-High-Performance Data Pipeline with Optimal Compression

### Overview

This system captures, processes, and stores Solana arbitrage transaction data with:
- **ClickHouse**: Time-series database with ZSTD compression (10-20x reduction)
- **Kafka**: Real-time streaming with LZ4 compression
- **Redis**: In-memory caching for hot data
- **Rust**: Zero-copy processing pipeline

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Solana Validator Node                        │
│                    (RPC: localhost:8899)                         │
└────────────────────────┬─────────────────────────────────────────┘
                         │ Transaction Data
┌────────────────────────▼─────────────────────────────────────────┐
│                 Rust Data Capture Service                         │
│         (Zero-copy parsing, Thread pinning, Async)               │
└──────────┬──────────────┬─────────────────┬──────────────────────┘
           │              │                  │
      ┌────▼────┐    ┌───▼────┐       ┌────▼────┐
      │  Redis  │    │  Kafka │       │ClickHouse│
      │ (Cache) │    │(Stream)│       │(Storage) │
      └─────────┘    └────┬───┘       └──────────┘
                          │
                    ┌─────▼──────┐
                    │ Consumers  │
                    └─────────────┘
```

## 📊 Compression Statistics

| Component | Algorithm | Compression Ratio | Use Case |
|-----------|-----------|------------------|----------|
| **ClickHouse** | ZSTD(3) | 10-20x | Long-term storage |
| **Kafka** | LZ4 | 2-4x | Streaming |
| **Redis** | LZF | 2-3x | Hot cache |

### ClickHouse Optimization Details

The schema uses multiple compression techniques:
- **ZSTD(3)**: For string fields (tx_signature, signer, program)
- **DoubleDelta + ZSTD(1)**: For timestamps and sequential IDs
- **Gorilla + ZSTD(1)**: For decimal/float fields (prices, amounts)
- **Enum8**: For categorical data (mev_type)

Expected storage savings:
- Raw transaction: ~5KB
- Compressed: ~250-500 bytes
- **Compression ratio: 10-20x**

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install missing components (Kafka, Redis)
cd /home/kidgordones/0solana/node/arbitrage-data-capture
./install-dependencies.sh
```

### 2. Configure Services

```bash
# Set up ClickHouse schema, Kafka topics, Redis config
./configure-services.sh
```

### 3. Start Data Capture

```bash
# Start all services and begin capturing data
./start-data-capture.sh start
```

### 4. Check Status

```bash
# View service status and statistics
./start-data-capture.sh status
```

## 📈 Data Schema

### Main Transaction Table

```sql
CREATE TABLE transactions (
    tx_signature String,          -- Transaction hash
    block_time DateTime,          -- Block timestamp
    slot UInt64,                  -- Slot number
    signer String,                -- Transaction signer
    program String,               -- Program ID
    path Array(String),           -- DEX path
    net_profit_sol Decimal64(9),  -- Profit in SOL
    roi Float32,                  -- Return on investment
    latency_ms UInt32,           -- Processing latency
    -- ... 30+ more fields
) ENGINE = MergeTree()
```

### Key Features

- **Partitioning**: By month for efficient queries
- **Indexing**: Bloom filters on high-cardinality columns
- **Projections**: Pre-aggregated views for dashboards
- **Materialized Views**: Auto-aggregation for analytics

## 🔧 Configuration

### ClickHouse Tuning

Edit compression settings in `clickhouse-setup.sql`:
```sql
SETTINGS 
    index_granularity = 8192,              -- Index precision
    min_bytes_for_wide_part = 10485760,    -- Wide part threshold
    min_compress_block_size = 65536,       -- Compression block size
    max_compress_block_size = 1048576;     -- Max compression block
```

### Kafka Configuration

Edit `kafka-server.properties`:
```properties
compression.type=lz4                # Compression algorithm
num.io.threads=8                   # I/O threads
socket.send.buffer.bytes=102400    # Send buffer
log.retention.hours=168             # 7-day retention
```

### Redis Configuration

Edit `redis.conf`:
```conf
maxmemory 4gb                    # Memory limit
maxmemory-policy allkeys-lru     # Eviction policy
appendonly yes                   # Persistence
```

## 📊 Monitoring

### View Real-time Statistics

```bash
# Monitor all services
./monitor-services.sh
```

### ClickHouse Queries

```bash
# Connect to ClickHouse
clickhouse-client

# View compression stats
SELECT 
    table,
    formatReadableSize(sum(data_compressed_bytes)) as compressed,
    formatReadableSize(sum(data_uncompressed_bytes)) as uncompressed,
    round(sum(data_uncompressed_bytes) / sum(data_compressed_bytes), 2) as ratio
FROM system.parts
WHERE database = 'solana_arbitrage'
GROUP BY table;

# Recent arbitrages
SELECT 
    tx_signature,
    block_time,
    net_profit_sol,
    roi,
    path
FROM solana_arbitrage.transactions
WHERE label_is_arb = 1
ORDER BY block_time DESC
LIMIT 10;

# Hourly statistics
SELECT * FROM solana_arbitrage.hourly_stats
ORDER BY hour DESC
LIMIT 24;
```

### Redis Monitoring

```bash
# Check cache status
redis-cli

# View top arbitrages
ZREVRANGE recent_arbs 0 10 WITHSCORES

# Memory usage
INFO memory
```

### Kafka Monitoring

```bash
# List topics
/opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092

# Check consumer lag
/opt/kafka/bin/kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --group clickhouse-consumer \
  --describe
```

## 🛠️ Management Commands

### Start/Stop Services

```bash
# Start all services
./start-data-capture.sh start

# Stop all services
./start-data-capture.sh stop

# View logs
./start-data-capture.sh logs
```

### Data Management

```bash
# Backup ClickHouse data
clickhouse-client --query "BACKUP TABLE solana_arbitrage.transactions TO '/backup/'"

# Clear old data (older than 30 days)
clickhouse-client --query "ALTER TABLE solana_arbitrage.transactions DELETE WHERE block_time < now() - INTERVAL 30 DAY"

# Optimize table
clickhouse-client --query "OPTIMIZE TABLE solana_arbitrage.transactions FINAL"
```

## 📈 Performance Metrics

### Expected Performance

- **Ingestion Rate**: 10,000+ transactions/second
- **Query Latency**: <100ms for recent data
- **Compression Ratio**: 10-20x reduction
- **Cache Hit Rate**: >90% for hot data
- **Stream Lag**: <1 second

### Resource Usage

- **CPU**: 2-4 cores for pipeline
- **Memory**: 8GB recommended (4GB Redis, 2GB ClickHouse, 2GB services)
- **Disk**: 100GB for 1 billion transactions (with compression)
- **Network**: 10-50 Mbps depending on activity

## 🔍 Troubleshooting

### ClickHouse Issues

```bash
# Check ClickHouse logs
sudo journalctl -u clickhouse-server -f

# Repair corrupted table
clickhouse-client --query "CHECK TABLE solana_arbitrage.transactions"
```

### Kafka Issues

```bash
# Check Kafka logs
tail -f logs/kafka.log

# Reset consumer offset
/opt/kafka/bin/kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --group clickhouse-consumer \
  --reset-offsets --to-earliest \
  --topic solana-transactions \
  --execute
```

### Redis Issues

```bash
# Check Redis logs
tail -f /var/log/redis/redis-server.log

# Clear cache
redis-cli FLUSHDB
```

## 🎯 Advanced Features

### Custom Arbitrage Detection

Edit `rust-services/src/main.rs` to customize arbitrage detection:
```rust
fn is_arbitrage(&self, tx: &Transaction) -> bool {
    // Add your custom logic
}
```

### Add New DEX

```rust
const NEW_DEX: &str = "YourDexProgramID";

fn is_dex_program(&self, program_id: &str) -> bool {
    matches!(program_id, 
        RAYDIUM_V4 | ORCA_WHIRLPOOL | NEW_DEX
    )
}
```

### Export Data

```bash
# Export to CSV
clickhouse-client --query "
SELECT * FROM solana_arbitrage.transactions 
WHERE block_time >= today()
FORMAT CSV" > arbitrage_data.csv

# Export to Parquet (better compression)
clickhouse-client --query "
SELECT * FROM solana_arbitrage.transactions
FORMAT Parquet" > arbitrage_data.parquet
```

## 📄 API Access

### REST API (Coming Soon)

```bash
# Get recent arbitrages
curl http://localhost:8080/api/arbitrages/recent

# Get statistics
curl http://localhost:8080/api/stats/hourly
```

### WebSocket Stream

```javascript
const ws = new WebSocket('ws://localhost:8080/stream');
ws.on('message', (data) => {
    const arb = JSON.parse(data);
    console.log('New arbitrage:', arb);
});
```

## 🔒 Security

- All services bound to localhost only
- Redis password protection available
- ClickHouse user authentication configurable
- Kafka SSL/SASL support available

## 📚 Additional Resources

- [ClickHouse Compression](https://clickhouse.com/docs/en/sql-reference/statements/create/table#compression-codecs)
- [Kafka Performance Tuning](https://kafka.apache.org/documentation/#performance)
- [Redis Optimization](https://redis.io/docs/manual/optimization/)
- [Solana RPC Methods](https://docs.solana.com/api/http)

## 🤝 Support

For issues or questions about the arbitrage data capture system, check the logs:
```bash
./start-data-capture.sh logs
```

---

**Built for maximum performance and minimal storage with enterprise-grade compression!** 🚀