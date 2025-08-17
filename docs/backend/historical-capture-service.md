# Historical Capture Service

High-performance FastAPI backend for Solana blockchain data capture and MEV detection.

## Features

- **High-Performance Data Capture**: Concurrent block fetching with configurable batch sizes
- **MEV Detection**: Advanced algorithms for arbitrage and sandwich attack detection
- **Time-Based Partitioning**: Daily, monthly, and yearly data organization
- **Real-time Updates**: WebSocket support for live job progress
- **Production Ready**: Redis caching, connection pooling, and comprehensive error handling

## Architecture

### Core Components

1. **RPC Client** (`rpc.py`)
   - Connection pooling (100+ concurrent connections)
   - Binary search for timestamp-to-slot mapping
   - Batch request optimization
   - Automatic retry with exponential backoff

2. **Storage Engine** (`storage.py`)
   - Parquet format with Snappy compression
   - Time-based partitioning (year/month/day)
   - DuckDB integration for fast analytics
   - Streaming writes to minimize memory usage

3. **Capture Engine** (`capture.py`)
   - Adaptive concurrency control
   - Checkpoint and resume capability
   - Real-time progress tracking
   - Graceful cancellation

4. **MEV Detection** (`detection.py`)
   - Graph-based arbitrage cycle detection
   - A-V-B pattern matching for sandwich attacks
   - Profit calculation with gas optimization
   - Multi-DEX support (Raydium, Orca, Phoenix, etc.)

## API Endpoints

### Data Capture

```http
POST /capture/start
```
Start a new data capture job with specified parameters.

### Job Management

```http
GET /jobs/{job_id}
POST /jobs/{job_id}/cancel
```
Monitor and control capture jobs.

### MEV Detection

```http
POST /convert/arbitrage/start
POST /convert/sandwich/start
```
Analyze captured data for MEV opportunities.

### Analytics

```http
GET /datasets/stats
```
Get comprehensive statistics about captured datasets.

### Health

```http
GET /health
```
Service health check with system metrics.

### WebSocket

```ws
WS /ws/jobs/{job_id}
```
Real-time job progress updates.

## Installation

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f historical-capture

# Stop services
docker-compose down
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis (required for caching)
redis-server

# Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Usage Examples

### Start Data Capture

```python
import requests

response = requests.post("http://localhost:8000/capture/start", json={
    "granularity": "day",
    "start": "2025-01-01",
    "end": "2025-01-10",
    "source": "rpc",
    "programs": ["675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"],  # Raydium
    "include_blocks": True,
    "include_transactions": True,
    "include_logs": True,
    "out_uri": "./data",
    "block_batch": 64,
    "json_parsed": True,
    "max_tx_version": 0
})

job_id = response.json()["job_id"]
print(f"Job started: {job_id}")
```

### Monitor Job Progress

```python
# Check job status
status = requests.get(f"http://localhost:8000/jobs/{job_id}")
print(f"Progress: {status.json()['progress']:.1f}%")

# Or use WebSocket for real-time updates
import websockets
import asyncio

async def monitor():
    async with websockets.connect(f"ws://localhost:8000/ws/jobs/{job_id}") as ws:
        while True:
            update = await ws.recv()
            print(f"Update: {update}")

asyncio.run(monitor())
```

### Detect Arbitrage

```python
response = requests.post("http://localhost:8000/convert/arbitrage/start", json={
    "raw_uri": "./data/raw",
    "out_uri": "./data/labels",
    "min_profit_usd": 10.0,
    "max_slot_gap": 3
})

print(f"Detection job: {response.json()['job_id']}")
```

## Storage Structure

```
data/
├── raw/
│   ├── blocks/
│   │   └── year=2025/month=01/day=01/
│   │       └── part-*.parquet
│   ├── transactions/
│   │   └── year=2025/month=01/day=01/
│   │       └── part-*.parquet
│   └── logs/
│       └── year=2025/month=01/day=01/
│           └── part-*.parquet
├── labels/
│   ├── arbitrage.parquet
│   └── sandwich.parquet
└── manifests/
    └── job_*.json
```

## Performance Optimization

### Recommended Settings

- **Block Batch Size**: 64-128 for optimal throughput
- **Max Connections**: 100 for mainnet RPC
- **Redis Memory**: 2GB minimum
- **Workers**: 4 for production deployment

### Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple instances behind a load balancer
2. **Storage**: Use NVMe SSDs for Parquet files
3. **Memory**: 8GB minimum for large capture jobs
4. **Network**: 1Gbps+ for optimal RPC performance

## Monitoring

### Prometheus Metrics

The service exposes metrics at `/metrics`:
- Request latency (p50, p95, p99)
- Active jobs count
- Blocks/transactions processed
- Storage usage
- System resources (CPU, memory)

### Grafana Dashboard

Import the provided dashboard for visualization:
1. Access Grafana at http://localhost:3001
2. Default credentials: admin/admin
3. Import dashboard from `./grafana-dashboard.json`

## Testing

```bash
# Run test script
python test_api.py

# Or use pytest for comprehensive tests
pytest tests/ -v
```

## Environment Variables

- `REDIS_URL`: Redis connection string (default: `redis://localhost:6379`)
- `RPC_ENDPOINT`: Solana RPC endpoint (default: mainnet-beta)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `MAX_CONNECTIONS`: Maximum RPC connections (default: 100)
- `STORAGE_PATH`: Data storage path (default: `./data`)

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Reduce block_batch size
2. **RPC Rate Limits**: Use private RPC endpoint
3. **Slow Queries**: Ensure DuckDB has sufficient memory
4. **Connection Errors**: Check Redis availability

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

## Security Considerations

1. **Authentication**: Implement API key authentication for production
2. **Rate Limiting**: Add request rate limiting
3. **Input Validation**: All inputs validated with Pydantic
4. **CORS**: Configure allowed origins appropriately

## License

MIT

## Support

For issues or questions, please open an issue on GitHub.