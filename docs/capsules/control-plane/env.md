# Control Plane Environment Configuration

## Required Environment Variables

### Core Configuration
```bash
# Service Configuration
CONTROL_PLANE_PORT=8000
CONTROL_PLANE_HOST=0.0.0.0
CONTROL_PLANE_WORKERS=4
CONTROL_PLANE_LOG_LEVEL=INFO

# Authentication & Security
ED25519_PUBLIC_KEYS_PATH=/secrets/public_keys.json
SIGNING_KEY_PATH=/secrets/control_plane_key.json
JWT_SECRET_KEY=<secure-random-key>
SESSION_TIMEOUT_SECONDS=3600
```

### Database Connections
```bash
# ClickHouse Configuration
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_DATABASE=mev
CLICKHOUSE_USER=control_plane
CLICKHOUSE_PASSWORD=<secure-password>
CLICKHOUSE_MAX_CONNECTIONS=50
CLICKHOUSE_TIMEOUT_SECONDS=30

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=100
REDIS_TIMEOUT_SECONDS=5
REDIS_RETRY_ATTEMPTS=3
```

### Messaging & Streaming
```bash
# Kafka Configuration
KAFKA_BROKERS=localhost:9092
KAFKA_COMMAND_TOPIC=mev.commands
KAFKA_AUDIT_TOPIC=mev.audit
KAFKA_CONSUMER_GROUP=control_plane
KAFKA_BATCH_SIZE=1000
KAFKA_LINGER_MS=10
```

### Performance Tuning
```bash
# Connection Pooling
DB_POOL_SIZE=50
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# Worker Configuration
WORKER_THREADS=8
ASYNC_TASKS_MAX=10000
COMMAND_QUEUE_SIZE=50000
BATCH_PROCESSING_SIZE=1000

# Timeouts
HTTP_TIMEOUT_SECONDS=30
COMMAND_TIMEOUT_SECONDS=60
MODEL_SWAP_TIMEOUT_SECONDS=120
HEALTH_CHECK_INTERVAL_SECONDS=10
```

### Monitoring & Observability  
```bash
# Prometheus Metrics
PROMETHEUS_PORT=9090
METRICS_NAMESPACE=mev_control_plane
METRICS_INTERVAL_SECONDS=15

# Distributed Tracing
JAEGER_ENDPOINT=http://localhost:14268/api/traces
TRACE_SAMPLING_RATE=0.1
SERVICE_NAME=control-plane

# Logging
LOG_FORMAT=json
LOG_FILE=/var/log/control_plane.log
LOG_ROTATION_SIZE=100MB
LOG_RETENTION_DAYS=30
```

### Development Configuration
```bash
# Development Mode
DEBUG=false
DEVELOPMENT_MODE=false
HOT_RELOAD=false
ENABLE_CORS=true
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Testing
TEST_MODE=false
MOCK_CLICKHOUSE=false
MOCK_REDIS=false
LOAD_TEST_USERS=1000
```

## Configuration Validation

The service validates all environment variables on startup:

- **Required Variables**: Service fails to start if missing
- **Format Validation**: URLs, ports, and formats are validated
- **Connection Testing**: Database connections tested on startup
- **Security Checks**: Keys and secrets validated for proper format

## Secrets Management

### Production Deployment
```bash
# Use HashiCorp Vault or Kubernetes Secrets
VAULT_ADDR=https://vault.example.com
VAULT_TOKEN=<vault-token>
VAULT_PATH=secret/mev/control-plane

# Or Kubernetes Secrets
kubectl create secret generic control-plane-secrets \
  --from-literal=clickhouse-password=$CLICKHOUSE_PASSWORD \
  --from-literal=redis-password=$REDIS_PASSWORD \
  --from-literal=jwt-secret=$JWT_SECRET_KEY
```

### Development Setup
```bash
# Copy example environment file
cp .env.example .env

# Generate required keys
python scripts/generate_keys.py --output /tmp/control_plane_keys

# Set development environment
export DEVELOPMENT_MODE=true
export DEBUG=true
```

## Health Check Configuration

```bash
# Health Check Endpoints
HEALTH_CHECK_TIMEOUT=5000  # milliseconds
DEEP_HEALTH_CHECK=true
HEALTH_CHECK_DEPENDENCIES=clickhouse,redis,kafka

# Readiness Probe
READINESS_CHECK_INTERVAL=10  # seconds
READINESS_TIMEOUT=30         # seconds

# Liveness Probe  
LIVENESS_CHECK_INTERVAL=30   # seconds
LIVENESS_TIMEOUT=10          # seconds
```

## Resource Limits

```bash
# Memory Configuration
MAX_MEMORY_USAGE=4GB
MEMORY_WARNING_THRESHOLD=85%
MEMORY_CRITICAL_THRESHOLD=95%

# CPU Configuration
MAX_CPU_CORES=4
CPU_WARNING_THRESHOLD=80%
CPU_CRITICAL_THRESHOLD=90%

# Network Configuration
MAX_CONNECTIONS=10000
CONNECTION_TIMEOUT=30000  # milliseconds
KEEPALIVE_TIMEOUT=60      # seconds
```