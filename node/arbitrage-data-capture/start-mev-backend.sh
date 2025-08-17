#!/bin/bash

################################################################################
# MEV Ultra-High-Performance Backend Startup Script
# Institutional-grade infrastructure with sub-millisecond latencies
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   MEV CONTROL PLANE - LEGENDARY BACKEND INFRASTRUCTURE${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION or higher required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip3 install --upgrade pip
pip3 install -r api/requirements.txt

# Generate protobuf files
echo -e "${YELLOW}Generating protobuf bindings...${NC}"
python3 -m grpc_tools.protoc -I protocol --python_out=api/proto_gen protocol/realtime.proto protocol/control.proto || true

# System optimizations
echo -e "${YELLOW}Applying system optimizations...${NC}"

# Increase file descriptors
ulimit -n 1000000

# TCP optimizations
if [ "$EUID" -eq 0 ]; then
    sysctl -w net.core.rmem_max=134217728
    sysctl -w net.core.wmem_max=134217728
    sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
    sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
    sysctl -w net.ipv4.tcp_congestion_control=bbr
    sysctl -w net.ipv4.tcp_notsent_lowat=16384
    sysctl -w net.ipv4.tcp_fastopen=3
    sysctl -w net.core.netdev_max_backlog=5000
    echo -e "${GREEN}✓ System optimizations applied${NC}"
else
    echo -e "${YELLOW}⚠ Running as non-root, skipping kernel optimizations${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p /tmp/models /tmp/exports logs

# Check services
echo -e "${YELLOW}Checking required services...${NC}"

# Check Kafka
if nc -z localhost 9092 2>/dev/null; then
    echo -e "${GREEN}✓ Kafka is running${NC}"
else
    echo -e "${YELLOW}⚠ Kafka not detected on localhost:9092${NC}"
fi

# Check Redis
if nc -z localhost 6379 2>/dev/null; then
    echo -e "${GREEN}✓ Redis is running${NC}"
else
    echo -e "${YELLOW}⚠ Redis not detected on localhost:6379${NC}"
fi

# Check ClickHouse
if nc -z localhost 8123 2>/dev/null; then
    echo -e "${GREEN}✓ ClickHouse is running${NC}"
else
    echo -e "${YELLOW}⚠ ClickHouse not detected on localhost:8123${NC}"
fi

# Export environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONASYNCIODEBUG=0
export MALLOC_ARENA_MAX=2

# API Configuration
export API_PORT=${API_PORT:-8000}
export API_WORKERS=${API_WORKERS:-4}
export KAFKA_BROKERS=${KAFKA_BROKERS:-localhost:9092}
export REDIS_URL=${REDIS_URL:-redis://localhost:6379}
export CLICKHOUSE_HOST=${CLICKHOUSE_HOST:-localhost}
export CLICKHOUSE_PORT=${CLICKHOUSE_PORT:-8123}
export CLICKHOUSE_DATABASE=${CLICKHOUSE_DATABASE:-mev}

# Security
export JWT_SECRET_KEY=${JWT_SECRET_KEY:-$(openssl rand -hex 32)}
export CTRL_SIGN_SK_HEX=${CTRL_SIGN_SK_HEX:-$(openssl rand -hex 32)}
export CTRL_PUBKEY_ID=${CTRL_PUBKEY_ID:-default}
export MASTER_API_KEY=${MASTER_API_KEY:-mev_$(openssl rand -hex 16)}

# Performance
export RATE_LIMIT_PER_SECOND=${RATE_LIMIT_PER_SECOND:-1000}
export RATE_LIMIT_BURST=${RATE_LIMIT_BURST:-2000}
export BATCH_WINDOW_MS=${BATCH_WINDOW_MS:-15}
export BATCH_MAX_SIZE=${BATCH_MAX_SIZE:-256}
export BACKPRESSURE_THRESHOLD=${BACKPRESSURE_THRESHOLD:-1000}

# WebTransport
export ENABLE_WEBTRANSPORT=${ENABLE_WEBTRANSPORT:-true}
export WT_HOST=${WT_HOST:-0.0.0.0}
export WT_PORT=${WT_PORT:-4433}

# Paths
export MODEL_DIR=${MODEL_DIR:-/tmp/models}
export EXPORT_DIR=${EXPORT_DIR:-/tmp/exports}

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Configuration:${NC}"
echo -e "  API Port: ${API_PORT}"
echo -e "  Workers: ${API_WORKERS}"
echo -e "  Kafka: ${KAFKA_BROKERS}"
echo -e "  Redis: ${REDIS_URL}"
echo -e "  ClickHouse: ${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}"
echo -e "  WebTransport: ${ENABLE_WEBTRANSPORT} (port ${WT_PORT})"
echo -e "  Master API Key: ${MASTER_API_KEY}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Function to handle shutdown
cleanup() {
    echo -e "\n${YELLOW}Shutting down MEV backend...${NC}"
    kill $UVICORN_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Shutdown complete${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start the application
echo -e "${GREEN}Starting MEV Control Plane...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Launch with uvicorn
python3 -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port ${API_PORT} \
    --workers ${API_WORKERS} \
    --loop uvloop \
    --log-level info \
    --access-log \
    --use-colors \
    --limit-concurrency 10000 \
    --limit-max-requests 1000000 \
    --timeout-keep-alive 5 \
    --timeout-graceful-shutdown 10 &

UVICORN_PID=$!

echo -e "${GREEN}✓ MEV Control Plane started (PID: $UVICORN_PID)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Access Points:${NC}"
echo -e "  API: http://localhost:${API_PORT}"
echo -e "  Docs: http://localhost:${API_PORT}/api/docs"
echo -e "  WebSocket: ws://localhost:${API_PORT}/api/realtime/ws"
echo -e "  Health: http://localhost:${API_PORT}/api/health"
echo -e "  Metrics: http://localhost:${API_PORT}/metrics"
if [ "$ENABLE_WEBTRANSPORT" = "true" ]; then
    echo -e "  WebTransport: https://localhost:${WT_PORT}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"

# Wait for the process
wait $UVICORN_PID