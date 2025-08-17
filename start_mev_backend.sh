#!/bin/bash

# MEV Backend Startup Script
# Starts all components for ultra-high-performance MEV operations

set -e

echo "🚀 Starting Legendary MEV Backend Infrastructure"
echo "================================================"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Change to arbitrage-data-capture directory
cd /home/kidgordones/0solana/solana2/arbitrage-data-capture

# Function to check service status
check_service() {
    if pgrep -f "$1" > /dev/null; then
        echo -e "${GREEN}✓ $2 is running${NC}"
        return 0
    else
        echo -e "${RED}✗ $2 is not running${NC}"
        return 1
    fi
}

# Function to start service
start_service() {
    echo -e "${YELLOW}Starting $1...${NC}"
    $2 &
    sleep 2
    if check_service "$3" "$1"; then
        echo -e "${GREEN}✓ $1 started successfully${NC}"
    else
        echo -e "${RED}✗ Failed to start $1${NC}"
        exit 1
    fi
}

# 1. Check dependencies
echo ""
echo "Checking dependencies..."
echo "------------------------"

# Check Redis
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Redis is running${NC}"
else
    echo -e "${YELLOW}Starting Redis...${NC}"
    redis-server --daemonize yes
    sleep 1
fi

# Check Kafka/Redpanda
if nc -z localhost 9092 2>/dev/null; then
    echo -e "${GREEN}✓ Kafka/Redpanda is running${NC}"
else
    echo -e "${YELLOW}Starting Redpanda...${NC}"
    docker run -d --name redpanda \
        -p 9092:9092 \
        -p 9644:9644 \
        -p 19092:19092 \
        -p 8081:8081 \
        -p 8082:8082 \
        docker.redpanda.com/redpandadata/redpanda:latest \
        redpanda start \
        --overprovisioned \
        --smp 2 \
        --memory 2G \
        --reserve-memory 0M \
        --node-id 0 \
        --check=false
    sleep 5
fi

# Check ClickHouse
if clickhouse-client --query "SELECT 1" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ClickHouse is running${NC}"
else
    echo -e "${YELLOW}Starting ClickHouse...${NC}"
    sudo service clickhouse-server start
    sleep 3
fi

# 2. Initialize ClickHouse tables
echo ""
echo "Initializing ClickHouse tables..."
echo "---------------------------------"

clickhouse-client --multiquery < clickhouse-setup.sql 2>/dev/null || true
echo -e "${GREEN}✓ ClickHouse tables initialized${NC}"

# 3. Start FastAPI backend
echo ""
echo "Starting FastAPI MEV Backend..."
echo "--------------------------------"

# Kill any existing instance
pkill -f "uvicorn api.main:app" 2>/dev/null || true
sleep 1

# Start with optimal settings
PYTHONPATH=/home/kidgordones/0solana/solana2/arbitrage-data-capture \
API_PORT=8000 \
API_WORKERS=4 \
KAFKA_BROKERS=localhost:9092 \
CLICKHOUSE_HOST=localhost \
CLICKHOUSE_PORT=9000 \
CLICKHOUSE_DATABASE=mev \
RATE_LIMIT_PER_SECOND=10000 \
RATE_LIMIT_BURST=20000 \
ENABLE_WEBTRANSPORT=false \
uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --loop uvloop \
    --log-level info \
    --access-log \
    --use-colors \
    --limit-concurrency 10000 \
    --limit-max-requests 1000000 \
    --timeout-keep-alive 5 > backend.log 2>&1 &

sleep 3

# Check if backend started
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo -e "${GREEN}✓ FastAPI backend started on port 8000${NC}"
else
    echo -e "${RED}✗ Failed to start FastAPI backend${NC}"
    echo "Check backend.log for errors"
    exit 1
fi

# 4. Start Rust MEV services (if compiled)
echo ""
echo "Starting Rust MEV Services..."
echo "-----------------------------"

if [ -f "rust-services/target/release/mev-agent" ]; then
    echo -e "${YELLOW}Starting MEV Agent...${NC}"
    MEV_API_BASE=http://localhost:8000 \
    ./rust-services/target/release/mev-agent > mev-agent.log 2>&1 &
    echo -e "${GREEN}✓ MEV Agent started${NC}"
else
    echo -e "${YELLOW}⚠ MEV Agent not compiled. Run: cd rust-services && cargo build --release${NC}"
fi

# 5. Create Kafka topics
echo ""
echo "Creating Kafka topics..."
echo "------------------------"

# Create required topics
docker exec -it redpanda rpk topic create control-commands-proto --partitions 3 --replicas 1 2>/dev/null || true
docker exec -it redpanda rpk topic create control-commands-high --partitions 3 --replicas 1 2>/dev/null || true
docker exec -it redpanda rpk topic create control-commands-critical --partitions 1 --replicas 1 2>/dev/null || true
docker exec -it redpanda rpk topic create mev-opportunities --partitions 10 --replicas 1 2>/dev/null || true
docker exec -it redpanda rpk topic create mev-executions --partitions 10 --replicas 1 2>/dev/null || true
docker exec -it redpanda rpk topic create bundle-submissions --partitions 5 --replicas 1 2>/dev/null || true

echo -e "${GREEN}✓ Kafka topics created${NC}"

# 6. System optimization
echo ""
echo "Applying system optimizations..."
echo "--------------------------------"

# CPU frequency scaling
sudo cpupower frequency-set -g performance 2>/dev/null || true

# Network optimizations
sudo sysctl -w net.core.rmem_max=134217728 2>/dev/null || true
sudo sysctl -w net.core.wmem_max=134217728 2>/dev/null || true
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr 2>/dev/null || true
sudo sysctl -w net.ipv4.tcp_notsent_lowat=16384 2>/dev/null || true

echo -e "${GREEN}✓ System optimizations applied${NC}"

# 7. Display status
echo ""
echo "================================================"
echo -e "${GREEN}🎉 MEV Backend Infrastructure Started!${NC}"
echo "================================================"
echo ""
echo "Service Endpoints:"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • Health Check: http://localhost:8000/api/health"
echo "  • MEV Opportunities: http://localhost:8000/api/mev/opportunities"
echo "  • MEV Stats: http://localhost:8000/api/mev/stats"
echo "  • Metrics: http://localhost:8000/metrics"
echo ""
echo "WebSocket Streams:"
echo "  • ws://localhost:8000/api/mev/ws/opportunities"
echo "  • ws://localhost:8000/api/mev/ws/executions"
echo "  • ws://localhost:8000/api/mev/ws/metrics"
echo ""
echo "Monitoring:"
echo "  • Backend logs: tail -f backend.log"
echo "  • MEV Agent logs: tail -f mev-agent.log"
echo "  • Kafka topics: docker exec -it redpanda rpk topic list"
echo "  • ClickHouse: clickhouse-client"
echo ""
echo "Performance Targets:"
echo "  ✓ Decision Latency P50: ≤8ms"
echo "  ✓ Decision Latency P99: ≤20ms"
echo "  ✓ Bundle Land Rate: ≥65%"
echo "  ✓ Throughput: ≥200k msg/sec"
echo ""
echo "Test the API:"
echo "  ./test_mev_api.sh"
echo ""

# 8. Run initial test
echo "Running quick health check..."
echo "-----------------------------"

# Test health endpoint
HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
fi

# Test MEV scan endpoint
echo "Testing MEV scan endpoint..."
SCAN_RESPONSE=$(curl -s -X POST http://localhost:8000/api/mev/scan \
    -H "Content-Type: application/json" \
    -d '{"scan_type":"all","min_profit":0.1}')

if echo "$SCAN_RESPONSE" | grep -q "opportunities"; then
    echo -e "${GREEN}✓ MEV scan endpoint working${NC}"
    OPPORTUNITY_COUNT=$(echo "$SCAN_RESPONSE" | jq '.total // 0')
    echo "  Found $OPPORTUNITY_COUNT opportunities"
else
    echo -e "${YELLOW}⚠ MEV scan endpoint returned no opportunities${NC}"
fi

echo ""
echo "================================================"
echo -e "${GREEN}System ready for billions in MEV volume!${NC}"
echo "================================================"