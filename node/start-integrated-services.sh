#!/bin/bash

# MEV Infrastructure Integrated Services Startup Script
# This script starts all integrated services for the MEV arbitrage system

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "MEV Infrastructure Integration Startup"
echo "========================================"

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found. Using defaults."
fi

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script requires elevated privileges for Docker."
    echo "Please run with: sudo $0"
    exit 1
fi

# Function to check if service is running
check_service() {
    local service=$1
    if docker ps | grep -q $service; then
        echo "✓ $service is running"
        return 0
    else
        echo "✗ $service is not running"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for $service on port $port..."
    while [ $attempt -lt $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo "✓ $service is ready on port $port"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo "✗ Timeout waiting for $service"
    return 1
}

echo ""
echo "1. Starting Infrastructure Services..."
echo "---------------------------------------"

# Start infrastructure services
echo "Starting Zookeeper..."
docker compose -f docker-compose.integrated.yml up -d zookeeper
wait_for_service "zookeeper" 2181

echo "Starting Kafka..."
docker compose -f docker-compose.integrated.yml up -d kafka
wait_for_service "kafka" 9092

echo "Starting Redis..."
docker compose -f docker-compose.integrated.yml up -d redis
wait_for_service "redis" 6379

echo "Starting ClickHouse..."
docker compose -f docker-compose.integrated.yml up -d clickhouse
wait_for_service "clickhouse" 8123

echo "Starting Prometheus..."
docker compose -f docker-compose.integrated.yml up -d prometheus
wait_for_service "prometheus" 9090

echo "Starting Grafana..."
docker compose -f docker-compose.integrated.yml up -d grafana
wait_for_service "grafana" 3001

echo ""
echo "2. Initializing ClickHouse Schema..."
echo "-------------------------------------"

# Wait for ClickHouse to be fully ready
sleep 5

# Apply ClickHouse schema
if [ -f infra/clickhouse/init_mev_integrated.sql ]; then
    echo "Applying MEV integrated schema to ClickHouse..."
    docker exec -i clickhouse clickhouse-client --password "$CLICKHOUSE_PASSWORD" < infra/clickhouse/init_mev_integrated.sql || {
        echo "Warning: Failed to apply schema. It may already exist."
    }
else
    echo "Warning: ClickHouse schema file not found"
fi

echo ""
echo "3. Creating Kafka Topics..."
echo "----------------------------"

# Create Kafka topics
docker exec kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --topic arbitrage-events \
    --partitions 3 \
    --replication-factor 1 || true

docker exec kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --topic mev-bundles \
    --partitions 3 \
    --replication-factor 1 || true

docker exec kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --topic mempool-events \
    --partitions 3 \
    --replication-factor 1 || true

docker exec kafka kafka-topics --create --if-not-exists \
    --bootstrap-server localhost:9092 \
    --topic pool-updates \
    --partitions 3 \
    --replication-factor 1 || true

echo ""
echo "4. Starting Backend Services..."
echo "--------------------------------"

# Build backend services if needed
if [ ! -f backend/target/release/arbitrage-detector ]; then
    echo "Building backend services..."
    cd backend && cargo build --release && cd ..
fi

# Start MEV backend services
echo "Starting MEV Backend..."
docker compose -f docker-compose.integrated.yml up -d mev-backend

echo "Starting Arbitrage Detector..."
docker compose -f docker-compose.integrated.yml up -d arbitrage-detector

echo ""
echo "5. Service Status Check..."
echo "---------------------------"

# Check all services
services=("zookeeper" "kafka" "redis" "clickhouse" "prometheus" "grafana")
all_running=true

for service in "${services[@]}"; do
    if ! check_service $service; then
        all_running=false
    fi
done

echo ""
echo "========================================"
if [ "$all_running" = true ]; then
    echo "✓ All services started successfully!"
    echo ""
    echo "Access Points:"
    echo "  - Grafana:        http://localhost:3001 (admin/admin)"
    echo "  - ClickHouse:     http://localhost:8123"
    echo "  - Prometheus:     http://localhost:9090"
    echo "  - Kafka:          localhost:9092"
    echo "  - Redis:          localhost:6379"
    echo "  - MEV API:        http://localhost:8080"
    echo "  - Arbitrage API:  http://localhost:8090"
    echo "  - Metrics:        http://localhost:9091/metrics"
    echo ""
    echo "To view logs:"
    echo "  docker compose -f docker-compose.integrated.yml logs -f [service-name]"
    echo ""
    echo "To stop all services:"
    echo "  docker compose -f docker-compose.integrated.yml down"
else
    echo "⚠ Some services failed to start. Check logs for details."
    echo "  docker compose -f docker-compose.integrated.yml logs"
fi
echo "========================================"