#!/bin/bash

# Elite Continuous Improvement System Startup Script
# Production-grade launcher with health checks

set -e

echo "════════════════════════════════════════════════════════════════"
echo "    STARTING ELITE CONTINUOUS IMPROVEMENT SYSTEM"
echo "════════════════════════════════════════════════════════════════"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION or higher is required${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python version check passed ($PYTHON_VERSION)${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade pip
pip install --upgrade pip wheel setuptools > /dev/null 2>&1

# Install requirements
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt > /dev/null 2>&1

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check service dependencies
echo -e "${YELLOW}Checking service dependencies...${NC}"

# Check Redis
if nc -z localhost 6379 2>/dev/null; then
    echo -e "${GREEN}✓ Redis is running${NC}"
else
    echo -e "${RED}✗ Redis is not running on port 6379${NC}"
    echo "  Start Redis with: redis-server"
fi

# Check Kafka
if nc -z localhost 9092 2>/dev/null; then
    echo -e "${GREEN}✓ Kafka is running${NC}"
else
    echo -e "${YELLOW}⚠ Kafka is not running on port 9092${NC}"
    echo "  Start Kafka with: docker-compose up -d kafka"
fi

# Check ClickHouse
if nc -z localhost 9000 2>/dev/null; then
    echo -e "${GREEN}✓ ClickHouse is running${NC}"
else
    echo -e "${YELLOW}⚠ ClickHouse is not running on port 9000${NC}"
    echo "  Start ClickHouse with: docker-compose up -d clickhouse"
fi

# Check MLflow
if nc -z localhost 5000 2>/dev/null; then
    echo -e "${GREEN}✓ MLflow is running${NC}"
else
    echo -e "${YELLOW}⚠ MLflow is not running on port 5000${NC}"
    echo "  Starting MLflow server in background..."
    mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns > /dev/null 2>&1 &
    sleep 2
fi

# Create necessary directories
mkdir -p logs
mkdir -p profiles
mkdir -p reports
mkdir -p models
mkdir -p checkpoints
mkdir -p tensorboard

echo -e "${GREEN}✓ Directories created${NC}"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Performance optimizations
export PYTHONUNBUFFERED=1
export ASYNCIO_EVENT_LOOP=uvloop

echo "════════════════════════════════════════════════════════════════"
echo "                    SYSTEM CONFIGURATION"
echo "════════════════════════════════════════════════════════════════"
echo "Performance Targets:"
echo "  • Latency: <5ms (P99)"
echo "  • Throughput: 100k+ TPS"
echo "  • Availability: 99.99%"
echo ""
echo "Monitoring:"
echo "  • Prometheus: http://localhost:8000/metrics"
echo "  • MLflow: http://localhost:5000"
echo ""
echo "Logs: ./logs/continuous-improvement.log"
echo "════════════════════════════════════════════════════════════════"

# Function to handle shutdown
cleanup() {
    echo -e "\n${YELLOW}Shutting down system...${NC}"
    # Kill background processes
    jobs -p | xargs -r kill 2>/dev/null
    deactivate 2>/dev/null || true
    echo -e "${GREEN}System stopped${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the master orchestrator
echo -e "\n${GREEN}Starting Master Orchestrator...${NC}\n"

# Run with optimal Python flags
python3 -O -u master_orchestrator.py 2>&1 | tee logs/continuous-improvement.log

# Keep script running
wait