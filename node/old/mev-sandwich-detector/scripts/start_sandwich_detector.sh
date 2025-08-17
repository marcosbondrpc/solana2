#!/bin/bash

# MEV Sandwich Detector - Ultra-High Performance Startup Script
# Completely independent from arbitrage module

set -e

SANDWICH_DIR="/home/kidgordones/0solana/node/mev-sandwich-detector"
LOG_DIR="$SANDWICH_DIR/logs"
PID_DIR="$SANDWICH_DIR/pids"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting MEV Sandwich Detector - Independent Ultra-Low Latency System${NC}"

# Create directories
mkdir -p $LOG_DIR $PID_DIR

# System optimizations
echo -e "${YELLOW}Applying system optimizations...${NC}"

# CPU frequency scaling - performance mode
sudo cpupower frequency-set -g performance 2>/dev/null || true

# Disable CPU throttling
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true

# Network optimizations
sudo sysctl -w net.core.rmem_max=134217728 2>/dev/null || true
sudo sysctl -w net.core.wmem_max=134217728 2>/dev/null || true
sudo sysctl -w net.core.netdev_max_backlog=50000 2>/dev/null || true
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr 2>/dev/null || true
sudo sysctl -w net.ipv4.tcp_notsent_lowat=16384 2>/dev/null || true

# Huge pages for memory performance
echo 2048 | sudo tee /proc/sys/vm/nr_hugepages 2>/dev/null || true

# Start ClickHouse if not running
if ! pgrep -x "clickhouse-serv" > /dev/null; then
    echo -e "${YELLOW}Starting ClickHouse...${NC}"
    sudo systemctl start clickhouse-server || true
    sleep 2
fi

# Initialize ClickHouse schema
echo -e "${YELLOW}Initializing ClickHouse schema...${NC}"
clickhouse-client < $SANDWICH_DIR/schemas/clickhouse_schema.sql 2>/dev/null || true

# Start Kafka if not running
if ! pgrep -f "kafka.Kafka" > /dev/null; then
    echo -e "${YELLOW}Starting Kafka...${NC}"
    sudo systemctl start kafka || {
        # Fallback to manual start
        /opt/kafka/bin/kafka-server-start.sh -daemon /opt/kafka/config/server.properties
    }
    sleep 3
fi

# Create Kafka topics
echo -e "${YELLOW}Creating Kafka topics...${NC}"
for topic in sandwich-raw sandwich-decisions sandwich-outcomes sandwich-metrics; do
    /opt/kafka/bin/kafka-topics.sh --create \
        --topic $topic \
        --bootstrap-server localhost:9092 \
        --partitions 16 \
        --replication-factor 1 \
        --if-not-exists 2>/dev/null || true
done

# Start Prometheus
if ! pgrep -x "prometheus" > /dev/null; then
    echo -e "${YELLOW}Starting Prometheus...${NC}"
    prometheus --config.file=$SANDWICH_DIR/monitoring/prometheus_config.yml \
        --storage.tsdb.path=$SANDWICH_DIR/data/prometheus \
        --web.listen-address=:9090 \
        > $LOG_DIR/prometheus.log 2>&1 &
    echo $! > $PID_DIR/prometheus.pid
fi

# Build the Rust binary with maximum optimizations
echo -e "${YELLOW}Building MEV Sandwich Detector...${NC}"
cd $SANDWICH_DIR

# Set build flags for maximum performance
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1"
export CARGO_BUILD_JOBS=8

cargo build --release

# Train/update ML model
echo -e "${YELLOW}Training ML model...${NC}"
python3 ml-pipeline/train_model.py > $LOG_DIR/ml_training.log 2>&1 &

# Start the sandwich detector with real-time priority
echo -e "${GREEN}Starting MEV Sandwich Detector with real-time priority...${NC}"

# Pin to specific CPU cores and set real-time priority
sudo nice -n -20 taskset -c 0-7 \
    chrt -f 99 \
    ./target/release/mev-sandwich-detector \
    > $LOG_DIR/sandwich_detector.log 2>&1 &

DETECTOR_PID=$!
echo $DETECTOR_PID > $PID_DIR/sandwich_detector.pid

# Wait for startup
sleep 2

# Verify process is running
if kill -0 $DETECTOR_PID 2>/dev/null; then
    echo -e "${GREEN}✓ MEV Sandwich Detector started successfully (PID: $DETECTOR_PID)${NC}"
else
    echo -e "${RED}✗ Failed to start MEV Sandwich Detector${NC}"
    exit 1
fi

# Start monitoring dashboard (optional)
echo -e "${YELLOW}Starting monitoring dashboard...${NC}"
cd $SANDWICH_DIR/monitoring
nohup python3 -m http.server 8080 > $LOG_DIR/dashboard.log 2>&1 &
echo $! > $PID_DIR/dashboard.pid

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}MEV Sandwich Detector System Started Successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Services running:"
echo "  • Sandwich Detector: http://localhost:9091/metrics"
echo "  • Prometheus: http://localhost:9090"
echo "  • ClickHouse: http://localhost:8123"
echo "  • Dashboard: http://localhost:8080"
echo ""
echo "Logs:"
echo "  • Detector: tail -f $LOG_DIR/sandwich_detector.log"
echo "  • ML Training: tail -f $LOG_DIR/ml_training.log"
echo ""
echo "Target SLOs:"
echo "  • Decision latency: ≤8ms median, ≤20ms P99"
echo "  • Bundle landing: ≥65% contested, ≥85% off-peak"
echo "  • Database writes: ≥200k rows/s"
echo "  • PnL: ≥0 over rolling 10k trades"
echo ""
echo -e "${GREEN}System is now hunting for sandwich opportunities!${NC}"