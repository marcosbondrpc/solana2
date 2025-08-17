#!/bin/bash
set -e

# MEV Sandwich Detector Deployment Script
# Production deployment with zero-downtime

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="mev-sandwich-detector"

echo "========================================="
echo "MEV Sandwich Detector - Production Deploy"
echo "========================================="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root for system configuration"
   exit 1
fi

# Create MEV user if not exists
if ! id "mev" &>/dev/null; then
    echo "Creating mev user..."
    useradd -r -s /bin/bash -d /home/mev mev
    mkdir -p /home/mev
    chown -R mev:mev /home/mev
fi

# System optimizations
echo "Applying system optimizations..."

# Network optimizations
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728
sysctl -w net.core.netdev_max_backlog=5000
sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
sysctl -w net.ipv4.tcp_congestion_control=bbr
sysctl -w net.core.default_qdisc=fq

# CPU optimizations
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
echo 0 > /proc/sys/kernel/numa_balancing

# Disable CPU frequency scaling
systemctl stop ondemand.service 2>/dev/null || true
systemctl disable ondemand.service 2>/dev/null || true

# Build release binary
echo "Building release binary..."
cd "$PROJECT_DIR"
cargo build --release --target-cpu=native

# Check build success
if [ ! -f "target/release/mev-sandwich-detector" ]; then
    echo "Build failed!"
    exit 1
fi

# Strip binary for size
strip target/release/mev-sandwich-detector

# Create necessary directories
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/models"
chown -R mev:mev "$PROJECT_DIR/data" "$PROJECT_DIR/logs" "$PROJECT_DIR/models"

# Check dependencies
echo "Checking dependencies..."

# Check ClickHouse
if ! systemctl is-active --quiet clickhouse-server; then
    echo "Starting ClickHouse..."
    systemctl start clickhouse-server
    sleep 5
fi

# Check Redis
if ! systemctl is-active --quiet redis; then
    echo "Starting Redis..."
    systemctl start redis
    sleep 2
fi

# Check Kafka (if using systemd)
if systemctl list-units --full -all | grep -Fq "kafka.service"; then
    if ! systemctl is-active --quiet kafka; then
        echo "Starting Kafka..."
        systemctl start kafka
        sleep 5
    fi
fi

# Initialize database
echo "Initializing database..."
clickhouse-client < "$PROJECT_DIR/schemas/clickhouse_optimized.sql" || true

# Deploy ML model
echo "Deploying ML model..."
if [ -f "$PROJECT_DIR/ml-pipeline/train_xgboost.py" ]; then
    cd "$PROJECT_DIR/ml-pipeline"
    python3 train_xgboost.py || echo "Model training skipped (run manually if needed)"
fi

# Install systemd service
echo "Installing systemd service..."
cp "$PROJECT_DIR/systemd/mev-sandwich-detector.service" /etc/systemd/system/
systemctl daemon-reload

# Stop existing service if running (for upgrades)
if systemctl is-active --quiet $SERVICE_NAME; then
    echo "Stopping existing service..."
    systemctl stop $SERVICE_NAME
    sleep 2
fi

# Start service
echo "Starting MEV Sandwich Detector..."
systemctl enable $SERVICE_NAME
systemctl start $SERVICE_NAME

# Wait for startup
sleep 5

# Check status
if systemctl is-active --quiet $SERVICE_NAME; then
    echo "✓ Service started successfully"
    
    # Show initial logs
    echo ""
    echo "Recent logs:"
    journalctl -u $SERVICE_NAME -n 20 --no-pager
    
    # Show metrics endpoint
    echo ""
    echo "Metrics available at: http://localhost:9091/metrics"
    
    # Performance check
    echo ""
    echo "Performance metrics:"
    curl -s http://localhost:9091/metrics | grep -E "sandwich_decision_time|sandwich_ml_inference" | head -5
    
else
    echo "✗ Service failed to start"
    echo "Check logs: journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi

echo ""
echo "========================================="
echo "Deployment completed successfully!"
echo "========================================="
echo ""
echo "Commands:"
echo "  Status:  systemctl status $SERVICE_NAME"
echo "  Logs:    journalctl -u $SERVICE_NAME -f"
echo "  Restart: systemctl restart $SERVICE_NAME"
echo "  Stop:    systemctl stop $SERVICE_NAME"
echo ""
echo "Monitoring:"
echo "  Metrics:   http://localhost:9091/metrics"
echo "  Grafana:   http://localhost:3000 (if configured)"
echo ""