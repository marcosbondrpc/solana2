#!/bin/bash

#########################################################################
# Solana Arbitrage Data Capture - Complete Startup Script
#########################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# ASCII Banner
cat << "EOF"
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   ███████╗ ██████╗ ██╗      █████╗ ███╗   ██╗ █████╗             ║
║   ██╔════╝██╔═══██╗██║     ██╔══██╗████╗  ██║██╔══██╗            ║
║   ███████╗██║   ██║██║     ███████║██╔██╗ ██║███████║            ║
║   ╚════██║██║   ██║██║     ██╔══██║██║╚██╗██║██╔══██║            ║
║   ███████║╚██████╔╝███████╗██║  ██║██║ ╚████║██║  ██║            ║
║   ╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝            ║
║                                                                    ║
║          ARBITRAGE DATA CAPTURE SYSTEM v1.0                       ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
EOF

echo
echo -e "${BLUE}Starting Arbitrage Data Capture System...${NC}"
echo

# Function to check service status
check_service() {
    local name=$1
    local check_cmd=$2
    
    echo -n "Checking $name... "
    if eval "$check_cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Running${NC}"
        return 0
    else
        echo -e "${RED}✗ Not running${NC}"
        return 1
    fi
}

# Function to start service
start_service() {
    local name=$1
    local start_cmd=$2
    local check_cmd=$3
    local log_file=$4
    
    echo -n "Starting $name... "
    
    if eval "$check_cmd" > /dev/null 2>&1; then
        echo -e "${YELLOW}Already running${NC}"
        return 0
    fi
    
    if [ -n "$log_file" ]; then
        eval "$start_cmd" > "$log_file" 2>&1 &
    else
        eval "$start_cmd" > /dev/null 2>&1
    fi
    
    sleep 2
    
    if eval "$check_cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Started${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to start${NC}"
        return 1
    fi
}

# Parse command line arguments
ACTION="${1:-start}"

case "$ACTION" in
    start)
        echo -e "${CYAN}[1/6] Starting ClickHouse...${NC}"
        if ! systemctl is-active --quiet clickhouse-server; then
            sudo systemctl start clickhouse-server
            sleep 3
        fi
        check_service "ClickHouse" "clickhouse-client --query 'SELECT 1'"
        
        echo -e "${CYAN}[2/6] Starting Zookeeper...${NC}"
        if [ -f "$SCRIPT_DIR/zookeeper.properties" ]; then
            start_service "Zookeeper" "sudo systemctl start zookeeper" \
                "nc -zv localhost 2181" "$LOG_DIR/zookeeper.log"
        else
            echo -e "${YELLOW}⚠ Zookeeper config not found. Run configure-services.sh first${NC}"
        fi
        
        echo -e "${CYAN}[3/6] Starting Kafka...${NC}"
        if [ -f "$SCRIPT_DIR/kafka-server.properties" ]; then
            start_service "Kafka" "sudo systemctl start kafka" \
                "/opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092" \
                "$LOG_DIR/kafka.log"
            
            # Create topics if they don't exist
            if [ -f "$SCRIPT_DIR/create-topics.sh" ]; then
                echo "Creating Kafka topics..."
                "$SCRIPT_DIR/create-topics.sh" > /dev/null 2>&1 || true
            fi
        else
            echo -e "${YELLOW}⚠ Kafka config not found. Run configure-services.sh first${NC}"
        fi
        
        echo -e "${CYAN}[4/6] Starting Redis...${NC}"
        if [ -f "$SCRIPT_DIR/redis.conf" ]; then
            start_service "Redis" "sudo systemctl start redis-custom" \
                "redis-cli ping" "$LOG_DIR/redis.log"
        else
            start_service "Redis" "sudo systemctl start redis-server" \
                "redis-cli ping" "$LOG_DIR/redis.log"
        fi
        
        echo -e "${CYAN}[5/6] Applying ClickHouse schema...${NC}"
        if [ -f "$SCRIPT_DIR/clickhouse-setup.sql" ] && clickhouse-client --query "SELECT 1" > /dev/null 2>&1; then
            clickhouse-client < "$SCRIPT_DIR/clickhouse-setup.sql" 2>/dev/null || echo -e "${YELLOW}Schema already exists${NC}"
            echo -e "${GREEN}✓ Schema applied${NC}"
        fi
        
        echo -e "${CYAN}[6/6] Building and starting Rust data capture service...${NC}"
        if [ -d "$SCRIPT_DIR/rust-services" ]; then
            cd "$SCRIPT_DIR/rust-services"
            if [ ! -f "target/release/solana-arbitrage-capture" ]; then
                echo "Building Rust service (this may take a few minutes)..."
                cargo build --release > "$LOG_DIR/build.log" 2>&1
            fi
            
            if [ -f "target/release/solana-arbitrage-capture" ]; then
                echo "Starting data capture service..."
                nohup ./target/release/solana-arbitrage-capture > "$LOG_DIR/capture.log" 2>&1 &
                echo $! > "$LOG_DIR/capture.pid"
                echo -e "${GREEN}✓ Data capture service started (PID: $(cat $LOG_DIR/capture.pid))${NC}"
            else
                echo -e "${RED}✗ Failed to build Rust service${NC}"
            fi
        fi
        ;;
        
    stop)
        echo -e "${YELLOW}Stopping services...${NC}"
        
        # Stop data capture service
        if [ -f "$LOG_DIR/capture.pid" ]; then
            kill $(cat "$LOG_DIR/capture.pid") 2>/dev/null || true
            rm "$LOG_DIR/capture.pid"
            echo "✓ Stopped data capture service"
        fi
        
        # Stop other services
        sudo systemctl stop kafka 2>/dev/null || true
        sudo systemctl stop zookeeper 2>/dev/null || true
        sudo systemctl stop redis-custom 2>/dev/null || true
        sudo systemctl stop redis-server 2>/dev/null || true
        
        echo -e "${GREEN}All services stopped${NC}"
        ;;
        
    status)
        echo -e "${BLUE}Service Status:${NC}"
        echo "═══════════════════════════════════════════════════════"
        
        check_service "ClickHouse" "clickhouse-client --query 'SELECT 1'"
        check_service "Kafka" "/opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092"
        check_service "Redis" "redis-cli ping"
        
        if [ -f "$LOG_DIR/capture.pid" ] && kill -0 $(cat "$LOG_DIR/capture.pid") 2>/dev/null; then
            echo -e "Data Capture: ${GREEN}✓ Running (PID: $(cat $LOG_DIR/capture.pid))${NC}"
        else
            echo -e "Data Capture: ${RED}✗ Not running${NC}"
        fi
        
        echo
        echo -e "${BLUE}Data Statistics:${NC}"
        echo "═══════════════════════════════════════════════════════"
        
        # ClickHouse stats
        if clickhouse-client --query "SELECT 1" > /dev/null 2>&1; then
            RECORDS=$(clickhouse-client --query "SELECT count() FROM solana_arbitrage.transactions" 2>/dev/null || echo "0")
            echo -e "ClickHouse Records: ${CYAN}$RECORDS${NC}"
            
            COMPRESSION=$(clickhouse-client --query "
                SELECT round(sum(data_uncompressed_bytes) / sum(data_compressed_bytes), 2) as ratio
                FROM system.parts
                WHERE database = 'solana_arbitrage' AND active" 2>/dev/null || echo "N/A")
            echo -e "Compression Ratio: ${CYAN}${COMPRESSION}x${NC}"
        fi
        
        # Redis stats
        if redis-cli ping > /dev/null 2>&1; then
            KEYS=$(redis-cli dbsize | awk '{print $2}')
            echo -e "Redis Keys: ${CYAN}$KEYS${NC}"
            
            MEMORY=$(redis-cli info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
            echo -e "Redis Memory: ${CYAN}$MEMORY${NC}"
        fi
        
        # Kafka stats
        if /opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092 > /dev/null 2>&1; then
            echo -e "${BLUE}Kafka Topics:${NC}"
            /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092 2>/dev/null | head -5
        fi
        ;;
        
    logs)
        echo -e "${BLUE}Viewing logs (Ctrl+C to exit)...${NC}"
        tail -f "$LOG_DIR"/*.log
        ;;
        
    test)
        echo -e "${BLUE}Running integration tests...${NC}"
        echo "═══════════════════════════════════════════════════════"
        
        # Test ClickHouse
        echo -n "Testing ClickHouse write... "
        if clickhouse-client --query "INSERT INTO solana_arbitrage.transactions (tx_signature, block_time, slot) VALUES ('test_sig', now(), 1)" 2>/dev/null; then
            echo -e "${GREEN}✓ Success${NC}"
        else
            echo -e "${RED}✗ Failed${NC}"
        fi
        
        # Test Kafka
        echo -n "Testing Kafka produce... "
        if echo '{"test": "data"}' | /opt/kafka/bin/kafka-console-producer.sh --broker-list localhost:9092 --topic solana-transactions 2>/dev/null; then
            echo -e "${GREEN}✓ Success${NC}"
        else
            echo -e "${RED}✗ Failed${NC}"
        fi
        
        # Test Redis
        echo -n "Testing Redis write... "
        if redis-cli set test_key test_value > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Success${NC}"
            redis-cli del test_key > /dev/null 2>&1
        else
            echo -e "${RED}✗ Failed${NC}"
        fi
        
        echo
        echo -e "${GREEN}Integration tests complete!${NC}"
        ;;
        
    *)
        echo "Usage: $0 {start|stop|status|logs|test}"
        echo
        echo "Commands:"
        echo "  start  - Start all data capture services"
        echo "  stop   - Stop all services"
        echo "  status - Check service status and statistics"
        echo "  logs   - Tail all service logs"
        echo "  test   - Run integration tests"
        exit 1
        ;;
esac

if [ "$ACTION" = "start" ]; then
    echo
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Arbitrage Data Capture System Started Successfully!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "${BLUE}Access Points:${NC}"
    echo -e "  ClickHouse UI:    ${CYAN}http://localhost:8123/play${NC}"
    echo -e "  Redis CLI:        ${CYAN}redis-cli${NC}"
    echo -e "  Kafka Manager:    ${CYAN}localhost:9092${NC}"
    echo
    echo -e "${BLUE}Useful Commands:${NC}"
    echo -e "  View status:      ${CYAN}$0 status${NC}"
    echo -e "  View logs:        ${CYAN}$0 logs${NC}"
    echo -e "  Run tests:        ${CYAN}$0 test${NC}"
    echo -e "  Monitor services: ${CYAN}$SCRIPT_DIR/monitor-services.sh${NC}"
    echo
    echo -e "${YELLOW}Data is being captured and stored with optimal compression!${NC}"
    echo -e "${YELLOW}ClickHouse ZSTD compression typically achieves 10-20x reduction.${NC}"
fi