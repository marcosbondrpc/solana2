#!/bin/bash
# Solana MEV System - All Services Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Base directory
BASE_DIR="/home/kidgordones/0solana/node"
LOG_DIR="/var/log/solana-mev"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR 2>/dev/null || true

echo -e "${GREEN}=== Starting Solana MEV System ===${NC}"

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}Port $1 already in use, killing existing process...${NC}"
        lsof -ti:$1 | xargs -r kill -9 2>/dev/null || true
        sleep 1
    fi
}

# Function to start a service
start_service() {
    local name=$1
    local port=$2
    local cmd=$3
    local dir=$4
    
    check_port $port
    echo -e "${GREEN}Starting $name on port $port...${NC}"
    cd $dir
    nohup $cmd > $LOG_DIR/${name}.log 2>&1 &
    echo $! > /tmp/${name}.pid
    sleep 2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}✓ $name started successfully${NC}"
    else
        echo -e "${RED}✗ Failed to start $name${NC}"
        return 1
    fi
}

# Start Backend Services
echo -e "\n${YELLOW}Starting Backend Services...${NC}"

# 1. MEV Engine Lite (Rust)
if [ -f "$BASE_DIR/backend/services/mev-engine-lite/target/release/mev-engine-lite" ]; then
    start_service "mev-engine-lite" 8081 \
        "./target/release/mev-engine-lite" \
        "$BASE_DIR/backend/services/mev-engine-lite"
else
    # Compile if not exists
    echo -e "${YELLOW}Building MEV Engine Lite...${NC}"
    cd $BASE_DIR/backend/services/mev-engine-lite
    cargo build --release 2>/dev/null || true
    if [ -f "target/release/mev-engine-lite" ]; then
        start_service "mev-engine-lite" 8081 \
            "./target/release/mev-engine-lite" \
            "$BASE_DIR/backend/services/mev-engine-lite"
    fi
fi

# 2. Mission Control Lite (Rust)
if [ -f "$BASE_DIR/backend/services/mission-control-lite/target/release/mission-control-lite" ]; then
    start_service "mission-control-lite" 8083 \
        "./target/release/mission-control-lite" \
        "$BASE_DIR/backend/services/mission-control-lite"
else
    # Compile if not exists
    echo -e "${YELLOW}Building Mission Control Lite...${NC}"
    cd $BASE_DIR/backend/services/mission-control-lite
    cargo build --release 2>/dev/null || true
    if [ -f "target/release/mission-control-lite" ]; then
        start_service "mission-control-lite" 8083 \
            "./target/release/mission-control-lite" \
            "$BASE_DIR/backend/services/mission-control-lite"
    fi
fi

# 3. Historical Capture API (Node.js)
start_service "historical-capture-api" 8055 \
    "node index.js" \
    "$BASE_DIR/backend/services/historical-capture-api"

# 4. API Proxy Gateway (Node.js)
start_service "api-proxy" 8085 \
    "node index.js" \
    "$BASE_DIR/backend/services/api-proxy"

# Start Frontend
echo -e "\n${YELLOW}Starting Frontend...${NC}"
check_port 3001

cd $BASE_DIR/frontend/apps/dashboard

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install --silent 2>/dev/null || true
fi

# Start frontend with specific port
nohup npm run dev -- --port 3001 > $LOG_DIR/frontend.log 2>&1 &
echo $! > /tmp/frontend.pid

# Wait for frontend to start
echo -e "${YELLOW}Waiting for frontend to start...${NC}"
sleep 5

# Check if all services are running
echo -e "\n${GREEN}=== Service Status ===${NC}"

services=(
    "frontend:3001"
    "historical-capture-api:8055"
    "mev-engine-lite:8081"
    "mission-control-lite:8083"
    "api-proxy:8085"
)

all_running=true
for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}✓ $name running on port $port${NC}"
    else
        echo -e "${RED}✗ $name NOT running on port $port${NC}"
        all_running=false
    fi
done

if $all_running; then
    echo -e "\n${GREEN}=== All Services Running Successfully ===${NC}"
    echo -e "${GREEN}Frontend: http://$(hostname -I | awk '{print $1}'):3001${NC}"
    echo -e "${GREEN}Logs available in: $LOG_DIR${NC}"
    exit 0
else
    echo -e "\n${RED}Some services failed to start. Check logs in $LOG_DIR${NC}"
    exit 1
fi