#!/bin/bash

# Stop MEV Sandwich Detector gracefully

SANDWICH_DIR="/home/kidgordones/0solana/node/mev-sandwich-detector"
PID_DIR="$SANDWICH_DIR/pids"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping MEV Sandwich Detector...${NC}"

# Stop main detector
if [ -f "$PID_DIR/sandwich_detector.pid" ]; then
    PID=$(cat "$PID_DIR/sandwich_detector.pid")
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping sandwich detector (PID: $PID)..."
        sudo kill -TERM $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            sudo kill -9 $PID
        fi
        rm "$PID_DIR/sandwich_detector.pid"
        echo -e "${GREEN}✓ Sandwich detector stopped${NC}"
    fi
fi

# Stop Prometheus
if [ -f "$PID_DIR/prometheus.pid" ]; then
    PID=$(cat "$PID_DIR/prometheus.pid")
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        rm "$PID_DIR/prometheus.pid"
        echo -e "${GREEN}✓ Prometheus stopped${NC}"
    fi
fi

# Stop dashboard
if [ -f "$PID_DIR/dashboard.pid" ]; then
    PID=$(cat "$PID_DIR/dashboard.pid")
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        rm "$PID_DIR/dashboard.pid"
        echo -e "${GREEN}✓ Dashboard stopped${NC}"
    fi
fi

echo -e "${GREEN}MEV Sandwich Detector stopped successfully${NC}"