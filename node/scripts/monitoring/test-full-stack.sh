#!/bin/bash

#########################################################################
# Full Stack Integration Test Script
# Tests all components of the Solana MEV monitoring system
#########################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        SOLANA MEV MONITORING - FULL STACK TEST               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo

# Function to check service
check_service() {
    local name=$1
    local url=$2
    local type=${3:-"http"}
    
    echo -n "Testing $name... "
    
    if [ "$type" = "ws" ]; then
        if timeout 1 bash -c "cat < /dev/null > /dev/tcp/localhost/$url" 2>/dev/null; then
            echo -e "${GREEN}✓ ONLINE${NC}"
            return 0
        else
            echo -e "${RED}✗ OFFLINE${NC}"
            return 1
        fi
    else
        if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|301\|302"; then
            echo -e "${GREEN}✓ ONLINE${NC}"
            return 0
        else
            echo -e "${RED}✗ OFFLINE${NC}"
            return 1
        fi
    fi
}

# Backend Services
echo -e "${YELLOW}[1/5] Testing Backend Services${NC}"
echo "================================"
check_service "MEV Backend API" "http://localhost:8080/health"
check_service "API Gateway" "http://localhost:3000/health"
check_service "RPC Probe" "http://localhost:3001/health"
check_service "Validator Agent" "http://localhost:3002/health"
check_service "Jito Probe" "http://localhost:3003/health"
check_service "Control Service" "http://localhost:3004/health"
check_service "Prometheus" "http://localhost:9090/-/healthy"
echo

# Frontend Services
echo -e "${YELLOW}[2/5] Testing Frontend Services${NC}"
echo "================================"
check_service "Next.js Dashboard" "http://localhost:42391"
check_service "MEV Dashboard" "http://localhost:42391/mev"
check_service "Integration Bridge" "http://localhost:4000/health"
echo

# WebSocket Connections
echo -e "${YELLOW}[3/5] Testing WebSocket Connections${NC}"
echo "================================"
check_service "MEV WebSocket" "8080" "ws"
check_service "Gateway WebSocket" "3000" "ws"
check_service "Bridge WebSocket" "4000" "ws"
echo

# API Endpoints
echo -e "${YELLOW}[4/5] Testing API Endpoints${NC}"
echo "================================"

# Test MEV endpoints
echo -n "Testing MEV Bundle Submission... "
response=$(curl -s -X POST http://localhost:8080/api/bundle/simulate \
  -H "Content-Type: application/json" \
  -d '{"transactions":[],"tip":1000000}' 2>/dev/null || echo "error")
if [[ "$response" != "error" ]]; then
    echo -e "${GREEN}✓ WORKING${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

# Test monitoring endpoints
echo -n "Testing Node Status... "
response=$(curl -s http://localhost:3000/api/node/status 2>/dev/null || echo "error")
if [[ "$response" != "error" ]]; then
    echo -e "${GREEN}✓ WORKING${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

echo

# Performance Metrics
echo -e "${YELLOW}[5/5] Performance Metrics${NC}"
echo "================================"

# Check latency
echo -n "API Gateway Latency: "
latency=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:3000/health)
latency_ms=$(echo "$latency * 1000" | bc 2>/dev/null || echo "N/A")
if (( $(echo "$latency < 0.05" | bc -l 2>/dev/null || echo 0) )); then
    echo -e "${GREEN}${latency_ms}ms ✓${NC}"
else
    echo -e "${YELLOW}${latency_ms}ms${NC}"
fi

# Check memory usage
echo -n "System Memory Usage: "
mem_usage=$(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')
echo -e "${BLUE}${mem_usage}${NC}"

# Check CPU usage
echo -n "System CPU Usage: "
cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
echo -e "${BLUE}${cpu_usage}${NC}"

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Summary
total_services=13
working_services=0

# Count working services (simplified for demo)
working_services=$(( $(check_service "MEV Backend" "http://localhost:8080/health" > /dev/null 2>&1 && echo 1 || echo 0) + \
                    $(check_service "API Gateway" "http://localhost:3000/health" > /dev/null 2>&1 && echo 1 || echo 0) + \
                    $(check_service "Dashboard" "http://localhost:42391" > /dev/null 2>&1 && echo 1 || echo 0) ))

if [ $working_services -eq $total_services ]; then
    echo -e "${GREEN}✓ ALL SYSTEMS OPERATIONAL${NC}"
    echo -e "${GREEN}Your Solana MEV monitoring stack is fully functional!${NC}"
else
    echo -e "${YELLOW}⚠ Some services may need attention${NC}"
    echo -e "Run ${BLUE}./start-monitoring-stack.sh${NC} to start all services"
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo
echo -e "${GREEN}Access Points:${NC}"
echo -e "  Main Dashboard:    ${BLUE}http://localhost:42391${NC}"
echo -e "  MEV Dashboard:     ${BLUE}http://localhost:42391/mev${NC}"
echo -e "  API Gateway:       ${BLUE}http://localhost:3000${NC}"
echo -e "  Prometheus:        ${BLUE}http://localhost:9090${NC}"
echo
echo -e "${YELLOW}For detailed logs, run:${NC} ${BLUE}./start-monitoring-stack.sh logs${NC}"