#!/bin/bash

echo "========================================="
echo "Solana MEV Infrastructure System Test"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Function to test a service
test_service() {
    local name=$1
    local command=$2
    local expected=$3
    
    echo -n "Testing $name... "
    
    if eval $command 2>/dev/null | grep -q "$expected" 2>/dev/null; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((TESTS_FAILED++))
    fi
}

# Function to check port
check_port() {
    local name=$1
    local port=$2
    
    echo -n "Checking $name (port $port)... "
    
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${GREEN}✓ ONLINE${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ OFFLINE${NC}"
        ((TESTS_FAILED++))
    fi
}

echo "1. Infrastructure Services"
echo "--------------------------"
check_port "Redis" 6379
check_port "ClickHouse HTTP" 8123
check_port "ClickHouse Native" 9000
check_port "Prometheus" 9090
check_port "Grafana" 3001
check_port "Kafka" 9092
check_port "Zookeeper" 2181

echo ""
echo "2. Docker Container Status"
echo "--------------------------"
CONTAINERS=$(sudo docker ps --format "{{.Names}}" 2>/dev/null)
for container in redis clickhouse prometheus grafana kafka zookeeper; do
    echo -n "Container $container... "
    if echo "$CONTAINERS" | grep -q "$container"; then
        STATUS=$(sudo docker ps --filter "name=$container" --format "{{.Status}}" | head -1)
        if echo "$STATUS" | grep -q "Up"; then
            echo -e "${GREEN}✓ Running${NC} ($STATUS)"
            ((TESTS_PASSED++))
        else
            echo -e "${YELLOW}⚠ Issue${NC} ($STATUS)"
            ((TESTS_FAILED++))
        fi
    else
        echo -e "${RED}✗ Not Found${NC}"
        ((TESTS_FAILED++))
    fi
done

echo ""
echo "3. Backend Services"
echo "-------------------"
echo -n "Checking Rust toolchain... "
if cargo --version >/dev/null 2>&1; then
    echo -e "${GREEN}✓ $(cargo --version)${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Not installed${NC}"
    ((TESTS_FAILED++))
fi

echo -n "Checking backend structure... "
if [ -f "/home/kidgordones/0solana/node/backend/Cargo.toml" ]; then
    echo -e "${GREEN}✓ Found${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Missing${NC}"
    ((TESTS_FAILED++))
fi

echo ""
echo "4. Frontend Services"
echo "--------------------"
echo -n "Checking Node.js... "
if node --version >/dev/null 2>&1; then
    echo -e "${GREEN}✓ $(node --version)${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Not installed${NC}"
    ((TESTS_FAILED++))
fi

echo -n "Checking frontend structure... "
if [ -f "/home/kidgordones/0solana/node/frontend2/package.json" ]; then
    echo -e "${GREEN}✓ Found${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Missing${NC}"
    ((TESTS_FAILED++))
fi

echo ""
echo "5. API Endpoints"
echo "----------------"
test_service "ClickHouse" "curl -s http://localhost:8123/" "Ok"
test_service "Prometheus" "curl -s http://localhost:9090/-/healthy" "Prometheus"
test_service "Grafana" "curl -s http://localhost:3001/api/health" "ok"

echo ""
echo "6. Database Connectivity"
echo "------------------------"
echo -n "Testing Redis connection... "
if redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo -e "${GREEN}✓ Connected${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Failed${NC}"
    ((TESTS_FAILED++))
fi

echo -n "Testing ClickHouse connection... "
if echo "SELECT 1" | curl -s "http://localhost:8123/" --data-binary @- 2>/dev/null | grep -q "1"; then
    echo -e "${GREEN}✓ Connected${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ Failed${NC}"
    ((TESTS_FAILED++))
fi

echo ""
echo "========================================="
echo "Test Results"
echo "========================================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo "The Solana MEV Infrastructure is ready to use."
    exit 0
else
    echo -e "${YELLOW}⚠️  Some tests failed. Please check the errors above.${NC}"
    echo ""
    echo "Quick fixes:"
    echo "1. For Docker permissions: sudo usermod -aG docker $USER && newgrp docker"
    echo "2. To start infrastructure: sudo make infra-up"
    echo "3. To check logs: sudo docker logs <container-name>"
    exit 1
fi