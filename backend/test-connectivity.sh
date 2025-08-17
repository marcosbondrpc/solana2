#!/bin/bash

# Test Backend MEV Services Connectivity

echo "Testing Backend MEV Services..."
echo "================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Test Main Service
echo -n "Testing Main Service (8080)... "
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

# Test Metrics
echo -n "Testing Metrics (9090)... "
if curl -s http://localhost:9090/metrics > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

# Test API Gateway
echo -n "Testing API Gateway (3000)... "
if curl -k -s https://localhost:3000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

# Test WebSocket
echo -n "Testing WebSocket (3000/ws)... "
if timeout 1 bash -c 'cat < /dev/null > /dev/tcp/localhost/3000' 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
fi

echo "================================"
echo "Backend connectivity test complete"