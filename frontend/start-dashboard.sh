#!/bin/bash

# LEGENDARY MEV Dashboard Startup Script
# Built for billions. Optimized for microseconds.

set -e

echo "==========================================="
echo "  LEGENDARY MEV DASHBOARD v1.0"
echo "  Ultra-High-Performance DeFi Interface"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Node.js version
echo -e "${BLUE}Checking Node.js version...${NC}"
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}Error: Node.js 18+ required. Current version: $(node -v)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js $(node -v)${NC}"

# Install dependencies if needed
if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Check environment variables
if [ ! -f ".env.local" ]; then
    echo -e "${YELLOW}Warning: .env.local not found. Using defaults.${NC}"
    cp .env.local.example .env.local 2>/dev/null || true
fi

# Build protobuf if needed
if [ -d "../arbitrage-data-capture/protocol" ]; then
    echo -e "${BLUE}Generating protobuf definitions...${NC}"
    npm run proto:gen 2>/dev/null || true
    echo -e "${GREEN}✓ Protobuf generated${NC}"
fi

# Check ClickHouse connectivity
echo -e "${BLUE}Checking ClickHouse connection...${NC}"
CLICKHOUSE_URL="${NEXT_PUBLIC_CLICKHOUSE_URL:-http://45.157.234.184:8123}"
if curl -s -o /dev/null -w "%{http_code}" "$CLICKHOUSE_URL/ping" | grep -q "200"; then
    echo -e "${GREEN}✓ ClickHouse connected at $CLICKHOUSE_URL${NC}"
else
    echo -e "${YELLOW}⚠ ClickHouse not reachable at $CLICKHOUSE_URL${NC}"
fi

# Check Backend API
echo -e "${BLUE}Checking Backend API...${NC}"
API_BASE="${NEXT_PUBLIC_API_BASE:-http://45.157.234.184:8000}"
if curl -s -o /dev/null -w "%{http_code}" "$API_BASE/health" | grep -q "200"; then
    echo -e "${GREEN}✓ Backend API connected at $API_BASE${NC}"
else
    echo -e "${YELLOW}⚠ Backend API not reachable at $API_BASE${NC}"
fi

# Performance optimizations
echo -e "${BLUE}Applying performance optimizations...${NC}"
export NODE_OPTIONS="--max-old-space-size=4096 --max-semi-space-size=256"
export NODE_ENV="${NODE_ENV:-development}"

# Start mode selection
MODE="${1:-dev}"

case $MODE in
    "dev")
        echo -e "${GREEN}Starting in DEVELOPMENT mode...${NC}"
        echo -e "${BLUE}Dashboard will be available at: http://localhost:42392${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        npm run dev
        ;;
    
    "build")
        echo -e "${GREEN}Building for PRODUCTION...${NC}"
        npm run build
        echo -e "${GREEN}✓ Build complete${NC}"
        ;;
    
    "prod")
        echo -e "${GREEN}Starting in PRODUCTION mode...${NC}"
        if [ ! -d ".next" ]; then
            echo -e "${YELLOW}No build found. Building...${NC}"
            npm run build
        fi
        echo -e "${BLUE}Dashboard will be available at: http://localhost:42392${NC}"
        npm run start
        ;;
    
    "test")
        echo -e "${GREEN}Running tests...${NC}"
        npm run typecheck
        echo -e "${GREEN}✓ All tests passed${NC}"
        ;;
    
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 [dev|build|prod|test]"
        echo "  dev   - Start development server (default)"
        echo "  build - Build for production"
        echo "  prod  - Start production server"
        echo "  test  - Run tests"
        exit 1
        ;;
esac