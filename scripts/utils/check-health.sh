#!/bin/bash
# Solana Node Health Check Utility

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

RPC_URL="${RPC_URL:-http://127.0.0.1:8899}"

echo "======================================"
echo "   Solana Node Health Check"
echo "======================================"
echo ""

# Check if node is responding
echo -n "Checking RPC endpoint... "
if curl -s -X POST -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' \
    $RPC_URL > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Online${NC}"
else
    echo -e "${RED}✗ Offline${NC}"
    exit 1
fi

# Get node version
echo -n "Node version: "
VERSION=$(curl -s -X POST -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"getVersion"}' \
    $RPC_URL | jq -r .result.version 2>/dev/null || echo "Unknown")
echo "$VERSION"

# Get current slot
echo -n "Current slot: "
SLOT=$(curl -s -X POST -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"getSlot"}' \
    $RPC_URL | jq -r .result 2>/dev/null || echo "Unknown")
echo "$SLOT"

# Get block height
echo -n "Block height: "
HEIGHT=$(curl -s -X POST -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"getBlockHeight"}' \
    $RPC_URL | jq -r .result 2>/dev/null || echo "Unknown")
echo "$HEIGHT"

# Check sync status
echo -n "Sync status: "
HEALTH=$(curl -s -X POST -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' \
    $RPC_URL 2>/dev/null)
if echo "$HEALTH" | grep -q '"result":"ok"'; then
    echo -e "${GREEN}✓ Synced${NC}"
else
    echo -e "${YELLOW}⚠ Syncing...${NC}"
fi

echo ""
echo "======================================"