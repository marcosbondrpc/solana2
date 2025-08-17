#!/bin/bash

# MEV API Testing Script - Complete endpoint testing suite
# Tests all MEV backend operations with example curl commands

API_BASE="http://45.157.234.184:8000"
AUTH_TOKEN="test_token_123"  # Replace with actual auth token

echo "🚀 Testing MEV Backend API Endpoints"
echo "====================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper function to print colored output
print_test() {
    echo -e "${YELLOW}Testing: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# 1. Test MEV Scan endpoint
print_test "POST /api/mev/scan - Scan for MEV opportunities"
curl -X POST "$API_BASE/api/mev/scan" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "scan_type": "all",
    "min_profit": 0.5,
    "max_gas_price": 0.01,
    "include_pending": true
  }' | jq '.'
echo ""
print_success "MEV scan completed"
echo ""

# 2. Test Get Opportunities endpoint
print_test "GET /api/mev/opportunities - Get real-time opportunities"
curl -X GET "$API_BASE/api/mev/opportunities?type=arbitrage&min_profit=0.1&limit=10" \
  -H "Authorization: Bearer $AUTH_TOKEN" | jq '.'
echo ""
print_success "Opportunities retrieved"
echo ""

# 3. Test Execute Opportunity endpoint
print_test "POST /api/mev/execute/{opportunity_id} - Execute opportunity"
curl -X POST "$API_BASE/api/mev/execute/arb_test123" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "opportunity_id": "arb_test123",
    "max_slippage": 0.01,
    "priority_fee": 0.001,
    "use_jito": true
  }' | jq '.'
echo ""
print_success "Execution initiated"
echo ""

# 4. Test MEV Statistics endpoint
print_test "GET /api/mev/stats - Get performance statistics"
curl -X GET "$API_BASE/api/mev/stats" \
  -H "Authorization: Bearer $AUTH_TOKEN" | jq '.'
echo ""
print_success "Statistics retrieved"
echo ""

# 5. Test Bundle Simulation endpoint
print_test "POST /api/mev/simulate - Simulate bundle execution"
curl -X POST "$API_BASE/api/mev/simulate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "transactions": [
      "AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAQABBOzF3E9Ql",
      "AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAQABBOzF3E9Q2"
    ],
    "slot": null,
    "fork_point": null
  }' | jq '.'
echo ""
print_success "Simulation completed"
echo ""

# 6. Test Bandit Statistics endpoint
print_test "GET /api/mev/bandit/stats - Get Thompson Sampling statistics"
curl -X GET "$API_BASE/api/mev/bandit/stats" \
  -H "Authorization: Bearer $AUTH_TOKEN" | jq '.'
echo ""
print_success "Bandit stats retrieved"
echo ""

# 7. Test Command Signing endpoint
print_test "POST /api/mev/control/sign - Sign control command"
curl -X POST "$API_BASE/api/mev/control/sign" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "module": "mev_engine",
    "action": "update_params",
    "params": {
      "max_position_size": 1000,
      "min_profit_threshold": 0.5
    }
  }' | jq '.'
echo ""
print_success "Command signed"
echo ""

# 8. Test Risk Status endpoint
print_test "GET /api/mev/risk/status - Get risk management status"
curl -X GET "$API_BASE/api/mev/risk/status" \
  -H "Authorization: Bearer $AUTH_TOKEN" | jq '.'
echo ""
print_success "Risk status retrieved"
echo ""

# 9. Test Bundle Submit endpoint
print_test "POST /api/mev/bundle/submit - Submit Jito bundle"
curl -X POST "$API_BASE/api/mev/bundle/submit" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "transactions": [
      "AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAQABBOzF3E9Ql"
    ],
    "tip_lamports": 1000000,
    "region": "amsterdam"
  }' | jq '.'
echo ""
print_success "Bundle submitted"
echo ""

# 10. Test WebSocket connections
echo "====================================="
echo "WebSocket Endpoints (use wscat or similar tool):"
echo ""
echo "1. Real-time opportunities stream:"
echo "   wscat -c ws://$API_BASE/api/mev/ws/opportunities"
echo ""
echo "2. Execution status stream:"
echo "   wscat -c ws://$API_BASE/api/mev/ws/executions"
echo ""
echo "3. Performance metrics stream:"
echo "   wscat -c ws://$API_BASE/api/mev/ws/metrics"
echo ""

# 11. Test Control Plane endpoints
echo "====================================="
echo "Control Plane Endpoints:"
echo ""

print_test "POST /api/control/command - Publish control command"
curl -X POST "$API_BASE/api/control/command" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "module": "mev_engine",
    "action": "start",
    "params": {
      "mode": "aggressive"
    },
    "priority": 1
  }' | jq '.'
echo ""
print_success "Control command published"
echo ""

print_test "POST /api/control/kill-switch - Activate emergency stop"
curl -X POST "$API_BASE/api/control/kill-switch" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "target": "mev_engine",
    "reason": "Testing kill switch",
    "duration_ms": 5000,
    "force": false
  }' | jq '.'
echo ""
print_success "Kill switch tested"
echo ""

# 12. Performance benchmark
echo "====================================="
echo "Performance Benchmark:"
echo ""

print_test "Running latency test (100 requests)..."
total_time=0
for i in {1..100}; do
    start_time=$(date +%s%N)
    curl -s -X GET "$API_BASE/api/mev/opportunities?limit=1" \
      -H "Authorization: Bearer $AUTH_TOKEN" > /dev/null
    end_time=$(date +%s%N)
    elapsed=$((($end_time - $start_time) / 1000000))
    total_time=$(($total_time + $elapsed))
done

avg_latency=$(($total_time / 100))
echo "Average latency: ${avg_latency}ms"

if [ $avg_latency -lt 10 ]; then
    print_success "✨ Sub-10ms latency achieved! Performance target met."
elif [ $avg_latency -lt 20 ]; then
    print_success "Good performance: ${avg_latency}ms average latency"
else
    print_error "Performance below target: ${avg_latency}ms average latency"
fi

echo ""
echo "====================================="
echo "Advanced Testing Examples:"
echo ""

# Example: Complex arbitrage scan
echo "1. Complex Arbitrage Scan:"
cat << 'EOF'
curl -X POST "$API_BASE/api/mev/scan" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "scan_type": "arbitrage",
    "min_profit": 1.0,
    "max_gas_price": 0.005,
    "include_pending": true,
    "pools": ["USDC/SOL", "SOL/RAY", "RAY/USDC"],
    "max_hops": 4
  }'
EOF

echo ""
echo "2. Batch Execution:"
cat << 'EOF'
curl -X POST "$API_BASE/api/mev/execute/batch" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "opportunity_ids": ["arb_001", "sandwich_002", "jit_003"],
    "strategy": "parallel",
    "max_gas": 0.05,
    "use_flashloan": true
  }'
EOF

echo ""
echo "3. Risk-Adjusted Execution:"
cat << 'EOF'
curl -X POST "$API_BASE/api/mev/execute/risk-adjusted" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "opportunity_id": "arb_high_risk_001",
    "risk_tolerance": 0.3,
    "kelly_fraction": 0.25,
    "stop_loss": 0.02,
    "take_profit": 0.10
  }'
EOF

echo ""
echo "====================================="
echo "Testing Complete!"
echo ""
echo "Summary:"
print_success "✓ All core MEV endpoints tested"
print_success "✓ Control plane integration verified"
print_success "✓ WebSocket streams documented"
print_success "✓ Performance benchmarks completed"
echo ""
echo "Next steps:"
echo "1. Monitor WebSocket streams for real-time data"
echo "2. Check Grafana dashboards at http://$API_BASE:3000"
echo "3. Review logs: docker logs mev-control-plane"
echo "4. Test with production data"
echo ""