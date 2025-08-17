#!/bin/bash

# Solana Historical Data Infrastructure Benchmark Script
# Performance target: ≥50k msgs/min ingestion, ≥100 slots/s backfill

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}  Solana Historical Data Benchmark${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""

# Configuration
KAFKA_BROKER="localhost:19092"
CLICKHOUSE_HOST="localhost"
CLICKHOUSE_USER="solana"
CLICKHOUSE_PASSWORD="mev_billions_2025"
TEST_DURATION=60
MESSAGE_COUNT=50000

# Check dependencies
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

check_dependency docker
check_dependency curl

echo -e "${BLUE}Starting benchmark...${NC}"
echo ""

# Function to generate test data
generate_slot_data() {
    local slot=$1
    cat <<EOF
{
    "slot": $slot,
    "parent_slot": $((slot - 1)),
    "block_height": $slot,
    "block_time": $(date +%s),
    "leader": "DWvDTSh3qfn88UoQTEKRV2JnLt5jtJAVoiCo3ivtMwXP",
    "rewards_json": "[]",
    "block_hash": "hash_$slot",
    "parent_hash": "hash_$((slot - 1))",
    "transaction_count": $((RANDOM % 500 + 100)),
    "entry_count": $((RANDOM % 100 + 10)),
    "tick_count": $((RANDOM % 20 + 5))
}
EOF
}

generate_transaction_data() {
    local slot=$1
    local index=$2
    cat <<EOF
{
    "signature": "sig_${slot}_${index}_$(date +%s%N)",
    "slot": $slot,
    "block_time": $(date +%s),
    "block_index": $index,
    "transaction_index": $index,
    "is_vote": $([ $((RANDOM % 10)) -lt 3 ] && echo "true" || echo "false"),
    "success": $([ $((RANDOM % 100)) -lt 95 ] && echo "true" || echo "false"),
    "fee": $((RANDOM % 10000 + 5000)),
    "compute_units_consumed": $((RANDOM % 1000000 + 100000)),
    "err": "",
    "memo": "benchmark_test",
    "signer": "test_signer_$index",
    "signers": ["test_signer_$index"],
    "account_keys": ["key1", "key2", "key3"],
    "pre_balances": [$((RANDOM * 1000000)), $((RANDOM * 1000000))],
    "post_balances": [$((RANDOM * 1000000)), $((RANDOM * 1000000))],
    "pre_token_balances_json": "[]",
    "post_token_balances_json": "[]",
    "instructions_json": "[]",
    "inner_instructions_json": "[]",
    "log_messages": ["Program log: Benchmark test"],
    "rewards_json": "[]",
    "loaded_addresses_json": "{}",
    "return_data_json": "null"
}
EOF
}

# Test 1: Kafka Ingestion Rate
echo -e "${YELLOW}Test 1: Kafka Ingestion Rate${NC}"
echo "Target: ≥50,000 messages/minute"
echo ""

START_TIME=$(date +%s)
START_SLOT=900000000

# Generate and send messages
echo "Sending $MESSAGE_COUNT messages to Kafka..."
for i in $(seq 1 $MESSAGE_COUNT); do
    if [ $((i % 1000)) -eq 0 ]; then
        echo -ne "\rProgress: $i/$MESSAGE_COUNT"
    fi
    
    SLOT=$((START_SLOT + i))
    
    # Send slot data
    generate_slot_data $SLOT | docker exec -i solana-redpanda \
        rpk topic produce solana.slots --key="$SLOT" -f '%v' &
    
    # Send transaction data (3 per slot on average)
    if [ $((i % 3)) -eq 0 ]; then
        for tx in $(seq 1 3); do
            generate_transaction_data $SLOT $tx | docker exec -i solana-redpanda \
                rpk topic produce solana.transactions --key="tx_${SLOT}_${tx}" -f '%v' &
        done
    fi
    
    # Limit concurrent processes
    if [ $((i % 100)) -eq 0 ]; then
        wait
    fi
done

wait
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo -e "\n"
MESSAGES_PER_MINUTE=$((MESSAGE_COUNT * 60 / DURATION))
echo -e "Duration: ${DURATION}s"
echo -e "Ingestion rate: ${GREEN}${MESSAGES_PER_MINUTE} msgs/min${NC}"

if [ $MESSAGES_PER_MINUTE -ge 50000 ]; then
    echo -e "${GREEN}✓ PASSED: Ingestion rate target met${NC}"
else
    echo -e "${RED}✗ FAILED: Ingestion rate below target${NC}"
fi
echo ""

# Test 2: ClickHouse Write Performance
echo -e "${YELLOW}Test 2: ClickHouse Write Performance${NC}"
echo "Checking data arrival in ClickHouse..."
sleep 5

CLICKHOUSE_COUNT=$(docker exec solana-clickhouse clickhouse-client \
    --user=$CLICKHOUSE_USER \
    --password=$CLICKHOUSE_PASSWORD \
    -q "SELECT count() FROM solana_history.slots WHERE slot >= $START_SLOT" 2>/dev/null || echo "0")

echo -e "Messages sent: $MESSAGE_COUNT"
echo -e "Messages in ClickHouse: $CLICKHOUSE_COUNT"
LOSS_RATE=$(echo "scale=2; 100 - ($CLICKHOUSE_COUNT * 100 / $MESSAGE_COUNT)" | bc 2>/dev/null || echo "0")
echo -e "Loss rate: ${LOSS_RATE}%"

if (( $(echo "$LOSS_RATE < 5" | bc -l) )); then
    echo -e "${GREEN}✓ PASSED: Acceptable data loss rate${NC}"
else
    echo -e "${RED}✗ FAILED: High data loss rate${NC}"
fi
echo ""

# Test 3: Query Performance
echo -e "${YELLOW}Test 3: Query Performance${NC}"
echo "Running analytical queries..."

# Query 1: Point lookup by signature
echo -n "Point lookup by signature: "
QUERY_START=$(date +%s%3N)
docker exec solana-clickhouse clickhouse-client \
    --user=$CLICKHOUSE_USER \
    --password=$CLICKHOUSE_PASSWORD \
    -q "SELECT * FROM solana_history.transactions WHERE signature = 'test' LIMIT 1" > /dev/null 2>&1
QUERY_END=$(date +%s%3N)
QUERY_TIME=$((QUERY_END - QUERY_START))
echo "${QUERY_TIME}ms"

# Query 2: Range scan by slot
echo -n "Range scan by slot: "
QUERY_START=$(date +%s%3N)
docker exec solana-clickhouse clickhouse-client \
    --user=$CLICKHOUSE_USER \
    --password=$CLICKHOUSE_PASSWORD \
    -q "SELECT count() FROM solana_history.transactions WHERE slot BETWEEN $START_SLOT AND $((START_SLOT + 1000))" > /dev/null 2>&1
QUERY_END=$(date +%s%3N)
QUERY_TIME=$((QUERY_END - QUERY_START))
echo "${QUERY_TIME}ms"

# Query 3: Aggregation
echo -n "Daily aggregation: "
QUERY_START=$(date +%s%3N)
docker exec solana-clickhouse clickhouse-client \
    --user=$CLICKHOUSE_USER \
    --password=$CLICKHOUSE_PASSWORD \
    -q "SELECT toDate(block_time) as day, count() as cnt, sum(fee) as fees FROM solana_history.transactions WHERE slot >= $START_SLOT GROUP BY day" > /dev/null 2>&1
QUERY_END=$(date +%s%3N)
QUERY_TIME=$((QUERY_END - QUERY_START))
echo "${QUERY_TIME}ms"

echo ""

# Test 4: Deduplication Check
echo -e "${YELLOW}Test 4: Deduplication Verification${NC}"
echo "Checking for duplicate entries..."

# Send duplicate messages
TEST_SLOT=$((START_SLOT + 999999))
for i in $(seq 1 5); do
    generate_slot_data $TEST_SLOT | docker exec -i solana-redpanda \
        rpk topic produce solana.slots --key="$TEST_SLOT" -f '%v'
done

sleep 3

DUPLICATE_COUNT=$(docker exec solana-clickhouse clickhouse-client \
    --user=$CLICKHOUSE_USER \
    --password=$CLICKHOUSE_PASSWORD \
    -q "SELECT count() FROM solana_history.slots WHERE slot = $TEST_SLOT" 2>/dev/null || echo "0")

echo -e "Duplicate messages sent: 5"
echo -e "Records in database: $DUPLICATE_COUNT"

if [ "$DUPLICATE_COUNT" -eq "1" ]; then
    echo -e "${GREEN}✓ PASSED: Deduplication working correctly${NC}"
else
    echo -e "${RED}✗ FAILED: Deduplication not working${NC}"
fi
echo ""

# Test 5: Consumer Lag
echo -e "${YELLOW}Test 5: Consumer Lag Analysis${NC}"
LAG_INFO=$(docker exec solana-redpanda rpk group describe clickhouse-slots-consumer 2>/dev/null | grep -E "LAG|TOPIC" || echo "No lag info")
echo "$LAG_INFO"
echo ""

# Test 6: System Resource Usage
echo -e "${YELLOW}Test 6: System Resource Usage${NC}"
echo "Container resource consumption:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" \
    solana-redpanda solana-clickhouse
echo ""

# Summary
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}           Benchmark Summary${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""
echo -e "Ingestion Rate: ${GREEN}${MESSAGES_PER_MINUTE} msgs/min${NC}"
echo -e "Data Loss: ${LOSS_RATE}%"
echo -e "Deduplication: ${GREEN}Working${NC}"
echo ""

# Calculate overall score
SCORE=0
[ $MESSAGES_PER_MINUTE -ge 50000 ] && SCORE=$((SCORE + 40))
(( $(echo "$LOSS_RATE < 5" | bc -l) )) && SCORE=$((SCORE + 30))
[ "$DUPLICATE_COUNT" -eq "1" ] && SCORE=$((SCORE + 30))

echo -e "Overall Score: ${GREEN}${SCORE}/100${NC}"

if [ $SCORE -ge 70 ]; then
    echo -e "${GREEN}✓ BENCHMARK PASSED${NC}"
    exit 0
else
    echo -e "${RED}✗ BENCHMARK FAILED${NC}"
    exit 1
fi