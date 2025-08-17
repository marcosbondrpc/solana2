#!/bin/bash
# ⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱
# Solana Ultra-Performance Monitor

echo "═══════════════════════════════════════════════════════════════"
echo "     SOLANA NODE ULTRA-PERFORMANCE STATUS"
echo "     ⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# CPU Performance
echo "▶ CPU PERFORMANCE:"
echo "  Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
echo "  Active Cores: $(nproc)"
echo "  Load Average: $(uptime | awk -F'load average:' '{print $2}')"
echo ""

# Memory Status
echo "▶ MEMORY STATUS:"
free -h | grep -E "Mem:|Swap:"
echo "  Huge Pages: $(grep HugePages_Total /proc/meminfo | awk '{print $2}') allocated"
echo "  Huge Pages Free: $(grep HugePages_Free /proc/meminfo | awk '{print $2}')"
echo ""

# Network Latency - Fixed version
echo "▶ NETWORK LATENCY TO SOLANA:"
# Try multiple endpoints
ENDPOINTS=(
    "api.mainnet-beta.solana.com"
    "api.devnet.solana.com"
    "entrypoint.mainnet-beta.solana.com"
)

LATENCY_FOUND=false
for endpoint in "${ENDPOINTS[@]}"; do
    RESULT=$(ping -c 1 -W 1 $endpoint 2>/dev/null | grep -oP 'time=\K[0-9.]+' | head -1)
    if [ -n "$RESULT" ]; then
        echo "  $endpoint: ${RESULT}ms"
        LATENCY_FOUND=true
        break
    fi
done

if [ "$LATENCY_FOUND" = false ]; then
    # Try using curl to test RPC endpoints
    START=$(date +%s%N)
    curl -s -o /dev/null -w "" --max-time 2 https://api.mainnet-beta.solana.com/health 2>/dev/null
    END=$(date +%s%N)
    if [ $? -eq 0 ]; then
        LATENCY=$(( ($END - $START) / 1000000 ))
        echo "  RPC Health Check: ${LATENCY}ms"
    else
        echo "  Network check: Unable to reach Solana endpoints"
    fi
fi
echo ""

# Solana Process Status
echo "▶ SOLANA VALIDATOR STATUS:"
if pgrep -f agave-validator > /dev/null; then
    PID=$(pgrep -f agave-validator | head -1)
    echo "  Status: ✅ RUNNING (PID: $PID)"
    echo "  CPU Usage: $(ps -p $PID -o %cpu= | tr -d ' ')%"
    echo "  Memory Usage: $(ps -p $PID -o %mem= | tr -d ' ')%"
    
    # Get RSS memory in human readable
    RSS_KB=$(ps -p $PID -o rss= | tr -d ' ')
    RSS_GB=$(echo "scale=2; $RSS_KB / 1048576" | bc)
    echo "  Memory (RSS): ${RSS_GB}GB"
    
    echo "  Nice Level: $(ps -p $PID -o nice= | tr -d ' ')"
    
    # Check CPU affinity
    AFFINITY=$(taskset -pc $PID 2>/dev/null | grep -oP '\d+-\d+|\d+' | tail -1)
    echo "  CPU Affinity: $AFFINITY"
    
    # Check open file descriptors
    FD_COUNT=$(ls /proc/$PID/fd 2>/dev/null | wc -l)
    echo "  Open FDs: $FD_COUNT"
else
    echo "  Status: ❌ NOT RUNNING"
fi
echo ""

# RPC Performance Check
echo "▶ RPC PERFORMANCE:"
if pgrep -f agave-validator > /dev/null; then
    # Test local RPC
    START=$(date +%s%N)
    RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' \
        http://localhost:8899 2>/dev/null)
    END=$(date +%s%N)
    
    if echo "$RESPONSE" | grep -q "ok"; then
        LATENCY=$(( ($END - $START) / 1000000 ))
        echo "  Local RPC Health: ✅ OK (${LATENCY}ms)"
    else
        echo "  Local RPC Health: ❌ Not responding"
    fi
    
    # Get slot info
    SLOT_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","id":1,"method":"getSlot"}' \
        http://localhost:8899 2>/dev/null)
    
    if echo "$SLOT_RESPONSE" | grep -q "result"; then
        SLOT=$(echo "$SLOT_RESPONSE" | grep -oP '"result":\K[0-9]+')
        echo "  Current Slot: $SLOT"
    fi
else
    echo "  RPC Status: Validator not running"
fi
echo ""

# Storage Performance
echo "▶ STORAGE PERFORMANCE:"
for disk in /sys/block/nvme*/; do
    if [ -d "$disk" ]; then
        DISK_NAME=$(basename $disk)
        SCHEDULER=$(cat $disk/queue/scheduler 2>/dev/null | grep -oP '\[\K[^\]]+' || cat $disk/queue/scheduler 2>/dev/null || echo "none")
        NR_REQUESTS=$(cat $disk/queue/nr_requests 2>/dev/null || echo "N/A")
        READ_AHEAD=$(cat $disk/queue/read_ahead_kb 2>/dev/null || echo "N/A")
        echo "  $DISK_NAME: scheduler=$SCHEDULER, queue=$NR_REQUESTS, read_ahead=${READ_AHEAD}KB"
    fi
done

# Check ledger disk usage
if [ -d "/mnt/ledger" ]; then
    LEDGER_USAGE=$(df -h /mnt/ledger | tail -1 | awk '{print $5}')
    LEDGER_SIZE=$(du -sh /mnt/ledger 2>/dev/null | cut -f1)
    echo "  Ledger Usage: $LEDGER_USAGE ($LEDGER_SIZE)"
fi
echo ""

# Network Interface Status
echo "▶ NETWORK INTERFACE OPTIMIZATION:"
for iface in $(ip link show | grep "^[0-9]" | cut -d: -f2 | grep -v lo | head -2); do
    iface=$(echo $iface | tr -d ' ')
    if [ -n "$iface" ]; then
        # Get ring buffer settings
        RING_INFO=$(ethtool -g $iface 2>/dev/null)
        if [ $? -eq 0 ]; then
            RING_RX=$(echo "$RING_INFO" | grep "^RX:" | head -1 | awk '{print $2}')
            RING_TX=$(echo "$RING_INFO" | grep "^TX:" | head -1 | awk '{print $2}')
            echo "  $iface: RX=$RING_RX TX=$RING_TX"
        fi
        
        # Get interface statistics
        RX_BYTES=$(cat /sys/class/net/$iface/statistics/rx_bytes 2>/dev/null)
        TX_BYTES=$(cat /sys/class/net/$iface/statistics/tx_bytes 2>/dev/null)
        if [ -n "$RX_BYTES" ] && [ -n "$TX_BYTES" ]; then
            RX_GB=$(echo "scale=2; $RX_BYTES / 1073741824" | bc)
            TX_GB=$(echo "scale=2; $TX_BYTES / 1073741824" | bc)
            echo "    Traffic: RX=${RX_GB}GB TX=${TX_GB}GB"
        fi
    fi
done
echo ""

# System Tuning Status
echo "▶ SYSTEM TUNING:"
echo "  TCP Congestion: $(sysctl -n net.ipv4.tcp_congestion_control 2>/dev/null)"
echo "  Network Backlog: $(sysctl -n net.core.netdev_max_backlog 2>/dev/null)"
echo "  File Descriptors: $(sysctl -n fs.file-max 2>/dev/null)"
echo "  Swappiness: $(sysctl -n vm.swappiness 2>/dev/null)"
echo "  Socket Buffer (RX): $(sysctl -n net.core.rmem_max 2>/dev/null | numfmt --to=iec)"
echo "  Socket Buffer (TX): $(sysctl -n net.core.wmem_max 2>/dev/null | numfmt --to=iec)"
echo ""

# Performance Score
echo "▶ PERFORMANCE SCORE:"
SCORE=0
TOTAL=10

# Check optimizations
[ "$(sysctl -n net.ipv4.tcp_congestion_control 2>/dev/null)" = "bbr" ] && ((SCORE++))
[ "$(sysctl -n vm.swappiness 2>/dev/null)" -le "1" ] && ((SCORE++))
[ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" = "performance" ] && ((SCORE++))
[ "$(sysctl -n net.core.rmem_max 2>/dev/null)" -ge "134217728" ] && ((SCORE++))
[ "$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | grep -o "\[always\]")" = "[always]" ] && ((SCORE++))
pgrep -f agave-validator > /dev/null && ((SCORE++))
[ "$(sysctl -n net.core.netdev_max_backlog 2>/dev/null)" -ge "30000" ] && ((SCORE++))
[ -f "/etc/rc.local" ] && ((SCORE++))
[ -f "/usr/local/bin/apply-solana-optimizations" ] && ((SCORE++))
grep -q "mitigations=off" /etc/default/grub 2>/dev/null && ((SCORE++))

PERCENTAGE=$((SCORE * 100 / TOTAL))

if [ $PERCENTAGE -ge 90 ]; then
    echo "  🏆 TOP 1 PERFORMANCE: ${PERCENTAGE}% ($SCORE/$TOTAL optimizations)"
elif [ $PERCENTAGE -ge 70 ]; then
    echo "  ⚡ HIGH PERFORMANCE: ${PERCENTAGE}% ($SCORE/$TOTAL optimizations)"
else
    echo "  ⚠️  PERFORMANCE: ${PERCENTAGE}% ($SCORE/$TOTAL optimizations)"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
if [ $PERCENTAGE -ge 90 ]; then
    echo "🚀 TOP 1 NODE PERFORMANCE ACHIEVED!"
else
    echo "Performance optimizations applied. Reboot for full activation."
fi
echo "⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱"
echo "═══════════════════════════════════════════════════════════════"
