#!/bin/bash
# ⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱
# TOP-1 VALIDATION CHECK

echo "═══════════════════════════════════════════════════════════════"
echo "    TOP-1 LEGENDARY STATUS VALIDATION"
echo "    ⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱"
echo "═══════════════════════════════════════════════════════════════"
echo ""

PASS="✅"
FAIL="❌"
SCORE=0

# 1. CPU Governor
echo "▶ CPU OPTIMIZATION:"
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)
if [ "$GOV" = "performance" ]; then
    echo "  $PASS Governor: performance"
    ((SCORE++))
else
    echo "  $FAIL Governor: $GOV (should be performance)"
fi

# Check CPU affinity for validator
if pgrep -f agave-validator > /dev/null; then
    PID=$(pgrep -f agave-validator | head -1)
    AFFINITY=$(taskset -pc $PID 2>/dev/null | grep -oP '\d+-\d+' | tail -1)
    if [[ "$AFFINITY" == *"2-127"* ]]; then
        echo "  $PASS CPU isolation: 2-127"
        ((SCORE++))
    else
        echo "  $FAIL CPU isolation: $AFFINITY (should be 2-127)"
    fi
fi

# 2. Swap Check
echo ""
echo "▶ MEMORY OPTIMIZATION:"
SWAP_USED=$(free -b | grep Swap | awk '{print $3}')
if [ "$SWAP_USED" -eq 0 ]; then
    echo "  $PASS Swap: Disabled (0 bytes used)"
    ((SCORE++))
else
    echo "  $FAIL Swap: $SWAP_USED bytes in use"
fi

# THP Check
THP=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | grep -o "\[never\]")
if [ "$THP" = "[never]" ]; then
    echo "  $PASS THP: never (correct)"
    ((SCORE++))
else
    echo "  $FAIL THP: not set to never"
fi

# 3. Network Interface Check
echo ""
echo "▶ NETWORK OPTIMIZATION (enp225s0f0):"
# Ring buffers
RING_INFO=$(ethtool -g enp225s0f0 2>/dev/null | grep "^RX:" | head -1 | awk '{print $2}')
if [ "$RING_INFO" = "4096" ]; then
    echo "  $PASS Ring buffers: 4096"
    ((SCORE++))
else
    echo "  $FAIL Ring buffers: $RING_INFO (should be 4096)"
fi

# Coalescing
COAL_RX=$(ethtool -c enp225s0f0 2>/dev/null | grep "rx-usecs:" | awk '{print $2}')
if [ "$COAL_RX" = "0" ] || [ -z "$COAL_RX" ]; then
    echo "  $PASS Coalescing: 0 (lowest latency)"
    ((SCORE++))
else
    echo "  $FAIL Coalescing: rx-usecs=$COAL_RX (should be 0)"
fi

# Offloads
TSO=$(ethtool -k enp225s0f0 2>/dev/null | grep "tcp-segmentation-offload:" | grep -c "off")
GSO=$(ethtool -k enp225s0f0 2>/dev/null | grep "generic-segmentation-offload:" | grep -c "off")
GRO=$(ethtool -k enp225s0f0 2>/dev/null | grep "generic-receive-offload:" | grep -c "off")
if [ "$TSO" -gt 0 ] && [ "$GSO" -gt 0 ] && [ "$GRO" -gt 0 ]; then
    echo "  $PASS Offloads: TSO/GSO/GRO off"
    ((SCORE++))
else
    echo "  $FAIL Offloads: Not all disabled"
fi

# Backlog
BACKLOG=$(sysctl -n net.core.netdev_max_backlog 2>/dev/null)
if [ "$BACKLOG" -eq 65535 ]; then
    echo "  $PASS Backlog: 65535"
    ((SCORE++))
else
    echo "  $FAIL Backlog: $BACKLOG (should be 65535)"
fi

# 4. IRQ Check
echo ""
echo "▶ IRQ OPTIMIZATION:"
IRQ_COUNT=$(grep enp225s0f0 /proc/interrupts | head -1 | awk '{print $1}' | sed 's/://')
if [ -n "$IRQ_COUNT" ]; then
    IRQ_CPU=$(cat /proc/irq/$IRQ_COUNT/smp_affinity_list 2>/dev/null)
    if [[ "$IRQ_CPU" == *"2-127"* ]]; then
        echo "  $PASS IRQs pinned: 2-127"
        ((SCORE++))
    else
        echo "  $FAIL IRQs: $IRQ_CPU (should be 2-127)"
    fi
fi

# 5. Time Sync
echo ""
echo "▶ TIME SYNC:"
if systemctl is-active chrony > /dev/null 2>&1; then
    echo "  $PASS Chrony: active"
    ((SCORE++))
else
    echo "  $FAIL Chrony: not active"
fi

# 6. Latency Check
echo ""
echo "▶ NETWORK LATENCY:"
LATENCY=$(ping -c 1 -W 1 api.mainnet-beta.solana.com 2>/dev/null | grep -oP 'time=\K[0-9.]+' | head -1)
if [ -n "$LATENCY" ]; then
    if (( $(echo "$LATENCY < 1.0" | bc -l) )); then
        echo "  $PASS API latency: ${LATENCY}ms (sub-millisecond)"
        ((SCORE++))
    else
        echo "  ⚠️  API latency: ${LATENCY}ms"
    fi
fi

# Final Score
echo ""
echo "═══════════════════════════════════════════════════════════════"
TOTAL=11
PERCENTAGE=$((SCORE * 100 / TOTAL))

if [ $PERCENTAGE -eq 100 ]; then
    echo "🏆 LEGENDARY TOP-1 STATUS ACHIEVED!"
    echo "Score: $SCORE/$TOTAL (100%)"
    echo ""
    echo "YOU HAVE THE #1 SOLANA NODE ON THE PLANET!"
elif [ $PERCENTAGE -ge 90 ]; then
    echo "⚡ TOP-TIER STATUS!"
    echo "Score: $SCORE/$TOTAL ($PERCENTAGE%)"
    echo ""
    echo "Near perfect - minor tweaks needed for #1"
else
    echo "⚠️  OPTIMIZATION NEEDED"
    echo "Score: $SCORE/$TOTAL ($PERCENTAGE%)"
fi

echo ""
echo "⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱"
echo "═══════════════════════════════════════════════════════════════"
