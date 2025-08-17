#!/bin/bash

# MEV Network Status Monitor
# Shows real-time network performance metrics

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Detect primary interface
PRIMARY_IFACE=$(ip route | grep default | awk '{print $5}' | head -n1)

clear

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    MEV Network Performance Monitor                    ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo

# Function to format numbers
format_number() {
    printf "%'d" $1
}

# Network Interface Status
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Network Interface: ${GREEN}$PRIMARY_IFACE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Get interface statistics
if [ -f "/sys/class/net/$PRIMARY_IFACE/statistics/rx_packets" ]; then
    RX_PACKETS=$(cat /sys/class/net/$PRIMARY_IFACE/statistics/rx_packets)
    TX_PACKETS=$(cat /sys/class/net/$PRIMARY_IFACE/statistics/tx_packets)
    RX_BYTES=$(cat /sys/class/net/$PRIMARY_IFACE/statistics/rx_bytes)
    TX_BYTES=$(cat /sys/class/net/$PRIMARY_IFACE/statistics/tx_bytes)
    RX_DROPPED=$(cat /sys/class/net/$PRIMARY_IFACE/statistics/rx_dropped)
    TX_DROPPED=$(cat /sys/class/net/$PRIMARY_IFACE/statistics/tx_dropped)
    
    echo -e "${GREEN}Interface Statistics:${NC}"
    echo -e "  RX Packets: $(format_number $RX_PACKETS) | TX Packets: $(format_number $TX_PACKETS)"
    echo -e "  RX Bytes:   $(format_number $RX_BYTES) | TX Bytes:   $(format_number $TX_BYTES)"
    
    if [ "$RX_DROPPED" -gt 0 ] || [ "$TX_DROPPED" -gt 0 ]; then
        echo -e "  ${YELLOW}⚠ Dropped:${NC} RX: $RX_DROPPED | TX: $TX_DROPPED"
    else
        echo -e "  ${GREEN}✓ No dropped packets${NC}"
    fi
fi

echo

# IRQ Affinity Status
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}IRQ Affinity Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check IRQ affinity for network interface
IRQS=$(grep "$PRIMARY_IFACE" /proc/interrupts | awk '{print $1}' | sed 's/://' | head -5)
if [ -n "$IRQS" ]; then
    echo -e "${GREEN}Network IRQs and CPU Affinity:${NC}"
    for irq in $IRQS; do
        if [ -f "/proc/irq/$irq/smp_affinity_list" ]; then
            AFFINITY=$(cat /proc/irq/$irq/smp_affinity_list 2>/dev/null || echo "N/A")
            echo -e "  IRQ $irq → CPUs: ${GREEN}$AFFINITY${NC}"
        fi
    done
else
    echo -e "${YELLOW}No specific IRQs found for $PRIMARY_IFACE${NC}"
fi

echo

# NIC Configuration
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}NIC Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Queue configuration
echo -e "${GREEN}Queue Configuration:${NC}"
QUEUES=$(ethtool -l "$PRIMARY_IFACE" 2>/dev/null | grep "Combined:" | tail -1 | awk '{print $2}')
MAX_QUEUES=$(ethtool -l "$PRIMARY_IFACE" 2>/dev/null | grep "Combined:" | head -1 | awk '{print $2}')
if [ -n "$QUEUES" ]; then
    echo -e "  Current Queues: ${GREEN}$QUEUES${NC} / Max: $MAX_QUEUES"
else
    echo -e "  ${YELLOW}Unable to query queue configuration${NC}"
fi

# Interrupt coalescing
echo -e "${GREEN}Interrupt Coalescing:${NC}"
RX_USECS=$(ethtool -c "$PRIMARY_IFACE" 2>/dev/null | grep "rx-usecs:" | awk '{print $2}')
RX_FRAMES=$(ethtool -c "$PRIMARY_IFACE" 2>/dev/null | grep "rx-frames:" | awk '{print $2}')
if [ -n "$RX_USECS" ]; then
    echo -e "  RX Usecs: ${GREEN}$RX_USECS${NC} | RX Frames: ${GREEN}$RX_FRAMES${NC}"
else
    echo -e "  ${YELLOW}Unable to query coalescing settings${NC}"
fi

# Offload status
echo -e "${GREEN}Offload Status:${NC}"
GRO=$(ethtool -k "$PRIMARY_IFACE" 2>/dev/null | grep "generic-receive-offload:" | awk '{print $2}')
LRO=$(ethtool -k "$PRIMARY_IFACE" 2>/dev/null | grep "large-receive-offload:" | awk '{print $2}')
TSO=$(ethtool -k "$PRIMARY_IFACE" 2>/dev/null | grep "tcp-segmentation-offload:" | awk '{print $2}')
GSO=$(ethtool -k "$PRIMARY_IFACE" 2>/dev/null | grep "generic-segmentation-offload:" | awk '{print $2}')

if [ -n "$GRO" ]; then
    [ "$GRO" = "off" ] && GRO="${GREEN}off ✓${NC}" || GRO="${YELLOW}on ⚠${NC}"
    [ "$LRO" = "off" ] && LRO="${GREEN}off ✓${NC}" || LRO="${YELLOW}on ⚠${NC}"
    [ "$TSO" = "off" ] && TSO="${GREEN}off ✓${NC}" || TSO="${YELLOW}on ⚠${NC}"
    [ "$GSO" = "off" ] && GSO="${GREEN}off ✓${NC}" || GSO="${YELLOW}on ⚠${NC}"
    
    echo -e "  GRO: $GRO | LRO: $LRO | TSO: $TSO | GSO: $GSO"
else
    echo -e "  ${YELLOW}Unable to query offload settings${NC}"
fi

echo

# Kernel Settings
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Kernel Network Settings${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Socket buffers
echo -e "${GREEN}Socket Buffers:${NC}"
RMEM_MAX=$(sysctl -n net.core.rmem_max 2>/dev/null)
WMEM_MAX=$(sysctl -n net.core.wmem_max 2>/dev/null)
echo -e "  RX Buffer: $(format_number $RMEM_MAX) bytes ($(($RMEM_MAX / 1048576))MB)"
echo -e "  TX Buffer: $(format_number $WMEM_MAX) bytes ($(($WMEM_MAX / 1048576))MB)"

# Busy polling
echo -e "${GREEN}Busy Polling:${NC}"
BUSY_POLL=$(sysctl -n net.core.busy_poll 2>/dev/null)
BUSY_READ=$(sysctl -n net.core.busy_read 2>/dev/null)
echo -e "  Busy Poll: ${GREEN}${BUSY_POLL}μs${NC} | Busy Read: ${GREEN}${BUSY_READ}μs${NC}"

# UDP settings
echo -e "${GREEN}UDP Settings:${NC}"
UDP_RMEM=$(sysctl -n net.ipv4.udp_rmem_min 2>/dev/null)
UDP_WMEM=$(sysctl -n net.ipv4.udp_wmem_min 2>/dev/null)
echo -e "  UDP RX Min: $(format_number $UDP_RMEM) | UDP TX Min: $(format_number $UDP_WMEM)"

# TCP congestion control
echo -e "${GREEN}TCP Settings:${NC}"
CONGESTION=$(sysctl -n net.ipv4.tcp_congestion_control 2>/dev/null)
QDISC=$(sysctl -n net.core.default_qdisc 2>/dev/null)
echo -e "  Congestion Control: ${GREEN}$CONGESTION${NC} | Queue Discipline: ${GREEN}$QDISC${NC}"

echo

# Time Synchronization
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Time Synchronization Status${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check Chrony status
if systemctl is-active --quiet chrony; then
    echo -e "${GREEN}Chrony Status: Active ✓${NC}"
    
    # Get tracking info
    TRACKING=$(chronyc tracking 2>/dev/null)
    if [ -n "$TRACKING" ]; then
        OFFSET=$(echo "$TRACKING" | grep "System time" | awk '{print $4, $5}')
        RMS=$(echo "$TRACKING" | grep "RMS offset" | awk '{print $4, $5}')
        FREQ=$(echo "$TRACKING" | grep "Frequency" | awk '{print $3, $4, $5}')
        
        echo -e "  System Time Offset: ${GREEN}$OFFSET${NC}"
        echo -e "  RMS Offset: ${GREEN}$RMS${NC}"
        echo -e "  Frequency: $FREQ"
    fi
else
    echo -e "${YELLOW}Chrony: Inactive ⚠${NC}"
fi

# Check PTP status
if systemctl is-active --quiet ptp4l 2>/dev/null; then
    echo -e "${GREEN}PTP Status: Active ✓${NC}"
    
    # Check for hardware timestamping
    if [ -e /dev/ptp0 ]; then
        echo -e "  Hardware Clock: ${GREEN}/dev/ptp0 detected ✓${NC}"
    fi
fi

echo

# CPU Performance
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}CPU Performance Settings${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# CPU Governor
GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "N/A")
if [ "$GOVERNOR" = "performance" ]; then
    echo -e "${GREEN}CPU Governor: performance ✓${NC}"
else
    echo -e "${YELLOW}CPU Governor: $GOVERNOR ⚠${NC}"
fi

# CPU frequency
if [ -f "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq" ]; then
    FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq)
    FREQ_GHZ=$(echo "scale=2; $FREQ / 1000000" | bc)
    echo -e "CPU Frequency: ${GREEN}${FREQ_GHZ} GHz${NC}"
fi

# Huge pages
HUGEPAGES=$(sysctl -n vm.nr_hugepages 2>/dev/null)
echo -e "Huge Pages: ${GREEN}$HUGEPAGES${NC}"

echo

# Performance Score
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}MEV Readiness Score${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

SCORE=0
MAX_SCORE=10

# Check each optimization
[ "$QUEUES" = "16" ] && ((SCORE++)) && echo -e "${GREEN}✓${NC} NIC queues optimized" || echo -e "${YELLOW}✗${NC} NIC queues not optimal (current: $QUEUES, target: 16)"
[ "$RX_USECS" = "0" ] && ((SCORE++)) && echo -e "${GREEN}✓${NC} Interrupt coalescing optimized" || echo -e "${YELLOW}✗${NC} Interrupt coalescing not optimal"
[ "$GRO" = "${GREEN}off ✓${NC}" ] && ((SCORE++)) && echo -e "${GREEN}✓${NC} Offloads disabled" || echo -e "${YELLOW}✗${NC} Offloads not optimal"
[ "$RMEM_MAX" -ge 33554432 ] && ((SCORE++)) && echo -e "${GREEN}✓${NC} Socket buffers optimized" || echo -e "${YELLOW}✗${NC} Socket buffers not optimal"
[ "$BUSY_POLL" = "50" ] && ((SCORE++)) && echo -e "${GREEN}✓${NC} Busy polling enabled" || echo -e "${YELLOW}✗${NC} Busy polling not optimal"
[ "$CONGESTION" = "bbr" ] && ((SCORE++)) && echo -e "${GREEN}✓${NC} BBR congestion control" || echo -e "${YELLOW}✗${NC} Congestion control not optimal"
systemctl is-active --quiet chrony && ((SCORE++)) && echo -e "${GREEN}✓${NC} Time sync active" || echo -e "${YELLOW}✗${NC} Time sync not active"
[ "$GOVERNOR" = "performance" ] && ((SCORE++)) && echo -e "${GREEN}✓${NC} CPU performance mode" || echo -e "${YELLOW}✗${NC} CPU not in performance mode"
[ "$HUGEPAGES" -ge 1024 ] && ((SCORE++)) && echo -e "${GREEN}✓${NC} Huge pages configured" || echo -e "${YELLOW}✗${NC} Huge pages not optimal"
[ "$RX_DROPPED" = "0" ] && [ "$TX_DROPPED" = "0" ] && ((SCORE++)) && echo -e "${GREEN}✓${NC} No packet drops" || echo -e "${YELLOW}✗${NC} Packet drops detected"

echo
PERCENTAGE=$((SCORE * 100 / MAX_SCORE))

if [ $PERCENTAGE -ge 90 ]; then
    COLOR=$GREEN
    STATUS="OPTIMAL"
elif [ $PERCENTAGE -ge 70 ]; then
    COLOR=$YELLOW
    STATUS="GOOD"
else
    COLOR=$RED
    STATUS="NEEDS OPTIMIZATION"
fi

echo -e "${COLOR}Overall Score: $SCORE/$MAX_SCORE ($PERCENTAGE%) - $STATUS${NC}"

echo
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Run 'sudo /home/kidgordones/0solana/node/scripts/optimize/network-optimization.sh'${NC}"
echo -e "${CYAN}to apply optimizations if score is not optimal.${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"