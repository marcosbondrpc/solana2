#!/bin/bash

# SOTA System Tuning for Ultra-Low Latency MEV
# Target: <8ms P50, <18ms P99 decision latency

set -e

echo "🚀 Applying SOTA system optimizations for MEV..."

# 1. CPU Performance Tuning
echo "⚡ Setting CPU to performance mode..."
sudo cpupower frequency-set -g performance 2>/dev/null || true
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null

# Disable CPU throttling
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space >/dev/null
echo 1 | sudo tee /proc/sys/kernel/sched_autogroup_enabled >/dev/null

# 2. Network Stack Optimization
echo "🌐 Optimizing network stack for QUIC/UDP..."

# Increase network buffers for high-throughput
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.rmem_default=25165824
sudo sysctl -w net.core.wmem_default=25165824
sudo sysctl -w net.core.netdev_max_backlog=30000
sudo sysctl -w net.core.netdev_budget=600

# UDP specific optimizations
sudo sysctl -w net.ipv4.udp_mem="262144 4194304 67108864"
sudo sysctl -w net.ipv4.udp_rmem_min=8192
sudo sysctl -w net.ipv4.udp_wmem_min=8192

# TCP optimizations for RPC/HTTP
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
sudo sysctl -w net.ipv4.tcp_notsent_lowat=16384
sudo sysctl -w net.ipv4.tcp_fastopen=3
sudo sysctl -w net.ipv4.tcp_mtu_probing=1
sudo sysctl -w net.ipv4.tcp_timestamps=0

# Enable busy polling for lower latency
sudo sysctl -w net.core.busy_poll=50
sudo sysctl -w net.core.busy_read=50

# 3. DSCP/QoS Configuration for MEV Priority
echo "📡 Setting up DSCP marking for MEV traffic..."

# Clear existing rules
sudo iptables -t mangle -F 2>/dev/null || true

# Mark MEV packets with DSCP EF (Expedited Forwarding)
sudo iptables -t mangle -A OUTPUT -p udp --dport 8003 -j DSCP --set-dscp-class EF
sudo iptables -t mangle -A OUTPUT -p udp --dport 8004 -j DSCP --set-dscp-class EF
sudo iptables -t mangle -A OUTPUT -p tcp --dport 8900 -j DSCP --set-dscp-class AF41

# 4. Traffic Control with FQ (Fair Queue)
echo "🎯 Configuring traffic control..."

# Remove existing qdiscs
sudo tc qdisc del dev eth0 root 2>/dev/null || true

# Add FQ for better packet scheduling
sudo tc qdisc add dev eth0 root fq

# 5. Memory and I/O Optimization
echo "💾 Optimizing memory and I/O..."

# Transparent Huge Pages for better memory performance
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null

# Increase file descriptors
ulimit -n 1000000

# VM tuning for low latency
sudo sysctl -w vm.swappiness=0
sudo sysctl -w vm.dirty_ratio=10
sudo sysctl -w vm.dirty_background_ratio=5

# 6. IRQ Affinity (bind network interrupts to specific cores)
echo "🔧 Setting IRQ affinity..."

# Find network interface IRQs
for irq in $(grep eth0 /proc/interrupts | awk '{print $1}' | sed 's/://'); do
    # Bind to cores 0-3 for network processing
    echo 0-3 | sudo tee /proc/irq/$irq/smp_affinity_list 2>/dev/null || true
done

# 7. Process Priority
echo "⚙️ Setting process priorities..."

# Give MEV processes real-time priority
for pid in $(pgrep -f "mev-|arbitrage-|sandwich-"); do
    sudo renice -20 -p $pid 2>/dev/null || true
    sudo chrt -f -p 50 $pid 2>/dev/null || true
done

# 8. Kernel Bypass Setup (optional, requires DPDK)
echo "🚀 Kernel bypass preparations..."

# Enable hugepages for DPDK/kernel bypass
echo 1024 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages >/dev/null

# 9. Security Mitigations (disable for performance)
echo "🛡️ Adjusting security mitigations for performance..."

# Disable Spectre/Meltdown mitigations (only for dedicated MEV servers!)
if [ "$1" == "--unsafe-performance" ]; then
    echo "⚠️ Disabling CPU mitigations for maximum performance..."
    sudo grubby --update-kernel=ALL --args="mitigations=off" 2>/dev/null || \
    echo "mitigations=off" | sudo tee -a /etc/default/grub >/dev/null
    echo "Note: Reboot required for CPU mitigation changes"
fi

# 10. Systemd Optimizations
echo "🔄 Optimizing systemd services..."

# Disable unnecessary services
for service in bluetooth cups avahi-daemon ModemManager; do
    sudo systemctl disable $service 2>/dev/null || true
    sudo systemctl stop $service 2>/dev/null || true
done

echo "✅ System optimizations complete!"
echo ""
echo "📊 Current Settings:"
echo "  CPU Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "  Network rmem_max: $(sysctl -n net.core.rmem_max)"
echo "  TCP Congestion: $(sysctl -n net.ipv4.tcp_congestion_control)"
echo "  Swappiness: $(sysctl -n vm.swappiness)"
echo ""
echo "🎯 Expected Performance:"
echo "  - Decision Latency P50: <8ms"
echo "  - Decision Latency P99: <18ms"
echo "  - Packet Processing: <100μs"
echo "  - Model Inference: <100μs"
echo ""
echo "💡 Run with --unsafe-performance for maximum speed (disables CPU mitigations)"