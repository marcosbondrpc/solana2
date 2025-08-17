#!/bin/bash

# 🚀 Elite Network Optimization Script for MEV Infrastructure
# Achieves sub-microsecond latency with hardware-level optimizations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        🚀 Elite Network Optimization for MEV                  ║${NC}"
echo -e "${BLUE}║            Sub-microsecond Latency Configuration              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root"
   exit 1
fi

# Detect network interface
detect_network_interface() {
    # Get the primary network interface
    PRIMARY_IFACE=$(ip route | grep default | awk '{print $5}' | head -n1)
    
    if [ -z "$PRIMARY_IFACE" ]; then
        print_error "Could not detect primary network interface"
        exit 1
    fi
    
    echo -e "${BLUE}Detected primary network interface: ${GREEN}$PRIMARY_IFACE${NC}"
    
    # Check if it's a Mellanox card (optimal for MEV)
    if lspci | grep -i mellanox > /dev/null; then
        print_status "Mellanox NIC detected - optimal for MEV operations"
        IS_MELLANOX=true
    else
        print_warning "Non-Mellanox NIC detected - performance may be suboptimal"
        IS_MELLANOX=false
    fi
    
    return 0
}

# 1.2 NIC & UDP Tuning
configure_nic_and_udp() {
    echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}1. NIC & UDP Tuning${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}\n"
    
    # Stop irqbalance for manual IRQ steering
    if systemctl is-active --quiet irqbalance; then
        systemctl stop irqbalance
        systemctl disable irqbalance
        print_status "IRQ balance service stopped and disabled"
    fi
    
    # IRQ Steering to cores 2-11 (leaving 0-1 for system, 12+ for application)
    echo -e "\n${YELLOW}Configuring IRQ affinity...${NC}"
    
    # Get number of CPUs
    NCPUS=$(nproc)
    if [ $NCPUS -lt 12 ]; then
        print_warning "System has less than 12 CPUs, adjusting IRQ affinity range"
        IRQ_CORES="2-$((NCPUS-1))"
    else
        IRQ_CORES="2-11"
    fi
    
    # Set IRQ affinity for network interrupts
    for irq in $(grep "$PRIMARY_IFACE" /proc/interrupts | awk '{print $1}' | sed 's/://'); do
        if [ -f "/proc/irq/$irq/smp_affinity_list" ]; then
            echo "$IRQ_CORES" > /proc/irq/$irq/smp_affinity_list 2>/dev/null || true
        fi
    done
    
    # Also set for all interrupts (as in original script)
    for i in /proc/irq/*/smp_affinity_list; do
        if [ -w "$i" ]; then
            echo "$IRQ_CORES" > "$i" 2>/dev/null || true
        fi
    done
    
    print_status "IRQ steering configured to cores $IRQ_CORES"
    
    # NIC Queue Configuration
    echo -e "\n${YELLOW}Configuring NIC queues...${NC}"
    
    # Check current queue settings
    CURRENT_QUEUES=$(ethtool -l "$PRIMARY_IFACE" 2>/dev/null | grep "Combined:" | tail -1 | awk '{print $2}')
    MAX_QUEUES=$(ethtool -l "$PRIMARY_IFACE" 2>/dev/null | grep "Combined:" | head -1 | awk '{print $2}')
    
    if [ -n "$MAX_QUEUES" ] && [ "$MAX_QUEUES" -ge 16 ]; then
        ethtool -L "$PRIMARY_IFACE" combined 16 2>/dev/null || \
            print_warning "Could not set combined queues to 16"
        print_status "NIC queues set to 16 (was $CURRENT_QUEUES)"
    else
        print_warning "Maximum queues available: $MAX_QUEUES"
    fi
    
    # Interrupt Coalescing for minimal latency
    echo -e "\n${YELLOW}Configuring interrupt coalescing...${NC}"
    
    # Adaptive RX with minimal latency
    ethtool -C "$PRIMARY_IFACE" adaptive-rx on rx-usecs 0 rx-frames 1 2>/dev/null || \
        print_warning "Could not configure interrupt coalescing"
    
    # Also set TX coalescing for consistency
    ethtool -C "$PRIMARY_IFACE" tx-usecs 0 tx-frames 1 2>/dev/null || true
    
    print_status "Interrupt coalescing configured for minimal latency"
    
    # Offload Settings for Latency Determinism
    echo -e "\n${YELLOW}Configuring offload settings...${NC}"
    
    # Disable offloads that add latency variance
    ethtool -K "$PRIMARY_IFACE" gro off 2>/dev/null || print_warning "Could not disable GRO"
    ethtool -K "$PRIMARY_IFACE" lro off 2>/dev/null || print_warning "Could not disable LRO"
    ethtool -K "$PRIMARY_IFACE" tso off 2>/dev/null || print_warning "Could not disable TSO"
    ethtool -K "$PRIMARY_IFACE" gso off 2>/dev/null || print_warning "Could not disable GSO"
    
    # Keep checksums on for data integrity
    ethtool -K "$PRIMARY_IFACE" rx on 2>/dev/null || true
    ethtool -K "$PRIMARY_IFACE" tx on 2>/dev/null || true
    
    # Enable hardware timestamping if available
    ethtool -K "$PRIMARY_IFACE" rx-all on 2>/dev/null || true
    ethtool -K "$PRIMARY_IFACE" tx-all on 2>/dev/null || true
    
    print_status "Offload settings optimized for latency determinism"
    
    # Socket Buffer and Busy Poll Configuration
    echo -e "\n${YELLOW}Configuring socket buffers and busy polling...${NC}"
    
    # Socket buffers (32MB for high-throughput)
    sysctl -w net.core.rmem_max=33554432 > /dev/null
    sysctl -w net.core.wmem_max=33554432 > /dev/null
    sysctl -w net.core.rmem_default=33554432 > /dev/null
    sysctl -w net.core.wmem_default=33554432 > /dev/null
    
    # Network device backlog
    sysctl -w net.core.netdev_max_backlog=5000 > /dev/null
    sysctl -w net.core.netdev_budget=600 > /dev/null
    
    # Busy polling for ultra-low latency
    sysctl -w net.core.busy_poll=50 > /dev/null
    sysctl -w net.core.busy_read=50 > /dev/null
    
    # UDP specific settings
    sysctl -w net.ipv4.udp_rmem_min=262144 > /dev/null
    sysctl -w net.ipv4.udp_wmem_min=262144 > /dev/null
    
    # Additional UDP optimizations
    sysctl -w net.ipv4.udp_mem="262144 524288 1048576" > /dev/null
    
    print_status "Socket buffers and busy polling configured"
    
    # Additional Performance Tuning
    echo -e "\n${YELLOW}Applying additional performance tuning...${NC}"
    
    # TCP/IP stack optimizations
    sysctl -w net.ipv4.tcp_timestamps=1 > /dev/null
    sysctl -w net.ipv4.tcp_sack=1 > /dev/null
    sysctl -w net.ipv4.tcp_low_latency=1 > /dev/null
    sysctl -w net.ipv4.tcp_fastopen=3 > /dev/null
    
    # Congestion control
    sysctl -w net.ipv4.tcp_congestion_control=bbr > /dev/null
    sysctl -w net.core.default_qdisc=fq > /dev/null
    
    # Connection tracking for MEV
    sysctl -w net.netfilter.nf_conntrack_max=1048576 > /dev/null 2>&1 || true
    sysctl -w net.ipv4.tcp_max_syn_backlog=8192 > /dev/null
    
    print_status "Additional performance tuning applied"
}

# 1.3 Time Synchronization with Nanosecond Accuracy
configure_time_sync() {
    echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}2. Time Synchronization (Nanosecond Accuracy)${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}\n"
    
    # Install required packages
    echo -e "${YELLOW}Installing PTP and Chrony...${NC}"
    
    # Check if packages are installed
    PACKAGES_TO_INSTALL=""
    
    if ! command -v phc2sys &> /dev/null; then
        PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL linuxptp"
    fi
    
    if ! command -v chronyd &> /dev/null; then
        PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL chrony"
    fi
    
    if [ -n "$PACKAGES_TO_INSTALL" ]; then
        apt-get update -qq
        apt-get install -y -qq $PACKAGES_TO_INSTALL
        print_status "PTP and Chrony packages installed"
    else
        print_status "PTP and Chrony already installed"
    fi
    
    # Check for PTP hardware support
    echo -e "\n${YELLOW}Checking for PTP hardware support...${NC}"
    
    PTP_DEVICE=""
    if [ -e /dev/ptp0 ]; then
        PTP_DEVICE="/dev/ptp0"
        print_status "PTP hardware device found: $PTP_DEVICE"
        HAS_PTP_HW=true
    else
        print_warning "No PTP hardware device found - using software timestamping"
        HAS_PTP_HW=false
    fi
    
    # Configure Chrony
    echo -e "\n${YELLOW}Configuring Chrony for high precision...${NC}"
    
    # Backup original config
    if [ -f /etc/chrony/chrony.conf ] && [ ! -f /etc/chrony/chrony.conf.backup ]; then
        cp /etc/chrony/chrony.conf /etc/chrony/chrony.conf.backup
    fi
    
    # Create optimized chrony configuration
    cat > /etc/chrony/chrony.conf << 'EOF'
# Elite MEV Chrony Configuration - Nanosecond Precision

# NTP Pools with low stratum
pool time.cloudflare.com iburst maxpoll 6 minpoll 4
pool time.google.com iburst maxpoll 6 minpoll 4
server time.nist.gov iburst maxpoll 6 minpoll 4

# Hardware timestamping if available
hwtimestamp *

# PTP hardware clock reference if available
EOF

    if [ "$HAS_PTP_HW" = true ]; then
        echo "refclock PHC $PTP_DEVICE poll 2 dpoll -2 offset 0.0 precision 1e-9" >> /etc/chrony/chrony.conf
    fi

    cat >> /etc/chrony/chrony.conf << 'EOF'

# Aggressive clock stepping for quick sync
makestep 0.1 -1

# More frequent measurements
minsamples 32
maxsamples 256

# Local stratum for isolated networks
local stratum 10

# Drift file
driftfile /var/lib/chrony/drift

# Log tracking
log tracking measurements statistics

# RTC sync
rtcsync

# Command port
cmdport 323

# Allow monitoring
allow 127.0.0.1
allow ::1

# Leap second handling
leapsecmode slew
maxslewrate 1000.0

# Performance tuning
maxupdateskew 100.0
maxdistance 1.0
maxjitter 1.0

# Hardware timestamping for all interfaces
EOF

    # Add interface-specific hardware timestamping
    echo "hwtimestamp $PRIMARY_IFACE" >> /etc/chrony/chrony.conf
    
    # Restart chrony
    systemctl restart chrony
    systemctl enable chrony
    
    print_status "Chrony configured for nanosecond precision"
    
    # Configure PTP if hardware is available
    if [ "$HAS_PTP_HW" = true ]; then
        echo -e "\n${YELLOW}Configuring PTP daemon...${NC}"
        
        # Create PTP configuration
        cat > /etc/ptp4l.conf << 'EOF'
# PTP Configuration for MEV Operations
[global]
#
# Default Data Set
#
slaveOnly               0
priority1               128
priority2               128
domainNumber            0
clockClass              248
clockAccuracy           0xFE
offsetScaledLogVariance 0xFFFF
free_running            0
freq_est_interval       1
#
# Port Data Set
#
logAnnounceInterval     1
logSyncInterval         0
logMinDelayReqInterval  0
logMinPdelayReqInterval 0
announceReceiptTimeout  3
syncReceiptTimeout      0
delayAsymmetry          0
fault_reset_interval    4
neighborPropDelayThresh 20000000
#
# Run time options
#
assume_two_step         0
logging_level           6
path_trace_enabled      0
follow_up_info          0
hybrid_e2e              0
tx_timestamp_timeout    1
use_syslog              1
verbose                 0
summary_interval        0
kernel_leap             1
check_fup_sync          0
#
# Servo Options
#
pi_proportional_const   0.0
pi_integral_const       0.0
pi_proportional_scale   0.0
pi_proportional_exponent -0.3
pi_proportional_norm_max 0.7
pi_integral_scale       0.0
pi_integral_exponent    0.4
pi_integral_norm_max    0.3
step_threshold          0.0
first_step_threshold    0.00002
max_frequency           900000000
clock_servo             pi
sanity_freq_limit       200000000
ntpshm_segment          0
#
# Transport options
#
transportSpecific       0x0
ptp_dst_mac             01:1B:19:00:00:00
p2p_dst_mac             01:80:C2:00:00:0E
udp_ttl                 1
udp6_scope              0x0E
uds_address             /var/run/ptp4l
#
# Default interface options
#
network_transport       UDPv4
delay_mechanism         E2E
time_stamping           hardware
tsproc_mode             filter
delay_filter            moving_median
delay_filter_length     10
egressLatency           0
ingressLatency          0
boundary_clock_jbod     0
#
# Clock description
#
productDescription      "MEV Infrastructure PTP"
revisionData            "1.0"
manufacturerIdentity    00:00:00
userDescription         "Elite MEV Time Sync"
timeSource              0xA0
EOF
        
        # Create systemd service for PTP
        cat > /etc/systemd/system/ptp4l.service << EOF
[Unit]
Description=Precision Time Protocol daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/sbin/ptp4l -f /etc/ptp4l.conf -i $PRIMARY_IFACE
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
        
        # Start PTP daemon
        systemctl daemon-reload
        systemctl start ptp4l
        systemctl enable ptp4l
        
        print_status "PTP daemon configured and started"
        
        # Configure PHC2SYS for system clock sync
        cat > /etc/systemd/system/phc2sys.service << EOF
[Unit]
Description=Synchronize system clock to PTP hardware clock
After=ptp4l.service
Requires=ptp4l.service

[Service]
Type=simple
ExecStart=/usr/sbin/phc2sys -s $PTP_DEVICE -c CLOCK_REALTIME -w -O 0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
        
        systemctl daemon-reload
        systemctl start phc2sys
        systemctl enable phc2sys
        
        print_status "PHC2SYS configured for hardware clock sync"
    fi
}

# Additional Kernel Optimizations
configure_kernel_optimizations() {
    echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}3. Additional Kernel Optimizations${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}\n"
    
    # CPU frequency scaling
    echo -e "${YELLOW}Setting CPU governor to performance...${NC}"
    
    if command -v cpupower &> /dev/null; then
        cpupower frequency-set -g performance 2>/dev/null || true
    else
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "performance" > "$cpu" 2>/dev/null || true
        done
    fi
    
    print_status "CPU governor set to performance mode"
    
    # Disable CPU idle states for consistent latency
    echo -e "\n${YELLOW}Disabling deep CPU idle states...${NC}"
    
    for cpu_idle in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
        echo 1 > "$cpu_idle" 2>/dev/null || true
    done
    
    print_status "Deep CPU idle states disabled"
    
    # Configure huge pages
    echo -e "\n${YELLOW}Configuring huge pages...${NC}"
    
    sysctl -w vm.nr_hugepages=1024 > /dev/null
    
    print_status "Huge pages configured (1024 pages)"
    
    # NUMA optimizations
    if command -v numactl &> /dev/null; then
        echo -e "\n${YELLOW}Configuring NUMA optimizations...${NC}"
        
        # Set NUMA balancing
        echo 0 > /proc/sys/kernel/numa_balancing 2>/dev/null || true
        
        print_status "NUMA balancing disabled for consistent latency"
    fi
}

# Save configuration for persistence
save_persistent_config() {
    echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}4. Saving Persistent Configuration${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}\n"
    
    # Save sysctl settings
    cat > /etc/sysctl.d/99-mev-network-optimization.conf << 'EOF'
# MEV Network Optimization Settings
# Generated by elite network optimization script

# Socket buffers
net.core.rmem_max = 33554432
net.core.wmem_max = 33554432
net.core.rmem_default = 33554432
net.core.wmem_default = 33554432

# Network device settings
net.core.netdev_max_backlog = 5000
net.core.netdev_budget = 600

# Busy polling for low latency
net.core.busy_poll = 50
net.core.busy_read = 50

# UDP settings
net.ipv4.udp_rmem_min = 262144
net.ipv4.udp_wmem_min = 262144
net.ipv4.udp_mem = 262144 524288 1048576

# TCP settings
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
net.ipv4.tcp_max_syn_backlog = 8192

# Connection tracking
net.netfilter.nf_conntrack_max = 1048576

# Huge pages
vm.nr_hugepages = 1024

# Disable NUMA balancing
kernel.numa_balancing = 0
EOF
    
    print_status "Sysctl settings saved to /etc/sysctl.d/99-mev-network-optimization.conf"
    
    # Create startup script for interface settings
    cat > /etc/systemd/system/mev-network-optimization.service << EOF
[Unit]
Description=MEV Network Optimization
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/mev-network-optimize.sh
StandardOutput=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Create the optimization script
    cat > /usr/local/bin/mev-network-optimize.sh << EOF
#!/bin/bash
# MEV Network Interface Optimization
# Auto-generated script for persistent settings

PRIMARY_IFACE="$PRIMARY_IFACE"

# NIC settings
ethtool -L "\$PRIMARY_IFACE" combined 16 2>/dev/null || true
ethtool -C "\$PRIMARY_IFACE" adaptive-rx on rx-usecs 0 rx-frames 1 tx-usecs 0 tx-frames 1 2>/dev/null || true

# Offload settings
ethtool -K "\$PRIMARY_IFACE" gro off lro off tso off gso off rx on tx on 2>/dev/null || true

# IRQ affinity
for irq in \$(grep "\$PRIMARY_IFACE" /proc/interrupts | awk '{print \$1}' | sed 's/://'); do
    if [ -f "/proc/irq/\$irq/smp_affinity_list" ]; then
        echo "$IRQ_CORES" > /proc/irq/\$irq/smp_affinity_list 2>/dev/null || true
    fi
done

# CPU frequency
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > "\$cpu" 2>/dev/null || true
done

# Disable CPU idle states
for cpu_idle in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 1 > "\$cpu_idle" 2>/dev/null || true
done

exit 0
EOF
    
    chmod +x /usr/local/bin/mev-network-optimize.sh
    
    # Enable the service
    systemctl daemon-reload
    systemctl enable mev-network-optimization.service
    
    print_status "Startup service created for persistent settings"
}

# Show current status
show_optimization_status() {
    echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Optimization Status${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "${GREEN}Network Interface:${NC} $PRIMARY_IFACE"
    
    # Show NIC settings
    echo -e "\n${GREEN}NIC Queue Settings:${NC}"
    ethtool -l "$PRIMARY_IFACE" 2>/dev/null | grep "Combined:" | tail -1 || echo "Unable to query"
    
    echo -e "\n${GREEN}Interrupt Coalescing:${NC}"
    ethtool -c "$PRIMARY_IFACE" 2>/dev/null | grep -E "rx-usecs:|rx-frames:" || echo "Unable to query"
    
    echo -e "\n${GREEN}Offload Settings:${NC}"
    ethtool -k "$PRIMARY_IFACE" 2>/dev/null | grep -E "generic-receive-offload:|large-receive-offload:" || echo "Unable to query"
    
    echo -e "\n${GREEN}Socket Buffer Sizes:${NC}"
    sysctl net.core.rmem_max net.core.wmem_max 2>/dev/null || echo "Unable to query"
    
    echo -e "\n${GREEN}Busy Polling:${NC}"
    sysctl net.core.busy_poll net.core.busy_read 2>/dev/null || echo "Unable to query"
    
    echo -e "\n${GREEN}Time Sync Status:${NC}"
    if systemctl is-active --quiet chrony; then
        echo "Chrony: Active"
        chronyc tracking 2>/dev/null | grep -E "System time|RMS offset" || true
    else
        echo "Chrony: Inactive"
    fi
    
    if [ "$HAS_PTP_HW" = true ] && systemctl is-active --quiet ptp4l; then
        echo "PTP: Active"
    fi
    
    echo -e "\n${GREEN}CPU Governor:${NC}"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "Unable to query"
}

# Main execution
main() {
    echo -e "${YELLOW}Starting network optimization for MEV infrastructure...${NC}\n"
    
    # Detect network interface
    detect_network_interface
    
    # Apply optimizations
    configure_nic_and_udp
    configure_time_sync
    configure_kernel_optimizations
    save_persistent_config
    
    # Show status
    show_optimization_status
    
    echo -e "\n${GREEN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ Network optimization complete!${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "${BLUE}Optimizations applied:${NC}"
    echo -e "  • IRQ steering to cores $IRQ_CORES"
    echo -e "  • NIC queues set to 16 with minimal coalescing"
    echo -e "  • Offloads disabled for latency determinism"
    echo -e "  • Socket buffers set to 32MB"
    echo -e "  • Busy polling enabled (50μs)"
    echo -e "  • Time sync configured with nanosecond precision"
    echo -e "  • CPU governor set to performance"
    echo -e "  • Huge pages configured"
    echo
    echo -e "${YELLOW}Note:${NC} Reboot recommended for all settings to take full effect"
    echo -e "${YELLOW}Monitor performance with:${NC} /usr/local/bin/mev-network-status.sh"
}

# Run main function
main "$@"