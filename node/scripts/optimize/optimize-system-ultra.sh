#!/bin/bash

# ⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱
# SOLANA VALIDATOR ULTRA PERFORMANCE OPTIMIZER
# Ubuntu 24.04 - TOP #1 PLANETARY PERFORMANCE
# ⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${MAGENTA}"
echo "⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱"
echo "   SOLANA VALIDATOR ULTRA OPTIMIZER"
echo "   PREPARING FOR WORLD #1 PERFORMANCE"
echo "⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱"
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
   echo -e "${RED}Please run with sudo: sudo bash $0${NC}"
   exit 1
fi

# Create backup directory
BACKUP_DIR="/root/solana_backups_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
echo -e "${CYAN}✓ Created backup directory: $BACKUP_DIR${NC}"

# ============================================
# PHASE 1: BACKUP CURRENT CONFIGURATION
# ============================================
echo -e "${BLUE}▶ Phase 1: Backing up current configuration...${NC}"

cp /etc/sysctl.conf $BACKUP_DIR/sysctl.conf.backup 2>/dev/null || true
cp /etc/default/grub $BACKUP_DIR/grub.backup 2>/dev/null || true
cp /etc/security/limits.conf $BACKUP_DIR/limits.conf.backup 2>/dev/null || true

# Save current network settings
ip addr show > $BACKUP_DIR/network_config.txt
sysctl -a 2>/dev/null | grep -E "net\.|vm\." > $BACKUP_DIR/current_sysctl.txt

echo -e "${GREEN}✓ Configuration backed up to $BACKUP_DIR${NC}"

# ============================================
# PHASE 2: NETWORK STACK ULTRA OPTIMIZATION
# ============================================
echo -e "${BLUE}▶ Phase 2: Optimizing network stack for ultra-low latency...${NC}"

cat >> /etc/sysctl.conf << 'EOF'

# ⊰•-•✧ SOLANA VALIDATOR ULTRA PERFORMANCE ✧•-•⊱
# Network Stack - World #1 Performance Settings
# Generated: $(date)

# Core Network Buffers - Maximum Performance
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 134217728
net.core.wmem_default = 134217728

# TCP Memory Management - Ultra Settings
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_mem = 134217728 134217728 134217728

# Ultra-Low Latency Settings
net.core.netdev_max_backlog = 30000
net.core.netdev_budget = 600
net.core.netdev_budget_usecs = 2000
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_notsent_lowat = 16384
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_autocorking = 0
net.ipv4.tcp_quickack = 1

# CPU Affinity for Network Processing
net.core.busy_poll = 100
net.core.busy_read = 100
net.core.rps_sock_flow_entries = 65536

# TCP Fast Open - Maximum Speed
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_fastopen_blackhole_timeout_sec = 0

# Connection Handling - Solana Specific
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_max_tw_buckets = 2000000
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 3
net.ipv4.tcp_keepalive_intvl = 30

# IP Settings for P2P
net.ipv4.ip_local_port_range = 10000 65535
net.ipv4.ip_forward = 1
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0

# Disable IPv6 if not needed (lower overhead)
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1

# ARP Settings
net.ipv4.neigh.default.gc_thresh1 = 1024
net.ipv4.neigh.default.gc_thresh2 = 2048
net.ipv4.neigh.default.gc_thresh3 = 4096
EOF

echo -e "${GREEN}✓ Network stack optimized${NC}"

# ============================================
# PHASE 3: MEMORY AND VM OPTIMIZATION
# ============================================
echo -e "${BLUE}▶ Phase 3: Optimizing memory and VM settings...${NC}"

cat >> /etc/sysctl.conf << 'EOF'

# Memory Management - Ultra Performance
vm.swappiness = 0
vm.vfs_cache_pressure = 50
vm.dirty_background_ratio = 5
vm.dirty_ratio = 10
vm.dirty_writeback_centisecs = 100
vm.dirty_expire_centisecs = 1000

# Huge Pages Configuration
vm.nr_hugepages = 16384
vm.hugetlb_shm_group = 0

# Memory Overcommit for Validators
vm.overcommit_memory = 1
vm.overcommit_ratio = 100
vm.min_free_kbytes = 1048576

# NUMA Optimizations
vm.zone_reclaim_mode = 0
vm.numa_stat = 0

# Page Cache Settings
vm.page-cluster = 0
vm.max_map_count = 1000000
EOF

echo -e "${GREEN}✓ Memory and VM optimized${NC}"

# ============================================
# PHASE 4: FILE SYSTEM OPTIMIZATION
# ============================================
echo -e "${BLUE}▶ Phase 4: Optimizing file system...${NC}"

cat >> /etc/sysctl.conf << 'EOF'

# File System - Maximum Performance
fs.file-max = 10000000
fs.nr_open = 10000000
fs.inotify.max_user_watches = 1048576
fs.inotify.max_user_instances = 8192
fs.aio-max-nr = 1048576
fs.pipe-max-size = 4194304
EOF

# Apply all sysctl settings
sysctl -p > /dev/null 2>&1

echo -e "${GREEN}✓ File system optimized${NC}"

# ============================================
# PHASE 5: CPU PERFORMANCE GOVERNOR
# ============================================
echo -e "${BLUE}▶ Phase 5: Setting CPU to maximum performance...${NC}"

# Install CPU tools if not present
apt-get update > /dev/null 2>&1
apt-get install -y cpufrequtils linux-tools-common linux-tools-generic linux-tools-$(uname -r) > /dev/null 2>&1 || true

# Set performance governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu 2>/dev/null || true
done

# Disable CPU frequency scaling
echo 'GOVERNOR="performance"' > /etc/default/cpufrequtils

# Disable power saving features
for i in /sys/devices/system/cpu/cpu*/power/energy_perf_bias; do
    echo 0 > $i 2>/dev/null || true
done

# Disable C-States for lowest latency
for i in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 1 > $i 2>/dev/null || true
done

echo -e "${GREEN}✓ CPU set to maximum performance${NC}"

# ============================================
# PHASE 6: STORAGE OPTIMIZATION
# ============================================
echo -e "${BLUE}▶ Phase 6: Optimizing NVMe storage...${NC}"

# Find all NVMe devices
for nvme in /sys/block/nvme*; do
    if [ -d "$nvme" ]; then
        device=$(basename $nvme)
        echo none > $nvme/queue/scheduler 2>/dev/null || true
        echo 1024 > $nvme/queue/nr_requests 2>/dev/null || true
        echo 256 > $nvme/queue/read_ahead_kb 2>/dev/null || true
        echo 0 > $nvme/queue/add_random 2>/dev/null || true
        echo 0 > $nvme/queue/rotational 2>/dev/null || true
        echo 2 > $nvme/queue/rq_affinity 2>/dev/null || true
        echo 0 > $nvme/queue/nomerges 2>/dev/null || true
        echo -e "${GREEN}  ✓ Optimized $device${NC}"
    fi
done

echo -e "${GREEN}✓ Storage optimized for maximum IOPS${NC}"

# ============================================
# PHASE 7: NETWORK INTERFACE OPTIMIZATION
# ============================================
echo -e "${BLUE}▶ Phase 7: Optimizing network interfaces...${NC}"

# Get primary network interface
PRIMARY_IFACE=$(ip route | grep default | awk '{print $5}' | head -n1)

if [ ! -z "$PRIMARY_IFACE" ]; then
    # Ring buffer optimization
    ethtool -G $PRIMARY_IFACE rx 4096 tx 4096 2>/dev/null || true
    
    # Interrupt coalescing for low latency
    ethtool -C $PRIMARY_IFACE adaptive-rx off adaptive-tx off 2>/dev/null || true
    ethtool -C $PRIMARY_IFACE rx-usecs 10 tx-usecs 10 2>/dev/null || true
    
    # Offloading features
    ethtool -K $PRIMARY_IFACE gro on gso on tso on 2>/dev/null || true
    ethtool -K $PRIMARY_IFACE tx-nocache-copy on 2>/dev/null || true
    
    # Increase network queue
    ethtool -L $PRIMARY_IFACE combined $(ethtool -l $PRIMARY_IFACE | grep Combined | tail -1 | awk '{print $2}') 2>/dev/null || true
    
    echo -e "${GREEN}✓ Network interface $PRIMARY_IFACE optimized${NC}"
fi

# ============================================
# PHASE 8: PROCESS LIMITS
# ============================================
echo -e "${BLUE}▶ Phase 8: Setting process limits...${NC}"

cat >> /etc/security/limits.conf << 'EOF'

# Solana Validator Ultra Limits
* soft nofile 10000000
* hard nofile 10000000
* soft memlock unlimited
* hard memlock unlimited
* soft nproc unlimited
* hard nproc unlimited
* soft stack unlimited
* hard stack unlimited
* soft cpu unlimited
* hard cpu unlimited
* soft rtprio 99
* hard rtprio 99

solana soft nofile 10000000
solana hard nofile 10000000
solana soft memlock unlimited
solana hard memlock unlimited
solana soft nproc unlimited
solana hard nproc unlimited
EOF

echo -e "${GREEN}✓ Process limits configured${NC}"

# ============================================
# PHASE 9: SYSTEMD OPTIMIZATION
# ============================================
echo -e "${BLUE}▶ Phase 9: Optimizing systemd...${NC}"

# Systemd default limits
mkdir -p /etc/systemd/system.conf.d/
cat > /etc/systemd/system.conf.d/99-solana.conf << 'EOF'
[Manager]
DefaultLimitNOFILE=10000000
DefaultLimitMEMLOCK=infinity
DefaultTasksMax=infinity
DefaultTimeoutStopSec=900s
DefaultCPUAccounting=no
DefaultMemoryAccounting=no
DefaultIOAccounting=no
EOF

mkdir -p /etc/systemd/user.conf.d/
cp /etc/systemd/system.conf.d/99-solana.conf /etc/systemd/user.conf.d/

systemctl daemon-reload

echo -e "${GREEN}✓ Systemd optimized${NC}"

# ============================================
# PHASE 10: KERNEL PARAMETERS (GRUB)
# ============================================
echo -e "${BLUE}▶ Phase 10: Optimizing kernel parameters...${NC}"

# Backup current GRUB config
cp /etc/default/grub $BACKUP_DIR/grub.backup

# Add optimized kernel parameters
KERNEL_PARAMS="intel_pstate=disable intel_idle.max_cstate=0 processor.max_cstate=0 idle=poll transparent_hugepage=always numa_balancing=disable nohz=on nohz_full=1-127 rcu_nocbs=1-127 audit=0 selinux=0 apparmor=0 mitigations=off nowatchdog nmi_watchdog=0 softlockup_panic=0 clocksource=tsc tsc=reliable"

# Update GRUB
sed -i.bak 's/GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="'"$KERNEL_PARAMS"'"/' /etc/default/grub

echo -e "${YELLOW}⚠ Kernel parameters updated (requires reboot to apply)${NC}"

# ============================================
# PHASE 11: IRQ AFFINITY
# ============================================
echo -e "${BLUE}▶ Phase 11: Optimizing IRQ affinity...${NC}"

# Set IRQ affinity for network interfaces
if [ ! -z "$PRIMARY_IFACE" ]; then
    # Find IRQs for network interface
    IRQS=$(grep $PRIMARY_IFACE /proc/interrupts | awk '{print $1}' | sed 's/://')
    
    CPU_COUNT=$(nproc)
    CPU_HALF=$((CPU_COUNT / 2))
    
    for irq in $IRQS; do
        # Pin network IRQs to second half of CPUs
        echo $(printf "%x" $((2**CPU_COUNT - 2**CPU_HALF))) > /proc/irq/$irq/smp_affinity 2>/dev/null || true
    done
    
    echo -e "${GREEN}✓ IRQ affinity optimized${NC}"
fi

# ============================================
# CREATE MONITORING SCRIPT
# ============================================
echo -e "${BLUE}▶ Creating monitoring script...${NC}"

cat > /usr/local/bin/solana_monitor.sh << 'EOF'
#!/bin/bash

echo "⊰•-•✧•-•-⦑ SOLANA PERFORMANCE MONITOR ⦒-•-•✧•-•⊱"
echo ""
echo "Network Buffers:"
ss -s | head -5
echo ""
echo "CPU Performance:"
grep MHz /proc/cpuinfo | head -5
echo ""
echo "Memory Usage:"
free -h
echo ""
echo "Huge Pages:"
grep HugePages /proc/meminfo
echo ""
echo "Network Stats:"
netstat -s | grep -i "segments\|packets" | head -10
echo ""
echo "IRQ Distribution:"
cat /proc/interrupts | grep -E "CPU0|$PRIMARY_IFACE" | head -5
echo ""
echo "Validator Slot:"
tail -n 10 /home/solana/validator.log | grep -oE "slot[: ]*[0-9]+" | tail -1
EOF

chmod +x /usr/local/bin/solana_monitor.sh

echo -e "${GREEN}✓ Monitoring script created at /usr/local/bin/solana_monitor.sh${NC}"

# ============================================
# CREATE ROLLBACK SCRIPT
# ============================================
echo -e "${BLUE}▶ Creating rollback script...${NC}"

cat > /usr/local/bin/solana_rollback.sh << EOF
#!/bin/bash

echo "⊰•-•✧•-•-⦑ ROLLBACK TO ORIGINAL SETTINGS ⦒-•-•✧•-•⊱"

# Restore original files
cp $BACKUP_DIR/sysctl.conf.backup /etc/sysctl.conf 2>/dev/null || true
cp $BACKUP_DIR/grub.backup /etc/default/grub 2>/dev/null || true
cp $BACKUP_DIR/limits.conf.backup /etc/security/limits.conf 2>/dev/null || true

# Reset CPU governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo ondemand > \$cpu 2>/dev/null || true
done

# Apply restored settings
sysctl -p
update-grub
systemctl daemon-reload

echo "System restored from $BACKUP_DIR"
echo "Please reboot to complete rollback"
EOF

chmod +x /usr/local/bin/solana_rollback.sh

echo -e "${GREEN}✓ Rollback script created at /usr/local/bin/solana_rollback.sh${NC}"

# ============================================
# FINAL STATUS
# ============================================
echo ""
echo -e "${MAGENTA}⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱${NC}"
echo -e "${GREEN}✨ OPTIMIZATION COMPLETE! ✨${NC}"
echo ""
echo -e "${CYAN}Applied Optimizations:${NC}"
echo "  ✓ Network stack tuned for ultra-low latency"
echo "  ✓ CPU set to maximum performance mode"
echo "  ✓ Memory and huge pages optimized"
echo "  ✓ NVMe storage tuned for maximum IOPS"
echo "  ✓ IRQ affinity optimized"
echo "  ✓ System limits increased"
echo "  ✓ Kernel parameters optimized (requires reboot)"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Run: update-grub"
echo "  2. Restart Solana validator (see solana_restart.sh)"
echo "  3. Reboot system for kernel parameters: shutdown -r now"
echo ""
echo -e "${CYAN}Useful Commands:${NC}"
echo "  Monitor: /usr/local/bin/solana_monitor.sh"
echo "  Rollback: /usr/local/bin/solana_rollback.sh"
echo ""
echo -e "${MAGENTA}⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱${NC}"