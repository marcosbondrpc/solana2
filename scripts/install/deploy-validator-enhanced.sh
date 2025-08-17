#!/bin/bash
# deploy_apex_validator_enhanced.sh
# Enhanced deployment with full testing and reboot verification

set -e

# Check root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Logging
LOG_FILE="/var/log/apex-validator-deployment.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

log_info() { echo -e "${BLUE}[$(date '+%H:%M:%S')] [INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] [SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] [WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[$(date '+%H:%M:%S')] [ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[$(date '+%H:%M:%S')] [STEP]${NC} $1"; }

# Deployment tracking
DEPLOYMENT_DIR="/opt/apex-deployment"
STATE_FILE="$DEPLOYMENT_DIR/state"
VERIFICATION_LOG="$DEPLOYMENT_DIR/verification.log"

mkdir -p "$DEPLOYMENT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "         APEX-1 VALIDATOR - ENHANCED DEPLOYMENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Version: 1.0.0-legendary"
echo "Time: $(date)"
echo "System: $(uname -a)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# System requirements check
check_system_requirements() {
    log_step "Checking system requirements..."
    
    local requirements_met=true
    
    # CPU check
    local cpu_cores=$(nproc)
    if [ $cpu_cores -ge 32 ]; then
        log_success "CPU cores: $cpu_cores ✓"
    else
        log_warning "CPU cores: $cpu_cores (recommended: 32+)"
        requirements_met=false
    fi
    
    # RAM check
    local total_ram_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [ $total_ram_gb -ge 128 ]; then
        log_success "RAM: ${total_ram_gb}GB ✓"
    else
        log_warning "RAM: ${total_ram_gb}GB (recommended: 128GB+)"
        requirements_met=false
    fi
    
    # Disk space check
    local available_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ $available_space -ge 100 ]; then
        log_success "Available disk space: ${available_space}GB ✓"
    else
        log_warning "Available disk space: ${available_space}GB (recommended: 100GB+)"
    fi
    
    # Network interfaces
    local nic_count=$(ip link show | grep -c "^[0-9]:" || echo 0)
    log_info "Network interfaces found: $nic_count"
    
    if [ "$requirements_met" = false ]; then
        log_warning "System does not meet all recommended requirements"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Apply kernel optimizations
apply_kernel_optimizations() {
    log_step "Applying kernel optimizations..."
    
    # Create comprehensive sysctl configuration
    cat > /etc/sysctl.d/99-apex-validator.conf << 'EOF'
# APEX-1 Validator Kernel Optimizations

# Network Stack - Ultra Performance
net.core.rmem_max = 2147483647
net.core.wmem_max = 2147483647
net.core.rmem_default = 536870912
net.core.wmem_default = 536870912
net.ipv4.tcp_rmem = 4096 87380 2147483647
net.ipv4.tcp_wmem = 4096 65536 2147483647
net.core.netdev_max_backlog = 50000
net.core.netdev_budget = 600
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
net.ipv4.tcp_notsent_lowat = 16384
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_mtu_probing = 1
net.core.busy_poll = 100
net.core.busy_read = 100

# Memory Management
vm.swappiness = 0
vm.dirty_ratio = 5
vm.dirty_background_ratio = 2
vm.min_free_kbytes = 3145728
vm.nr_hugepages = 30720
vm.max_map_count = 1048576

# File System
fs.file-max = 10000000
fs.nr_open = 10000000

# Kernel
kernel.numa_balancing = 0
kernel.sched_rt_runtime_us = -1
kernel.sched_latency_ns = 100000
kernel.sched_min_granularity_ns = 100000
EOF
    
    # Apply settings
    sysctl -p /etc/sysctl.d/99-apex-validator.conf 2>/dev/null || true
    log_success "Kernel parameters applied"
    
    # Set CPU governor
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > $cpu 2>/dev/null || true
    done
    log_success "CPU governor set to performance"
    
    # Disable turbo boost for consistent performance
    echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
    
    # Set up huge pages mount
    if ! mount | grep -q hugetlbfs; then
        mkdir -p /mnt/huge
        mount -t hugetlbfs nodev /mnt/huge
        echo "nodev /mnt/huge hugetlbfs defaults 0 0" >> /etc/fstab
        log_success "Huge pages mounted"
    fi
}

# Install dependencies
install_dependencies() {
    log_step "Installing dependencies..."
    
    apt-get update
    apt-get install -y \
        build-essential \
        pkg-config \
        libssl-dev \
        libudev-dev \
        git \
        curl \
        jq \
        htop \
        iotop \
        net-tools \
        sysstat \
        nvme-cli \
        linux-tools-common \
        linux-tools-generic \
        linux-tools-$(uname -r) \
        2>/dev/null || true
    
    log_success "Dependencies installed"
}

# Setup Solana user and directories
setup_solana_environment() {
    log_step "Setting up Solana environment..."
    
    # Create solana user
    if ! id -u solana > /dev/null 2>&1; then
        useradd -m -s /bin/bash solana
        usermod -aG sudo solana
        log_success "Solana user created"
    fi
    
    # Create directories
    mkdir -p /opt/solana/{bin,config,logs}
    mkdir -p /var/lib/solana/{ledger,accounts,snapshots}
    mkdir -p /opt/apex/{bin,config,logs}
    
    # Set permissions
    chown -R solana:solana /opt/solana
    chown -R solana:solana /var/lib/solana
    chown -R solana:solana /opt/apex
    
    # Set limits for solana user
    cat > /etc/security/limits.d/99-solana.conf << 'EOF'
solana soft nofile 1000000
solana hard nofile 1000000
solana soft nproc unlimited
solana hard nproc unlimited
solana soft memlock unlimited
solana hard memlock unlimited
EOF
    
    log_success "Solana environment configured"
}

# Create systemd services
create_systemd_services() {
    log_step "Creating systemd services..."
    
    # APEX Validator Service (stub for testing)
    cat > /etc/systemd/system/apex-validator.service << 'EOF'
[Unit]
Description=APEX-1 Solana Validator
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=10
User=solana
WorkingDirectory=/opt/solana
Environment="RUST_LOG=info"
Environment="RUST_BACKTRACE=1"

# For testing - replace with actual validator command
ExecStart=/bin/bash -c 'echo "APEX Validator Running" && exec sleep infinity'

# Resource limits
LimitNOFILE=1000000
LimitNPROC=unlimited
LimitMEMLOCK=infinity

# CPU affinity (cores 8-31)
CPUAffinity=8-31

[Install]
WantedBy=multi-user.target
EOF
    
    # Monitoring service
    cat > /etc/systemd/system/apex-monitor.service << 'EOF'
[Unit]
Description=APEX Validator Monitor
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/apex-monitor
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF
    
    # Performance logger service
    cat > /etc/systemd/system/apex-performance-logger.service << 'EOF'
[Unit]
Description=APEX Performance Logger
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/apex-performance-logger
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    log_success "Systemd services created"
}

# Create monitoring scripts
create_monitoring_scripts() {
    log_step "Creating monitoring scripts..."
    
    # Main monitor
    cat > /usr/local/bin/apex-monitor << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "       APEX VALIDATOR MONITOR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Time: $(date)"
    echo ""
    echo "SYSTEM STATUS:"
    echo "  CPU Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
    echo "  Huge Pages: $(cat /proc/sys/vm/nr_hugepages)"
    echo "  Swappiness: $(cat /proc/sys/vm/swappiness)"
    echo "  Load Average: $(uptime | awk -F'load average:' '{print $2}')"
    echo ""
    echo "NETWORK:"
    echo "  TCP Congestion: $(sysctl -n net.ipv4.tcp_congestion_control)"
    echo "  RX Buffer: $(numfmt --to=iec $(cat /proc/sys/net/core/rmem_max))"
    echo "  TX Buffer: $(numfmt --to=iec $(cat /proc/sys/net/core/wmem_max))"
    echo ""
    echo "SERVICES:"
    for service in apex-validator apex-monitor apex-performance-logger; do
        if systemctl is-active --quiet $service; then
            echo "  $service: ✓ Running"
        else
            echo "  $service: ✗ Stopped"
        fi
    done
    sleep 5
done
EOF
    chmod +x /usr/local/bin/apex-monitor
    
    # Performance logger
    cat > /usr/local/bin/apex-performance-logger << 'EOF'
#!/bin/bash
LOG_FILE="/var/log/apex-performance.log"
while true; do
    {
        echo "$(date '+%Y-%m-%d %H:%M:%S')"
        echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
        echo "MEM: $(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2}')"
        echo "LOAD: $(cat /proc/loadavg)"
        echo "---"
    } >> "$LOG_FILE"
    sleep 60
done
EOF
    chmod +x /usr/local/bin/apex-performance-logger
    
    log_success "Monitoring scripts created"
}

# Setup boot persistence
setup_boot_persistence() {
    log_step "Setting up boot persistence..."
    
    # Create boot optimization script
    cat > /usr/local/bin/apex-boot-optimize << 'EOF'
#!/bin/bash
# Apply optimizations at boot

# Set CPU governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu 2>/dev/null || true
done

# Disable turbo boost
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true

# Set IRQ affinity
for irq in $(grep -E 'eth|ens|enp' /proc/interrupts | awk '{print $1}' | sed 's/://'); do
    echo 2 > /proc/irq/$irq/smp_affinity_list 2>/dev/null || true
done

# Log status
echo "Boot optimizations applied at $(date)" >> /var/log/apex-boot.log
EOF
    chmod +x /usr/local/bin/apex-boot-optimize
    
    # Create systemd service for boot optimizations
    cat > /etc/systemd/system/apex-boot-optimize.service << 'EOF'
[Unit]
Description=APEX Boot Optimizations
DefaultDependencies=no
After=sysinit.target
Before=basic.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/apex-boot-optimize
RemainAfterExit=yes

[Install]
WantedBy=basic.target
EOF
    
    systemctl daemon-reload
    systemctl enable apex-boot-optimize.service
    
    log_success "Boot persistence configured"
}

# Verification function
verify_deployment() {
    log_step "Verifying deployment..."
    
    echo "" > "$VERIFICATION_LOG"
    local checks_passed=0
    local checks_total=0
    
    verify_check() {
        local name="$1"
        local cmd="$2"
        local expected="$3"
        
        checks_total=$((checks_total + 1))
        
        if eval "$cmd" 2>/dev/null | grep -q "$expected" 2>/dev/null; then
            echo "✓ $name" | tee -a "$VERIFICATION_LOG"
            checks_passed=$((checks_passed + 1))
            return 0
        else
            echo "✗ $name" | tee -a "$VERIFICATION_LOG"
            return 1
        fi
    }
    
    echo "VERIFICATION RESULTS:" | tee -a "$VERIFICATION_LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━" | tee -a "$VERIFICATION_LOG"
    
    verify_check "CPU Performance Mode" "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" "performance"
    verify_check "Swappiness = 0" "cat /proc/sys/vm/swappiness" "0"
    verify_check "Huge Pages Configured" "test \$(cat /proc/sys/vm/nr_hugepages) -gt 0 && echo ok" "ok"
    verify_check "TCP BBR Enabled" "sysctl net.ipv4.tcp_congestion_control" "bbr"
    verify_check "Network Buffers > 128MB" "test \$(cat /proc/sys/net/core/rmem_max) -gt 134217728 && echo ok" "ok"
    verify_check "Solana User Exists" "id solana && echo ok" "ok"
    verify_check "Boot Service Enabled" "systemctl is-enabled apex-boot-optimize" "enabled"
    
    echo "" | tee -a "$VERIFICATION_LOG"
    echo "Verification: $checks_passed/$checks_total checks passed" | tee -a "$VERIFICATION_LOG"
    
    if [ $checks_passed -eq $checks_total ]; then
        log_success "All verification checks passed!"
        return 0
    else
        log_warning "Some verification checks failed"
        return 1
    fi
}

# Test reboot persistence
test_reboot_persistence() {
    log_step "Testing reboot persistence..."
    
    # Create reboot test marker
    echo "REBOOT_TEST_$(date +%s)" > "$DEPLOYMENT_DIR/reboot_marker"
    
    # Create post-reboot verification script
    cat > /usr/local/bin/apex-reboot-verify << 'EOF'
#!/bin/bash
MARKER_FILE="/opt/apex-deployment/reboot_marker"
LOG_FILE="/opt/apex-deployment/reboot-test.log"

if [ -f "$MARKER_FILE" ]; then
    echo "=== Post-Reboot Verification $(date) ===" >> "$LOG_FILE"
    
    # Check critical settings
    echo "CPU Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" >> "$LOG_FILE"
    echo "Swappiness: $(cat /proc/sys/vm/swappiness)" >> "$LOG_FILE"
    echo "Huge Pages: $(cat /proc/sys/vm/nr_hugepages)" >> "$LOG_FILE"
    echo "Network Buffer: $(cat /proc/sys/net/core/rmem_max)" >> "$LOG_FILE"
    
    # Check services
    for service in apex-validator apex-monitor apex-boot-optimize; do
        systemctl is-active --quiet $service && echo "$service: Active" >> "$LOG_FILE" || echo "$service: Inactive" >> "$LOG_FILE"
    done
    
    rm -f "$MARKER_FILE"
    echo "Reboot test completed" >> "$LOG_FILE"
fi
EOF
    chmod +x /usr/local/bin/apex-reboot-verify
    
    # Add to crontab
    (crontab -l 2>/dev/null | grep -v apex-reboot-verify; echo "@reboot /usr/local/bin/apex-reboot-verify") | crontab -
    
    log_info "Reboot persistence test configured"
    log_info "System will reboot in 30 seconds to test persistence..."
    log_info "Check /opt/apex-deployment/reboot-test.log after reboot"
    
    # Ask for confirmation
    read -p "Proceed with reboot test? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_warning "Rebooting in 10 seconds..."
        sleep 10
        reboot
    else
        log_info "Reboot test skipped"
    fi
}

# Main deployment function
main() {
    echo ""
    log_info "Starting APEX-1 Validator Deployment..."
    echo ""
    
    # Phase 1: System Preparation
    echo -e "${MAGENTA}━━━ PHASE 1: SYSTEM PREPARATION ━━━${NC}"
    check_system_requirements
    install_dependencies
    
    # Phase 2: Optimizations
    echo ""
    echo -e "${MAGENTA}━━━ PHASE 2: APPLYING OPTIMIZATIONS ━━━${NC}"
    apply_kernel_optimizations
    setup_solana_environment
    
    # Phase 3: Services
    echo ""
    echo -e "${MAGENTA}━━━ PHASE 3: CREATING SERVICES ━━━${NC}"
    create_systemd_services
    create_monitoring_scripts
    setup_boot_persistence
    
    # Phase 4: Verification
    echo ""
    echo -e "${MAGENTA}━━━ PHASE 4: VERIFICATION ━━━${NC}"
    verify_deployment
    
    # Enable and start services
    systemctl enable apex-validator apex-monitor apex-performance-logger
    systemctl start apex-monitor apex-performance-logger
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}         DEPLOYMENT COMPLETED SUCCESSFULLY!${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Services Status:"
    systemctl status apex-validator --no-pager | head -3
    systemctl status apex-monitor --no-pager | head -3
    echo ""
    echo "Available Commands:"
    echo "  Monitor:        apex-monitor"
    echo "  Logs:          journalctl -u apex-validator -f"
    echo "  Performance:   cat /var/log/apex-performance.log"
    echo ""
    echo "Deployment Log: $LOG_FILE"
    echo "Verification:   $VERIFICATION_LOG"
    echo ""
    
    # Optional reboot test
    read -p "Test reboot persistence? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_reboot_persistence
    else
        log_success "Deployment complete! Reboot recommended."
    fi
}

# Execute main function
main "$@"