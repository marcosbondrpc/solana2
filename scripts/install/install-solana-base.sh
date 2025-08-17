#!/bin/bash
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root"
    exit 1
fi

echo "==========================================="
echo "Ultra-Low-Latency Agave RPC Node Setup"
echo "==========================================="

ACCOUNTS_DEV="${ACCOUNTS_DEV:-/dev/nvme0n1}"
LEDGER_DEV="${LEDGER_DEV:-/dev/nvme1n1}"

echo "[1/15] Installing required packages..."
apt-get update
apt-get install -y \
    tuned tuned-utils \
    chrony \
    ethtool \
    linux-tools-common linux-tools-$(uname -r) \
    xfsprogs \
    nvme-cli \
    fio \
    mtr-tiny \
    jq \
    curl \
    wget \
    build-essential \
    pkg-config \
    libssl-dev \
    libudev-dev \
    screen \
    htop \
    iotop \
    sysstat \
    net-tools

echo "[2/15] Applying sysctl tuning for low latency..."
cat > /etc/sysctl.d/99-solana-low-latency.conf << 'EOF'
# Network tuning for ultra-low latency
net.core.rmem_default = 134217728
net.core.rmem_max = 134217728
net.core.wmem_default = 134217728
net.core.wmem_max = 134217728
net.core.netdev_max_backlog = 30000
net.core.netdev_budget = 600
net.core.netdev_budget_usecs = 2000
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_mtu_probing = 1
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_no_metrics_save = 1
net.ipv4.tcp_moderate_rcvbuf = 1
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 15
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_tw_reuse = 1
net.ipv4.ip_local_port_range = 8000 65535

# Memory and VM tuning
vm.swappiness = 1
vm.vfs_cache_pressure = 50
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5
vm.zone_reclaim_mode = 0
vm.max_map_count = 1000000

# File system tuning
fs.file-max = 2097152
fs.nr_open = 2097152

# Kernel tuning
kernel.numa_balancing = 0
kernel.sched_min_granularity_ns = 10000000
kernel.sched_wakeup_granularity_ns = 15000000
EOF
sysctl -p /etc/sysctl.d/99-solana-low-latency.conf

echo "[3/15] Disabling Transparent Huge Pages..."
cat > /etc/systemd/system/disable-thp.service << 'EOF'
[Unit]
Description=Disable Transparent Huge Pages
Before=solana-rpc.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled'
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/defrag'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now disable-thp.service

echo "[4/15] Setting CPU governor to performance..."
cpupower frequency-set -g performance || echo "cpupower not available, skipping"

echo "[5/15] Configuring TuneD for latency-performance..."
systemctl enable --now tuned
tuned-adm profile latency-performance

echo "[6/15] Configuring fstrim timer..."
systemctl enable --now fstrim.timer

echo "[7/15] Provisioning storage with XFS..."
if [ -b "$ACCOUNTS_DEV" ]; then
    echo "Formatting $ACCOUNTS_DEV for accounts..."
    wipefs -a "$ACCOUNTS_DEV" || true
    mkfs.xfs -f -L ACCOUNTS -m bigtime=1 -d agcount=8 "$ACCOUNTS_DEV"
else
    echo "WARNING: $ACCOUNTS_DEV not found, skipping accounts disk setup"
fi

if [ -b "$LEDGER_DEV" ]; then
    echo "Formatting $LEDGER_DEV for ledger..."
    wipefs -a "$LEDGER_DEV" || true
    mkfs.xfs -f -L LEDGER -m bigtime=1 -d agcount=8 "$LEDGER_DEV"
else
    echo "WARNING: $LEDGER_DEV not found, skipping ledger disk setup"
fi

echo "[8/15] Creating mount points and mounting filesystems..."
mkdir -p /mnt/accounts /mnt/ledger

cat >> /etc/fstab << 'EOF'
LABEL=ACCOUNTS /mnt/accounts xfs nodiratime,noatime,inode64,logbufs=8,logbsize=256k 0 0
LABEL=LEDGER /mnt/ledger xfs nodiratime,noatime,inode64,logbufs=8,logbsize=256k 0 0
EOF

mount /mnt/accounts || echo "Failed to mount accounts, continuing..."
mount /mnt/ledger || echo "Failed to mount ledger, continuing..."

mkdir -p /mnt/ledger/snapshots
chmod 755 /mnt/ledger/snapshots

echo "[9/15] Creating solana user..."
useradd -m -s /bin/bash solana || echo "User solana already exists"
usermod -aG sudo solana || true

echo "[10/15] Installing Agave validator..."
mkdir -p /opt/solana
chown solana:solana /opt/solana

if ! command -v agave-validator &> /dev/null; then
    sudo -u solana sh -c "curl --proto '=https' --tlsv1.2 -sSf https://release.anza.xyz/stable/install | sh -s -- --no-modify-path"
    ln -sf /home/solana/.local/share/solana/install/active_release/bin/* /usr/local/bin/
else
    echo "Agave validator already installed"
fi

echo "[11/15] Setting up file limits..."
cat > /etc/security/limits.d/solana.conf << 'EOF'
solana soft nofile 1000000
solana hard nofile 1000000
solana soft nproc 1000000
solana hard nproc 1000000
* soft nofile 1000000
* hard nofile 1000000
EOF

echo "[12/15] Configuring NIC tuning..."
IFACE=$(ip -o -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
if [ -n "$IFACE" ]; then
    echo "Configuring interface: $IFACE"
    
    ethtool -C "$IFACE" adaptive-rx off adaptive-tx off rx-usecs 0 tx-usecs 0 2>/dev/null || echo "Coalescing not fully supported"
    ethtool -K "$IFACE" gro off lro off 2>/dev/null || echo "Offload settings not fully supported"
    ethtool -G "$IFACE" rx 4096 tx 4096 2>/dev/null || echo "Ring buffer resize not supported"
    
    cat > /etc/systemd/system/ethtool@.service << 'EOF'
[Unit]
Description=Apply ethtool settings for %i
After=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/apply-nic-tuning.sh %i

[Install]
WantedBy=multi-user.target
EOF

    cat > /usr/local/sbin/apply-nic-tuning.sh << 'EOF'
#!/bin/bash
IFACE=$1
ethtool -C "$IFACE" adaptive-rx off adaptive-tx off rx-usecs 0 tx-usecs 0 2>/dev/null || true
ethtool -K "$IFACE" gro off lro off 2>/dev/null || true
ethtool -G "$IFACE" rx 4096 tx 4096 2>/dev/null || true
EOF
    chmod +x /usr/local/sbin/apply-nic-tuning.sh
    
    systemctl daemon-reload
    systemctl enable --now "ethtool@${IFACE}.service"
else
    echo "WARNING: Could not detect primary network interface"
fi

echo "[13/15] Configuring chrony for time synchronization..."
cat > /etc/chrony/chrony.conf << 'EOF'
pool time.google.com iburst maxsources 4
pool time.cloudflare.com iburst maxsources 4
pool time.facebook.com iburst maxsources 4

server time1.google.com iburst
server time2.google.com iburst
server time3.google.com iburst
server time4.google.com iburst

makestep 1.0 3
rtcsync

driftfile /var/lib/chrony/chrony.drift
logdir /var/log/chrony

local stratum 10
allow 127.0.0.1
EOF
systemctl restart chrony

echo "[14/15] Setting permissions..."
chown -R solana:solana /opt/solana
chown -R solana:solana /mnt/accounts || true
chown -R solana:solana /mnt/ledger || true

echo "[15/15] Creating identity keypair if needed..."
if [ ! -f /opt/solana/identity.json ]; then
    sudo -u solana /usr/local/bin/solana-keygen new --no-bip39-passphrase --silent --outfile /opt/solana/identity.json
fi

echo "==========================================="
echo "Base system setup complete!"
echo "Next steps:"
echo "1. Review and edit /etc/default/solana-rpc"
echo "2. Start the service: systemctl start solana-rpc"
echo "3. Monitor logs: journalctl -u solana-rpc -f"
echo "4. Check health: /usr/local/sbin/solana-rpc-health.sh"
echo "==========================================="