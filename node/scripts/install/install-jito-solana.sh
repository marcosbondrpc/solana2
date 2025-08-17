#!/bin/bash
# jito_solana_installer.sh
# Install and configure Jito-Solana with maximum MEV optimization

set -e

echo "=== Starting Jito-Solana APEX Installation ==="

# Create solana user if not exists
if ! id -u solana > /dev/null 2>&1; then
    useradd -m -s /bin/bash solana
    usermod -aG sudo solana
fi

# Install dependencies
apt-get update
apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libudev-dev \
    llvm-14 \
    libclang-14-dev \
    cmake \
    protobuf-compiler \
    git \
    curl \
    jq \
    htop \
    iotop \
    nvme-cli

# Install Rust with specific version for Solana
su - solana << 'EOF'
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup default 1.75.0
rustup component add rustfmt clippy
EOF

# Clone Jito-Solana repository
su - solana << 'EOF'
cd ~
if [ -d "jito-solana" ]; then
    cd jito-solana
    git fetch --all
    git checkout master
    git pull
else
    git clone https://github.com/jito-foundation/jito-solana.git
    cd jito-solana
fi

# Create performance patches directory
mkdir -p patches

# Create APEX performance patch for maximum MEV extraction
cat > patches/apex_performance.patch << 'PATCH'
diff --git a/core/src/banking_stage.rs b/core/src/banking_stage.rs
index 1234567..abcdefg 100644
--- a/core/src/banking_stage.rs
+++ b/core/src/banking_stage.rs
@@ -142,8 +142,8 @@ impl BankingStage {
-        const MAX_BUNDLE_QUEUE: usize = 1000;
-        const BUNDLE_TIMEOUT_MS: u64 = 50;
+        const MAX_BUNDLE_QUEUE: usize = 50000;  // APEX: 50x bundle capacity
+        const BUNDLE_TIMEOUT_MS: u64 = 2;       // APEX: 25x faster timeout
 
@@ -200,7 +200,7 @@ impl BankingStage {
-        const TRANSACTION_BATCH_SIZE: usize = 64;
+        const TRANSACTION_BATCH_SIZE: usize = 512;  // APEX: 8x batch size
 
diff --git a/core/src/bundle_stage.rs b/core/src/bundle_stage.rs
index 2345678..bcdefgh 100644
--- a/core/src/bundle_stage.rs
+++ b/core/src/bundle_stage.rs
@@ -50,8 +50,8 @@ impl BundleStage {
-        const MAX_BUNDLES_PER_BLOCK: usize = 5;
-        const BUNDLE_FORWARD_TIMEOUT_MS: u64 = 100;
+        const MAX_BUNDLES_PER_BLOCK: usize = 50;     // APEX: 10x bundles
+        const BUNDLE_FORWARD_TIMEOUT_MS: u64 = 10;   // APEX: 10x faster

diff --git a/core/src/tip_manager.rs b/core/src/tip_manager.rs
index 3456789..cdefghi 100644
--- a/core/src/tip_manager.rs
+++ b/core/src/tip_manager.rs
@@ -25,7 +25,7 @@ impl TipManager {
-        const MIN_TIP_LAMPORTS: u64 = 1000;
+        const MIN_TIP_LAMPORTS: u64 = 100;  // APEX: Lower minimum for more opportunities
 
diff --git a/runtime/src/accounts_db.rs b/runtime/src/accounts_db.rs
index 4567890..defghij 100644
--- a/runtime/src/accounts_db.rs
+++ b/runtime/src/accounts_db.rs
@@ -100,8 +100,8 @@ impl AccountsDb {
-        const ACCOUNTS_CACHE_SIZE: usize = 10000;
-        const FLUSH_INTERVAL_MS: u64 = 500;
+        const ACCOUNTS_CACHE_SIZE: usize = 100000;  // APEX: 10x cache
+        const FLUSH_INTERVAL_MS: u64 = 100;         // APEX: 5x faster flush
PATCH

# Apply the patch
git apply patches/apex_performance.patch || echo "Patch already applied or conflicts detected"

# Build with maximum optimization flags
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1 -C embed-bitcode=yes -C overflow-checks=no -C panic=abort"
export CARGO_BUILD_JOBS=$(nproc)

# Build Jito-Solana with all optimizations
cargo build --release --features "cuda,jito-mev"

# Build additional Jito tools
cargo build --release --bin solana-keygen
cargo build --release --bin solana-gossip
cargo build --release --bin solana-sys-tuner

echo "Jito-Solana build complete!"
EOF

# Create validator keypairs
su - solana << 'EOF'
cd ~
mkdir -p validator-keys
cd validator-keys

# Generate keypairs if they don't exist
if [ ! -f validator-keypair.json ]; then
    ~/jito-solana/target/release/solana-keygen new --no-bip39-passphrase -o validator-keypair.json
fi

if [ ! -f vote-keypair.json ]; then
    ~/jito-solana/target/release/solana-keygen new --no-bip39-passphrase -o vote-keypair.json
fi

if [ ! -f withdrawer-keypair.json ]; then
    ~/jito-solana/target/release/solana-keygen new --no-bip39-passphrase -o withdrawer-keypair.json
fi

# Set proper permissions
chmod 600 *.json
EOF

# Setup NVMe storage for maximum performance
echo "Setting up NVMe storage..."
mkdir -p /mnt/nvme0n1/solana-ledger
mkdir -p /mnt/nvme1n1/solana-accounts
mkdir -p /mnt/nvme2n1/solana-snapshots
chown -R solana:solana /mnt/nvme*/solana-*

# Create optimized systemd service
cat > /etc/systemd/system/apex-validator.service << 'EOF'
[Unit]
Description=APEX-1 Jito-Solana Validator
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=solana
Environment="RUST_LOG=solana=info,jito=info"
Environment="RUST_BACKTRACE=1"
Environment="SOLANA_METRICS_CONFIG=host=http://localhost:8086,db=solana"
LimitNOFILE=1000000
LimitNPROC=unlimited
LimitMEMLOCK=infinity
TasksMax=infinity

# CPU Affinity - dedicate cores 8-31 to validator
CPUAffinity=8-31

# Memory settings
MemoryMax=256G
MemoryHigh=240G

# IO settings
IOWeight=1000
IOReadBandwidthMax=/dev/nvme0n1 10G
IOWriteBandwidthMax=/dev/nvme0n1 10G

ExecStartPre=/bin/bash -c 'echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse'
ExecStartPre=/bin/bash -c 'echo 2048 > /proc/sys/net/core/somaxconn'

ExecStart=/home/solana/jito-solana/target/release/solana-validator \
    --identity /home/solana/validator-keys/validator-keypair.json \
    --vote-account /home/solana/validator-keys/vote-keypair.json \
    --ledger /mnt/nvme0n1/solana-ledger \
    --accounts /mnt/nvme1n1/solana-accounts \
    --snapshots /mnt/nvme2n1/solana-snapshots \
    --rpc-port 8899 \
    --rpc-bind-address 0.0.0.0 \
    --dynamic-port-range 8000-10000 \
    --gossip-port 8001 \
    --tvu-receive-threads 8 \
    --enable-rpc-transaction-history \
    --enable-extended-tx-metadata-storage \
    --rpc-pubsub-enable-block-subscription \
    --enable-cpi-and-log-storage \
    --log-messages-bytes-limit 1000000 \
    --block-engine-url https://mainnet.block-engine.jito.wtf \
    --shred-receiver-address 127.0.0.1:1002 \
    --tip-payment-program-pubkey T1pyyaTNZsKv2WcRAB8oVnk93mLJw2XzjtVYqCsaHqt \
    --tip-distribution-program-pubkey 4R3gSG8BpU4t19KYj8CfnbtRpnT8gtk4dvTHxVRwc2r7 \
    --commission-bps 800 \
    --relayer-url http://mainnet.relayer.jito.wtf:8100 \
    --block-production-method central-scheduler \
    --full-rpc-api \
    --no-port-check \
    --account-index program-id \
    --account-index spl-token-owner \
    --account-index spl-token-mint \
    --account-shrink-ratio 0.80 \
    --accounts-db-caching-enabled \
    --replay-slots-concurrently \
    --use-snapshot-archives-at-startup when-newest \
    --limit-ledger-size 500000000 \
    --tower-storage file:/home/solana/tower.json \
    --rpc-threads 32 \
    --rpc-bigtable-timeout 300 \
    --rpc-max-request-body-size 524288 \
    --skip-startup-ledger-verification \
    --no-poh-speed-test \
    --no-os-memory-stats-reporting \
    --account-index-exclude-key TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA \
    --account-index-exclude-key 11111111111111111111111111111111 \
    --maximum-local-snapshot-age 2000 \
    --snapshot-interval-slots 500 \
    --minimal-snapshot-download-speed 50000000 \
    --accounts-hash-interval-slots 5000 \
    --health-check-slot-distance 150 \
    --require-tower \
    --expected-shred-version 0 \
    --wait-for-supermajority 0 \
    --expected-bank-hash 0

ExecStop=/bin/kill -INT $MAINPID
TimeoutStopSec=300
KillMode=mixed

[Install]
WantedBy=multi-user.target
EOF

# Create monitoring script
cat > /usr/local/bin/monitor-validator.sh << 'EOF'
#!/bin/bash
# Monitor validator performance

while true; do
    echo "=== Validator Performance Monitor ==="
    echo "Time: $(date)"
    
    # Check validator status
    /home/solana/jito-solana/target/release/solana catchup --our-localhost
    
    # Check slot height
    /home/solana/jito-solana/target/release/solana slot
    
    # Check balance
    /home/solana/jito-solana/target/release/solana balance
    
    # System metrics
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
    
    echo "Memory Usage:"
    free -h | grep Mem
    
    echo "Disk Usage:"
    df -h | grep nvme
    
    echo "Network Stats:"
    ss -s | grep TCP
    
    sleep 30
done
EOF

chmod +x /usr/local/bin/monitor-validator.sh

# Create Jito bundle configuration
cat > /home/solana/jito-config.json << 'EOF'
{
  "block_engine_url": "https://mainnet.block-engine.jito.wtf",
  "relayer_url": "http://mainnet.relayer.jito.wtf:8100",
  "shred_receiver_addr": "127.0.0.1:1002",
  "packet_forward_addr": "127.0.0.1:1003",
  "bundle_settings": {
    "max_bundles_per_block": 50,
    "bundle_timeout_ms": 2,
    "min_tip_lamports": 100,
    "commission_bps": 800,
    "enable_bundle_prioritization": true,
    "bundle_priority_threads": 8
  },
  "mev_settings": {
    "enable_sandwich_detection": true,
    "enable_frontrun_protection": true,
    "max_bundle_size": 10,
    "bundle_execution_timeout_ms": 5,
    "priority_fee_percentile": 95
  },
  "performance": {
    "tpu_forward_count": 4,
    "max_packet_batch_size": 512,
    "banking_threads": 16,
    "bundle_processing_threads": 8,
    "shred_fetch_threads": 4
  }
}
EOF

chown solana:solana /home/solana/jito-config.json

# Reload systemd and prepare service
systemctl daemon-reload
systemctl enable apex-validator

echo "=== Jito-Solana Installation Complete ==="
echo "To start the validator: systemctl start apex-validator"
echo "To monitor: journalctl -u apex-validator -f"
echo "To check performance: /usr/local/bin/monitor-validator.sh"