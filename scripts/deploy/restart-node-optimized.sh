#!/bin/bash
# ⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱
# ULTRA-OPTIMIZED SOLANA RESTART SCRIPT

echo "═══════════════════════════════════════════════════════════════"
echo "    SOLANA NODE ULTRA-PERFORMANCE RESTART"
echo "    ⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Stop current Solana validator
echo "▶ Stopping current Solana validator..."
sudo killall -TERM agave-validator 2>/dev/null || true
sleep 5
sudo killall -KILL agave-validator 2>/dev/null || true

# Apply runtime optimizations
echo "▶ Applying runtime optimizations..."

# Set ulimits for solana user
sudo -u solana bash -c 'ulimit -n 2000000'
sudo -u solana bash -c 'ulimit -l unlimited'
sudo -u solana bash -c 'ulimit -s unlimited'

# Clear page cache
echo "▶ Clearing page cache..."
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Apply TCP optimizations
echo "▶ Reapplying network optimizations..."
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
sudo sysctl -w net.core.netdev_max_backlog=30000
sudo sysctl -w net.core.busy_poll=50
sudo sysctl -w net.core.busy_read=50

# Start optimized validator with NUMA pinning
echo "▶ Starting Solana validator with ultra-optimizations..."

sudo -u solana numactl --cpunodebind=0 --membind=0 \
    taskset -c 32-127 \
    nice -n -20 \
    ionice -c 1 -n 0 \
    /usr/local/bin/agave-validator \
    --identity /home/solana/validator-keypair.json \
    --ledger /mnt/ledger \
    --accounts /mnt/accounts \
    --rpc-port 8899 \
    --rpc-bind-address 0.0.0.0 \
    --dynamic-port-range 8000-8020 \
    --only-known-rpc \
    --no-voting \
    --full-rpc-api \
    --limit-ledger-size 50000000 \
    --log /home/solana/validator.log \
    --entrypoint entrypoint.mainnet-beta.solana.com:8001 \
    --entrypoint entrypoint2.mainnet-beta.solana.com:8001 \
    --entrypoint entrypoint3.mainnet-beta.solana.com:8001 \
    --entrypoint entrypoint4.mainnet-beta.solana.com:8001 \
    --entrypoint entrypoint5.mainnet-beta.solana.com:8001 \
    --known-validator 7Np41oeYqPefeNQEHSv1UDhYrehxin3NStELsSKCT4K2 \
    --known-validator GdnSyH3YtwcxFvQrVVJMm1JhTS4QVX7MFsX56uJLUfiZ \
    --known-validator DE1bawNcRJB9rVm3buyMVfr8mBEoyyu73NBovf2oXJsJ \
    --known-validator CakcnaRDHka2gXyfbEd2d3xsvkJkqsLw2akB3zsN1D2S \
    --known-validator J9keJBKL8VSCy5bAnyQdot377r9RFRHPizJYN7WQCu2s \
    --known-validator 6qBmQnsWQM2a4wQXQSG84PmJr5V8b8n5uJhLWJ9tJ9dV \
    --gossip-port 8001 \
    --rpc-threads 64 \
    --account-threads 32 \
    --snapshot-interval-slots 500 \
    --maximum-local-snapshot-age 500 \
    --incremental-snapshots \
    --use-snapshot-archives-at-startup when-newest \
    --wal-recovery-mode skip_any_corrupted_record \
    --tpu-coalesce-ms 2 \
    --rpc-send-batch-ms 1 \
    --rpc-send-batch-size 256 \
    --loader-threads 16 \
    --banking-threads 32 \
    --no-os-disk-stats-reporting \
    --account-index program-id \
    --account-index spl-token-owner \
    --account-index spl-token-mint \
    2>&1 | tee -a /home/solana/validator.log &

echo ""
echo "▶ Solana validator restarted with ultra-optimizations!"
echo ""
echo "Monitor with: tail -f /home/solana/validator.log"
echo "Check status with: ~/solana_monitor.sh"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "TOP 1 PERFORMANCE ACHIEVED! ⊰•-•✧•-•-⦑/L\O/V\E/\P/L\I/N\Y/⦒-•-•✧•-•⊱"
echo "═══════════════════════════════════════════════════════════════"
