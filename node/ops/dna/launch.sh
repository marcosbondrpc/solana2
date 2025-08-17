#!/usr/bin/env bash
set -euo pipefail

# 1) Kernel/stack knobs
sudo sysctl -w net.core.busy_poll=50 net.core.busy_read=50 \
  net.core.rmem_max=134217728 net.core.wmem_max=134217728 \
  net.core.netdev_max_backlog=250000
sudo sysctl -w net.ipv4.udp_rmem_min=4194304 net.ipv4.udp_wmem_min=4194304

# 2) NIC features
NIC="${NIC:-eth0}"
sudo ethtool -K "$NIC" gro off lro off rxhash on || true

# 3) NUMA/CPU pinning example
export RUSTFLAGS="-C target-cpu=native -C lto=fat -C codegen-units=1 -C strip=symbols"
export TOKIO_WORKER_THREADS=6

taskset -c 2-5 ./target/release/data-ingestion &
taskset -c 6-7 ./target/release/sandwich-detector &
taskset -c 8-9 ./target/release/jito-engine &
wait