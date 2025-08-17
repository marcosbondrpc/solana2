#!/usr/bin/env bash
set -euo pipefail
while true; do
  echo "=== $(date) ==="
  echo "Current Slot: $(solana slot || true)"
  echo "Transaction Count: $(solana transaction-count || true)"
  echo "Cluster Version: $(solana cluster-version || true)"
  echo "Memory Usage:"
  free -h || true
  echo "Disk Usage:"
  df -h /mnt/ledger /mnt/accounts || true
  echo "------------------------"
  sleep 60
done
