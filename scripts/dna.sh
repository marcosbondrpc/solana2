#!/usr/bin/env bash
set -euo pipefail

# Network buffers
sysctl -w net.core.rmem_max=33554432 >/dev/null || true
sysctl -w net.core.wmem_max=33554432 >/dev/null || true

# Busy poll (kernel bypass hints for NIC drivers that support it)
sysctl -w net.core.busy_poll=50 >/dev/null || true
sysctl -w net.core.busy_read=50 >/dev/null || true

# If NIC is provided as $NIC, disable GRO/LRO to reduce latency (validate first)
if [[ "${NIC:-}" != "" ]]; then
  ethtool -K "$NIC" gro off lro off || true
fi

# PTP time sync if PHC clock is available via $NIC
if command -v ptp4l >/dev/null 2>&1 && command -v phc2sys >/dev/null 2>&1 && [[ "${NIC:-}" != "" ]]; then
  (ptp4l -2 -i "$NIC" -m & disown) || true
  (phc2sys -s "$NIC" -w -m & disown) || true
fi

echo "Applied low-latency system tuning."


