#!/usr/bin/env bash
set -euo pipefail

# Find validator binary
ROOT_BIN=/root/.local/share/solana/install/active_release/bin
SOL_BIN=/home/solana/.local/share/solana/install/active_release/bin
USR_BIN=/usr/local/bin

if [ -x "$USR_BIN/agave-validator" ]; then 
    BIN="$USR_BIN/agave-validator"
elif [ -x "$ROOT_BIN/agave-validator" ]; then 
    BIN="$ROOT_BIN/agave-validator"
elif [ -x "$ROOT_BIN/solana-validator" ]; then 
    BIN="$ROOT_BIN/solana-validator"
elif [ -x "$SOL_BIN/agave-validator" ]; then 
    BIN="$SOL_BIN/agave-validator"
elif [ -x "$SOL_BIN/solana-validator" ]; then 
    BIN="$SOL_BIN/solana-validator"
else 
    BIN="$(command -v agave-validator || command -v solana-validator)"
fi

IDENTITY="/home/solana/validator-keypair.json"

# Base arguments
ARGS=(
    --identity "$IDENTITY"
    --ledger /mnt/ledger
    --accounts /mnt/accounts
    --rpc-port 8899
    --rpc-bind-address 0.0.0.0
    --dynamic-port-range 8000-8020
    --only-known-rpc
    --no-voting
    --full-rpc-api
    --limit-ledger-size 50000000
    --log "/home/solana/validator.log"
)

# Add entrypoints
ARGS+=(
    --entrypoint entrypoint.mainnet-beta.solana.com:8001
    --entrypoint entrypoint2.mainnet-beta.solana.com:8001
    --entrypoint entrypoint3.mainnet-beta.solana.com:8001
)

# Add known validators (these are well-known mainnet validators)
ARGS+=(
    --known-validator 7Np41oeYqPefeNQEHSv1UDhYrehxin3NStELsSKCT4K2
    --known-validator GdnSyH3YtwcxFvQrVVJMm1JhTS4QVX7MFsX56uJLUfiZ
    --known-validator DE1bawNcRJB9rVm3buyMVfr8mBEoyyu73NBovf2oXJsJ
    --known-validator CakcnaRDHka2gXyfbEd2d3xsvkJkqsLw2akB3zsN1D2S
    --known-validator J9keJBKL8VSCy5bAnyQdot377r9RFRHPizJYN7WQCu2s
    --known-validator 6qBmQnsWQM2a4wQXQSG84PmJr5V8b8n5uJhLWJ9tJ9dV
)

# Check for CUDA support
if command -v nvidia-smi >/dev/null 2>&1 || [ -e /dev/nvidia0 ]; then 
    ARGS+=(--cuda)
fi

echo "Starting validator with binary: $BIN"
exec "$BIN" "${ARGS[@]}"
