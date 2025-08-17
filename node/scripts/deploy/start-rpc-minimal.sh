#!/bin/bash
# Minimal RPC setup for testing

echo "Starting minimal Solana RPC node..."

# Clean up
sudo pkill -f agave-validator 2>/dev/null
sudo rm -f /mnt/ledger/ledger.lock /mnt/ledger/admin.rpc 2>/dev/null

# Start with absolute minimum configuration
sudo -u solana /usr/local/bin/agave-validator \
    --identity /home/solana/validator-keypair.json \
    --ledger /mnt/ledger \
    --rpc-port 8899 \
    --rpc-bind-address 127.0.0.1 \
    --no-voting \
    --entrypoint entrypoint.mainnet-beta.solana.com:8001 \
    --expected-genesis-hash 5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d \
    --log -

echo "Validator started. Check logs above for status."