#!/bin/bash
# Clean old logs and maintain disk space

set -e

LOG_DIR="/home/kidgordones/0solana/node/logs"
VALIDATOR_LOG="/home/solana/validator.log"
DAYS_TO_KEEP=7

echo "Starting log cleanup..."

# Rotate and compress validator log
if [ -f "$VALIDATOR_LOG" ]; then
    SIZE=$(du -h "$VALIDATOR_LOG" | cut -f1)
    echo "Current validator log size: $SIZE"
    
    if [ -s "$VALIDATOR_LOG" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        cp "$VALIDATOR_LOG" "${LOG_DIR}/validator_${TIMESTAMP}.log"
        gzip "${LOG_DIR}/validator_${TIMESTAMP}.log"
        > "$VALIDATOR_LOG"
        echo "✓ Validator log rotated"
    fi
fi

# Clean old compressed logs
find "$LOG_DIR" -name "*.gz" -mtime +$DAYS_TO_KEEP -delete 2>/dev/null
echo "✓ Old logs cleaned (kept last $DAYS_TO_KEEP days)"

# Clean systemd journal
sudo journalctl --vacuum-time=${DAYS_TO_KEEP}d 2>/dev/null || true
echo "✓ Systemd journal cleaned"

# Check disk space
echo ""
echo "Current disk usage:"
df -h /mnt/ledger /mnt/accounts / | grep -v Filesystem

echo ""
echo "Log cleanup completed!"