#!/bin/bash
# Complete setup verification script

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================"
echo "   Solana Node Setup Verification"
echo "======================================"
echo ""

PASS=0
FAIL=0

# Function to check and report
check() {
    if eval "$2" &>/dev/null; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $1"
        ((FAIL++))
    fi
}

echo "Directory Structure:"
echo "-------------------"
check "node/ directory exists" "[ -d /home/kidgordones/0solana/node ]"
check "configs/ directory exists" "[ -d /home/kidgordones/0solana/node/configs ]"
check "scripts/ directory exists" "[ -d /home/kidgordones/0solana/node/scripts ]"
check "systemd/ directory exists" "[ -d /home/kidgordones/0solana/node/systemd ]"
check "keys/ directory exists" "[ -d /home/kidgordones/0solana/node/keys ]"
check "logs/ directory exists" "[ -d /home/kidgordones/0solana/node/logs ]"
check "backups/ directory exists" "[ -d /home/kidgordones/0solana/node/backups ]"

echo ""
echo "Script Files:"
echo "-------------"
check "Master control script" "[ -x /home/kidgordones/0solana/node/solana-node.sh ]"
check "Health check script" "[ -x /home/kidgordones/0solana/node/scripts/utils/check-health.sh ]"
check "Backup script" "[ -x /home/kidgordones/0solana/node/scripts/utils/backup-keys.sh ]"
check "Monitor script" "[ -x /home/kidgordones/0solana/node/scripts/monitor/monitor-performance.sh ]"

echo ""
echo "Configuration Files:"
echo "-------------------"
check "Mainnet config" "[ -f /home/kidgordones/0solana/node/configs/mainnet/solana-rpc.env ]"
check "Testnet config" "[ -f /home/kidgordones/0solana/node/configs/testnet/solana-rpc.env ]"
check "Devnet config" "[ -f /home/kidgordones/0solana/node/configs/devnet/solana-rpc.env ]"

echo ""
echo "Service Files:"
echo "--------------"
check "Local service" "[ -f /home/kidgordones/0solana/node/systemd/services/solana-local.service ]"
check "RPC service" "[ -f /home/kidgordones/0solana/node/systemd/services/solana-rpc.service ]"

echo ""
echo "System Requirements:"
echo "-------------------"
check "Agave validator installed" "command -v agave-validator"
check "Ledger directory exists" "[ -d /mnt/ledger ]"
check "Accounts directory exists" "[ -d /mnt/accounts ]"
check "Solana user exists" "id solana"

echo ""
echo "Documentation:"
echo "--------------"
check "Main README exists" "[ -f /home/kidgordones/0solana/README.md ]"

echo ""
echo "======================================"
echo "Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo "======================================"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ Setup verification completed successfully!${NC}"
    echo ""
    echo "You can now start using the Solana node:"
    echo "  cd /home/kidgordones/0solana"
    echo "  ./node/solana-node.sh"
else
    echo -e "${YELLOW}⚠ Some checks failed. Please review and fix the issues.${NC}"
fi