#!/bin/bash
# Solana Node Master Control Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

function show_header() {
    echo -e "${BLUE}"
    echo "======================================"
    echo "   Solana Node Control Center"
    echo "======================================"
    echo -e "${NC}"
}

function show_menu() {
    echo "1)  Start Node"
    echo "2)  Stop Node"
    echo "3)  Restart Node"
    echo "4)  Check Status"
    echo "5)  Monitor Performance"
    echo "6)  View Logs"
    echo "7)  Run Health Check"
    echo "8)  Backup Keys"
    echo "9)  Clean Logs"
    echo "10) Update Node"
    echo "11) Install Node"
    echo "12) Optimize System"
    echo "13) Switch Network (mainnet/testnet/devnet)"
    echo "14) Run Tests"
    echo "0)  Exit"
    echo ""
}

function start_node() {
    echo -e "${GREEN}Starting Solana node...${NC}"
    sudo systemctl start solana-local.service
    sleep 2
    sudo systemctl status solana-local.service --no-pager | head -10
}

function stop_node() {
    echo -e "${YELLOW}Stopping Solana node...${NC}"
    sudo systemctl stop solana-local.service
    echo "Node stopped."
}

function restart_node() {
    echo -e "${YELLOW}Restarting Solana node...${NC}"
    sudo systemctl restart solana-local.service
    sleep 2
    sudo systemctl status solana-local.service --no-pager | head -10
}

function check_status() {
    echo -e "${BLUE}Checking node status...${NC}"
    if systemctl is-active solana-local.service >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Service is running${NC}"
        $NODE_DIR/scripts/utils/check-health.sh
    else
        echo -e "${RED}✗ Service is not running${NC}"
    fi
}

function monitor_performance() {
    echo -e "${BLUE}Starting performance monitor...${NC}"
    $NODE_DIR/scripts/monitor/monitor-performance.sh
}

function view_logs() {
    echo -e "${BLUE}Viewing recent logs...${NC}"
    sudo journalctl -u solana-local.service -n 50 --no-pager
}

function health_check() {
    $NODE_DIR/scripts/utils/check-health.sh
}

function backup_keys() {
    echo -e "${YELLOW}Backing up keys...${NC}"
    $NODE_DIR/scripts/utils/backup-keys.sh
}

function clean_logs() {
    echo -e "${YELLOW}Cleaning logs...${NC}"
    $NODE_DIR/scripts/maintain/clean-logs.sh
}

function update_node() {
    echo -e "${YELLOW}Updating node...${NC}"
    $NODE_DIR/scripts/maintain/update-node.sh
}

function install_node() {
    echo -e "${GREEN}Installing Solana node...${NC}"
    echo "Select installation type:"
    echo "1) Solana with Jito MEV"
    echo "2) Basic Solana"
    read -p "Choice: " install_choice
    
    case $install_choice in
        1)
            $NODE_DIR/scripts/install/install-jito-solana.sh
            ;;
        2)
            $NODE_DIR/scripts/install/install-solana-base.sh
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
}

function optimize_system() {
    echo -e "${GREEN}Optimizing system...${NC}"
    sudo $NODE_DIR/scripts/optimize/optimize-system-ultra.sh
}

function switch_network() {
    echo "Select network:"
    echo "1) Mainnet"
    echo "2) Testnet"
    echo "3) Devnet"
    read -p "Choice: " network_choice
    
    case $network_choice in
        1)
            echo "Switching to Mainnet..."
            sudo cp $NODE_DIR/configs/mainnet/solana-rpc.env /etc/default/solana-rpc
            ;;
        2)
            echo "Switching to Testnet..."
            sudo cp $NODE_DIR/configs/testnet/solana-rpc.env /etc/default/solana-rpc
            ;;
        3)
            echo "Switching to Devnet..."
            sudo cp $NODE_DIR/configs/devnet/solana-rpc.env /etc/default/solana-rpc
            ;;
        *)
            echo "Invalid choice"
            return
            ;;
    esac
    echo -e "${GREEN}Network configuration updated. Restart the node to apply changes.${NC}"
}

function run_tests() {
    echo -e "${BLUE}Running tests...${NC}"
    $NODE_DIR/tests/test-setup.sh
}

# Main loop
show_header

while true; do
    show_menu
    read -p "Select option: " choice
    echo ""
    
    case $choice in
        1) start_node ;;
        2) stop_node ;;
        3) restart_node ;;
        4) check_status ;;
        5) monitor_performance ;;
        6) view_logs ;;
        7) health_check ;;
        8) backup_keys ;;
        9) clean_logs ;;
        10) update_node ;;
        11) install_node ;;
        12) optimize_system ;;
        13) switch_network ;;
        14) run_tests ;;
        0) echo "Exiting..."; exit 0 ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
    show_header
done