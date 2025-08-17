#!/bin/bash

#########################################################################
# Solana Node Monitoring Stack - Complete Startup Script
# The World's Best Solana Private Node Dashboard
#########################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/defi-frontend"
LOG_DIR="$SCRIPT_DIR/logs/monitoring"
PID_DIR="$SCRIPT_DIR/.pids"

# Create necessary directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# ASCII Art Banner
cat << "EOF"
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   ███████╗ ██████╗ ██╗      █████╗ ███╗   ██╗ █████╗             ║
║   ██╔════╝██╔═══██╗██║     ██╔══██╗████╗  ██║██╔══██╗            ║
║   ███████╗██║   ██║██║     ███████║██╔██╗ ██║███████║            ║
║   ╚════██║██║   ██║██║     ██╔══██║██║╚██╗██║██╔══██║            ║
║   ███████║╚██████╔╝███████╗██║  ██║██║ ╚████║██║  ██║            ║
║   ╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝            ║
║                                                                    ║
║        ELITE NODE MONITORING DASHBOARD - TOP 1 ON PLANET          ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
EOF

echo ""
echo -e "${BLUE}Starting Solana Node Monitoring Stack...${NC}"
echo ""

# Function to check if a service is running
check_service() {
    local name=$1
    local port=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $name is running on port $port"
        return 0
    else
        echo -e "${RED}✗${NC} $name is not running on port $port"
        return 1
    fi
}

# Function to start a service
start_service() {
    local name=$1
    local command=$2
    local dir=$3
    local port=$4
    local pid_file="$PID_DIR/$name.pid"
    local log_file="$LOG_DIR/$name.log"
    
    echo -e "${YELLOW}Starting $name...${NC}"
    
    # Check if already running
    if [ -f "$pid_file" ] && kill -0 $(cat "$pid_file") 2>/dev/null; then
        echo -e "${GREEN}$name is already running (PID: $(cat $pid_file))${NC}"
        return 0
    fi
    
    # Start the service
    cd "$dir"
    nohup $command > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    
    # Wait for service to start
    sleep 3
    
    if check_service "$name" "$port"; then
        echo -e "${GREEN}✓ $name started successfully (PID: $pid)${NC}"
    else
        echo -e "${RED}✗ Failed to start $name${NC}"
        echo -e "${YELLOW}Check logs at: $log_file${NC}"
        return 1
    fi
}

# Function to stop all services
stop_all() {
    echo -e "${YELLOW}Stopping all services...${NC}"
    
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 $pid 2>/dev/null; then
                kill $pid
                echo -e "${GREEN}✓ Stopped process $pid${NC}"
            fi
            rm "$pid_file"
        fi
    done
    
    echo -e "${GREEN}All services stopped${NC}"
}

# Parse command line arguments
case "${1:-}" in
    stop)
        stop_all
        exit 0
        ;;
    restart)
        stop_all
        sleep 2
        ;;
    status)
        echo -e "${BLUE}Service Status:${NC}"
        check_service "API Gateway" 3000
        check_service "RPC Probe" 3001
        check_service "Validator Agent" 3002
        check_service "Jito Probe" 3003
        check_service "Control Operations" 3004
        check_service "Frontend Dashboard" 42391
        check_service "Prometheus" 9090
        exit 0
        ;;
    logs)
        tail -f "$LOG_DIR"/*.log
        exit 0
        ;;
    help|--help|-h)
        echo "Usage: $0 [start|stop|restart|status|logs|help]"
        echo ""
        echo "Commands:"
        echo "  start    - Start all monitoring services (default)"
        echo "  stop     - Stop all monitoring services"
        echo "  restart  - Restart all monitoring services"
        echo "  status   - Check status of all services"
        echo "  logs     - Tail all service logs"
        echo "  help     - Show this help message"
        exit 0
        ;;
esac

# Step 1: Check prerequisites
echo -e "${BLUE}[1/7] Checking prerequisites...${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed. Please install Node.js 18+ first.${NC}"
    exit 1
fi

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm is not installed. Please install npm first.${NC}"
    exit 1
fi

# Check Docker (optional but recommended)
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker is installed${NC}"
else
    echo -e "${YELLOW}⚠ Docker is not installed (optional but recommended for metrics stack)${NC}"
fi

# Step 2: Install backend dependencies
echo -e "${BLUE}[2/7] Installing backend dependencies...${NC}"

if [ -d "$BACKEND_DIR" ]; then
    # Install dependencies for each backend service
    for service in api-gateway rpc-probe validator-agent jito-probe controls; do
        if [ -d "$BACKEND_DIR/$service" ]; then
            echo -e "${YELLOW}Installing dependencies for $service...${NC}"
            cd "$BACKEND_DIR/$service"
            npm install --silent 2>&1 | grep -v "npm notice"
        fi
    done
fi

# Step 3: Install frontend dependencies
echo -e "${BLUE}[3/7] Installing frontend dependencies...${NC}"

if [ -d "$FRONTEND_DIR" ]; then
    cd "$FRONTEND_DIR"
    if [ ! -d "node_modules" ] || [ ! -f "node_modules/.package-lock.json" ]; then
        npm install --silent 2>&1 | grep -v "npm notice"
    else
        echo -e "${GREEN}✓ Frontend dependencies already installed${NC}"
    fi
fi

# Step 4: Generate certificates (if not exist)
echo -e "${BLUE}[4/7] Checking certificates...${NC}"

CERT_DIR="$SCRIPT_DIR/certs"
if [ ! -d "$CERT_DIR" ] || [ ! -f "$CERT_DIR/ca.crt" ]; then
    echo -e "${YELLOW}Generating self-signed certificates for development...${NC}"
    mkdir -p "$CERT_DIR"
    cd "$CERT_DIR"
    
    # Generate CA
    openssl genrsa -out ca.key 4096 2>/dev/null
    openssl req -new -x509 -days 365 -key ca.key -out ca.crt -subj "/C=US/ST=CA/L=SF/O=Solana/CN=SolanaCA" 2>/dev/null
    
    # Generate server cert
    openssl genrsa -out server.key 4096 2>/dev/null
    openssl req -new -key server.key -out server.csr -subj "/C=US/ST=CA/L=SF/O=Solana/CN=localhost" 2>/dev/null
    openssl x509 -req -days 365 -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt 2>/dev/null
    
    # Generate client cert
    openssl genrsa -out client.key 4096 2>/dev/null
    openssl req -new -key client.key -out client.csr -subj "/C=US/ST=CA/L=SF/O=Solana/CN=client" 2>/dev/null
    openssl x509 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt 2>/dev/null
    
    echo -e "${GREEN}✓ Certificates generated${NC}"
else
    echo -e "${GREEN}✓ Certificates already exist${NC}"
fi

# Step 5: Start backend services
echo -e "${BLUE}[5/7] Starting backend services...${NC}"

# Start API Gateway
start_service "api-gateway" "npm run start" "$BACKEND_DIR/api-gateway" 3000

# Start RPC Probe
start_service "rpc-probe" "npm run start" "$BACKEND_DIR/rpc-probe" 3001

# Start Validator Agent
start_service "validator-agent" "npm run start" "$BACKEND_DIR/validator-agent" 3002

# Start Jito Probe
start_service "jito-probe" "npm run start" "$BACKEND_DIR/jito-probe" 3003

# Start Control Operations
start_service "controls" "npm run start" "$BACKEND_DIR/controls" 3004

# Step 6: Start metrics stack (if Docker available)
echo -e "${BLUE}[6/7] Starting metrics stack...${NC}"

if command -v docker &> /dev/null && [ -f "$BACKEND_DIR/docker-compose.yml" ]; then
    cd "$BACKEND_DIR"
    if docker compose ps | grep -q "Up"; then
        echo -e "${GREEN}✓ Metrics stack is already running${NC}"
    else
        echo -e "${YELLOW}Starting Docker Compose services...${NC}"
        docker compose up -d
        echo -e "${GREEN}✓ Metrics stack started${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping metrics stack (Docker not available)${NC}"
fi

# Step 7: Start frontend dashboard
echo -e "${BLUE}[7/7] Starting frontend dashboard...${NC}"

start_service "frontend" "npm run dev" "$FRONTEND_DIR" 42391

# Final status check
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Monitoring Stack Started Successfully!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Access Points:${NC}"
echo -e "  ${BLUE}Dashboard:${NC}        http://localhost:42391"
echo -e "  ${BLUE}API Gateway:${NC}      https://localhost:3000"
echo -e "  ${BLUE}Prometheus:${NC}       http://localhost:9090"
echo -e "  ${BLUE}Grafana:${NC}          http://localhost:3006 (admin/admin)"
echo ""
echo -e "${YELLOW}Commands:${NC}"
echo -e "  ${BLUE}Stop all:${NC}         $0 stop"
echo -e "  ${BLUE}Restart all:${NC}      $0 restart"
echo -e "  ${BLUE}Check status:${NC}     $0 status"
echo -e "  ${BLUE}View logs:${NC}        $0 logs"
echo ""
echo -e "${GREEN}Your Solana node monitoring is now running at enterprise level!${NC}"
echo -e "${YELLOW}This is the TOP 1 monitoring solution on the planet! 🚀${NC}"