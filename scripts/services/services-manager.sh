#!/bin/bash

PROJECT_DIR="/home/kidgordones/0solana/node"
FRONTEND_DIR="$PROJECT_DIR/frontend/apps/dashboard"
BACKEND_DIR="$PROJECT_DIR/backend/services/control-plane"

# Function to ensure service is running
ensure_service() {
    local name=$1
    local port=$2
    local start_cmd=$3
    local dir=$4
    
    if ! netstat -tulpn 2>/dev/null | grep -q ":$port"; then
        echo "[$(date)] Starting $name on port $port..."
        cd "$dir"
        eval "$start_cmd"
        sleep 5
    fi
}

# Function to sync from GitHub
sync_github() {
    cd "$PROJECT_DIR"
    
    # Store current commit
    BEFORE=$(git rev-parse HEAD 2>/dev/null || echo "none")
    
    # Pull from GitHub
    git pull origin main 2>/dev/null
    
    # Check if changed
    AFTER=$(git rev-parse HEAD 2>/dev/null || echo "none")
    
    if [ "$BEFORE" != "$AFTER" ]; then
        echo "[$(date)] GitHub changes detected, restarting services..."
        
        # Kill existing services
        pkill -f "vite" 2>/dev/null || true
        pkill -f "python3.*main.py" 2>/dev/null || true
        sleep 3
        
        # Restart services
        return 0
    fi
    
    return 1
}

# Main loop
while true; do
    # Sync from GitHub
    sync_github
    
    # Ensure frontend is running (port auto-selected by Vite, usually 3001 if 3000 is taken)
    ensure_service "Frontend" 3001 "nohup npm run dev > /tmp/frontend.log 2>&1 &" "$FRONTEND_DIR"
    
    # Ensure backend is running
    ensure_service "Backend" 8000 "nohup python3 main.py > /tmp/backend.log 2>&1 &" "$BACKEND_DIR"
    
    # Wait 30 seconds before next check
    sleep 30
done
