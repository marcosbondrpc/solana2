#!/bin/bash

# GitHub Pull Sync Service
# Pulls changes from GitHub and restarts services if files changed

PROJECT_DIR="/home/kidgordones/0solana/node"
FRONTEND_DIR="$PROJECT_DIR/frontend/apps/dashboard"
BACKEND_DIR="$PROJECT_DIR/backend/services/control-plane"
LOG_FILE="$PROJECT_DIR/github-sync.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if service is running
is_running() {
    local port=$1
    netstat -tulpn 2>/dev/null | grep -q ":$port"
}

# Function to stop frontend
stop_frontend() {
    log_message "Stopping frontend..."
    pkill -f "vite.*--port 3000" 2>/dev/null || true
    sleep 2
}

# Function to start frontend
start_frontend() {
    if is_running 3000; then
        log_message "Frontend already running on port 3000"
    else
        log_message "Starting frontend on port 3000..."
        cd "$FRONTEND_DIR"
        nohup npm run dev > "$PROJECT_DIR/frontend.log" 2>&1 &
        sleep 5
        if is_running 3000; then
            log_message "✅ Frontend started successfully"
        else
            log_message "❌ Failed to start frontend"
        fi
    fi
}

# Function to stop backend
stop_backend() {
    log_message "Stopping backend..."
    pkill -f "python3.*main.py" 2>/dev/null || true
    sleep 2
}

# Function to start backend
start_backend() {
    if is_running 8000; then
        log_message "Backend already running on port 8000"
    else
        log_message "Starting backend on port 8000..."
        cd "$BACKEND_DIR"
        nohup python3 main.py > "$PROJECT_DIR/backend.log" 2>&1 &
        sleep 3
        if is_running 8000; then
            log_message "✅ Backend started successfully"
        else
            log_message "❌ Failed to start backend"
        fi
    fi
}

# Function to pull from GitHub
pull_changes() {
    cd "$PROJECT_DIR"
    
    # Get current commit
    BEFORE_PULL=$(git rev-parse HEAD)
    
    # Pull changes
    log_message "Pulling from GitHub..."
    git pull origin main 2>&1 | tee -a "$LOG_FILE"
    
    # Get new commit
    AFTER_PULL=$(git rev-parse HEAD)
    
    # Check if there were changes
    if [ "$BEFORE_PULL" != "$AFTER_PULL" ]; then
        log_message "📥 New changes pulled from GitHub"
        
        # Get list of changed files
        CHANGED_FILES=$(git diff --name-only "$BEFORE_PULL" "$AFTER_PULL")
        
        # Check if frontend files changed
        if echo "$CHANGED_FILES" | grep -q "frontend/"; then
            log_message "Frontend files changed, restarting frontend..."
            stop_frontend
            start_frontend
        fi
        
        # Check if backend files changed
        if echo "$CHANGED_FILES" | grep -q "backend/"; then
            log_message "Backend files changed, restarting backend..."
            stop_backend
            start_backend
        fi
        
        return 0
    else
        log_message "No changes from GitHub"
        return 1
    fi
}

# Main sync loop
main() {
    log_message "🚀 GitHub Sync Service Started"
    log_message "Monitoring: $PROJECT_DIR"
    
    # Initial service start
    start_frontend
    start_backend
    
    # Continuous sync loop
    while true; do
        pull_changes
        
        # Wait 30 seconds before next check
        sleep 30
    done
}

# Handle script termination
cleanup() {
    log_message "Shutting down sync service..."
    stop_frontend
    stop_backend
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run main function
main