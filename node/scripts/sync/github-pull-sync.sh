#!/bin/bash

cd /home/kidgordones/0solana/node

# Store current commit
BEFORE=$(git rev-parse HEAD 2>/dev/null || echo "none")

# Pull changes
git pull origin main > /dev/null 2>&1

# Check if changed
AFTER=$(git rev-parse HEAD 2>/dev/null || echo "none")

if [ "$BEFORE" != "$AFTER" ]; then
    echo "[$(date)] New changes pulled from GitHub"
    
    # Check if frontend files changed
    if git diff --name-only "$BEFORE" "$AFTER" | grep -q "frontend/"; then
        echo "[$(date)] Frontend changes detected, restarting..."
        pkill -f "vite" 2>/dev/null || true
        sleep 2
        cd /home/kidgordones/0solana/node/frontend/apps/dashboard
        nohup npm run dev > /tmp/frontend.log 2>&1 &
    fi
    
    # Check if backend files changed
    if git diff --name-only "$BEFORE" "$AFTER" | grep -q "backend/"; then
        echo "[$(date)] Backend changes detected, restarting..."
        pkill -f "python3.*main.py" 2>/dev/null || true
        sleep 2
        cd /home/kidgordones/0solana/node/backend/services/control-plane
        nohup python3 main.py > /tmp/backend.log 2>&1 &
    fi
else
    echo "[$(date)] No changes from GitHub"
fi
