#!/bin/bash

# Ensure GitHub Sync is Working
# This script ensures bidirectional sync with GitHub

echo "🔄 Ensuring GitHub sync is working..."

# Function to setup cron if missing
setup_cron() {
    echo "📅 Setting up cron jobs..."
    
    # Remove existing GitHub sync jobs
    crontab -l 2>/dev/null | grep -v "auto-commit\|github-pull" > /tmp/crontab.tmp
    
    # Add new jobs
    echo "# GitHub Auto Sync" >> /tmp/crontab.tmp
    echo "*/5 * * * * /home/kidgordones/0solana/node/scripts/sync/auto-commit.sh >> /home/kidgordones/0solana/node/sync.log 2>&1" >> /tmp/crontab.tmp
    echo "* * * * * /home/kidgordones/0solana/node/scripts/sync/github-pull-sync.sh >> /home/kidgordones/0solana/node/sync.log 2>&1" >> /tmp/crontab.tmp
    
    # Install new crontab
    crontab /tmp/crontab.tmp
    rm /tmp/crontab.tmp
    
    echo "✅ Cron jobs installed"
}

# Check if cron jobs exist
if ! crontab -l 2>/dev/null | grep -q "auto-commit"; then
    echo "⚠️ Push cron job missing, installing..."
    setup_cron
fi

if ! crontab -l 2>/dev/null | grep -q "github-pull"; then
    echo "⚠️ Pull cron job missing, installing..."
    setup_cron
fi

# Ensure scripts are executable
chmod +x /home/kidgordones/0solana/node/scripts/sync/auto-commit.sh
chmod +x /home/kidgordones/0solana/node/scripts/sync/github-pull-sync.sh
chmod +x /home/kidgordones/0solana/node/scripts/monitoring/monitor-sync.sh

# Test GitHub connectivity
cd /home/kidgordones/0solana/node
if git ls-remote origin HEAD > /dev/null 2>&1; then
    echo "✅ GitHub connection successful"
else
    echo "❌ GitHub connection failed"
    echo "Setting up remote..."
    git remote set-url origin https://marcosbondrpc:${GITHUB_TOKEN}@github.com/marcosbondrpc/solana2.git
fi

# Perform immediate sync
echo ""
echo "📤 Pushing local changes to GitHub..."
if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
    git push origin main
    echo "✅ Changes pushed"
else
    echo "✅ No local changes to push"
fi

echo ""
echo "📥 Pulling remote changes from GitHub..."
BEFORE=$(git rev-parse HEAD)
git pull origin main
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" != "$AFTER" ]; then
    echo "✅ New changes pulled from GitHub"
    
    # Restart services if needed
    if git diff --name-only "$BEFORE" "$AFTER" | grep -q "frontend/"; then
        echo "🔄 Frontend changes detected, restarting..."
        pkill -f "vite" 2>/dev/null || true
        cd /home/kidgordones/0solana/node/frontend2
        nohup npm run dev > /tmp/frontend.log 2>&1 &
        cd /home/kidgordones/0solana/node
    fi
    
    if git diff --name-only "$BEFORE" "$AFTER" | grep -q "backend/"; then
        echo "🔄 Backend changes detected, restarting..."
        pkill -f "python3.*main.py" 2>/dev/null || true
        cd /home/kidgordones/0solana/node/backend/services/control-plane
        nohup python3 main.py > /tmp/backend.log 2>&1 &
        cd /home/kidgordones/0solana/node
    fi
else
    echo "✅ Already up to date with GitHub"
fi

# Show current status
echo ""
echo "📊 Sync Status:"
echo "   Local:  $(git rev-parse HEAD | cut -c1-7)"
echo "   Remote: $(git ls-remote origin HEAD | cut -c1-7)"
echo "   Branch: $(git branch --show-current)"

# Verify cron is running
echo ""
echo "⏰ Cron Status:"
if pgrep cron > /dev/null; then
    echo "   ✅ Cron daemon is running"
    echo "   📋 Active sync jobs:"
    crontab -l | grep -E "auto-commit|github-pull" | while read line; do
        echo "      $line"
    done
else
    echo "   ❌ Cron daemon not running"
    echo "   Starting cron..."
    sudo service cron start
fi

echo ""
echo "✅ GitHub sync is now active!"
echo ""
echo "📝 Sync Schedule:"
echo "   • Pull from GitHub: Every minute"
echo "   • Push to GitHub: Every 5 minutes"
echo ""
echo "🔍 Monitor sync activity:"
echo "   ./scripts/monitoring/monitor-sync.sh      # Check sync status"
echo "   tail -f sync.log       # Watch live sync log"