#!/bin/bash

SYNC_DIR="/home/kidgordones/0solana/solana2"
LOG_FILE="/tmp/github-sync.log"
LOCK_FILE="/tmp/github-sync.lock"

# Prevent multiple instances
if [ -f "$LOCK_FILE" ]; then
    PID=$(cat "$LOCK_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "[$(date)] Another sync is already running (PID: $PID)" >> $LOG_FILE
        exit 0
    fi
fi

echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

echo "[$(date)] Starting GitHub sync..." >> $LOG_FILE

cd $SYNC_DIR

# Pull latest changes
echo "[$(date)] Pulling from GitHub..." >> $LOG_FILE
git pull origin master --no-edit 2>&1 >> $LOG_FILE || {
    echo "[$(date)] Pull failed, attempting to resolve..." >> $LOG_FILE
    git stash >> $LOG_FILE 2>&1
    git pull origin master --no-edit >> $LOG_FILE 2>&1
    git stash pop >> $LOG_FILE 2>&1 || true
}

# Check if there are changes to commit
if [[ -n $(git status -s) ]]; then
    echo "[$(date)] Changes detected, committing..." >> $LOG_FILE
    
    # Add all changes
    git add -A
    
    # Create commit message with timestamp
    COMMIT_MSG="Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Get brief summary of changes
    CHANGES=$(git status --short | head -5)
    if [ ! -z "$CHANGES" ]; then
        COMMIT_MSG="$COMMIT_MSG

Changes:
$CHANGES"
    fi
    
    # Commit
    git commit -m "$COMMIT_MSG" >> $LOG_FILE 2>&1
    
    # Push to GitHub
    echo "[$(date)] Pushing to GitHub..." >> $LOG_FILE
    git push origin master >> $LOG_FILE 2>&1 || {
        echo "[$(date)] Push failed, will retry on next sync" >> $LOG_FILE
    }
    
    echo "[$(date)] Sync completed successfully" >> $LOG_FILE
else
    echo "[$(date)] No changes to sync" >> $LOG_FILE
fi

# Only restart services if we just pushed changes
if [ "$?" -eq 0 ] && git diff HEAD~1 --name-only 2>/dev/null | grep -qE "(backend/|frontend/|Cargo.toml|package.json)"; then
    echo "[$(date)] Core files changed in last commit, considering restart..." >> $LOG_FILE
    
    # Check if this is a new change (not already restarted)
    LAST_RESTART_FILE="/tmp/last-restart-commit"
    CURRENT_COMMIT=$(git rev-parse HEAD)
    
    if [ -f "$LAST_RESTART_FILE" ]; then
        LAST_RESTART=$(cat "$LAST_RESTART_FILE")
        if [ "$LAST_RESTART" = "$CURRENT_COMMIT" ]; then
            echo "[$(date)] Already restarted for this commit, skipping..." >> $LOG_FILE
            echo "[$(date)] Sync cycle complete" >> $LOG_FILE
            exit 0
        fi
    fi
    
    # Restart backend if needed
    if git diff HEAD~1 --name-only 2>/dev/null | grep -q "backend/"; then
        echo "[$(date)] Restarting backend services..." >> $LOG_FILE
        systemctl restart mev-control-plane 2>/dev/null || true
        systemctl restart mev-arbitrage 2>/dev/null || true
    fi
    
    # Restart frontend if needed
    if git diff HEAD~1 --name-only 2>/dev/null | grep -q "frontend/"; then
        echo "[$(date)] Restarting frontend..." >> $LOG_FILE
        pkill -f "npm run dev" 2>/dev/null || true
        cd /home/kidgordones/0solana/solana2/frontend
        nohup npm run dev -- --port 3001 > /tmp/frontend.log 2>&1 &
    fi
    
    echo "$CURRENT_COMMIT" > "$LAST_RESTART_FILE"
fi

echo "[$(date)] Sync cycle complete" >> $LOG_FILE
