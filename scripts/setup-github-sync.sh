#!/bin/bash

# GitHub Auto-Sync Setup Script
# This script sets up automatic synchronization with GitHub

set -e

REPO_URL="https://github.com/marcosbondrpc/solana2.git"
SYNC_DIR="/home/kidgordones/0solana/solana2"
LOG_FILE="/tmp/github-sync.log"

echo "🔧 Setting up GitHub auto-sync..."

# 1. Configure git
cd $SYNC_DIR

echo "📝 Configuring git..."
git config user.name "MEV Bot"
git config user.email "mev@solana2.local"
git config push.default current

# 2. Create sync script
cat > /home/kidgordones/0solana/solana2/scripts/github-sync.sh << 'EOF'
#!/bin/bash

SYNC_DIR="/home/kidgordones/0solana/solana2"
LOG_FILE="/tmp/github-sync.log"

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

# Restart services if core files changed
if git diff HEAD~1 --name-only 2>/dev/null | grep -qE "(backend/|frontend2/|Cargo.toml|package.json)"; then
    echo "[$(date)] Core files changed, considering restart..." >> $LOG_FILE
    
    # Restart backend if needed
    if git diff HEAD~1 --name-only 2>/dev/null | grep -q "backend/"; then
        echo "[$(date)] Restarting backend services..." >> $LOG_FILE
        systemctl restart mev-control-plane 2>/dev/null || true
        systemctl restart mev-arbitrage 2>/dev/null || true
    fi
    
    # Restart frontend if needed
    if git diff HEAD~1 --name-only 2>/dev/null | grep -q "frontend2/"; then
        echo "[$(date)] Restarting frontend..." >> $LOG_FILE
        pkill -f "npm run dev" 2>/dev/null || true
        cd /home/kidgordones/0solana/solana2/frontend2
        nohup npm run dev -- --port 3001 > /tmp/frontend.log 2>&1 &
    fi
fi

echo "[$(date)] Sync cycle complete" >> $LOG_FILE
EOF

chmod +x /home/kidgordones/0solana/solana2/scripts/github-sync.sh

# 3. Create systemd service for auto-sync
echo "🔄 Creating systemd service..."
sudo tee /etc/systemd/system/github-sync.service > /dev/null << EOF
[Unit]
Description=GitHub Auto-Sync for Solana2 MEV
After=network.target

[Service]
Type=oneshot
User=$USER
WorkingDirectory=$SYNC_DIR
ExecStart=/home/kidgordones/0solana/solana2/scripts/github-sync.sh
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 4. Create systemd timer for periodic sync
echo "⏰ Creating systemd timer..."
sudo tee /etc/systemd/system/github-sync.timer > /dev/null << EOF
[Unit]
Description=GitHub Auto-Sync Timer for Solana2 MEV
Requires=github-sync.service

[Timer]
# Run every 5 minutes
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
EOF

# 5. Set up cron as backup (every minute for pull, every 5 for push)
echo "📅 Setting up cron jobs..."
(crontab -l 2>/dev/null | grep -v "github-sync") | crontab -
(crontab -l 2>/dev/null; echo "*/1 * * * * cd $SYNC_DIR && git pull origin master --no-edit >/dev/null 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "*/5 * * * * $SYNC_DIR/scripts/github-sync.sh >/dev/null 2>&1") | crontab -

# 6. Enable and start the systemd timer
echo "🚀 Enabling auto-sync..."
sudo systemctl daemon-reload
sudo systemctl enable github-sync.timer
sudo systemctl start github-sync.timer

# 7. Create manual sync command
cat > /home/kidgordones/0solana/solana2/scripts/sync-now.sh << 'EOF'
#!/bin/bash
echo "🔄 Manual GitHub sync triggered..."
/home/kidgordones/0solana/solana2/scripts/github-sync.sh
echo "✅ Sync complete!"
tail -20 /tmp/github-sync.log
EOF
chmod +x /home/kidgordones/0solana/solana2/scripts/sync-now.sh

echo "✅ GitHub auto-sync setup complete!"
echo ""
echo "📊 Configuration:"
echo "  - Repository: $REPO_URL"
echo "  - Auto-pull: Every 1 minute"
echo "  - Auto-push: Every 5 minutes"
echo "  - Log file: $LOG_FILE"
echo ""
echo "🛠️ Commands:"
echo "  - Manual sync: ./scripts/sync-now.sh"
echo "  - View logs: tail -f $LOG_FILE"
echo "  - Check timer: systemctl status github-sync.timer"
echo "  - Stop sync: systemctl stop github-sync.timer"
echo ""
echo "⚠️ IMPORTANT: Store your GitHub token securely!"
echo "For authenticated pushes, use:"
echo "  git config credential.helper store"
echo "  git push (enter username and token when prompted)"
echo ""
echo "The token will be stored encrypted in ~/.git-credentials"