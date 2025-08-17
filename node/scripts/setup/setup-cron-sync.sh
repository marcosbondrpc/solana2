#!/bin/bash

# Setup GitHub Auto-sync Cron Jobs

echo "📅 Setting up GitHub auto-sync cron jobs..."

# Create auto-commit script
cat > /home/kidgordones/0solana/node/auto-commit.sh << 'EOF'
#!/bin/bash

cd /home/kidgordones/0solana/node

# Check if there are changes
if [ -n "$(git status --porcelain)" ]; then
    # Add all changes
    git add -A
    
    # Create commit message with timestamp
    COMMIT_MSG="Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Commit changes
    git commit -m "$COMMIT_MSG" > /dev/null 2>&1
    
    # Push to GitHub
    git push origin main > /dev/null 2>&1
    
    echo "[$(date)] Changes pushed to GitHub"
else
    echo "[$(date)] No changes to commit"
fi
EOF

# Create pull sync script
cat > /home/kidgordones/0solana/node/github-pull-sync.sh << 'EOF'
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
        cd /home/kidgordones/0solana/node/frontend2
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
EOF

# Make scripts executable
chmod +x /home/kidgordones/0solana/node/auto-commit.sh
chmod +x /home/kidgordones/0solana/node/github-pull-sync.sh

# Setup cron jobs
(crontab -l 2>/dev/null || true; echo "# GitHub Auto Sync") | crontab -
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/kidgordones/0solana/node/auto-commit.sh >> /home/kidgordones/0solana/node/sync.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "* * * * * /home/kidgordones/0solana/node/github-pull-sync.sh >> /home/kidgordones/0solana/node/sync.log 2>&1") | crontab -

echo "✅ Cron jobs installed!"
echo ""
echo "📋 Installed cron jobs:"
crontab -l | grep -E "auto-commit|github-pull"
echo ""
echo "🔄 Sync schedule:"
echo "  - Pull from GitHub: Every minute"
echo "  - Push to GitHub: Every 5 minutes"
echo ""
echo "📄 Logs will be saved to: /home/kidgordones/0solana/node/sync.log"