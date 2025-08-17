#!/bin/bash

# Setup automatic syncing to GitHub

echo "🔧 Setting up automatic GitHub sync..."

# Install fswatch if not installed (for file watching)
if ! command -v inotifywait &> /dev/null; then
    echo "Installing inotify-tools for file watching..."
    sudo apt-get update
    sudo apt-get install -y inotify-tools
fi

# Create systemd service for real-time sync
sudo tee /etc/systemd/system/github-auto-sync.service > /dev/null <<EOF
[Unit]
Description=GitHub Auto Sync for Solana MEV Infrastructure
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/kidgordones/0solana/node
ExecStart=/home/kidgordones/0solana/node/watch-and-sync.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create the watch and sync script
cat > /home/kidgordones/0solana/node/watch-and-sync.sh <<'EOF'
#!/bin/bash

# Watch for file changes and auto-commit
cd /home/kidgordones/0solana/node

echo "👁️ Watching for changes..."

# Function to commit and push
sync_changes() {
    if [[ -n $(git status -s) ]]; then
        echo "📝 Changes detected, committing..."
        git add .
        git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
        git push origin main
        echo "✅ Synced to GitHub"
    fi
}

# Watch for changes (excluding .git directory and build files)
while true; do
    inotifywait -r -e modify,create,delete,move \
        --exclude '\.git|node_modules|target|dist|build|\.log' \
        . 2>/dev/null
    
    # Wait a bit to batch changes
    sleep 5
    
    # Sync changes
    sync_changes
done
EOF

chmod +x /home/kidgordones/0solana/node/watch-and-sync.sh

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable github-auto-sync.service
sudo systemctl start github-auto-sync.service

echo "✅ Auto-sync service installed and started"
echo ""
echo "📊 Service Status:"
sudo systemctl status github-auto-sync.service --no-pager

echo ""
echo "🎯 Usage:"
echo "  - Check status: sudo systemctl status github-auto-sync"
echo "  - Stop service: sudo systemctl stop github-auto-sync"
echo "  - Start service: sudo systemctl start github-auto-sync"
echo "  - View logs: sudo journalctl -u github-auto-sync -f"
echo ""
echo "✨ All changes will now be automatically synced to GitHub!"