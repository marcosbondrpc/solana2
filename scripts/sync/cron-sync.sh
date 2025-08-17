#!/bin/bash

cd /home/kidgordones/0solana/node

# Check if there are changes
if [[ -n $(git status -s) ]]; then
    # Add all changes
    git add .
    
    # Commit with timestamp
    git commit -m "Cron auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Push to GitHub
    git push origin main
    
    # Log the sync
    echo "[$(date)] Synced to GitHub" >> /home/kidgordones/0solana/node/sync.log
fi
