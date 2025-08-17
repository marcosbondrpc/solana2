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
