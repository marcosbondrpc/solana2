#!/bin/bash

# GitHub Sync Monitor
# Shows real-time sync status and health

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              📊 GitHub Sync Monitor                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if cron jobs are installed
echo "🔧 Cron Jobs Status:"
if crontab -l 2>/dev/null | grep -q "auto-commit"; then
    echo "   ✅ Push job: Every 5 minutes"
else
    echo "   ❌ Push job: Not installed"
fi

if crontab -l 2>/dev/null | grep -q "github-pull"; then
    echo "   ✅ Pull job: Every minute"
else
    echo "   ❌ Pull job: Not installed"
fi

echo ""
echo "📝 Recent Sync Activity (last 10 entries):"
tail -10 /home/kidgordones/0solana/node/sync.log | while read line; do
    echo "   $line"
done

echo ""
echo "🔄 Current Repository Status:"
cd /home/kidgordones/0solana/node
BRANCH=$(git branch --show-current)
REMOTE=$(git remote -v | grep push | awk '{print $2}')
LOCAL_COMMIT=$(git rev-parse HEAD | cut -c1-7)
REMOTE_COMMIT=$(git ls-remote origin HEAD | cut -c1-7)

echo "   Branch: $BRANCH"
echo "   Remote: $REMOTE"
echo "   Local commit: $LOCAL_COMMIT"
echo "   Remote commit: $REMOTE_COMMIT"

if [ "$LOCAL_COMMIT" == "$REMOTE_COMMIT" ]; then
    echo "   Status: ✅ In sync"
else
    echo "   Status: ⚠️ Out of sync"
fi

# Check for uncommitted changes
CHANGES=$(git status --porcelain | wc -l)
if [ $CHANGES -gt 0 ]; then
    echo "   Uncommitted changes: $CHANGES files"
    echo ""
    echo "   Modified files:"
    git status --porcelain | head -5 | while read line; do
        echo "      $line"
    done
else
    echo "   Uncommitted changes: None"
fi

echo ""
echo "⏰ Next Sync Times:"
CURRENT_MIN=$(date +%M)
NEXT_PULL=$((60 - CURRENT_MIN % 1))
NEXT_PUSH=$((5 - CURRENT_MIN % 5))
[ $NEXT_PUSH -eq 0 ] && NEXT_PUSH=5
[ $NEXT_PULL -eq 60 ] && NEXT_PULL=1

echo "   Next pull from GitHub: in $NEXT_PULL minute(s)"
echo "   Next push to GitHub: in $NEXT_PUSH minute(s)"

echo ""
echo "💡 Manual Commands:"
echo "   Force push now:  ./auto-commit.sh"
echo "   Force pull now:  ./github-pull-sync.sh"
echo "   View full log:   tail -f sync.log"

echo ""
echo "═══════════════════════════════════════════════════════════════"