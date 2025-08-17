#!/bin/bash

# Monitor GitHub Auto-Sync Status
echo "=========================================="
echo "🔄 GitHub Auto-Sync Monitor"
echo "=========================================="
echo ""

# Check systemd timer status
echo "📅 Systemd Timer Status:"
systemctl status github-sync.timer --no-pager | grep -E "(Active:|Trigger:)" | sed 's/^/  /'
echo ""

# Check last sync from log
echo "📝 Last Sync Activity:"
if [ -f /tmp/github-sync.log ]; then
    tail -5 /tmp/github-sync.log | sed 's/^/  /'
else
    echo "  No sync log found"
fi
echo ""

# Check cron job
echo "⏰ Cron Job Status:"
crontab -l 2>/dev/null | grep github-sync | sed 's/^/  /'
echo ""

# Check last 3 commits
echo "📊 Recent Commits:"
git log --oneline -3 | sed 's/^/  /'
echo ""

# Check if there are uncommitted changes
echo "📂 Working Directory Status:"
CHANGES=$(git status --porcelain)
if [ -z "$CHANGES" ]; then
    echo "  ✅ Clean - no uncommitted changes"
else
    echo "  ⚠️  Uncommitted changes detected:"
    git status --short | head -5 | sed 's/^/    /'
fi
echo ""

# Check remote status
echo "🌐 Remote Repository:"
git remote get-url origin | sed 's/ghp_[^@]*@/[TOKEN]@/' | sed 's/^/  URL: /'
BEHIND=$(git rev-list HEAD..origin/master --count 2>/dev/null)
AHEAD=$(git rev-list origin/master..HEAD --count 2>/dev/null)
if [ "$BEHIND" = "0" ] && [ "$AHEAD" = "0" ]; then
    echo "  ✅ In sync with remote"
elif [ "$BEHIND" != "0" ]; then
    echo "  ⬇️  Behind remote by $BEHIND commits"
elif [ "$AHEAD" != "0" ]; then
    echo "  ⬆️  Ahead of remote by $AHEAD commits"
fi
echo ""

# Next sync time
echo "⏭️  Next Sync:"
NEXT_SYNC=$(systemctl status github-sync.timer --no-pager | grep "Trigger:" | awk '{for(i=2;i<=NF;i++) printf "%s ", $i; print ""}')
echo "  $NEXT_SYNC"
echo ""

echo "=========================================="
echo "💡 Tips:"
echo "  - Auto-sync runs every 5 minutes"
echo "  - Manual sync: make sync"
echo "  - View full log: tail -f /tmp/github-sync.log"
echo "  - Force sync now: sudo systemctl start github-sync.service"
echo "==========================================" 