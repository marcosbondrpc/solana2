#!/bin/bash

echo "=== Deploying MEV Frontend ==="

# Kill any existing frontend processes
echo "Stopping existing frontend processes..."
ps aux | grep -E "vite|node.*dashboard" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null

# Start frontend on port 3001
echo "Starting frontend on port 3001..."
cd /home/kidgordones/0solana/node/frontend2
nohup npm run dev -- --port 3001 --host 0.0.0.0 > /tmp/frontend-3001.log 2>&1 &

echo "Waiting for frontend to start..."
sleep 5

# Check if frontend is running
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3001 | grep -q "200"; then
    echo "✅ Frontend successfully deployed on http://45.157.234.184:3001/"
    echo ""
    echo "Available pages:"
    echo "  • Dashboard: http://45.157.234.184:3001/"
    echo "  • Node (Mission Control): http://45.157.234.184:3001/node"
    echo "  • Scrapper (Historical Capture): http://45.157.234.184:3001/scrapper"
    echo ""
    echo "Monitoring moved to:"
    echo "  • Grafana: http://45.157.234.184:3003/ (admin/admin)"
else
    echo "❌ Frontend failed to start. Check /tmp/frontend-3001.log for errors"
fi