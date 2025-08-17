#!/bin/bash
# Check status of Solana MEV System

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           SOLANA MEV SYSTEM STATUS CHECK                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check services
echo "📊 SERVICE STATUS:"
echo "─────────────────────────────────────────"

check_service() {
    local name=$1
    local port=$2
    local url=$3
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        if [ ! -z "$url" ]; then
            if curl -s "$url" >/dev/null 2>&1; then
                echo "✅ $name (Port $port) - RUNNING & HEALTHY"
            else
                echo "⚠️  $name (Port $port) - RUNNING but NOT RESPONDING"
            fi
        else
            echo "✅ $name (Port $port) - RUNNING"
        fi
    else
        echo "❌ $name (Port $port) - NOT RUNNING"
    fi
}

check_service "Frontend Dashboard" 3001 "http://localhost:3001"
check_service "Historical Capture API" 8055 "http://localhost:8055/health"
check_service "MEV Engine Lite" 8081 "http://localhost:8081/health"
check_service "Mission Control" 8083 "http://localhost:8083/health"
check_service "API Gateway" 8085 "http://localhost:8085/api/node/metrics"

echo ""
echo "🌐 ACCESS URLS:"
echo "─────────────────────────────────────────"
IP=$(hostname -I | awk '{print $1}')
echo "Local Frontend:    http://localhost:3001"
echo "Remote Frontend:   http://$IP:3001"
echo "Public Frontend:   http://45.157.234.184:3001"

echo ""
echo "⚙️  SYSTEMD SERVICE:"
echo "─────────────────────────────────────────"
if systemctl is-enabled solana-mev.service >/dev/null 2>&1; then
    echo "✅ Auto-startup ENABLED (will start on reboot)"
    
    if systemctl is-active solana-mev.service >/dev/null 2>&1; then
        echo "✅ Service is ACTIVE"
    else
        echo "⚠️  Service is INACTIVE (manually started)"
    fi
else
    echo "❌ Auto-startup DISABLED"
fi

echo ""
echo "💡 USEFUL COMMANDS:"
echo "─────────────────────────────────────────"
echo "Start all:         bash /home/kidgordones/0solana/node/scripts/start-services.sh"
echo "Stop all:          lsof -ti:3001,8055,8081,8083,8085 | xargs -r kill -9"
echo "View logs:         tail -f /tmp/*.log"
echo "Service status:    sudo systemctl status solana-mev"
echo "Restart service:   sudo systemctl restart solana-mev"
echo ""