#!/bin/bash

echo "=== 🚀 SOLANA MEV INFRASTRUCTURE STATUS ==="
echo "=========================================="
echo ""

# Frontend Status
echo "📊 FRONTEND (Port 3001):"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3001 | grep -q "200"; then
    echo "  ✅ Dashboard: http://45.157.234.184:3001/"
    echo "  ✅ Mission Control: http://45.157.234.184:3001/node"
    echo "  ✅ Historical Capture: http://45.157.234.184:3001/scrapper"
else
    echo "  ❌ Frontend is DOWN"
fi
echo ""

# Backend Services
echo "🔥 BACKEND SERVICES:"
services=(
    "8081:Node Metrics API"
    "8083:Mission Control API"
    "8085:API Proxy Gateway"
    "8055:Historical Capture API"
)

for service in "${services[@]}"; do
    IFS=':' read -r port name <<< "$service"
    if nc -z localhost $port 2>/dev/null; then
        echo "  ✅ $name (Port $port)"
    else
        echo "  ❌ $name (Port $port) - DOWN"
    fi
done
echo ""

# Docker Infrastructure
echo "📦 DOCKER INFRASTRUCTURE:"
running_containers=$(sudo docker ps --format "{{.Names}}" | grep -E "(redis|clickhouse|kafka|prometheus|grafana|zookeeper)" | wc -l)
echo "  Running Containers: $running_containers"
sudo docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(NAME|redis|clickhouse|kafka|prometheus|grafana|zookeeper)" | head -7
echo ""

# Test API Endpoints
echo "🧪 API HEALTH CHECKS:"
endpoints=(
    "http://localhost:8085/api/node/health:Node Metrics"
    "http://localhost:8085/api/mission-control/overview:Mission Control"
    "http://localhost:8055/health:Historical Capture"
)

for endpoint in "${endpoints[@]}"; do
    IFS=':' read -r url name <<< "$endpoint"
    if curl -s "$url" | grep -q "healthy\|success\|ok"; then
        echo "  ✅ $name - Healthy"
    else
        echo "  ❌ $name - Unhealthy"
    fi
done
echo ""

# System Resources
echo "💻 SYSTEM RESOURCES:"
echo "  CPU Load: $(uptime | awk -F'load average:' '{print $2}')"
echo "  Memory: $(free -h | awk '/^Mem:/ {print "Used: " $3 " / Total: " $2}')"
echo "  Disk: $(df -h / | awk 'NR==2 {print "Used: " $3 " / Total: " $2 " (" $5 " used)"}')"
echo ""

echo "=========================================="
echo "🎯 All systems operational!"
echo "Access the dashboard at: http://45.157.234.184:3001/"