#!/bin/bash
# MEV Control Plane - Production Service Installation
# Ultra-high-performance deployment script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="mev-control-plane"
SERVICE_FILE="mev-control-plane.service"
INSTALL_DIR="/home/kidgordones/0solana/solana2/arbitrage-data-capture"
API_DIR="$INSTALL_DIR/api"
VENV_DIR="/home/kidgordones/0solana/solana2/venv"
USER="$(whoami)"
GROUP="$(id -gn)"

echo -e "${BLUE}🚀 MEV Control Plane - Production Installation${NC}"
echo -e "${BLUE}=============================================${NC}"

# Check if running as root for service installation
if [[ $EUID -eq 0 ]]; then
    echo -e "${RED}❌ Do not run this script as root. Run as the mev user.${NC}"
    exit 1
fi

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check if API directory exists
if [[ ! -d "$API_DIR" ]]; then
    print_error "API directory not found: $API_DIR"
    exit 1
fi

# Check if main.py exists
if [[ ! -f "$API_DIR/main.py" ]]; then
    print_error "main.py not found in $API_DIR"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -d "$VENV_DIR" ]]; then
    print_warning "Virtual environment not found. Creating..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment and install dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install requirements
if [[ -f "$API_DIR/requirements.txt" ]]; then
    pip install -r "$API_DIR/requirements.txt"
    print_status "Dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Test import
echo -e "${BLUE}🧪 Testing imports...${NC}"
cd "$INSTALL_DIR"
python -c "
import uvicorn
import fastapi
import uvloop
import prometheus_fastapi_instrumentator
print('✅ All imports successful')
" || {
    print_error "Import test failed"
    exit 1
}

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
sudo mkdir -p /tmp/models /tmp/exports
sudo chown -R "$USER:$GROUP" /tmp/models /tmp/exports
sudo mkdir -p "$INSTALL_DIR/logs"
sudo chown -R "$USER:$GROUP" "$INSTALL_DIR/logs"

# Set permissions
echo -e "${BLUE}🔒 Setting permissions...${NC}"
chmod +x "$API_DIR/main.py" 2>/dev/null || true
chmod 644 "$API_DIR/requirements.txt"

# Copy and install systemd service
echo -e "${BLUE}⚙️  Installing systemd service...${NC}"

# Update service file with current user
SERVICE_TEMP="/tmp/$SERVICE_NAME.service"
sed "s/User=mev/User=$USER/g; s/Group=mev/Group=$GROUP/g" "$API_DIR/$SERVICE_FILE" > "$SERVICE_TEMP"

# Install service file
sudo cp "$SERVICE_TEMP" "/etc/systemd/system/$SERVICE_NAME.service"
sudo chown root:root "/etc/systemd/system/$SERVICE_NAME.service"
sudo chmod 644 "/etc/systemd/system/$SERVICE_NAME.service"

# Clean up temp file
rm "$SERVICE_TEMP"

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable "$SERVICE_NAME"

print_status "Service installed and enabled"

# Performance tuning
echo -e "${BLUE}⚡ Applying performance tuning...${NC}"

# CPU governor
if command -v cpupower >/dev/null 2>&1; then
    sudo cpupower frequency-set -g performance 2>/dev/null || print_warning "Could not set CPU governor to performance"
fi

# Network tuning
sudo sysctl -w net.core.rmem_max=134217728 2>/dev/null || true
sudo sysctl -w net.core.wmem_max=134217728 2>/dev/null || true
sudo sysctl -w net.core.netdev_max_backlog=5000 2>/dev/null || true
sudo sysctl -w net.core.somaxconn=65535 2>/dev/null || true

print_status "Performance tuning applied"

# Create startup script
echo -e "${BLUE}📜 Creating management scripts...${NC}"

cat > "$API_DIR/start-production.sh" << 'EOF'
#!/bin/bash
# Start MEV Control Plane in production mode

set -euo pipefail

SERVICE_NAME="mev-control-plane"

echo "🚀 Starting MEV Control Plane..."

# Check if service is already running
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "⚠️  Service is already running"
    systemctl status "$SERVICE_NAME" --no-pager -l
    exit 0
fi

# Start service
sudo systemctl start "$SERVICE_NAME"

# Wait for startup
sleep 3

# Check status
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "✅ MEV Control Plane started successfully"
    echo "📊 Status:"
    systemctl status "$SERVICE_NAME" --no-pager -l
    echo ""
    echo "🌐 API available at: http://localhost:8000"
    echo "📈 Metrics available at: http://localhost:8000/metrics"
    echo "📖 API docs at: http://localhost:8000/api/docs"
else
    echo "❌ Failed to start service"
    systemctl status "$SERVICE_NAME" --no-pager -l
    exit 1
fi
EOF

cat > "$API_DIR/stop-production.sh" << 'EOF'
#!/bin/bash
# Stop MEV Control Plane

set -euo pipefail

SERVICE_NAME="mev-control-plane"

echo "🛑 Stopping MEV Control Plane..."

if systemctl is-active --quiet "$SERVICE_NAME"; then
    sudo systemctl stop "$SERVICE_NAME"
    echo "✅ Service stopped successfully"
else
    echo "⚠️  Service was not running"
fi
EOF

cat > "$API_DIR/restart-production.sh" << 'EOF'
#!/bin/bash
# Restart MEV Control Plane

set -euo pipefail

SERVICE_NAME="mev-control-plane"

echo "🔄 Restarting MEV Control Plane..."

sudo systemctl restart "$SERVICE_NAME"

# Wait for startup
sleep 3

# Check status
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "✅ MEV Control Plane restarted successfully"
    systemctl status "$SERVICE_NAME" --no-pager -l
else
    echo "❌ Failed to restart service"
    systemctl status "$SERVICE_NAME" --no-pager -l
    exit 1
fi
EOF

cat > "$API_DIR/status-production.sh" << 'EOF'
#!/bin/bash
# Check MEV Control Plane status

set -euo pipefail

SERVICE_NAME="mev-control-plane"

echo "📊 MEV Control Plane Status"
echo "=========================="

# Service status
echo "🔧 Service Status:"
systemctl status "$SERVICE_NAME" --no-pager -l

echo ""

# API health check
echo "🏥 API Health Check:"
if curl -s -f http://localhost:8000/api/health/ >/dev/null 2>&1; then
    echo "✅ API is responding"
    curl -s http://localhost:8000/api/health/ | python3 -m json.tool
else
    echo "❌ API is not responding"
fi

echo ""

# Performance metrics
echo "📈 Performance Metrics:"
if curl -s -f http://localhost:8000/api/health/metrics/performance >/dev/null 2>&1; then
    curl -s http://localhost:8000/api/health/metrics/performance | python3 -m json.tool
else
    echo "❌ Performance metrics not available"
fi
EOF

# Make scripts executable
chmod +x "$API_DIR"/*.sh

print_status "Management scripts created"

# Test configuration
echo -e "${BLUE}🧪 Testing configuration...${NC}"

# Validate service file
sudo systemd-analyze verify "/etc/systemd/system/$SERVICE_NAME.service" || {
    print_error "Service file validation failed"
    exit 1
}

print_status "Service file is valid"

# Final status
echo -e "${GREEN}🎉 Installation completed successfully!${NC}"
echo ""
echo -e "${BLUE}📚 Quick Start Guide:${NC}"
echo "1. Start service:    $API_DIR/start-production.sh"
echo "2. Check status:     $API_DIR/status-production.sh"
echo "3. Stop service:     $API_DIR/stop-production.sh"
echo "4. Restart service:  $API_DIR/restart-production.sh"
echo ""
echo -e "${BLUE}🌐 Endpoints:${NC}"
echo "• API:        http://localhost:8000"
echo "• Health:     http://localhost:8000/api/health/"
echo "• Metrics:    http://localhost:8000/metrics"
echo "• Docs:       http://localhost:8000/api/docs"
echo ""
echo -e "${BLUE}📊 Monitoring:${NC}"
echo "• Logs:       journalctl -u $SERVICE_NAME -f"
echo "• Status:     systemctl status $SERVICE_NAME"
echo ""
echo -e "${YELLOW}⚡ Performance Notes:${NC}"
echo "• Service runs with high priority scheduling"
echo "• Memory limit: 8GB"
echo "• CPU affinity optimized for low latency"
echo "• Network buffers tuned for high throughput"
echo ""
echo -e "${GREEN}Ready for billions in volume! 🚀${NC}"