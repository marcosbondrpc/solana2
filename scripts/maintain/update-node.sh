#!/bin/bash
# Update Solana/Agave validator to latest version

set -e

echo "======================================"
echo "   Solana Node Update Script"
echo "======================================"
echo ""

# Check current version
CURRENT_VERSION=$(agave-validator --version 2>/dev/null | head -1 || echo "Not installed")
echo "Current version: $CURRENT_VERSION"
echo ""

# Backup current binary
if [ -f "/usr/local/bin/agave-validator" ]; then
    sudo cp /usr/local/bin/agave-validator /usr/local/bin/agave-validator.backup
    echo "✓ Current binary backed up"
fi

# Download latest release
echo "Fetching latest release information..."
LATEST_RELEASE=$(curl -s https://api.github.com/repos/anza-xyz/agave/releases/latest | grep tag_name | cut -d'"' -f4)
echo "Latest release: $LATEST_RELEASE"

read -p "Do you want to update to $LATEST_RELEASE? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Update cancelled"
    exit 0
fi

# Stop validator
echo "Stopping validator service..."
sudo systemctl stop solana-local.service 2>/dev/null || true
sudo systemctl stop solana-rpc.service 2>/dev/null || true

# Download and install new version
echo "Downloading new version..."
cd /tmp
wget -q "https://github.com/anza-xyz/agave/releases/download/${LATEST_RELEASE}/agave-${LATEST_RELEASE}-x86_64-unknown-linux-gnu.tar.bz2"
tar jxf "agave-${LATEST_RELEASE}-x86_64-unknown-linux-gnu.tar.bz2"
sudo cp -f agave-release/bin/agave-validator /usr/local/bin/
sudo chmod +x /usr/local/bin/agave-validator

# Verify installation
NEW_VERSION=$(agave-validator --version 2>/dev/null | head -1)
echo "✓ Updated to: $NEW_VERSION"

# Clean up
rm -rf agave-release "agave-${LATEST_RELEASE}-x86_64-unknown-linux-gnu.tar.bz2"

# Restart validator
echo "Starting validator service..."
sudo systemctl start solana-local.service 2>/dev/null || true

echo ""
echo "Update completed successfully!"
echo "Check service status: sudo systemctl status solana-local.service"