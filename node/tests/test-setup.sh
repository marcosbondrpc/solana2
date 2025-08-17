#!/bin/bash

echo "================================"
echo "Solana Node Setup Test Report"
echo "================================"
echo ""

# Check directory structure
echo "📁 Directory Structure Check:"
echo "----------------------------"
for dir in config scripts services docs backups; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/ exists"
    else
        echo "❌ $dir/ missing"
    fi
done
echo ""

# Check important files
echo "📄 Configuration Files Check:"
echo "-----------------------------"
if [ -f "config/solana-rpc.env" ]; then
    echo "✅ config/solana-rpc.env exists"
else
    echo "❌ config/solana-rpc.env missing"
fi

if [ -f "config/validator-keypair.json" ]; then
    echo "✅ config/validator-keypair.json exists"
else
    echo "❌ config/validator-keypair.json missing"
fi
echo ""

# Check scripts
echo "📜 Scripts Check:"
echo "-----------------"
for script in scripts/*/*.sh; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        echo "✅ $script (executable)"
    elif [ -f "$script" ]; then
        echo "⚠️  $script (not executable)"
    fi
done
echo ""

# Check services
echo "🔧 Service Files Check:"
echo "-----------------------"
for service in services/*.service; do
    if [ -f "$service" ]; then
        echo "✅ $service"
    fi
done
echo ""

# Check Solana binary
echo "🔨 Solana Binary Check:"
echo "-----------------------"
if command -v agave-validator &> /dev/null; then
    version=$(agave-validator --version 2>&1)
    echo "✅ Agave validator installed: $version"
else
    echo "❌ Agave validator not found"
fi
echo ""

# Check network connectivity
echo "🌐 Network Check:"
echo "-----------------"
if ping -c 1 entrypoint.mainnet-beta.solana.com &> /dev/null; then
    echo "✅ Can reach Solana mainnet entrypoint"
else
    echo "❌ Cannot reach Solana mainnet entrypoint"
fi
echo ""

# Check storage directories
echo "💾 Storage Directories Check:"
echo "-----------------------------"
for dir in /mnt/ledger /mnt/accounts /opt/solana; do
    if [ -d "$dir" ]; then
        echo "✅ $dir exists"
        df -h "$dir" | tail -1 | awk '{print "   Space: "$4" available"}'
    else
        echo "❌ $dir missing"
    fi
done
echo ""

echo "================================"
echo "Test Complete!"
echo "================================"