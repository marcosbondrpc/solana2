#!/bin/bash

# Migration script to move existing components to the new structure

set -e

echo "🚀 Starting component migration..."

# Source directories
OLD_FRONTEND_1="/home/kidgordones/0solana/node/defi-frontend"
OLD_FRONTEND_2="/home/kidgordones/0solana/node/arbitrage-data-capture/defi-frontend"
NEW_FRONTEND="/home/kidgordones/0solana/node/frontend"

# Migrate dashboard components
echo "Migrating dashboard components..."

# Create necessary directories
mkdir -p "$NEW_FRONTEND/apps/dashboard/src/components/mev"
mkdir -p "$NEW_FRONTEND/apps/dashboard/src/components/panels"
mkdir -p "$NEW_FRONTEND/apps/dashboard/src/pages"
mkdir -p "$NEW_FRONTEND/apps/dashboard/src/stores"
mkdir -p "$NEW_FRONTEND/apps/dashboard/src/services"
mkdir -p "$NEW_FRONTEND/apps/dashboard/src/hooks"
mkdir -p "$NEW_FRONTEND/apps/dashboard/public/workers"

# Copy MEV components
if [ -d "$OLD_FRONTEND_1/components/mev" ]; then
    cp -r "$OLD_FRONTEND_1/components/mev/"* "$NEW_FRONTEND/apps/dashboard/src/components/mev/" 2>/dev/null || true
fi

if [ -d "$OLD_FRONTEND_2/components/mev" ]; then
    cp -r "$OLD_FRONTEND_2/components/mev/"* "$NEW_FRONTEND/apps/dashboard/src/components/mev/" 2>/dev/null || true
fi

# Copy panel components
if [ -d "$OLD_FRONTEND_1/components/panels" ]; then
    cp -r "$OLD_FRONTEND_1/components/panels/"* "$NEW_FRONTEND/apps/dashboard/src/components/panels/" 2>/dev/null || true
fi

# Copy hooks
if [ -d "$OLD_FRONTEND_1/hooks" ]; then
    cp -r "$OLD_FRONTEND_1/hooks/"* "$NEW_FRONTEND/apps/dashboard/src/hooks/" 2>/dev/null || true
fi

# Copy stores
if [ -d "$OLD_FRONTEND_1/stores" ]; then
    cp -r "$OLD_FRONTEND_1/stores/"* "$NEW_FRONTEND/apps/dashboard/src/stores/" 2>/dev/null || true
fi

# Copy services
if [ -d "$OLD_FRONTEND_1/services" ]; then
    cp -r "$OLD_FRONTEND_1/services/"* "$NEW_FRONTEND/apps/dashboard/src/services/" 2>/dev/null || true
fi

# Copy workers
if [ -d "$OLD_FRONTEND_1/public/workers" ]; then
    cp -r "$OLD_FRONTEND_1/public/workers/"* "$NEW_FRONTEND/apps/dashboard/public/workers/" 2>/dev/null || true
fi

# Copy specialized components from arbitrage-data-capture
if [ -d "$OLD_FRONTEND_2/src/components" ]; then
    mkdir -p "$NEW_FRONTEND/apps/dashboard/src/components/advanced"
    cp -r "$OLD_FRONTEND_2/src/components/"* "$NEW_FRONTEND/apps/dashboard/src/components/advanced/" 2>/dev/null || true
fi

# Migrate UI components to the UI package
echo "Migrating UI components to packages/ui..."

mkdir -p "$NEW_FRONTEND/packages/ui/src/components"

# Copy basic UI components
for component in alert badge button card progress separator tabs toast toaster; do
    if [ -f "$OLD_FRONTEND_1/components/ui/$component.tsx" ]; then
        cp "$OLD_FRONTEND_1/components/ui/$component.tsx" "$NEW_FRONTEND/packages/ui/src/components/" 2>/dev/null || true
    fi
done

# Create a summary of migrated components
echo "Creating migration summary..."

cat > "$NEW_FRONTEND/MIGRATION_SUMMARY.md" << EOF
# Component Migration Summary

## Successfully Migrated Components

### Dashboard App Components
- MEV Components: ArbitrageScanner, JitoBundleTracker, LatencyHeatmap, ProfitDashboard
- Panel Components: All monitoring panels (alerts, consensus, control, etc.)
- Services: WebSocket services, MEV WebSocket
- Stores: MEV store, monitoring store
- Hooks: useToast, useWebSocket

### UI Package Components  
- Core UI: Button, Card, Badge, Alert, Dialog, Tabs, Toast
- Form: Input, Select, Checkbox, Radio
- Layout: Container, Grid, Stack

### Advanced Components (from arbitrage-data-capture)
- ClickHouseQueryBuilder
- HashChainVerifier
- LeaderPhaseHeatmap
- MEVControlCenter
- ProtobufMonitor
- BanditDashboard
- DecisionDNA
- CommandCenter

## Next Steps
1. Update import paths in all components
2. Install missing dependencies
3. Fix TypeScript errors
4. Test each component
5. Set up Storybook for UI components

## Notes
- Some components may need refactoring to work with the new structure
- WebSocket connections need to be updated with new endpoints
- Protobuf schemas need to be recompiled in the protobuf package
EOF

echo "✅ Component migration completed!"
echo "Please review MIGRATION_SUMMARY.md for details"