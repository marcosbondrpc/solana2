# Frontend Integration Requirements

## Critical Missing Components from arbitrage-data-capture

### 1. MEV Control Center Dashboard
**Source**: `/arbitrage-data-capture/defi-frontend/src/components/MEVControlCenter.tsx`
**Target**: `/frontend/src/components/MEVControlCenter.tsx`

**Features to Add**:
- Real-time system health monitoring (ClickHouse, Kafka, Grafana status)
- MEV performance metrics dashboard (lands/min, EV/min, success rate)
- Bandit optimization tracking with UCB scores
- Decision DNA visualization with cryptographic verification
- Lab smoke test runner with one-click execution
- Auto-refresh with configurable intervals
- Keyboard shortcuts (Alt+1-5 for tabs, Alt+R refresh, Alt+S tests)

**API Dependencies**:
```typescript
GET http://localhost:8001/api/provisioning/health
GET http://localhost:8001/api/health/metrics/performance
POST http://localhost:8001/api/control/command
WS ws://localhost:8001/api/realtime/ws
```

### 2. ClickHouse Query Builder
**Source**: `/arbitrage-data-capture/defi-frontend/src/components/ClickHouseQueryBuilder.tsx`
**Target**: `/frontend/src/components/ClickHouseQueryBuilder.tsx`

**Features to Add**:
- Visual query builder with drag-and-drop
- SQL editor with syntax highlighting and auto-completion
- Pre-built query templates for MEV analytics
- Result visualization with pagination
- Export to CSV functionality
- Query history tracking
- Saved queries with persistent storage

**API Dependencies**:
```typescript
POST http://localhost:8001/api/clickhouse/query
GET http://localhost:8001/api/clickhouse/tables
GET http://localhost:8001/api/clickhouse/schema
```

### 3. Grafana Dashboard Provisioning
**Source**: `/arbitrage-data-capture/defi-frontend/src/components/GrafanaProvisioning.tsx`
**Target**: `/frontend/src/components/GrafanaProvisioning.tsx`

**Features to Add**:
- One-click provisioning of all dashboards
- Automated datasource configuration
- Dashboard catalog with 6 pre-built templates
- Connection testing and validation
- Progress tracking with status indicators
- Smoke test integration

**Dashboards to Provision**:
1. Operator Command Center (24 panels)
2. Bandit Dashboard (18 panels)
3. Decision DNA Panel (12 panels)
4. MEV Performance Analytics (20 panels)
5. System Health Monitor (16 panels)
6. Kafka Topics Monitor (14 panels)

### 4. Protobuf Monitor
**Source**: `/arbitrage-data-capture/defi-frontend/src/components/ProtobufMonitor.tsx`
**Target**: `/frontend/src/components/ProtobufMonitor.tsx`

**Features to Add**:
- Real-time protobuf message decoding
- Message flow visualization
- Topic filtering and search
- Message rate monitoring
- Hex/Binary/JSON view modes
- Message replay capability

**WebSocket Connection**:
```typescript
const ws = new WebSocket('ws://localhost:8001/api/realtime/ws');
ws.binaryType = 'arraybuffer';
```

### 5. Bandit Dashboard
**Source**: `/arbitrage-data-capture/defi-frontend/src/components/bandit/BanditDashboard.tsx`
**Target**: `/frontend/src/components/bandit/BanditDashboard.tsx`

**Features to Add**:
- Multi-armed bandit visualization
- UCB (Upper Confidence Bound) score tracking
- Arm performance metrics with heat maps
- Exploration vs exploitation balance chart
- Thompson sampling visualization
- Regret tracking over time
- Context-aware bandit statistics

### 6. Decision DNA Visualizer
**Source**: `/arbitrage-data-capture/defi-frontend/src/components/dna/DecisionDNA.tsx`
**Target**: `/frontend/src/components/dna/DecisionDNA.tsx`

**Features to Add**:
- Cryptographic hash chain visualization
- Strategy evolution timeline
- Decision lineage tracking
- Hash verification interface
- Fork detection and resolution
- Performance attribution by decision

### 7. WebSocket Libraries
**Source**: `/arbitrage-data-capture/defi-frontend/lib/`
**Target**: `/frontend/src/lib/`

**Files to Copy**:
- `ws.ts` - High-performance WebSocket client (100k+ msg/sec)
- `ws-proto.ts` - Protobuf encoder/decoder
- `wt.ts` - WebTransport QUIC client

**Key Features**:
- Zero-copy ArrayBuffer operations
- Micro-batching (10-25ms windows)
- Backpressure handling
- Ring buffer for message queueing
- Worker pool for parallel decoding

### 8. Worker Scripts
**Source**: `/arbitrage-data-capture/defi-frontend/workers/`
**Target**: `/frontend/public/workers/`

**Files to Copy**:
- `wsDecoder.worker.ts` - Parallel WebSocket decoding
- `protobufDecoder.worker.ts` - Protobuf message processing

### 9. Stores and State Management
**Source**: `/arbitrage-data-capture/defi-frontend/stores/`
**Target**: `/frontend/src/stores/`

**Files to Integrate**:
- `feed.ts` - Real-time data feed management

## API Endpoint Configuration

Create new configuration file:
`/frontend/src/config/arbitrage-api.ts`

```typescript
export const ARBITRAGE_API = {
  baseUrl: process.env.NEXT_PUBLIC_ARBITRAGE_API_URL || 'http://localhost:8001',
  
  endpoints: {
    // Control Plane
    control: {
      command: '/api/control/command',
      policy: '/api/control/policy',
      modelSwap: '/api/control/model-swap',
      killSwitch: '/api/control/kill-switch',
      status: '/api/control/status',
      auditLog: '/api/control/audit-log'
    },
    
    // Health & Monitoring
    health: {
      system: '/api/health/metrics/system',
      performance: '/api/health/metrics/performance',
      latency: '/api/health/metrics/latency',
      connections: '/api/health/debug/connections'
    },
    
    // Provisioning
    provisioning: {
      health: '/api/provisioning/health',
      datasource: '/api/provisioning/datasource',
      dashboard: '/api/provisioning/dashboard',
      provision: '/api/provisioning/provision/all',
      benchmark: '/api/provisioning/benchmark'
    },
    
    // Training & ML
    training: {
      train: '/api/training/train',
      jobs: '/api/training/jobs',
      models: '/api/training/models',
      deploy: '/api/training/models/{id}/deploy'
    },
    
    // Datasets
    datasets: {
      export: '/api/datasets/export',
      schema: '/api/datasets/schema/{type}',
      download: '/api/datasets/export/{id}/download'
    }
  },
  
  websockets: {
    realtime: '/api/realtime/ws',
    control: '/ws',
    webtransport: '/wt'
  }
};
```

## Environment Variables

Add to `.env.local`:
```bash
# Arbitrage Data Capture API
NEXT_PUBLIC_ARBITRAGE_API_URL=http://localhost:8001
NEXT_PUBLIC_CLICKHOUSE_URL=http://localhost:8123
NEXT_PUBLIC_GRAFANA_URL=http://localhost:3001
NEXT_PUBLIC_KAFKA_METRICS_URL=http://localhost:9090

# WebSocket Configuration
NEXT_PUBLIC_WS_BATCH_WINDOW=15
NEXT_PUBLIC_WS_MAX_BATCH_SIZE=256
NEXT_PUBLIC_WS_WORKER_POOL_SIZE=4

# Feature Flags
NEXT_PUBLIC_ENABLE_PROTOBUF=true
NEXT_PUBLIC_ENABLE_WEBTRANSPORT=true
NEXT_PUBLIC_ENABLE_BANDIT_TRACKING=true
```

## Package Dependencies

Add to `package.json`:
```json
{
  "dependencies": {
    "@bufbuild/protobuf": "^1.3.0",
    "@connectrpc/connect-web": "^1.1.0",
    "comlink": "^4.4.1",
    "antd": "^5.11.0",
    "@ant-design/charts": "^1.4.0",
    "@ant-design/pro-components": "^2.6.0",
    "recharts": "^2.10.0",
    "react-use-websocket": "^4.5.0",
    "uuid": "^9.0.0",
    "dayjs": "^1.11.10",
    "lodash": "^4.17.21",
    "@tanstack/react-query": "^5.0.0"
  }
}
```

## Integration Priority

### Phase 1 - Core Infrastructure (Critical)
1. WebSocket libraries (ws.ts, ws-proto.ts)
2. Worker scripts for parallel processing
3. API configuration

### Phase 2 - Monitoring (High Priority)
1. MEV Control Center Dashboard
2. System health monitoring components
3. Real-time metrics display

### Phase 3 - Analytics (Medium Priority)
1. ClickHouse Query Builder
2. Grafana Dashboard Provisioning
3. Performance analytics views

### Phase 4 - Advanced Features (Lower Priority)
1. Bandit Dashboard
2. Decision DNA Visualizer
3. Protobuf Monitor

## Testing Requirements

### Unit Tests
- WebSocket connection handling
- Protobuf encoding/decoding
- API endpoint integration
- Worker pool management

### Integration Tests
- Real-time data flow
- Dashboard provisioning
- Query builder functionality
- Control command execution

### Performance Tests
- 100k+ messages/second WebSocket handling
- Sub-100ms UI updates
- Worker pool scaling
- Memory leak detection

## Migration Script

```bash
#!/bin/bash
# Frontend component migration script

SOURCE="/home/kidgordones/0solana/node/arbitrage-data-capture/defi-frontend"
TARGET="/home/kidgordones/0solana/node/frontend"

# Copy components
cp -r $SOURCE/src/components/* $TARGET/src/components/

# Copy libraries
cp -r $SOURCE/lib/* $TARGET/src/lib/

# Copy workers
cp -r $SOURCE/workers/* $TARGET/public/workers/

# Copy stores
cp -r $SOURCE/stores/* $TARGET/src/stores/

# Install dependencies
cd $TARGET
npm install @bufbuild/protobuf @connectrpc/connect-web comlink antd @ant-design/charts

echo "Frontend migration complete!"
```

## Validation Checklist

- [ ] All WebSocket connections established
- [ ] Protobuf messages decoded correctly
- [ ] Dashboard components render without errors
- [ ] API endpoints return expected data
- [ ] Worker pools initialized
- [ ] Real-time data flow verified
- [ ] Performance metrics within targets
- [ ] No memory leaks detected
- [ ] Keyboard shortcuts functional
- [ ] Auto-refresh working

This integration will provide a comprehensive MEV monitoring and control interface with institutional-grade performance and reliability.