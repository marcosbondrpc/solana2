# MEV Detection Dashboard

## Overview

Ultra-sophisticated **DETECTION-ONLY** dashboard for MEV monitoring on Solana. This dashboard provides real-time visualization and analysis of MEV patterns with ZERO execution capabilities.

## Features

### 1. Behavioral Spectrum Analysis
- Interactive radar charts for entity profiling
- Attack style visualization (surgical vs shotgun patterns)
- Risk appetite heat maps
- Fee posture distribution graphs
- 24-hour activity patterns

### 2. Detection Models
- Real-time ROC curves for GNN/Transformer/Hybrid models
- Live confusion matrices with automatic updates
- Latency histograms (P50/P95/P99 metrics)
- Model ensemble voting visualization

### 3. Entity Analytics (Tracking 10 Key Addresses)
- B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi
- 6gAnjderE13TGGFeqdPVQ438jp2FPVeyXAszxKu9y338
- CaCZgxpiEDZtCgXABN9X8PTwkRk81Y9PKdc6s66frH7q
- D9Akv6CQyExvjtEA22g6AAEgCs4GYWzxVoC5UA4fWXEC
- E6YoRP3adE5XYneSseLee15wJshDxCsmyD2WtLvAmfLi
- GGG4BBhgAYKXwjpHjTJyKXWaUnPnHvJ5fnXpG1jvZcNQ
- EAJ1DPoYR9GhaBbSWHmFiyAtssy8Qi7dPvTzBdwkCYMW
- 2brXWR3RYHXsAcbo58LToDV9KVd7D2dyH9iP3qa9PZTy
- CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C (Raydium)
- pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA (PumpSwap)

### 4. Economic Impact Visualization
- 7/30/90-day SOL extraction trends
- Tip/fee burden analysis charts
- Network congestion externality metrics
- Victim impact aggregations

### 5. Hypothesis Testing
- Landing rate anomaly detection
- Latency distribution skew analysis
- Wallet fleet coordination visualization
- Transaction ordering quirk detection

### 6. Decision DNA & Verification
- Ed25519 signature verification display
- Merkle tree visualization
- Immutable audit trail timeline
- Feature hash explorer

## Technology Stack

- **Frontend Framework**: React 18 with TypeScript
- **Build Tool**: Vite 5.4
- **State Management**: Valtio & Zustand
- **Data Visualization**: D3.js, Three.js, Framer Motion
- **Real-time Communication**: WebSocket with auto-reconnect
- **Styling**: Tailwind CSS with glass morphism
- **Performance**: WebWorkers, Virtual scrolling, Binary protobuf

## Installation

```bash
# Install dependencies
npm install --legacy-peer-deps

# Start development server
npm run dev
```

## Access Points

- **Local**: http://localhost:3002
- **Network**: http://45.157.234.184:3002

## Performance Targets

- **Rendering**: 60 FPS with 10k+ data points
- **Bundle Size**: <500KB
- **Initial Load**: <2s
- **WebSocket Latency**: <50ms

## Architecture

```
frontend/
├── src/
│   ├── components/
│   │   ├── BehavioralSpectrum.tsx    # Entity behavior analysis
│   │   ├── DetectionModels.tsx       # ML model performance
│   │   ├── EntityAnalytics.tsx       # Target address tracking
│   │   ├── EconomicImpact.tsx        # Economic metrics
│   │   ├── HypothesisTesting.tsx     # Statistical analysis
│   │   └── DecisionDNA.tsx           # Cryptographic verification
│   ├── pages/
│   │   └── MEVDetectionDashboard.tsx # Main dashboard page
│   └── App.tsx                        # Application entry
├── styles/
│   └── globals.css                    # Global styles & themes
└── vite.config.ts                     # Build configuration
```

## Key Features

### Real-time Monitoring
- Live WebSocket connection to detection API
- Auto-reconnecting with exponential backoff
- Binary protobuf message handling
- Sub-50ms update latency

### Advanced Visualizations
- 3D transaction graphs using Three.js
- Slot-aligned timelines with WebGL
- Behavioral clustering with t-SNE/UMAP
- Venue migration flow diagrams
- Real-time sandwich detection streams

### Detection-Only Design
- **NO EXECUTION**: Pure read-only analytics
- **NO TRADING**: Only detection and visualization
- **NO CONTROL**: Observation without intervention
- **FULL TRANSPARENCY**: All metrics visible

## Dashboard Tabs

1. **Behavioral Spectrum**: Entity profiling and pattern recognition
2. **Detection Models**: ML model performance and ensemble voting
3. **Entity Analytics**: Detailed tracking of key addresses
4. **Economic Impact**: Network-wide MEV impact metrics
5. **Hypothesis Testing**: Statistical anomaly detection
6. **Decision DNA**: Cryptographic verification and audit trails

## Styling Theme

- **Primary Gradient**: #00ff88 to #ff00ff
- **Background**: Pure black (#000000)
- **Glass Morphism**: 50% opacity with backdrop blur
- **Accent Colors**: Cyan, Purple, Green, Yellow
- **Typography**: Monospace for addresses, sans-serif for UI

## Important Notes

- This is a **DETECTION-ONLY** system
- No execution capabilities included
- Read-only access to blockchain data
- Designed for monitoring and analysis only

## Status

✅ Dashboard Running at: http://45.157.234.184:3002/
✅ All components operational
✅ WebSocket ready for connection
✅ 100% detection-only implementation