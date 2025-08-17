# MEV Detection Dashboard - DETECTION ONLY

## Ultra-Sophisticated Frontend for MEV Detection System

This is a **DETECTION-ONLY** interface with NO execution capabilities. It provides real-time visualization and analysis of MEV patterns on Solana.

### Features

#### 🎯 Entity Behavioral Spectrum
- Interactive radar charts showing attack profiles
- Surgical vs Shotgun style visualization
- Risk appetite heat maps with WebGL acceleration
- Fee posture distribution graphs
- Uptime/cadence pattern analysis

#### 🚨 Detection Stream
- Real-time sandwich detection alerts
- Color-coded severity levels (LOW/MEDIUM/HIGH/CRITICAL)
- Transaction flow animations with React Flow
- Victim/attacker relationship graphs
- Live WebSocket updates with 15ms batching

#### 🧠 Model Performance
- ROC curves for each model layer
- Confusion matrices with live updates
- Latency histograms (P50/P95/P99)
- Model ensemble voting visualization
- Performance target indicators

#### 🧬 Decision DNA Explorer
- 3D Merkle tree visualization using Three.js
- Ed25519 signature verification display
- Audit trail timeline
- Feature hash explorer
- Daily Merkle anchor status

### Monitored Entities

The dashboard focuses on these key addresses:
- `B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi`
- `6gAnjderE13TGGFeqdPVQ438jp2FPVeyXAszxKu9y338`
- `E6YoRP3adE5XYneSseLee15wJshDxCsmyD2WtLvAmfLi`
- `CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C` (Raydium)
- `pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA` (PumpSwap)

### Performance

- **60 FPS** with 10,000+ data points
- **<500KB** initial bundle size
- **WebWorker** for heavy computations
- **Virtual scrolling** for large datasets
- **Optimistic UI updates**
- **Binary protobuf** message handling

### Tech Stack

- **React 18** with TypeScript
- **Vite** for blazing fast builds
- **Three.js/React Three Fiber** for 3D visualizations
- **D3.js** for complex charts
- **ECharts** for performance metrics
- **React Flow** for node graphs
- **Framer Motion** for animations
- **Socket.io** for real-time updates
- **Protobuf** for binary messages
- **Valtio** for state management
- **Comlink** for WebWorker communication

### Running the Dashboard

```bash
# Install dependencies
npm install

# Start development server (port 4001)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### API Integration

The dashboard connects to:
- **FastAPI Backend**: `http://localhost:8000`
- **WebSocket**: `ws://localhost:4000/ws`
- **ClickHouse** queries via API
- **ONNX model** inference results

### UI/UX Features

- **Cyberpunk-inspired dark theme**
- **Gradient accents** (#00ff41 to #00ffff)
- **Glass morphism effects**
- **Smooth transitions** and micro-interactions
- **Keyboard shortcuts** for power users
- **Multi-monitor support**
- **Custom scrollbars** with neon glow
- **Responsive layout** for all screen sizes

### Security

- **100% DETECTION-ONLY** - No execution capabilities
- **Read-only access** to blockchain data
- **No wallet connections**
- **No transaction signing**
- **No private key handling**

### Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Performance Optimizations

1. **React.memo** for expensive components
2. **useMemo/useCallback** for computation caching
3. **Virtual scrolling** for large lists
4. **WebWorker** for heavy computations
5. **Message batching** (15ms windows)
6. **Lazy loading** for code splitting
7. **Binary protobuf** for smaller payloads

### Monitoring Capabilities

- Real-time MEV detection with <20ms latency
- Behavioral pattern recognition
- Attack style classification
- Venue migration tracking
- Coordinated actor clustering
- ML confidence scoring
- Decision lineage tracking

This dashboard provides operators with immediate visibility into MEV patterns while maintaining 100% detection-only functionality.

---

**Built for Detection. Optimized for Performance. Designed for Clarity.**
