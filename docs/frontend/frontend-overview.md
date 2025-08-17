# Solana MEV Frontend Infrastructure

Ultra-high-performance, production-ready frontend monorepo for Solana MEV operations. Built with cutting-edge web technologies to handle millions of real-time blockchain events while maintaining sub-10ms render times.

## 🚀 Architecture

```
frontend/
├── apps/                    # Application packages
│   ├── dashboard/          # Main MEV dashboard (port 3000)
│   ├── operator/           # Operator command center (port 3001)
│   └── analytics/          # Analytics dashboard (port 3002)
├── packages/               # Shared packages
│   ├── ui/                # Core UI component library
│   ├── charts/            # High-performance chart components
│   ├── websocket/         # WebSocket client with protobuf support
│   ├── protobuf/          # Protocol buffer definitions
│   └── utils/             # Shared utilities
├── configs/               # Shared configurations
│   ├── vite/             # Vite configurations
│   ├── tsconfig/         # TypeScript configurations
│   └── eslint/           # ESLint configurations
├── scripts/              # Build and deployment scripts
└── docs/                 # Documentation
```

## 🎯 Key Features

### Performance
- **Sub-10ms render times** for critical paths
- **WebAssembly** integration for compute-intensive operations
- **Virtual scrolling** for massive datasets
- **Web Workers** for parallel processing
- **SharedArrayBuffer** for zero-copy data transfer
- **Brotli compression** for optimal payload sizes
- **Edge-optimized** with Cloudflare Workers support

### Real-time Capabilities
- **Binary WebSocket** with protobuf encoding
- **Automatic reconnection** with exponential backoff
- **Message queuing** and batching
- **Heartbeat monitoring** for connection health
- **Multi-channel subscriptions**
- **Sub-millisecond latency** for critical updates

### DeFi Specific
- **MEV Dashboard** with real-time arbitrage tracking
- **Jito Bundle Monitoring** with submission tracking
- **Gas optimization** visualizations
- **Liquidity depth** charts
- **Cross-chain** analytics
- **Validator performance** metrics

## 🛠 Tech Stack

- **Framework**: React 18 with Concurrent Features
- **Build Tool**: Vite 5 with SWC
- **Language**: TypeScript 5 (strict mode)
- **State Management**: Zustand + Valtio
- **Data Fetching**: TanStack Query v5
- **Styling**: Tailwind CSS + Emotion
- **Charts**: D3.js + Visx + Lightweight Charts
- **WebSocket**: Custom client with protobuf
- **Testing**: Vitest + Testing Library
- **Monorepo**: Turborepo + npm workspaces

## 🚀 Quick Start

### Prerequisites
- Node.js >= 20.0.0
- npm >= 10.0.0
- Git

### Installation

```bash
# Clone the repository
cd /home/kidgordones/0solana/node/frontend

# Install dependencies
npm install

# Start development servers (all apps)
npm run dev

# Or start specific app
npm run dev --filter=@solana-mev/dashboard
```

### Development

```bash
# Run all development servers
npm run dev

# Run specific app
npm run dev --filter=@solana-mev/dashboard

# Run tests
npm run test

# Run tests in watch mode
npm run test:watch

# Type checking
npm run type-check

# Linting
npm run lint

# Format code
npm run format
```

### Building

```bash
# Build all applications
npm run build

# Build specific app
npm run build:dashboard

# Build with analysis
ANALYZE=true npm run build

# Production build with optimizations
BUILD_ENV=production npm run build
```

### Deployment

```bash
# Deploy to Vercel
./scripts/deploy.sh production vercel

# Deploy to Cloudflare Pages
./scripts/deploy.sh production cloudflare

# Build and run Docker container
docker build -t solana-mev-frontend .
docker run -p 3000:3000 solana-mev-frontend

# Deploy to Kubernetes
./scripts/deploy.sh production k8s
```

## 📦 Package Structure

### Apps

#### Dashboard (`@solana-mev/dashboard`)
Main MEV monitoring dashboard with real-time data visualization.

- Real-time MEV opportunity tracking
- Arbitrage path visualization
- Bundle submission monitoring
- P&L tracking
- Gas optimization metrics

#### Operator (`@solana-mev/operator`)
Command center for MEV operators.

- Strategy configuration
- Risk management controls
- Manual intervention tools
- System health monitoring
- Alert management

#### Analytics (`@solana-mev/analytics`)
Advanced analytics and reporting dashboard.

- Historical performance analysis
- Strategy backtesting
- Custom report generation
- Data export capabilities
- ML model insights

### Packages

#### UI (`@solana-mev/ui`)
Core UI component library with 50+ production-ready components.

```typescript
import { Button, Card, Chart } from '@solana-mev/ui';
```

#### Charts (`@solana-mev/charts`)
High-performance charting library optimized for financial data.

```typescript
import { CandlestickChart, HeatmapChart } from '@solana-mev/charts';
```

#### WebSocket (`@solana-mev/websocket`)
Enterprise-grade WebSocket client with automatic reconnection.

```typescript
import { WebSocketClient } from '@solana-mev/websocket';

const client = new WebSocketClient({
  url: 'wss://api.solana-mev.com',
  protocols: ['protobuf'],
  enableCompression: true,
});
```

#### Utils (`@solana-mev/utils`)
Shared utilities for common operations.

```typescript
import { formatSOL, calculateAPY } from '@solana-mev/utils';
```

## 🎨 Design System

### Theme
- Dark mode by default with light mode support
- Consistent color palette across all apps
- Accessible color contrasts (WCAG AA)
- Custom Tailwind configuration

### Typography
- Inter for UI text
- JetBrains Mono for code/data
- Responsive font scaling
- Optimized line heights

### Components
- Consistent spacing system (4px base)
- Rounded corners (4px, 8px, 12px)
- Subtle shadows for depth
- Smooth animations (Framer Motion)

## 🔧 Configuration

### Environment Variables

```env
# API Configuration
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8001

# Solana Configuration
VITE_RPC_URL=https://api.mainnet-beta.solana.com
VITE_COMMITMENT_LEVEL=confirmed

# Features
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_PWA=true
VITE_ENABLE_WORKER=true
```

### TypeScript Paths

```json
{
  "paths": {
    "@/*": ["./src/*"],
    "@solana-mev/ui": ["../../packages/ui/src"],
    "@solana-mev/charts": ["../../packages/charts/src"],
    "@solana-mev/websocket": ["../../packages/websocket/src"],
    "@solana-mev/utils": ["../../packages/utils/src"]
  }
}
```

## 📊 Performance Metrics

### Target Metrics
- **First Contentful Paint**: < 500ms
- **Time to Interactive**: < 2s
- **Largest Contentful Paint**: < 1s
- **Cumulative Layout Shift**: < 0.1
- **First Input Delay**: < 100ms

### Bundle Size Targets
- **Main bundle**: < 200KB (gzipped)
- **Vendor bundle**: < 300KB (gzipped)
- **Total initial load**: < 500KB (gzipped)
- **Code splitting**: Automatic for routes

## 🧪 Testing

### Unit Testing
```bash
# Run unit tests
npm run test

# Coverage report
npm run test:coverage
```

### E2E Testing
```bash
# Run E2E tests
npm run test:e2e

# Run in headed mode
npm run test:e2e:headed
```

### Performance Testing
```bash
# Run performance benchmarks
npm run bench

# Lighthouse CI
npm run lighthouse
```

## 🔒 Security

- **Content Security Policy** enforced
- **Subresource Integrity** for CDN assets
- **HTTPS only** in production
- **Input sanitization** for all user inputs
- **Secure WebSocket** connections
- **Rate limiting** on API calls

## 📚 Documentation

- [Architecture Guide](./docs/architecture.md)
- [Component Library](./docs/components.md)
- [Performance Guide](./docs/performance.md)
- [Deployment Guide](./docs/deployment.md)
- [Contributing Guide](./docs/contributing.md)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details

## 🙏 Acknowledgments

Built with cutting-edge technologies and best practices from:
- React Core Team
- Vite Team
- Vercel
- Cloudflare
- Solana Labs
- Jito Labs

---

**Built for speed. Optimized for profit. Ready for scale.**