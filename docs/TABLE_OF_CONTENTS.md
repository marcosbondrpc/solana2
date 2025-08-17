# 📚 Complete Documentation Table of Contents

## Table of Contents

### 🏠 Root Documentation
- [`README.md`](../README.md) - Project overview and getting started
- [`CLAUDE.md`](../CLAUDE.md) - AI assistant configuration and memory

### 📁 Documentation by Category

#### Backend Documentation (`/docs/backend/`)
| Document | Description |
|----------|-------------|
| [mev-backend-architecture.md](./backend/mev-backend-architecture.md) | Core MEV engine architecture with Jito integration |
| [backend-overview.md](./backend/backend-overview.md) | General backend services and API structure |
| [arbitrage-backend-system.md](./backend/arbitrage-backend-system.md) | Arbitrage detection and execution system |
| [legendary-backend-architecture.md](./backend/legendary-backend-architecture.md) | Advanced patterns for ultra-low latency |
| [historical-capture-service.md](./backend/historical-capture-service.md) | Blockchain data capture and storage service |

#### Frontend Documentation (`/docs/frontend/`)
| Document | Description |
|----------|-------------|
| [frontend-overview.md](./frontend/frontend-overview.md) | Dashboard architecture and component structure |
| [integration-requirements.md](./frontend/integration-requirements.md) | Frontend-backend integration specifications |
| [legendary-frontend-integration.md](./frontend/legendary-frontend-integration.md) | Advanced UI/UX patterns and real-time updates |
| [defi-frontend-guide.md](./frontend/defi-frontend-guide.md) | DeFi-specific components and workflows |
| [migration-summary.md](./frontend/migration-summary.md) | Frontend technology migration notes |

#### Arbitrage Documentation (`/docs/arbitrage/`)
| Document | Description |
|----------|-------------|
| [integration-plan.md](./arbitrage/integration-plan.md) | Complete arbitrage system integration strategy |
| [system-overview.md](./arbitrage/system-overview.md) | Arbitrage detection algorithms and execution |
| [data-capture-overview.md](./arbitrage/data-capture-overview.md) | Historical data capture for training |
| [ml-data-structure.md](./arbitrage/ml-data-structure.md) | Machine learning data organization |
| [continuous-improvement.md](./arbitrage/continuous-improvement.md) | Optimization and performance tuning |

#### Deployment Documentation (`/docs/deployment/`)
| Document | Description |
|----------|-------------|
| [integration-complete.md](./deployment/integration-complete.md) | Final system integration checklist |
| [frontend-deployment-status.md](./deployment/frontend-deployment-status.md) | Frontend deployment and monitoring |
| [scripts-documentation.md](./deployment/scripts-documentation.md) | Utility and deployment scripts guide |

## 🎯 Quick Start Guides

### For Developers
1. Start with [README.md](../README.md)
2. Review [backend-overview.md](./backend/backend-overview.md)
3. Check [frontend-overview.md](./frontend/frontend-overview.md)
4. Read [integration-requirements.md](./frontend/integration-requirements.md)

### For DevOps
1. Read [scripts-documentation.md](./deployment/scripts-documentation.md)
2. Review [integration-complete.md](./deployment/integration-complete.md)
3. Check deployment configurations

### For Data Scientists
1. Study [ml-data-structure.md](./arbitrage/ml-data-structure.md)
2. Review [data-capture-overview.md](./arbitrage/data-capture-overview.md)
3. Understand [historical-capture-service.md](./backend/historical-capture-service.md)

## 📊 System Components

```
┌─────────────────────────────────────────────────┐
│                   Frontend                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │Dashboard │ │  Node    │ │ Scrapper │        │
│  └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│                 API Gateway                      │
│              (Port 8085)                         │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│               Backend Services                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │   MEV    │ │Historical│ │ Mission  │        │
│  │  Engine  │ │ Capture  │ │ Control  │        │
│  └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│              Infrastructure                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │  Redis   │ │ClickHouse│ │  Kafka   │        │
│  └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────────────────────────────┘
```

## 🔗 External Resources

- [Solana Documentation](https://docs.solana.com)
- [Jito Labs Documentation](https://docs.jito.wtf)
- [Rust Documentation](https://doc.rust-lang.org)
- [React Documentation](https://react.dev)

---

*Documentation Organization Complete - All MD files properly categorized*
*Last Updated: 2025-08-16*