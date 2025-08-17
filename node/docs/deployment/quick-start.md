# Quick Start Guide

Get the Solana MEV Infrastructure running in under 10 minutes.

## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04+ or macOS 12+
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 100GB SSD minimum
- **CPU**: 8 cores minimum (16 recommended)
- **Network**: 1Gbps connection

### Software Requirements

```bash
# Check versions
docker --version      # 20.10+
node --version        # v20+
npm --version         # 10+
cargo --version       # 1.70+
python3 --version     # 3.11+
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/solana-mev/infrastructure.git
cd infrastructure
```

### 2. Fix Docker Permissions

If you encounter Docker permission errors:

```bash
# Run the helper script
./fix-docker-permissions.sh

# Or manually add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### 3. Install Dependencies

```bash
# Install all dependencies
make install-deps

# Or install manually
npm install -g turbo
pip3 install -r requirements.txt
cargo install --path backend/tools
```

## Development Setup

### Start Infrastructure

```bash
# Start Docker services (Redis, ClickHouse, Grafana)
make infra-up

# Check status
make infra-status
```

### Start Services

#### Option 1: Start Everything

```bash
# Start complete development environment
make dev
```

#### Option 2: Start Services Individually

```bash
# Terminal 1: Start frontend
make frontend-dev

# Terminal 2: Start backend
make backend-dev

# Terminal 3: Monitor logs
make logs-follow
```

### Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend Dashboard | http://localhost:3000 | - |
| API Documentation | http://localhost:8000/docs | - |
| Grafana | http://localhost:3000 | admin/admin |
| ClickHouse | http://localhost:8123 | default/- |

## Configuration

### Environment Variables

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:

```bash
# Solana Configuration
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_WS_URL=wss://api.mainnet-beta.solana.com

# Jito Configuration
JITO_BLOCK_ENGINE_URL=https://mainnet.block-engine.jito.wtf
JITO_AUTH_TOKEN=your_token_here

# Database Configuration
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000
CLICKHOUSE_DATABASE=mev

# Redis Configuration
REDIS_URL=redis://localhost:6390

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
JWT_SECRET=your-secret-key-minimum-32-chars

# MEV Configuration
MIN_PROFIT_THRESHOLD=0.5
MAX_GAS_PRICE=0.01
ENABLE_SANDWICH_DETECTION=true
ENABLE_ARBITRAGE=true
```

### Strategy Configuration

Edit `configs/strategies.yaml`:

```yaml
arbitrage:
  enabled: true
  min_profit: 0.5
  max_gas: 0.01
  pools:
    - raydium
    - orca
    - meteora

sandwich:
  enabled: false
  min_victim_amount: 100
  max_priority_fee: 0.01

liquidation:
  enabled: true
  min_collateral: 1000
  protocols:
    - solend
    - mango
```

## Testing the Setup

### 1. Health Check

```bash
# Check all services are healthy
make health-check

# Expected output:
# ✅ Backend API responding
# ✅ Frontend responding
# ✅ Redis responding
# ✅ ClickHouse responding
```

### 2. Run Tests

```bash
# Run all tests
make test

# Run specific tests
make frontend-test
make backend-test
```

### 3. Test MEV Detection

```bash
# Submit a test transaction
curl -X POST http://localhost:8000/v1/test/opportunity \
  -H "Content-Type: application/json" \
  -d '{
    "type": "arbitrage",
    "pools": ["USDC/SOL", "SOL/USDT"],
    "amount": 100
  }'
```

## Common Issues

### Docker Permission Denied

```bash
# Error: permission denied while trying to connect to Docker daemon
# Solution:
sudo usermod -aG docker $USER
newgrp docker
# Or run with sudo:
sudo make infra-up
```

### Port Already in Use

```bash
# Error: bind: address already in use
# Solution: Find and kill the process
lsof -i :3000  # Find process using port 3000
kill -9 <PID>  # Kill the process
```

### Node Version Mismatch

```bash
# Error: The engine "node" is incompatible
# Solution: Use nvm to install correct version
nvm install 20
nvm use 20
```

### Rust Compilation Errors

```bash
# Error: could not compile package
# Solution: Update Rust and clean build
rustup update
cd backend && cargo clean
cargo build --release
```

### Database Connection Failed

```bash
# Error: ClickHouse connection refused
# Solution: Ensure Docker services are running
docker ps  # Check running containers
make infra-up  # Restart infrastructure
```

## Next Steps

### 1. Configure Wallets

```bash
# Generate new keypair
solana-keygen new -o ~/.config/solana/mev-wallet.json

# Set in .env
MEV_WALLET_PATH=~/.config/solana/mev-wallet.json
```

### 2. Fund Accounts

```bash
# Check balance
solana balance ~/.config/solana/mev-wallet.json

# Request airdrop (devnet only)
solana airdrop 10 ~/.config/solana/mev-wallet.json --url devnet
```

### 3. Configure RPC Endpoints

For production, use premium RPC providers:

```bash
# Premium RPC providers (in .env)
SOLANA_RPC_URL=https://your-premium-rpc.com
SOLANA_WS_URL=wss://your-premium-ws.com
```

### 4. Enable Strategies

```bash
# Edit strategy configuration
nano configs/strategies.yaml

# Restart services
make restart
```

### 5. Monitor Performance

```bash
# Open Grafana dashboards
make grafana-open

# View real-time logs
make logs-follow

# Check metrics
curl http://localhost:9090/metrics
```

## Production Deployment

For production deployment, see [Production Deployment Guide](./production-deployment.md).

## Troubleshooting

For more troubleshooting help, see:
- [Common Issues](../TROUBLESHOOTING.md)
- [FAQ](../FAQ.md)
- [Discord Community](https://discord.gg/solana-mev)

## Support

- **GitHub Issues**: [Report bugs](https://github.com/solana-mev/infrastructure/issues)
- **Discord**: [Join community](https://discord.gg/solana-mev)
- **Email**: support@solana-mev.io