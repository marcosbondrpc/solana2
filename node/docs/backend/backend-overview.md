# Elite MEV Backend Infrastructure

Production-grade Solana MEV infrastructure with ultra-low latency and maximum value extraction capabilities. This backend is architected for institutional-grade MEV operations with sub-millisecond opportunity detection and execution.

## Architecture Overview

This backend system consists of seven microservices working in concert:

### 1. API Gateway Service (Port 3000)
- **mTLS Authentication**: Client certificate validation for admin operations
- **OIDC Integration**: OpenID Connect for user authentication
- **RBAC**: Role-based access control with granular permissions
- **2FA**: Two-factor authentication for critical operations
- **CSRF Protection**: Token-based CSRF prevention
- **Rate Limiting**: Redis-backed rate limiting per IP/user
- **Audit Logging**: Cryptographically signed audit trails

### 2. RPC Probe Service (Port 3010)
- Measures Solana RPC method latencies in real-time
- Tracks success/error rates for all RPC methods
- Monitors critical methods: getAccountInfo, getProgramAccounts, getBlock
- Correlates performance with network/kernel counters
- Health scoring algorithms with ML-based anomaly detection
- WebSocket support for real-time metrics streaming

### 3. Validator Agent (Port 3020)
- **systemd D-Bus Integration**: Direct control without shell execution
- Safe restart/stop/start operations with state verification
- Vote account toggle functionality
- Dynamic flag management and configuration updates
- Journal log parsing for structured metrics extraction
- Automatic error classification and alerting

### 4. Jito MEV Probe (Port 3030)
- gRPC/WebSocket connection to Jito Block Engine
- Bundle acceptance and landing rate tracking
- Tip spend analysis with profitability calculations
- Relay RTT measurements across regions
- Expected Value (EV) calculations for strategies
- Priority fee tracking and optimization

### 5. Geyser Plugin Probe (Port 3040)
- Kafka consumer group lag monitoring
- Database sink metrics (Postgres/ClickHouse)
- Event throughput tracking (accounts, transactions, blocks)
- Plugin health monitoring with automatic recovery

### 6. Metrics Collector (Port 3050)
- Prometheus exporters for all custom metrics
- VictoriaMetrics integration for long-term storage
- Loki log aggregation with structured queries
- AlertManager configuration with PagerDuty/Slack
- Custom Solana-specific exporters

### 7. Control Operations Service (Port 3060)
- **Snapshot Management**: Create/verify/prune with checksums
- **Ledger Repair**: Automated repair with slot range support
- **Configuration Editor**: Diff validation and rollback
- **Rolling Restart**: Orchestrated restarts with health checks
- All operations require RBAC + 2FA + audit logging

## Security Features

### Authentication & Authorization
- **mTLS**: Mutual TLS for service-to-service communication
- **OIDC**: Enterprise SSO integration
- **RBAC**: Three-tier role system (admin, operator, viewer)
- **2FA**: TOTP-based two-factor authentication
- **Session Management**: Redis-backed sessions with timeout

### Network Security
- **Rate Limiting**: Configurable per-endpoint limits
- **IP Allowlisting**: Restrict access by IP ranges
- **CSRF Protection**: Token-based CSRF prevention
- **CORS**: Configurable cross-origin policies

### Audit & Compliance
- **Signed Audit Logs**: HMAC-SHA256 signed events
- **Immutable Trail**: Redis-backed with retention policies
- **Operation Tracking**: Every privileged operation logged
- **Compliance Reports**: Automated report generation

## Performance Specifications

- **API Response Time**: <50ms p99 latency
- **Metrics Collection**: 1-second granularity
- **Bundle Tracking**: Real-time with <100ms delay
- **Health Scoring**: ML-based with 30-second updates
- **Concurrent Connections**: 10,000+ WebSocket clients
- **Data Retention**: 90 days hot, unlimited cold storage

## Deployment

### Prerequisites
```bash
# Required software
- Docker 20.10+
- Docker Compose 2.0+
- Node.js 20+
- Redis 7+
- PostgreSQL 14+ (optional)
- ClickHouse (optional)
```

### Quick Start
```bash
# Clone and navigate to backend directory
cd /home/kidgordones/0solana/node/backend

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env

# Deploy all services
chmod +x deploy.sh
./deploy.sh

# Check health
curl -k https://localhost/health
```

### Production Deployment
```bash
# Generate production certificates
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -keyout certs/server.key -out certs/server.crt

# Configure environment
export NODE_ENV=production
export MTLS_ENABLED=true
export REQUIRE_2FA=true

# Deploy with Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose scale rpc-probe=3 jito-probe=2
```

## Configuration

### Environment Variables
Key configuration variables in `.env`:

```bash
# Security
JWT_SECRET=minimum-32-character-secret
AUDIT_SIGNATURE_KEY=audit-signing-key
CSRF_SECRET=csrf-token-secret

# OIDC
OIDC_ISSUER=https://auth.yourdomain.com
OIDC_CLIENT_ID=solana-monitor
OIDC_CLIENT_SECRET=your-secret

# Solana
SOLANA_RPC_URL=http://localhost:8899
VALIDATOR_CONFIG_PATH=/path/to/config.yaml
VALIDATOR_LEDGER_PATH=/path/to/ledger

# Jito
JITO_BLOCK_ENGINE_URL=grpc://mainnet.block-engine.jito.wtf:443
```

### Service Configuration
Each service has its own configuration in `<service>/src/config.ts`

## API Endpoints

### Authentication
- `GET /auth/login` - Initiate OIDC login
- `GET /auth/callback` - OIDC callback
- `POST /auth/refresh` - Refresh JWT token
- `POST /auth/logout` - Terminate session
- `POST /auth/2fa/setup` - Setup 2FA
- `POST /auth/2fa/verify` - Verify 2FA token

### RPC Monitoring
- `GET /api/rpc/metrics` - Current RPC metrics
- `GET /api/rpc/latency/:method` - Method-specific latency
- `GET /api/rpc/health` - Health scores
- `WS /api/rpc/ws` - Real-time metrics stream

### Validator Control
- `GET /api/validator/status` - Current status
- `POST /api/validator/restart` - Restart validator
- `POST /api/validator/vote/:action` - Toggle voting
- `PUT /api/validator/config/flags` - Update flags

### MEV/Jito
- `POST /api/jito/bundle/submit` - Submit bundle
- `GET /api/jito/bundle/:id` - Bundle status
- `GET /api/jito/priority-fees` - Current fees
- `GET /api/jito/expected-value` - Calculate EV

### Control Operations
- `POST /api/controls/snapshot/create` - Create snapshot
- `POST /api/controls/ledger/repair` - Repair ledger
- `POST /api/controls/config/apply` - Apply configuration
- `POST /api/controls/restart/rolling` - Rolling restart

## Monitoring

### Prometheus Metrics
Access at `http://localhost:3050/metrics`

Key metrics:
- `solana_rpc_latency_seconds` - RPC latencies
- `jito_bundle_acceptance_rate` - Bundle acceptance
- `solana_validator_status` - Validator status
- `solana_rpc_health_score` - Health scores

### Grafana Dashboards
Import provided dashboards from `./dashboards/`:
- `rpc-performance.json` - RPC monitoring
- `mev-tracking.json` - MEV/Jito metrics
- `validator-health.json` - Validator status
- `system-overview.json` - System metrics

## Advanced Features

### MEV Optimization
The Jito probe includes sophisticated MEV strategies:
- Dynamic tip calculation based on network congestion
- Multi-region relay selection for optimal latency
- Bundle success prediction using ML models
- Automatic retry with exponential backoff

### Health Scoring Algorithm
```typescript
score = (
  latencyScore * 0.3 +
  successRateScore * 0.4 +
  consistencyScore * 0.2 +
  availabilityScore * 0.1
) * 100
```

### Circuit Breaker Pattern
All external connections implement circuit breakers:
- **Closed**: Normal operation
- **Open**: Failures exceed threshold, requests blocked
- **Half-Open**: Test requests to check recovery

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose logs <service-name>

# Verify Redis connection
redis-cli -h localhost ping

# Check port availability
netstat -tulpn | grep <port>
```

### Authentication Issues
```bash
# Verify OIDC configuration
curl https://your-issuer/.well-known/openid-configuration

# Check JWT secret
echo $JWT_SECRET | wc -c  # Should be >= 32
```

### Performance Issues
```bash
# Check Redis memory
redis-cli info memory

# Monitor Docker resources
docker stats

# Check connection pool
curl http://localhost:3010/api/rpc/connections
```

## Security Considerations

1. **Never expose services directly** - Always use the API Gateway
2. **Rotate secrets regularly** - Use secret management tools
3. **Monitor audit logs** - Set up alerts for suspicious activity
4. **Update dependencies** - Run `npm audit` regularly
5. **Use production certificates** - Never use self-signed in production

## License

Proprietary - All rights reserved

## Support

For enterprise support, contact: support@solana-monitor.io