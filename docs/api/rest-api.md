# REST API Reference

## Base URL

```
Production: https://api.solana-mev.io/v1
Staging: https://staging-api.solana-mev.io/v1
Development: http://localhost:8000/v1
```

## Authentication

All API requests require authentication using JWT tokens or API keys.

### JWT Authentication

```http
Authorization: Bearer <jwt_token>
```

### API Key Authentication

```http
X-API-Key: <api_key>
```

## Rate Limiting

| Tier | Requests/Second | Requests/Day | Burst |
|------|----------------|--------------|-------|
| Free | 10 | 10,000 | 20 |
| Basic | 100 | 100,000 | 200 |
| Pro | 1,000 | 1,000,000 | 2,000 |
| Enterprise | Custom | Unlimited | Custom |

## Endpoints

### Health & Status

#### GET /health

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "kafka": "healthy",
    "ml_service": "healthy"
  }
}
```

#### GET /status

Get detailed system status.

**Response:**
```json
{
  "version": "1.0.0",
  "uptime": 864000,
  "metrics": {
    "tps": 150000,
    "latency_ms": 0.95,
    "success_rate": 0.963,
    "active_strategies": 5
  }
}
```

### MEV Opportunities

#### GET /opportunities

List current MEV opportunities.

**Query Parameters:**
- `type` (string): Filter by opportunity type (arbitrage, sandwich, liquidation)
- `min_profit` (number): Minimum profit threshold in SOL
- `limit` (number): Maximum results (default: 100, max: 1000)
- `offset` (number): Pagination offset

**Response:**
```json
{
  "data": [
    {
      "id": "opp_123456",
      "type": "arbitrage",
      "timestamp": "2024-01-15T10:30:00Z",
      "pools": ["USDC/SOL", "SOL/USDT"],
      "estimated_profit": 1.5,
      "confidence": 0.92,
      "gas_cost": 0.005,
      "net_profit": 1.495,
      "expires_at": "2024-01-15T10:30:01Z"
    }
  ],
  "meta": {
    "total": 450,
    "limit": 100,
    "offset": 0
  }
}
```

#### GET /opportunities/{id}

Get detailed opportunity information.

**Response:**
```json
{
  "id": "opp_123456",
  "type": "arbitrage",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z",
  "details": {
    "path": ["USDC", "SOL", "USDT", "USDC"],
    "pools": [
      {
        "address": "pool_address_1",
        "protocol": "raydium",
        "reserve0": "1000000",
        "reserve1": "50000"
      }
    ],
    "input_amount": "1000",
    "expected_output": "1015",
    "price_impact": 0.001,
    "slippage_tolerance": 0.005
  },
  "simulation": {
    "success": true,
    "profit": 1.5,
    "gas_used": 250000,
    "logs": ["..."]
  }
}
```

### Bundle Management

#### POST /bundles

Submit a new bundle for execution.

**Request Body:**
```json
{
  "transactions": [
    {
      "instruction": "base64_encoded_instruction",
      "signer": "public_key"
    }
  ],
  "tip_lamports": 1000000,
  "strategy": "ladder",
  "priority": "high"
}
```

**Response:**
```json
{
  "bundle_id": "bundle_789",
  "status": "submitted",
  "submitted_at": "2024-01-15T10:30:00Z",
  "tip_amount": 0.001,
  "estimated_landing": "2024-01-15T10:30:02Z"
}
```

#### GET /bundles/{id}

Get bundle status and details.

**Response:**
```json
{
  "bundle_id": "bundle_789",
  "status": "landed",
  "created_at": "2024-01-15T10:30:00Z",
  "landed_at": "2024-01-15T10:30:02Z",
  "slot": 180000000,
  "transactions": [
    {
      "signature": "tx_signature",
      "status": "success",
      "fee": 0.00025
    }
  ],
  "tip_paid": 0.001,
  "profit_realized": 1.494
}
```

### Strategy Management

#### GET /strategies

List available trading strategies.

**Response:**
```json
{
  "strategies": [
    {
      "id": "strat_arb_v2",
      "name": "Arbitrage V2",
      "type": "arbitrage",
      "status": "active",
      "performance": {
        "total_profit": 1523.45,
        "success_rate": 0.934,
        "avg_profit": 1.23,
        "total_executions": 1240
      }
    }
  ]
}
```

#### POST /strategies/{id}/activate

Activate a trading strategy.

**Request Body:**
```json
{
  "config": {
    "min_profit": 0.5,
    "max_gas": 0.01,
    "slippage": 0.005,
    "position_size": 100
  }
}
```

**Response:**
```json
{
  "strategy_id": "strat_arb_v2",
  "status": "active",
  "activated_at": "2024-01-15T10:30:00Z",
  "config": {...}
}
```

### Analytics & Metrics

#### GET /analytics/pnl

Get profit and loss analytics.

**Query Parameters:**
- `start_date` (string): ISO 8601 date
- `end_date` (string): ISO 8601 date
- `granularity` (string): hour, day, week, month

**Response:**
```json
{
  "summary": {
    "total_profit": 15234.56,
    "total_loss": 234.56,
    "net_pnl": 15000.00,
    "roi": 0.234,
    "sharpe_ratio": 2.1
  },
  "timeline": [
    {
      "timestamp": "2024-01-15T00:00:00Z",
      "profit": 500.23,
      "loss": 10.23,
      "net": 490.00,
      "executions": 45
    }
  ]
}
```

#### GET /analytics/performance

Get strategy performance metrics.

**Response:**
```json
{
  "metrics": {
    "success_rate": 0.943,
    "avg_profit": 1.23,
    "max_drawdown": 45.67,
    "win_loss_ratio": 3.2,
    "profit_factor": 2.8
  },
  "by_strategy": {
    "arbitrage": {
      "executions": 5430,
      "success_rate": 0.956,
      "total_profit": 6789.12
    },
    "sandwich": {
      "executions": 2340,
      "success_rate": 0.923,
      "total_profit": 3456.78
    }
  }
}
```

### Configuration

#### GET /config

Get current system configuration.

**Response:**
```json
{
  "trading": {
    "enabled": true,
    "max_position_size": 1000,
    "default_slippage": 0.005,
    "gas_limit": 0.01
  },
  "strategies": {
    "arbitrage": {
      "enabled": true,
      "min_profit": 0.5
    },
    "sandwich": {
      "enabled": false,
      "max_priority_fee": 0.01
    }
  },
  "risk": {
    "max_daily_loss": 100,
    "position_limit": 10000,
    "correlation_threshold": 0.7
  }
}
```

#### PUT /config

Update system configuration.

**Request Body:**
```json
{
  "trading": {
    "enabled": true,
    "max_position_size": 2000
  }
}
```

**Response:**
```json
{
  "status": "updated",
  "config": {...},
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### WebSocket Subscriptions

#### WS /ws/opportunities

Real-time opportunity stream.

**Subscribe Message:**
```json
{
  "type": "subscribe",
  "channels": ["opportunities"],
  "filters": {
    "min_profit": 0.5,
    "types": ["arbitrage", "sandwich"]
  }
}
```

**Stream Message:**
```json
{
  "type": "opportunity",
  "data": {
    "id": "opp_123456",
    "type": "arbitrage",
    "profit": 1.5,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Error Responses

### Error Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid request parameters",
    "details": {
      "field": "min_profit",
      "reason": "Must be a positive number"
    }
  },
  "request_id": "req_abc123"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| UNAUTHORIZED | 401 | Invalid or missing authentication |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| INVALID_REQUEST | 400 | Invalid request parameters |
| RATE_LIMITED | 429 | Rate limit exceeded |
| INTERNAL_ERROR | 500 | Internal server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

## SDK Examples

### JavaScript/TypeScript

```typescript
import { SolanaMEVClient } from '@solana-mev/sdk';

const client = new SolanaMEVClient({
  apiKey: 'your_api_key',
  environment: 'production'
});

// Get opportunities
const opportunities = await client.opportunities.list({
  type: 'arbitrage',
  minProfit: 0.5
});

// Submit bundle
const bundle = await client.bundles.submit({
  transactions: [...],
  tipLamports: 1000000
});

// Subscribe to real-time updates
client.ws.subscribe('opportunities', (opp) => {
  console.log('New opportunity:', opp);
});
```

### Python

```python
from solana_mev import Client

client = Client(
    api_key="your_api_key",
    environment="production"
)

# Get opportunities
opportunities = client.opportunities.list(
    type="arbitrage",
    min_profit=0.5
)

# Submit bundle
bundle = client.bundles.submit(
    transactions=[...],
    tip_lamports=1000000
)

# Subscribe to real-time updates
@client.on("opportunity")
def handle_opportunity(opp):
    print(f"New opportunity: {opp}")

client.connect()
```

### Rust

```rust
use solana_mev_sdk::{Client, OpportunityFilter};

#[tokio::main]
async fn main() {
    let client = Client::new(
        "your_api_key",
        Environment::Production
    );

    // Get opportunities
    let opportunities = client
        .opportunities()
        .list(OpportunityFilter {
            opportunity_type: Some("arbitrage"),
            min_profit: Some(0.5),
            ..Default::default()
        })
        .await?;

    // Submit bundle
    let bundle = client
        .bundles()
        .submit(Bundle {
            transactions: vec![...],
            tip_lamports: 1000000,
        })
        .await?;
}
```