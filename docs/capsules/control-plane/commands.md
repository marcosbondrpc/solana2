# Control Plane Commands & API

## Command Categories

### 1. Policy Management Commands

#### Update Risk Policy
```bash
curl -X POST http://localhost:8000/api/v1/policy/risk \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "policy_id": "risk_v2.1",
    "thresholds": {
      "max_position_size": 1000000,
      "max_slippage": 0.05,
      "min_profit_threshold": 0.001
    },
    "rules": {
      "stop_loss_enabled": "true",
      "position_timeout": "300"
    },
    "enabled": true
  }'
```

#### Update Execution Policy  
```bash
curl -X POST http://localhost:8000/api/v1/policy/execution \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "policy_id": "exec_v1.5",
    "thresholds": {
      "max_gas_price": 100,
      "max_bundle_size": 5,
      "priority_fee_multiplier": 1.2
    },
    "rules": {
      "bundle_optimization": "true",
      "dynamic_gas_pricing": "true"
    }
  }'
```

### 2. Model Management Commands

#### Hot Swap Model
```bash
curl -X POST http://localhost:8000/api/v1/model/swap \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "model_id": "arbitrage_v3.2",
    "model_path": "/models/arbitrage_v3.2.treelite",
    "model_type": "treelite_xgboost",
    "version": "3.2.0",
    "metadata": {
      "training_date": "2024-01-15",
      "accuracy": 0.892,
      "latency_us": 45
    }
  }'
```

#### Model A/B Test
```bash
curl -X POST http://localhost:8000/api/v1/model/ab-test \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "test_id": "arbitrage_ab_001",
    "model_a": "arbitrage_v3.1", 
    "model_b": "arbitrage_v3.2",
    "traffic_split": 0.1,
    "duration_hours": 24,
    "success_metrics": ["accuracy", "profit", "latency"]
  }'
```

### 3. Emergency Control Commands

#### Emergency Kill Switch
```bash
curl -X POST http://localhost:8000/api/v1/emergency/kill \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "target": "all_trading",
    "reason": "Market anomaly detected",
    "duration_ms": 300000,
    "force": true
  }'
```

#### Throttle System
```bash
curl -X POST http://localhost:8000/api/v1/emergency/throttle \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "target": "arbitrage_engine",
    "throttle_percent": 25,
    "reason": "High latency detected",
    "duration_ms": 600000
  }'
```

### 4. Configuration Commands

#### Update System Config
```bash
curl -X POST http://localhost:8000/api/v1/config/update \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "config_key": "max_concurrent_bundles",
    "config_value": "50",
    "config_type": "integer",
    "hot_reload": true
  }'
```

#### Bulk Config Update
```bash
curl -X POST http://localhost:8000/api/v1/config/bulk-update \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "configs": [
      {
        "key": "decision_timeout_ms",
        "value": "8",
        "type": "integer"
      },
      {
        "key": "bundle_land_threshold",
        "value": "0.65",
        "type": "float"
      }
    ],
    "hot_reload": true
  }'
```

## Query APIs

### System Status
```bash
# Get overall system health
curl http://localhost:8000/api/v1/status/health

# Get detailed system metrics
curl http://localhost:8000/api/v1/status/metrics

# Get service dependencies status
curl http://localhost:8000/api/v1/status/dependencies
```

### Policy Information
```bash
# Get active policies
curl http://localhost:8000/api/v1/policy/active

# Get policy history
curl http://localhost:8000/api/v1/policy/history?hours=24

# Get policy effectiveness metrics
curl http://localhost:8000/api/v1/policy/metrics?policy_id=risk_v2.1
```

### Model Information
```bash
# Get active models
curl http://localhost:8000/api/v1/model/active

# Get model performance metrics
curl http://localhost:8000/api/v1/model/performance?model_id=arbitrage_v3.2

# Get A/B test results
curl http://localhost:8000/api/v1/model/ab-test/results?test_id=arbitrage_ab_001
```

## Command Authentication

### Ed25519 Signature Format
```python
import ed25519
import json
import time

# Create command payload
command = {
    "id": "cmd_001",
    "module": "policy", 
    "action": "update_risk",
    "params": {"max_position_size": 1000000},
    "nonce": int(time.time() * 1000),
    "timestamp_ns": int(time.time() * 1e9),
    "pubkey_id": "control_key_001"
}

# Sign command
private_key = ed25519.SigningKey(key_bytes)
message = json.dumps(command, sort_keys=True).encode()
signature = private_key.sign(message)

# Add signature to command
command["signature"] = signature.hex()
```

### JWT Token Authentication
```bash
# Obtain JWT token
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "secure_password",
    "mfa_token": "123456"
  }' | jq -r '.access_token')

# Use token in subsequent requests
curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/api/v1/status/health
```

## Batch Operations

### Batch Command Execution
```bash
curl -X POST http://localhost:8000/api/v1/command/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "commands": [
      {
        "id": "cmd_001",
        "module": "policy",
        "action": "update_risk",
        "params": {"max_position_size": 1000000}
      },
      {
        "id": "cmd_002", 
        "module": "model",
        "action": "swap",
        "params": {"model_id": "arbitrage_v3.2"}
      }
    ],
    "atomic": true,
    "timeout_ms": 30000
  }'
```

## WebSocket Real-time API

### Connect to Command Stream
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/commands');
ws.onmessage = (event) => {
  const command = JSON.parse(event.data);
  console.log('Command executed:', command);
};
```

### Subscribe to System Events
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/events');
ws.send(JSON.stringify({
  "subscribe": ["policy_updates", "model_swaps", "emergency_actions"]
}));
```

## CLI Interface

### Installation
```bash
pip install mev-control-cli
mev-control configure --endpoint http://localhost:8000 --auth-token $TOKEN
```

### Common Commands
```bash
# Policy management
mev-control policy update --file risk_policy.json
mev-control policy list --active-only

# Model management  
mev-control model swap --id arbitrage_v3.2 --path /models/arbitrage_v3.2.treelite
mev-control model status --id arbitrage_v3.2

# Emergency controls
mev-control emergency kill --target all_trading --reason "manual_intervention"
mev-control emergency throttle --target arbitrage_engine --percent 50

# System monitoring
mev-control status health
mev-control status metrics --service arbitrage_engine
```