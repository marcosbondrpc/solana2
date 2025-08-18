# Dashboard API Service

## Overview

The Dashboard API is a high-performance Rust service providing real-time data streaming and analytics for the SOTA MEV dashboard. It handles 50,000+ requests/second with sub-100ms latency while streaming live MEV data to thousands of concurrent WebSocket connections.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │───▶│  Dashboard API   │───▶│   ClickHouse    │
│   Dashboard     │    │                  │    │    Database     │
└─────────────────┘    │ ┌──────────────┐ │    └─────────────────┘
                       │ │  REST API    │ │
┌─────────────────┐    │ └──────────────┘ │    ┌─────────────────┐
│   Mobile App    │───▶│                  │───▶│     Redis       │
└─────────────────┘    │ ┌──────────────┐ │    │     Cache       │
                       │ │ WebSocket    │ │    └─────────────────┘
┌─────────────────┐    │ │  Streaming   │ │
│   Grafana       │───▶│ └──────────────┘ │    ┌─────────────────┐
└─────────────────┘    └──────────────────┘    │     Kafka       │
                                               │   Event Bus     │
                                               └─────────────────┘
```

## Core Features

### 1. Real-time Data API
- **Live MEV Opportunities**: Stream detected opportunities in real-time
- **Bundle Outcomes**: Real-time bundle success/failure notifications
- **Performance Metrics**: Live system performance indicators
- **Decision DNA**: Complete audit trail of MEV decisions

### 2. Historical Analytics
- **Time-series Queries**: Efficient historical data aggregation
- **Performance Analysis**: Detailed performance breakdowns
- **Profit Analytics**: Comprehensive profit and loss analysis
- **Risk Metrics**: Historical risk assessment data

### 3. WebSocket Streaming
- **Ultra-low Latency**: Sub-millisecond message delivery
- **High Throughput**: 100k+ messages/second capability
- **Smart Filtering**: Client-side filtering to reduce bandwidth
- **Automatic Reconnection**: Robust connection management

### 4. Caching Layer
- **Redis Integration**: Multi-level caching strategy
- **Smart Cache Keys**: Intelligent cache invalidation
- **Cache Warming**: Proactive cache population
- **Cache Analytics**: Cache hit ratio monitoring

## API Endpoints

### Real-time Data Endpoints

#### Get Live MEV Opportunities
```http
GET /api/v1/opportunities/live?limit=100&filter=arbitrage
```

#### Get System Performance Metrics
```http
GET /api/v1/metrics/performance?window=1h
```

#### Get Bundle Success Rate
```http
GET /api/v1/metrics/bundles?timeframe=24h&granularity=1m
```

### Historical Analytics Endpoints

#### Query Decision DNA
```http
GET /api/v1/decisions?start_time=2024-01-01&end_time=2024-01-02&model=arbitrage_v3.2
```

#### Get Profit Analysis
```http
GET /api/v1/analytics/profit?period=7d&group_by=strategy
```

#### Get Risk Metrics
```http
GET /api/v1/analytics/risk?period=30d&metric=var_95
```

### Performance Endpoints

#### Get Latency Metrics
```http
GET /api/v1/performance/latency?service=mev-engine&period=1h
```

#### Get Throughput Metrics
```http
GET /api/v1/performance/throughput?period=24h&granularity=5m
```

## WebSocket Streaming

### Connection Establishment
```javascript
const ws = new WebSocket('wss://api.mev.example.com/ws');

ws.onopen = () => {
  // Subscribe to real-time opportunities
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['opportunities', 'bundles', 'metrics'],
    filters: {
      min_profit: 0.001,
      strategies: ['arbitrage', 'sandwich']
    }
  }));
};
```

### Message Types

#### Opportunity Events
```json
{
  "type": "opportunity",
  "timestamp": 1704067200000000000,
  "data": {
    "id": "opp_12345",
    "strategy": "arbitrage",
    "profit_estimate": 0.0052,
    "confidence": 0.89,
    "dexes": ["raydium", "orca"],
    "token_pair": "SOL/USDC"
  }
}
```

#### Bundle Outcome Events
```json
{
  "type": "bundle_outcome",
  "timestamp": 1704067200500000000,
  "data": {
    "bundle_id": "bundle_67890",
    "success": true,
    "profit_actual": 0.0048,
    "gas_used": 150000,
    "slot": 12345678,
    "latency_ms": 15.2
  }
}
```

#### Performance Metrics Events
```json
{
  "type": "metrics",
  "timestamp": 1704067200000000000,
  "data": {
    "decision_latency_p99": 18.5,
    "bundle_land_rate": 0.672,
    "throughput_ops_per_sec": 45230,
    "active_connections": 1247
  }
}
```

## High-Performance Implementation

### Async Request Handling
```rust
use axum::{Router, extract::Query, response::Json};
use tokio::time::Instant;

#[derive(Deserialize)]
struct OpportunityQuery {
    limit: Option<u32>,
    strategy: Option<String>,
    min_profit: Option<f64>,
}

async fn get_opportunities(
    Query(params): Query<OpportunityQuery>,
    State(app_state): State<AppState>,
) -> Result<Json<Vec<Opportunity>>, ApiError> {
    let start = Instant::now();
    
    // Check cache first
    if let Some(cached) = app_state.cache.get_opportunities(&params).await? {
        observe_latency("cache_hit", start.elapsed());
        return Ok(Json(cached));
    }
    
    // Query database
    let opportunities = app_state.db
        .query_opportunities(params.into())
        .await?;
    
    // Update cache
    app_state.cache.set_opportunities(&params, &opportunities).await?;
    
    observe_latency("db_query", start.elapsed());
    Ok(Json(opportunities))
}
```

### WebSocket Connection Management
```rust
use axum::extract::ws::{WebSocket, Message};
use tokio::sync::broadcast;

struct ConnectionManager {
    connections: DashMap<ConnectionId, WebSocketSender>,
    event_channel: broadcast::Sender<Event>,
}

impl ConnectionManager {
    async fn handle_connection(&self, socket: WebSocket) {
        let (sender, mut receiver) = socket.split();
        let connection_id = Uuid::new_v4();
        
        // Store connection
        self.connections.insert(connection_id, sender);
        
        // Listen for events and forward to client
        let mut event_receiver = self.event_channel.subscribe();
        
        tokio::spawn(async move {
            while let Ok(event) = event_receiver.recv().await {
                if let Ok(message) = serde_json::to_string(&event) {
                    let _ = sender.send(Message::Text(message)).await;
                }
            }
        });
        
        // Handle incoming messages
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    self.handle_client_message(connection_id, text).await;
                }
                Ok(Message::Close(_)) => break,
                _ => {}
            }
        }
        
        // Clean up connection
        self.connections.remove(&connection_id);
    }
}
```

### Database Query Optimization
```rust
use sqlx::{Pool, Postgres, Row};

struct PerformanceQueries {
    pool: Pool<Postgres>,
}

impl PerformanceQueries {
    async fn get_bundle_metrics(&self, timeframe: &str) -> Result<BundleMetrics> {
        // Optimized query with proper indexing
        let rows = sqlx::query(r#"
            SELECT 
                toStartOfInterval(timestamp, INTERVAL 1 MINUTE) as time_bucket,
                countIf(landed = 1) as successful_bundles,
                count() as total_bundles,
                avgIf(profit_lamports, landed = 1) as avg_profit,
                quantileExact(0.99)(decision_latency_us) as p99_latency
            FROM bundles 
            WHERE timestamp >= now() - INTERVAL ?
            GROUP BY time_bucket
            ORDER BY time_bucket
        "#)
        .bind(timeframe)
        .fetch_all(&self.pool)
        .await?;
        
        let metrics = rows.into_iter()
            .map(|row| BundleMetric {
                timestamp: row.get("time_bucket"),
                success_rate: row.get::<f64, _>("successful_bundles") / row.get::<f64, _>("total_bundles"),
                avg_profit: row.get("avg_profit"),
                p99_latency: row.get("p99_latency"),
            })
            .collect();
            
        Ok(BundleMetrics { data: metrics })
    }
}
```

### Redis Caching Strategy
```rust
use redis::{AsyncCommands, Connection};
use serde::{Serialize, Deserialize};

struct CacheManager {
    connection: redis::aio::Connection,
}

impl CacheManager {
    async fn get_cached<T>(&mut self, key: &str) -> Result<Option<T>> 
    where 
        T: for<'de> Deserialize<'de>
    {
        let cached: Option<String> = self.connection.get(key).await?;
        
        match cached {
            Some(data) => {
                let deserialized = serde_json::from_str(&data)?;
                Ok(Some(deserialized))
            }
            None => Ok(None)
        }
    }
    
    async fn set_cached<T>(&mut self, key: &str, value: &T, ttl: u64) -> Result<()>
    where 
        T: Serialize
    {
        let serialized = serde_json::to_string(value)?;
        self.connection.setex(key, ttl, serialized).await?;
        Ok(())
    }
    
    // Smart cache key generation
    fn generate_cache_key(&self, endpoint: &str, params: &impl Serialize) -> String {
        let params_hash = seahash::hash(
            &serde_json::to_vec(params).unwrap()
        );
        format!("api:{}:{:x}", endpoint, params_hash)
    }
}
```

## Performance Monitoring

### Custom Metrics Collection
```rust
use prometheus::{Counter, Histogram, Gauge, register_counter, register_histogram};

lazy_static! {
    static ref API_REQUESTS: Counter = register_counter!(
        "api_requests_total", 
        "Total number of API requests"
    ).unwrap();
    
    static ref REQUEST_LATENCY: Histogram = register_histogram!(
        "api_request_duration_seconds",
        "Request latency in seconds"
    ).unwrap();
    
    static ref ACTIVE_WEBSOCKETS: Gauge = register_gauge!(
        "websocket_connections_active",
        "Number of active WebSocket connections"
    ).unwrap();
}

fn observe_request(endpoint: &str, duration: Duration, status: u16) {
    API_REQUESTS.inc();
    REQUEST_LATENCY.observe(duration.as_secs_f64());
    
    // Custom metrics per endpoint
    let endpoint_latency = register_histogram!(
        &format!("api_{}_duration_seconds", endpoint),
        &format!("Latency for {} endpoint", endpoint)
    ).unwrap();
    
    endpoint_latency.observe(duration.as_secs_f64());
}
```

### Health Check Implementation
```rust
async fn health_check(State(app_state): State<AppState>) -> Result<Json<HealthStatus>> {
    let mut health = HealthStatus::new();
    
    // Check database connectivity
    match app_state.db.ping().await {
        Ok(_) => health.services.insert("clickhouse".to_string(), "healthy".to_string()),
        Err(e) => health.services.insert("clickhouse".to_string(), format!("error: {}", e))
    };
    
    // Check Redis connectivity  
    match app_state.cache.ping().await {
        Ok(_) => health.services.insert("redis".to_string(), "healthy".to_string()),
        Err(e) => health.services.insert("redis".to_string(), format!("error: {}", e))
    };
    
    // Check system resources
    health.metrics.insert("memory_usage_mb".to_string(), get_memory_usage());
    health.metrics.insert("cpu_usage_pct".to_string(), get_cpu_usage());
    health.metrics.insert("active_connections".to_string(), ACTIVE_WEBSOCKETS.get());
    
    // Overall health determination
    health.status = if health.services.values().all(|s| s == "healthy") {
        "healthy".to_string()
    } else {
        "degraded".to_string()
    };
    
    Ok(Json(health))
}
```

## Security Features

### JWT Authentication
```rust
use jsonwebtoken::{decode, DecodingKey, Validation, Algorithm};

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
    role: String,
    permissions: Vec<String>,
}

async fn authenticate(
    headers: HeaderMap,
    request: Request<Body>,
    next: Next<Body>,
) -> Result<Response, AuthError> {
    let token = headers
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or(AuthError::MissingToken)?;
    
    let claims = decode::<Claims>(
        token,
        &DecodingKey::from_secret(JWT_SECRET.as_ref()),
        &Validation::new(Algorithm::HS256),
    )
    .map_err(|_| AuthError::InvalidToken)?;
    
    // Add claims to request extensions
    request.extensions_mut().insert(claims.claims);
    
    Ok(next.run(request).await)
}
```

### Rate Limiting
```rust
use tower::limit::RateLimitLayer;
use std::time::Duration;

fn create_rate_limiter() -> RateLimitLayer {
    RateLimitLayer::new(
        1000, // requests per window
        Duration::from_secs(60) // 1 minute window
    )
}

// Per-user rate limiting
async fn per_user_rate_limit(
    claims: Extension<Claims>,
    State(rate_limiter): State<RateLimiter>,
) -> Result<(), RateLimitError> {
    let user_id = &claims.sub;
    
    if !rate_limiter.check_rate_limit(user_id, 100, Duration::from_secs(60)).await? {
        return Err(RateLimitError::ExceededLimit);
    }
    
    Ok(())
}
```

## Integration Points

- **ClickHouse**: Primary data source for all analytics and historical queries
- **Redis**: Caching layer for frequently accessed data and session management
- **Kafka**: Real-time event streaming for live dashboard updates
- **Control Plane**: Authentication and authorization service integration
- **MEV Engine**: Direct integration for real-time opportunity and outcome data