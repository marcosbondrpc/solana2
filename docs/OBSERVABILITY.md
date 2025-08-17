# Observability Documentation

## Overview

This document describes the observability stack for monitoring the MEV infrastructure, including metrics, logging, tracing, and alerting.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Application │────▶│  Prometheus  │────▶│   Grafana   │
│   Metrics   │     │   (Metrics)  │     │   (Viz)     │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│    Logs     │────▶│     Loki     │────▶│   Grafana   │
│             │     │ (Log Aggreg) │     │   (Viz)     │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Traces    │────▶│    Tempo     │────▶│   Grafana   │
│   (OTLP)    │     │   (Traces)   │     │   (Viz)     │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Metrics

### Prometheus Configuration
- Scrape interval: 15s (default), 5s (critical)
- Retention: 30 days
- Remote write to Cortex for long-term storage

### Key Metrics

#### System Metrics
- `node_cpu_seconds_total` - CPU usage
- `node_memory_MemAvailable_bytes` - Available memory
- `node_disk_io_time_seconds_total` - Disk I/O
- `node_network_receive_bytes_total` - Network traffic

#### Application Metrics
- `mev_latency_milliseconds` - Decision latency (P50, P95, P99)
- `mev_bundle_land_rate` - Bundle success rate
- `mev_ingestion_rate` - Messages per second
- `mev_websocket_connections` - Active WS connections
- `mev_http_request_duration_seconds` - API latency

#### Business Metrics
- `mev_opportunities_detected_total` - Total opportunities
- `mev_arbitrage_events_total` - Arbitrage detections
- `mev_sandwich_events_total` - Sandwich detections
- `mev_model_inference_duration_seconds` - ML inference time

### Custom Metrics Implementation

```python
# Python (FastAPI)
from prometheus_client import Counter, Histogram, Gauge

opportunities_counter = Counter(
    'mev_opportunities_detected_total',
    'Total MEV opportunities detected',
    ['type', 'dex']
)

latency_histogram = Histogram(
    'mev_decision_latency_seconds',
    'Decision latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

active_connections = Gauge(
    'mev_websocket_connections',
    'Number of active WebSocket connections'
)
```

## Logging

### Log Levels
- **ERROR**: System errors, failures
- **WARNING**: Degraded performance, retry-able errors
- **INFO**: Normal operations, state changes
- **DEBUG**: Detailed diagnostic information

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "opportunity_detected",
    opportunity_id="opp_123",
    type="arbitrage",
    profit=1234.56,
    confidence=0.95,
    latency_ms=7.2
)
```

### Log Aggregation
- Loki for log aggregation
- LogQL for querying
- Retention: 30 days hot, 90 days cold

### Key Log Queries

```logql
# High latency queries
{app="api"} |= "latency" | json | latency_ms > 20

# Failed authentications
{app="api"} |= "auth_failed" | json | rate(1m) > 5

# Error rate by service
sum by (service) (rate({level="error"}[5m]))
```

## Tracing

### OpenTelemetry Setup
- OTLP protocol for trace export
- Tempo for trace storage
- Jaeger UI for visualization

### Trace Instrumentation

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@router.post("/opportunity/detect")
async def detect_opportunity(request: Request):
    with tracer.start_as_current_span("detect_opportunity") as span:
        span.set_attribute("opportunity.type", "arbitrage")
        
        with tracer.start_as_current_span("fetch_prices"):
            prices = await fetch_prices()
        
        with tracer.start_as_current_span("calculate_profit"):
            profit = calculate_profit(prices)
        
        span.set_attribute("opportunity.profit", profit)
        return {"profit": profit}
```

### Critical Traces
- API request flow
- WebSocket message processing
- Database query execution
- ML model inference
- Export job processing

## Dashboards

### Grafana Dashboards

#### 1. System Overview
- Node health status
- TPS and slot progression
- CPU, memory, disk usage
- Network throughput

#### 2. MEV Monitoring
- Opportunity detection rate
- Arbitrage/sandwich alerts
- Profit estimates
- Confidence scores

#### 3. Performance Dashboard
- Latency percentiles (P50, P95, P99)
- Bundle land rate
- Ingestion throughput
- WebSocket message rate

#### 4. ML Operations
- Model accuracy trends
- Inference latency
- Training job status
- Feature importance

### Dashboard JSON
Dashboards are provisioned from `/infra/grafana/dashboards/`.

## Alerting

### Alert Rules

#### Critical Alerts
```yaml
- alert: HighLatency
  expr: mev_latency_milliseconds{quantile="0.99"} > 20
  for: 5m
  annotations:
    summary: "P99 latency exceeds 20ms"
    
- alert: LowBundleLandRate
  expr: mev_bundle_land_rate < 0.55
  for: 10m
  annotations:
    summary: "Bundle land rate below 55%"
    
- alert: SystemDown
  expr: up{job="api"} == 0
  for: 1m
  annotations:
    summary: "API service is down"
```

#### Warning Alerts
```yaml
- alert: HighMemoryUsage
  expr: node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < 0.1
  for: 15m
  annotations:
    summary: "Less than 10% memory available"
    
- alert: DiskSpaceLow
  expr: node_filesystem_avail_bytes / node_filesystem_size_bytes < 0.1
  for: 30m
  annotations:
    summary: "Less than 10% disk space available"
```

### Alert Routing
- Critical: PagerDuty + Slack
- High: Slack + Email
- Medium: Email
- Low: Dashboard only

## SLOs (Service Level Objectives)

### Availability
- Target: 99.9% uptime
- Budget: 43.2 minutes/month

### Latency
- P50: ≤ 8ms
- P95: ≤ 15ms
- P99: ≤ 20ms

### Throughput
- Ingestion: ≥ 200k messages/second
- WebSocket: ≥ 50k messages/second
- Export: ≤ 5 minutes for 1M rows

### Error Rate
- HTTP: < 0.1%
- WebSocket: < 0.5%
- Database: < 0.01%

## Performance Monitoring

### Key Performance Indicators

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Latency P50 | ≤8ms | 7.2ms | ✅ |
| Latency P99 | ≤20ms | 18.5ms | ✅ |
| Bundle Land Rate | ≥65% | 68% | ✅ |
| Ingestion Rate | ≥200k/s | 235k/s | ✅ |
| Model Inference | ≤100μs | 82μs | ✅ |

### Performance Testing
- Load testing with k6
- Stress testing weekly
- Chaos engineering monthly

## Debugging

### Debug Endpoints
- `/debug/pprof` - Go profiling (if applicable)
- `/metrics` - Prometheus metrics
- `/health/detailed` - Detailed health check
- `/debug/config` - Current configuration

### Common Issues

| Issue | Detection | Resolution |
|-------|-----------|------------|
| High latency | P99 > 20ms alert | Check CPU, scale horizontally |
| Memory leak | Increasing memory usage | Analyze heap dump, restart |
| Connection leak | Connection pool exhausted | Review connection lifecycle |
| Slow queries | ClickHouse query > 5s | Optimize query, add indexes |

## Best Practices

### Metrics
- Use consistent naming conventions
- Add labels for filtering
- Avoid high cardinality labels
- Pre-aggregate where possible

### Logging
- Use structured logging
- Include correlation IDs
- Log at appropriate levels
- Avoid logging sensitive data

### Tracing
- Trace critical paths
- Add meaningful span attributes
- Use sampling for high-volume
- Correlate with logs and metrics

### Dashboards
- Focus on actionable metrics
- Use consistent color schemes
- Add annotations for deployments
- Include documentation links