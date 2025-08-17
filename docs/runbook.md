## Prerequisites
- Ubuntu 22.04+ or similar
- ClickHouse server reachable
- Python 3.10+, Node 18+ (for frontend build, optional)
- Nginx installed and TLS certs available
- Systemd available
- Prometheus and Grafana (optional but recommended)

## Environment layout
- App code: /opt/solana-realtime/app (git checkout here)
- Env file: /opt/solana-realtime/uvicorn.env (from infra/uvicorn/uvicorn.env)
- Nginx conf: /etc/nginx/conf.d/solana-realtime.conf (from infra/nginx/solana-realtime.conf)
- Systemd units: /etc/systemd/system/{backend-api.service,solana-onlogs.service}

## Configure environment
Create /opt/solana-realtime/uvicorn.env:
```ini
HOST=0.0.0.0
PORT=8080
UVICORN_WORKERS=4
UVICORN_LOG_LEVEL=info
CH_URL=http://localhost:8123
CH_DB=solana_rt_dev
CH_TIMEOUT_S=1.0
CORS_ORIGINS=["https://your.app"]
API_REQUIRE_AUTH=false
REST_JWT_SECRET=
WS_REQUIRE_TOKEN=false
WS_TOKEN_SECRET=
RATE_LIMIT_PER_MIN=600
```

## ClickHouse: create databases, tables, MVs
Dev:
```bash
clickhouse-client --multiquery < clickhouse/ddl/dev/000_db.sql
clickhouse-client --multiquery < clickhouse/ddl/dev/010_tables.sql
clickhouse-client --multiquery < clickhouse/ddl/dev/020_mv_1m.sql
clickhouse-client --multiquery < clickhouse/ddl/dev/040_roles.sql
clickhouse-client --multiquery < clickhouse/ddl/dev/050_quotas.sql
clickhouse-client --multiquery < clickhouse/ddl/dev/060_ttls.sql
```
Prod: run corresponding files under clickhouse/ddl/prod in the same order.

Verify:
```bash
clickhouse-client -q "SHOW TABLES FROM solana_rt_dev"
clickhouse-client -q "DESCRIBE TABLE solana_rt_dev.detections_1m"
```

## Run backend (manual)
```bash
cd /opt/solana-realtime/app
source /opt/solana-realtime/uvicorn.env >/dev/null 2>&1 || true
uvicorn backend.app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8080} --workers ${UVICORN_WORKERS:-4} --http httptools --loop uvloop --log-level ${UVICORN_LOG_LEVEL:-info}
```

## Nginx
- Copy infra/nginx/solana-realtime.conf to /etc/nginx/conf.d/solana-realtime.conf
- Set ssl_certificate and ssl_certificate_key paths
- Test and reload:
```bash
nginx -t && systemctl reload nginx
```

## Systemd services
Install units:
```bash
install -d /opt/solana-realtime/app
cp -r . /opt/solana-realtime/app
install -D -m 0644 infra/uvicorn/uvicorn.env /opt/solana-realtime/uvicorn.env
install -D -m 0644 systemd/backend-api.service /etc/systemd/system/backend-api.service
install -D -m 0644 systemd/solana-onlogs.service /etc/systemd/system/solana-onlogs.service
systemctl daemon-reload
systemctl enable --now backend-api.service solana-onlogs.service
systemctl status backend-api.service --no-pager
systemctl status solana-onlogs.service --no-pager
```

## Prometheus scraping
API metrics:
```yaml
- job_name: solana-realtime-api
  static_configs:
  - targets: ['your.api.host:443']
  metrics_path: /metrics
  scheme: https
```
Ingestion metrics:
```yaml
- job_name: solana-realtime-ingest
  static_configs:
  - targets: ['your.ingest.host:9108']
```

## Grafana
- Add Prometheus datasource
- Build dashboards using these metrics:
  - api_http_latency_ms (p50/p95/p99 by route)
  - ch_query_latency_ms
  - ws_clients, ws_backlog_size
  - ingest_batches_total, ingest_rows_dropped_total, ch_insert_latency_ms, ingest_lag_ms
- Alerts (examples):
  - API p99 > 200ms for 10m
  - WS backlog > 50% for 1m
  - CH insert p95 > 200ms for 10m

## Frontend integration (quick start)
- Ensure VITE_API_BASE points to Nginx origin (e.g., https://your.api.host)
- At app bootstrap:
```ts
import { API_BASE, fetchSnapshot } from './src/api/client';
import { bootstrapSnapshot } from './src/store/detectionsStore';
import { connectWS } from './src/realtime/ws';

(async () => {
  const snap = await fetchSnapshot(200);
  bootstrapSnapshot(snap.detections, snap.as_of_seq);
  connectWS(API_BASE, "/ws");
})();
```

## Synthetic validation

1) Health and metrics
```bash
curl -sSf https://your.api.host/health
curl -sSf https://your.api.host/metrics | head
```

2) Seed synthetic detections
```bash
clickhouse-client -q "
INSERT INTO solana_rt_dev.detections (seq, ts, slot, kind, sig, address, score, payload)
SELECT number, now64(3), 0, 'test', toString(number), 'addr_'||toString(number%10), 0.5, '{}'
FROM numbers(500);
"
```

3) Snapshot endpoint
```bash
curl -sSf "https://your.api.host/api/snapshot?limit=200" | jq '.detections | length'
```

4) WebSocket replay and live
- Use websocat:
```bash
websocat -b --binary wss://your.api.host/ws
# send subscribe (JSON fallback accepted)
{"t":"subscribe","last_seq":0}
```
- Expect hello, then subsequent detection frames when broadcasted.

5) Gap replay test
```bash
curl -sSf "https://your.api.host/api/detections/range?from_seq=100&to_seq=200" | jq 'length'
```

6) Load test (k6)
k6 script save as k6-api.js:
```js
import http from 'k6/http';
import { sleep } from 'k6';
export let options = { vus: 50, duration: '60s' };
export default function () {
  http.get('https://your.api.host/api/snapshot?limit=200', { timeout: '2s' });
  sleep(0.1);
}
```
Run:
```bash
k6 run k6-api.js
```
- Acceptance: p99 < 200ms.

7) WS soak test (manual)
- Open multiple tabs or run multiple websocat clients; observe ws_clients and ws_backlog_size in /metrics; verify no unbounded growth over 30 minutes.

## Troubleshooting
- 502 from Nginx: check backend-api.service status and PORT in uvicorn.env
- /metrics empty: ensure PROM_MULTIPROC_DIR unset or configured properly if using multiple workers
- CH timeouts: increase CH_TIMEOUT_S and verify ClickHouse availability
- CORS blocked: update CORS_ORIGINS JSON list in uvicorn.env and restart backend

## Rollback plan
- Revert DDL by switching to previous tables/views (use separate _v2 and swap MVs if needed)
- Disable ingestion via systemctl stop solana-onlogs.service
- Reduce WS client load by temporarily setting WS_REQUIRE_TOKEN=true and distributing limited tokens