from typing import Optional
import os
from prometheus_client import CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest, Histogram, Gauge, Counter
from fastapi import APIRouter, Response

router = APIRouter()

# Multiprocess registry if configured
_registry: CollectorRegistry
if os.getenv("PROM_MULTIPROC_DIR"):
    from prometheus_client import CollectorRegistry, multiprocess, Gauge as _Gauge
    _registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(_registry)
else:
    from prometheus_client import REGISTRY as _registry  # type: ignore

API_LATENCY_MS = Histogram(
    "api_http_latency_ms",
    "API request latency in milliseconds",
    ["route", "method", "status"],
    buckets=(5,10,20,50,100,200,400,800,1600),
    registry=_registry,
)
WS_CLIENTS = Gauge(
    "ws_clients",
    "Number of connected WebSocket clients",
    registry=_registry,
)
WS_BACKLOG = Gauge(
    "ws_backlog_size",
    "Approx backlog size per WebSocket client",
    ["client_id"],
    registry=_registry,
)
INGEST_LAG_MS = Gauge(
    "ingest_lag_ms",
    "Ingestion lag in milliseconds",
    registry=_registry,
)
CH_QUERY_LAT_MS = Histogram(
    "ch_query_latency_ms",
    "ClickHouse query latency in milliseconds",
    buckets=(5,10,20,50,100,200,400,800,1600),
    registry=_registry,
)
WS_DROPS = Counter(
    "ws_send_drops_total",
    "Total dropped WS messages due to backpressure",
    registry=_registry,
)

@router.get("/metrics")
def metrics() -> Response:
    data = generate_latest(_registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)