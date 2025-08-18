"""
High-performance FastAPI backend for MEV infrastructure
DEFENSIVE-ONLY: No execution/trading capabilities
"""

import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    from prometheus_client.core import CollectorRegistry
except Exception:
    class _DummyMetric:
        def labels(self, *args, **kwargs): return self
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
    def Counter(*args, **kwargs): return _DummyMetric()
    def Histogram(*args, **kwargs): return _DummyMetric()
    def Gauge(*args, **kwargs): return _DummyMetric()
    class CollectorRegistry: pass
    def generate_latest(_): return b""
import uvicorn

# Import routes
from routes.datasets import router as datasets_router
from routes.clickhouse import router as clickhouse_router
from routes.training import router as training_router
from routes.control import router as control_router
from routes.realtime import router as realtime_router

# Import defensive services
from defensive_integration import router as defensive_router

# Import services
from services.clickhouse_client import initialize_clickhouse

# Import security
from security.audit import AuditMiddleware, audit_logger

# Metrics
registry = CollectorRegistry()
request_count = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry
)
request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
    registry=registry
)
active_connections = Gauge(
    "websocket_active_connections",
    "Active WebSocket connections",
    registry=registry
)

def get_env(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # Startup
    print("🚀 Starting MEV API Server...")
    
    # Initialize services (non-fatal)
    # Initialize ClickHouse
    try:
        await initialize_clickhouse(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
            database=os.getenv("CLICKHOUSE_DB", "mev"),
            user=os.getenv("CLICKHOUSE_USER", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", "")
        )
        print("✅ ClickHouse initialized")
    except Exception as e:
        print(f"⚠️ ClickHouse init failed: {e}")

    # Initialize Kafka bridge
    try:
        from services.kafka_bridge import initialize_kafka_bridge
        await initialize_kafka_bridge(
            bootstrap_servers=get_env("KAFKA_SERVERS", "KAFKA") or "localhost:9092",
            group_id="mev-api-consumer"
        )
        print("✅ Kafka bridge initialized")
    except Exception as e:
        print(f"⚠️ Kafka bridge init failed: {e}")

    # Verify audit chain integrity
    try:
        chain_valid = await audit_logger.verify_chain()
        print("✅ Audit chain verified" if chain_valid else "⚠️ WARNING: Audit chain integrity check failed")
    except Exception as e:
        print(f"⚠️ Audit chain verification error: {e}")
    
    print("✅ MEV API Server ready")
    print(f"📊 Metrics available at /metrics")
    print(f"📚 API docs available at /docs")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down MEV API Server...")
    
    # Cleanup services
    from services.clickhouse_client import clickhouse_pool
    try:
        from services.kafka_bridge import kafka_bridge
    except Exception:
        kafka_bridge = None
    
    if clickhouse_pool:
        await clickhouse_pool.close()
    
    if kafka_bridge:
        await kafka_bridge.stop()
    
    print("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="MEV Infrastructure API",
    description="Defensive-only MEV detection and analysis system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://45.157.234.184:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(AuditMiddleware)


# Metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics"""
    import time
    
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Record metrics
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check with dependency status"""
    from api.services.clickhouse_client import get_clickhouse_pool
    from api.models.schemas import HealthResponse, DependencyStatus
    
    dependencies = []
    overall_status = "healthy"
    
    # Check ClickHouse
    try:
        pool = await get_clickhouse_pool()
        start = datetime.now()
        await pool.execute_query("SELECT 1", use_cache=False)
        latency = (datetime.now() - start).total_seconds() * 1000
        
        dependencies.append(DependencyStatus(
            name="clickhouse",
            status="healthy",
            latency_ms=latency
        ))
    except Exception as e:
        dependencies.append(DependencyStatus(
            name="clickhouse",
            status="down",
            error=str(e)
        ))
        overall_status = "degraded"
    # Check Kafka
    try:
        from services.kafka_bridge import get_kafka_bridge
        bridge = await get_kafka_bridge()
        lag = await bridge.get_consumer_lag()
        total_lag = sum(lag.values())
        
        
        status = "healthy" if total_lag < 10000 else "degraded"
        dependencies.append(DependencyStatus(
            name="kafka",
            status=status,
            metadata={"total_lag": total_lag}
        ))
        
        if status == "degraded":
            overall_status = "degraded"
    except Exception as e:
        dependencies.append(DependencyStatus(
            name="kafka",
            status="down",
            error=str(e)
        ))
        overall_status = "critical" if overall_status == "degraded" else "degraded"
    
    # Check audit log
    try:
        chain_valid = await audit_logger.verify_chain()
        dependencies.append(DependencyStatus(
            name="audit_log",
            status="healthy" if chain_valid else "degraded",
            metadata={"chain_valid": chain_valid}
        ))
    except Exception as e:
        dependencies.append(DependencyStatus(
            name="audit_log",
            status="down",
            error=str(e)
        ))
    
    metrics = {}
    uptime = 0.0
    try:
        import psutil
        process = psutil.Process()
        uptime = (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds()
        metrics = {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads()
        }
    except Exception as e:
        metrics = {"error": str(e)}
    
    return HealthResponse(
        success=True,
        status=overall_status,
        version="1.0.0",
        uptime_seconds=uptime,
        dependencies=dependencies,
        metrics=metrics
    ).dict()


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(generate_latest(registry))


# Include routers
app.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
app.include_router(clickhouse_router, prefix="/clickhouse", tags=["clickhouse"])
app.include_router(training_router, prefix="/training", tags=["training"])
app.include_router(control_router, prefix="/control", tags=["control"])
app.include_router(realtime_router, prefix="/realtime", tags=["realtime"])
app.include_router(defensive_router, prefix="/defensive", tags=["defensive"])


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError):
    return JSONResponse(
        status_code=403,
        content={
            "success": False,
            "message": "Permission denied",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    import traceback
    
    # Log the full traceback for debugging
    print(f"Unhandled exception: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable in production
        workers=4,
        loop="uvloop",
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )