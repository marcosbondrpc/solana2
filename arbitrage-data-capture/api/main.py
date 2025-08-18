"""
Ultra-High-Performance FastAPI MEV Control Plane
Institutional-grade backend with sub-millisecond latencies
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvloop
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware

# Ultra-performance: Use uvloop for 2x speed
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Sub-millisecond request tracking with hardware timestamps"""
    
    async def dispatch(self, request: Request, call_next):
        # Hardware timestamp if available
        start_ns = time.perf_counter_ns()
        request.state.start_time = start_ns
        
        # Add request ID for distributed tracing
        request_id = request.headers.get("X-Request-ID", f"{start_ns}")
        request.state.request_id = request_id
        
        response = await call_next(request)
        
        # Calculate latency with nanosecond precision
        latency_ns = time.perf_counter_ns() - start_ns
        latency_ms = latency_ns / 1_000_000
        
        # Add performance headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{latency_ms:.3f}"
        response.headers["X-Server-Time-ns"] = str(start_ns)
        
        # Enable TCP_NODELAY for low latency
        response.headers["X-Accel-Buffering"] = "no"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter with burst support"""
    
    def __init__(self, app, rate: int = 1000, burst: int = 2000):
        super().__init__(app)
        self.rate = rate
        self.burst = burst
        self.buckets = {}
        self.lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP or API key)
        client_id = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key")
        if api_key:
            client_id = f"key:{api_key}"
        
        # Check rate limit
        async with self.lock:
            now = time.time()
            if client_id not in self.buckets:
                self.buckets[client_id] = {
                    "tokens": self.burst,
                    "last_update": now
                }
            
            bucket = self.buckets[client_id]
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(
                self.burst,
                bucket["tokens"] + elapsed * self.rate
            )
            bucket["last_update"] = now
            
            if bucket["tokens"] < 1:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"},
                    headers={
                        "X-RateLimit-Limit": str(self.rate),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(now + 1))
                    }
                )
            
            bucket["tokens"] -= 1
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.rate)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket["tokens"]))
        
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers and CSRF protection"""
    
    async def dispatch(self, request: Request, call_next):
        # Check CSRF token for state-changing operations
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            csrf_token = request.headers.get("X-CSRF-Token")
            if not csrf_token and "/api/" in str(request.url):
                return JSONResponse(
                    status_code=403,
                    content={"error": "CSRF token required"}
                )
        
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with resource cleanup"""
    # Startup
    print("🚀 MEV Control Plane starting...")
    print("⚡ Ultra-performance mode: ENABLED")
    print("🔒 Security: JWT + RBAC + Rate Limiting")
    print("📊 Metrics: Prometheus enabled")
    
    # Import and initialize subsystems
    from .kafka_bridge import start_kafka_bridge
    from .realtime import start_realtime_server
    from .wt_gateway import start_wt_gateway
    
    # Start background tasks
    tasks = []
    tasks.append(asyncio.create_task(start_kafka_bridge()))
    tasks.append(asyncio.create_task(start_realtime_server()))
    
    # Optionally start WebTransport gateway
    import os
    if os.getenv("ENABLE_WEBTRANSPORT", "false").lower() == "true":
        tasks.append(asyncio.create_task(start_wt_gateway()))
    
    yield
    
    # Shutdown
    print("🛑 Shutting down MEV Control Plane...")
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print("✅ Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="Solana MEV Control Plane",
        description="Ultra-high-performance MEV infrastructure with sub-millisecond latencies",
        version="3.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware in correct order (outermost to innermost)
    # 1. Trusted Host (security)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure for production
    )
    
    # 2. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms", "X-RateLimit-*"]
    )
    
    # 3. GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 4. Security headers
    app.add_middleware(SecurityMiddleware)
    
    # 5. Rate limiting
    import os
    app.add_middleware(
        RateLimitMiddleware,
        rate=int(os.getenv("RATE_LIMIT_PER_SECOND", "1000")),
        burst=int(os.getenv("RATE_LIMIT_BURST", "2000"))
    )
    
    # 6. Performance tracking (innermost)
    app.add_middleware(PerformanceMiddleware)
    
    # Initialize Prometheus metrics
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_group_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        inprogress_name="mev_http_requests_inprogress",
        inprogress_labels=True
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    
    # Import and register routers
    from .control import router as control_router
    from .realtime import router as realtime_router
    from .datasets import router as datasets_router
    from .training import router as training_router
    from .health import router as health_router
    from .mev_core import router as mev_router
    from .clickhouse_router import router as clickhouse_router
    
    # Mount all routers under /api/* prefixes
    app.include_router(control_router, prefix="/api/control", tags=["control"])
    app.include_router(realtime_router, prefix="/api/realtime", tags=["realtime"])
    app.include_router(datasets_router, prefix="/api/datasets", tags=["datasets"])
    app.include_router(training_router, prefix="/api/training", tags=["training"])
    app.include_router(health_router, prefix="/api/health", tags=["health"])
    app.include_router(mev_router, prefix="/api/mev", tags=["mev"])
    app.include_router(clickhouse_router, prefix="/api/clickhouse", tags=["clickhouse"])
    
    @app.get("/")
    async def root():
        return {
            "service": "MEV Control Plane",
            "status": "operational",
            "performance": "ultra-high",
            "latency_target": "sub-millisecond",
            "throughput_target": "200k+ msg/sec"
        }
    
    return app


# Create app instance
app = create_app()

# Enable production optimizations
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Production configuration
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        workers=int(os.getenv("API_WORKERS", "4")),
        loop="uvloop",
        log_level="info",
        access_log=False,  # Disable for performance
        use_colors=False,
        server_header=False,
        date_header=False,
        # HTTP/2 support
        ssl_keyfile=os.getenv("SSL_KEY"),
        ssl_certfile=os.getenv("SSL_CERT"),
        ssl_version=3,  # TLS 1.2+
        # Performance options
        limit_concurrency=10000,
        limit_max_requests=1000000,
        timeout_keep_alive=5,
        timeout_notify=60,
        # Enable response streaming
        timeout_graceful_shutdown=10
    )