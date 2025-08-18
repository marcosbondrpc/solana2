"""
Health and monitoring endpoints
Real-time system metrics with nanosecond precision
"""

import os
import time
import psutil
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import aioredis

from .deps import get_redis


router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "MEV Control Plane",
        "version": "3.0.0"
    }


@router.get("/live")
async def liveness_probe() -> Dict[str, Any]:
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp_ns": time.time_ns()}


@router.get("/ready")
async def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe - checks dependencies"""
    checks = {
        "api": True,
        "kafka": False,
        "redis": False,
        "clickhouse": False
    }
    
    # Check Redis
    try:
        redis_client = await get_redis()
        if redis_client:
            await redis_client.ping()
            checks["redis"] = True
    except:
        pass
    
    # Check Kafka (import here to avoid circular dependency)
    try:
        from .control import get_kafka_producer
        producer = await get_kafka_producer()
        if producer:
            checks["kafka"] = True
    except:
        pass
    
    # All checks must pass for readiness
    ready = all(checks.values())
    
    return {
        "ready": ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics/system")
async def system_metrics() -> Dict[str, Any]:
    """Detailed system metrics"""
    
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    cpu_count = psutil.cpu_count()
    
    # Memory metrics
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    # Disk metrics
    disk = psutil.disk_usage('/')
    
    # Network metrics
    net_io = psutil.net_io_counters()
    
    # Process metrics
    process = psutil.Process()
    
    return {
        "timestamp_ns": time.time_ns(),
        "cpu": {
            "percent": cpu_percent,
            "frequency_mhz": cpu_freq.current if cpu_freq else 0,
            "cores": cpu_count,
            "load_average": os.getloadavg()
        },
        "memory": {
            "total_gb": memory.total / (1024**3),
            "used_gb": memory.used / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent": memory.percent,
            "swap_used_gb": swap.used / (1024**3),
            "swap_percent": swap.percent
        },
        "disk": {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent": disk.percent
        },
        "network": {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "errin": net_io.errin,
            "errout": net_io.errout,
            "dropin": net_io.dropin,
            "dropout": net_io.dropout
        },
        "process": {
            "pid": process.pid,
            "threads": process.num_threads(),
            "memory_mb": process.memory_info().rss / (1024**2),
            "cpu_percent": process.cpu_percent(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }
    }


@router.get("/metrics/performance")
async def performance_metrics() -> Dict[str, Any]:
    """Application performance metrics"""
    
    # Get metrics from various components
    metrics = {
        "timestamp_ns": time.time_ns(),
        "api": {},
        "kafka": {},
        "websocket": {},
        "bridges": {}
    }
    
    # WebSocket metrics
    try:
        from .realtime import active_connections, kafka_consumers
        metrics["websocket"] = {
            "active_connections": len(active_connections),
            "kafka_consumers": len(kafka_consumers)
        }
    except:
        pass
    
    # Bridge metrics
    try:
        from .kafka_bridge import bridges
        bridge_stats = {}
        for source, bridge in bridges.items():
            bridge_stats[source] = bridge.stats
        metrics["bridges"] = bridge_stats
    except:
        pass
    
    # Redis metrics
    try:
        redis_client = await get_redis()
        if redis_client:
            info = await redis_client.info()
            metrics["redis"] = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": info.get("used_memory", 0) / (1024**2),
                "ops_per_sec": info.get("instantaneous_ops_per_sec", 0)
            }
    except:
        pass
    
    return metrics


@router.get("/metrics/latency")
async def latency_metrics() -> Dict[str, Any]:
    """Latency metrics for various operations"""
    
    latencies = {}
    
    # Measure Redis latency
    try:
        redis_client = await get_redis()
        if redis_client:
            start_ns = time.perf_counter_ns()
            await redis_client.ping()
            latencies["redis_ping_us"] = (time.perf_counter_ns() - start_ns) / 1000
    except:
        pass
    
    # Measure Kafka latency
    try:
        from .control import get_kafka_producer
        producer = await get_kafka_producer()
        if producer:
            start_ns = time.perf_counter_ns()
            # Just check if producer is connected
            latencies["kafka_check_us"] = (time.perf_counter_ns() - start_ns) / 1000
    except:
        pass
    
    return {
        "timestamp_ns": time.time_ns(),
        "latencies": latencies
    }


@router.get("/debug/connections")
async def debug_connections() -> Dict[str, Any]:
    """Debug endpoint for connection information"""
    
    connections = {
        "timestamp": datetime.utcnow().isoformat(),
        "websocket": {},
        "kafka": {},
        "redis": {}
    }
    
    # WebSocket connections
    try:
        from .realtime import active_connections
        connections["websocket"]["active"] = len(active_connections)
    except:
        pass
    
    # Kafka connections
    try:
        from .kafka_bridge import bridges
        connections["kafka"]["bridges"] = len(bridges)
    except:
        pass
    
    return connections


@router.post("/debug/gc")
async def trigger_gc() -> Dict[str, str]:
    """Trigger garbage collection (for debugging)"""
    import gc
    
    before = len(gc.get_objects())
    collected = gc.collect()
    after = len(gc.get_objects())
    
    return {
        "collected": collected,
        "objects_before": before,
        "objects_after": after,
        "reduction": before - after
    }