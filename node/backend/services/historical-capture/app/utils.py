"""
Utility functions for the historical capture service
Performance monitoring, caching, and system metrics
"""

import os
import psutil
import time
import asyncio
from typing import Dict, Any, Optional, List
from functools import wraps, lru_cache
from datetime import datetime, timedelta
import logging
import hashlib
import json
import aioredis
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            "requests_total": 0,
            "requests_failed": 0,
            "avg_response_time_ms": 0,
            "p95_response_time_ms": 0,
            "p99_response_time_ms": 0,
            "response_times": []
        }
        
    def record_request(self, duration_ms: float, success: bool = True):
        """Record request metrics"""
        self.metrics["requests_total"] += 1
        if not success:
            self.metrics["requests_failed"] += 1
            
        self.metrics["response_times"].append(duration_ms)
        
        # Keep only last 1000 response times for percentile calculation
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]
            
        # Update statistics
        self._update_stats()
        
    def _update_stats(self):
        """Update performance statistics"""
        if not self.metrics["response_times"]:
            return
            
        times = sorted(self.metrics["response_times"])
        self.metrics["avg_response_time_ms"] = sum(times) / len(times)
        
        # Calculate percentiles
        p95_idx = int(len(times) * 0.95)
        p99_idx = int(len(times) * 0.99)
        
        self.metrics["p95_response_time_ms"] = times[min(p95_idx, len(times) - 1)]
        self.metrics["p99_response_time_ms"] = times[min(p99_idx, len(times) - 1)]
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            **self.metrics,
            "uptime_seconds": time.time() - self.start_time,
            "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "storage_available_gb": psutil.disk_usage('/').free / 1024 / 1024 / 1024
        }


class RedisCache:
    """Redis-based caching for job management and RPC responses"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        
    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self.redis:
            return None
        return await self.redis.get(key)
        
    async def set(
        self,
        key: str,
        value: str,
        expire: int = 900  # 15 minutes default
    ):
        """Set value in cache with expiration"""
        if not self.redis:
            return
        await self.redis.set(key, value, ex=expire)
        
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self.redis:
            return
        await self.redis.delete(key)
        
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job metadata from cache"""
        data = await self.get(f"job:{job_id}")
        if data:
            return json.loads(data)
        return None
        
    async def set_job(self, job_id: str, job_data: Dict[str, Any]):
        """Store job metadata in cache"""
        await self.set(
            f"job:{job_id}",
            json.dumps(job_data, default=str),
            expire=3600  # 1 hour
        )
        
    async def publish_job_update(self, job_id: str, update: Dict[str, Any]):
        """Publish job update for WebSocket subscribers"""
        if not self.redis:
            return
        channel = f"job_updates:{job_id}"
        await self.redis.publish(channel, json.dumps(update, default=str))
        
    @asynccontextmanager
    async def lock(self, key: str, timeout: int = 10):
        """Distributed lock for coordination"""
        lock_key = f"lock:{key}"
        lock_value = str(time.time())
        
        try:
            # Try to acquire lock
            acquired = await self.redis.set(
                lock_key,
                lock_value,
                nx=True,
                ex=timeout
            )
            
            if not acquired:
                raise Exception(f"Failed to acquire lock for {key}")
                
            yield
            
        finally:
            # Release lock
            await self.redis.delete(lock_key)


def measure_time(func):
    """Decorator to measure async function execution time"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000
            logger.debug(f"{func.__name__} took {duration_ms:.2f}ms")
            return result
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            logger.error(f"{func.__name__} failed after {duration_ms:.2f}ms: {e}")
            raise
    return wrapper


def batch_iterator(items: List[Any], batch_size: int):
    """Yield batches from a list"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def calculate_slot_range(
    start_date: datetime,
    end_date: datetime,
    slots_per_second: float = 2.0
) -> tuple[int, int]:
    """Estimate slot range for date range"""
    duration_seconds = (end_date - start_date).total_seconds()
    total_slots = int(duration_seconds * slots_per_second)
    
    # Rough estimation - actual implementation uses RPC
    current_slot = int(time.time() * slots_per_second)
    start_slot = current_slot - int((datetime.now() - start_date).total_seconds() * slots_per_second)
    end_slot = start_slot + total_slots
    
    return start_slot, end_slot


def format_number(n: int) -> str:
    """Format large numbers with commas"""
    return f"{n:,}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


@lru_cache(maxsize=1000)
def get_program_name(program_id: str) -> str:
    """Get human-readable name for program ID"""
    known_programs = {
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium",
        "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "Orca Whirlpool",
        "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "Orca V2",
        "PHoeNiX582Ywqzb2x9B5pSkgCDPoKEpDNk2Q5UVEXx": "Phoenix",
        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "Jupiter V6",
        "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB": "Jupiter V4",
    }
    return known_programs.get(program_id, program_id[:8] + "...")


def validate_slot_range(start_slot: int, end_slot: int, max_slots: int = 10_000_000):
    """Validate slot range is reasonable"""
    if start_slot < 0:
        raise ValueError("Start slot must be non-negative")
    if end_slot < start_slot:
        raise ValueError("End slot must be >= start slot")
    if end_slot - start_slot > max_slots:
        raise ValueError(f"Slot range too large (max {max_slots} slots)")
        

class RateLimiter:
    """Token bucket rate limiter for RPC calls"""
    
    def __init__(self, rate: int = 100, per: float = 1.0):
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.updated_at = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self, tokens: int = 1):
        """Acquire tokens, waiting if necessary"""
        async with self.lock:
            while self.tokens < tokens:
                now = time.time()
                elapsed = now - self.updated_at
                
                # Add new tokens based on elapsed time
                self.tokens = min(
                    self.rate,
                    self.tokens + elapsed * (self.rate / self.per)
                )
                self.updated_at = now
                
                if self.tokens < tokens:
                    # Wait for tokens to regenerate
                    sleep_time = (tokens - self.tokens) * (self.per / self.rate)
                    await asyncio.sleep(sleep_time)
                    
            self.tokens -= tokens


def estimate_capture_time(
    total_slots: int,
    batch_size: int = 64,
    rpc_latency_ms: float = 100
) -> float:
    """Estimate time to capture data"""
    num_batches = total_slots / batch_size
    
    # Assume 10 concurrent batches
    concurrent_batches = 10
    sequential_rounds = num_batches / concurrent_batches
    
    # Time per round (RPC latency + processing)
    time_per_round = (rpc_latency_ms + 50) / 1000  # 50ms processing overhead
    
    return sequential_rounds * time_per_round


def generate_correlation_id() -> str:
    """Generate unique correlation ID for request tracing"""
    return hashlib.sha256(
        f"{time.time()}{os.getpid()}".encode()
    ).hexdigest()[:16]