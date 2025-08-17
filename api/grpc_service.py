"""
Ultra-Performance gRPC Service for MEV Detection
Target: <10ms response time, 100k+ RPS
DEFENSIVE-ONLY: Read-only monitoring APIs
"""

import grpc
from concurrent import futures
import asyncio
import time
import numpy as np
from typing import List, Dict, Optional, AsyncIterator
import redis.asyncio as redis
from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer
import msgpack
import lz4.frame
import xxhash
from dataclasses import dataclass
import uvloop
from grpc_reflection.v1alpha import reflection
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import logging
from functools import wraps
import clickhouse_connect
from clickhouse_connect.driver.asyncclient import AsyncClient
import pyarrow.flight as flight
import pyarrow as pa

# Import protobuf definitions (would be generated)
# import mev_detection_pb2
# import mev_detection_pb2_grpc

# Set up ultra-performance event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
grpc_latency = Histogram('grpc_request_latency_ms', 'gRPC request latency',
                         ['method'], buckets=[1, 5, 10, 25, 50, 100, 250, 500])
grpc_requests = Counter('grpc_requests_total', 'Total gRPC requests', ['method', 'status'])
cache_hits = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
cache_misses = Counter('cache_misses_total', 'Cache misses', ['cache_type'])
active_connections = Gauge('grpc_active_connections', 'Active gRPC connections')

# Cache configuration
CACHE_TTL = 60  # seconds
CACHE_SIZE = 10000

class CircuitBreaker:
    """Circuit breaker for downstream services"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
            
            raise e

class ConnectionPool:
    """Ultra-fast connection pooling"""
    
    def __init__(self, factory, max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool = asyncio.Queue(maxsize=max_size)
        self.size = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        try:
            conn = self.pool.get_nowait()
        except asyncio.QueueEmpty:
            async with self._lock:
                if self.size < self.max_size:
                    conn = await self.factory()
                    self.size += 1
                else:
                    conn = await self.pool.get()
        
        return conn
    
    async def release(self, conn):
        await self.pool.put(conn)

class MEVDetectionService:  # (mev_detection_pb2_grpc.MEVDetectionServicer):
    """High-performance gRPC service implementation"""
    
    def __init__(self):
        # Initialize caches
        self.local_cache = Cache(Cache.MEMORY, serializer=PickleSerializer())
        self.redis_client = None
        self.clickhouse_client = None
        
        # Connection pools
        self.redis_pool = None
        self.clickhouse_pool = None
        
        # Circuit breakers
        self.redis_breaker = CircuitBreaker()
        self.clickhouse_breaker = CircuitBreaker()
        
        # Request coalescing
        self.pending_requests = {}
        
        # Initialize connections
        asyncio.create_task(self._init_connections())
    
    async def _init_connections(self):
        """Initialize connection pools"""
        # Redis connection pool
        self.redis_pool = redis.ConnectionPool(
            host='localhost',
            port=6379,
            db=0,
            max_connections=50,
            decode_responses=False
        )
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        
        # ClickHouse async client
        self.clickhouse_client = await clickhouse_connect.get_async_client(
            host='localhost',
            port=8123,
            database='mev_detection',
            compress=True,
            pool_size=20
        )
        
        logger.info("Initialized connection pools")
    
    @cached(ttl=CACHE_TTL, cache=Cache.MEMORY, key_builder=lambda *args, **kwargs: str(args))
    async def GetSandwichDetections(self, request, context):
        """Get recent sandwich attack detections"""
        start_time = time.perf_counter()
        
        try:
            # Check Redis cache first
            cache_key = f"sandwich:{request.start_slot}:{request.end_slot}"
            cached_result = await self._get_redis_cache(cache_key)
            
            if cached_result:
                cache_hits.labels(cache_type='redis').inc()
                return self._deserialize_response(cached_result)
            
            cache_misses.labels(cache_type='redis').inc()
            
            # Query ClickHouse with circuit breaker
            query = """
                SELECT
                    detection_id,
                    slot,
                    frontrun_tx,
                    victim_tx,
                    backrun_tx,
                    confidence_score,
                    victim_loss_lamports,
                    attacker_profit_lamports,
                    detection_latency_us
                FROM sandwich_detections
                WHERE slot >= {start_slot:UInt64}
                  AND slot <= {end_slot:UInt64}
                  AND confidence_score >= {min_confidence:Float32}
                ORDER BY slot DESC
                LIMIT {limit:UInt32}
                SETTINGS
                    max_threads = 8,
                    max_execution_time = 5
            """
            
            params = {
                'start_slot': request.start_slot,
                'end_slot': request.end_slot,
                'min_confidence': request.min_confidence or 0.8,
                'limit': request.limit or 100
            }
            
            result = await self.clickhouse_breaker.call(
                self.clickhouse_client.query,
                query,
                parameters=params
            )
            
            # Build response
            detections = []
            for row in result.result_rows:
                detection = {
                    'detection_id': str(row[0]),
                    'slot': row[1],
                    'frontrun_tx': row[2].hex(),
                    'victim_tx': row[3].hex(),
                    'backrun_tx': row[4].hex(),
                    'confidence': row[5],
                    'victim_loss': row[6],
                    'attacker_profit': row[7],
                    'latency_us': row[8]
                }
                detections.append(detection)
            
            # Cache result
            await self._set_redis_cache(cache_key, detections, ttl=60)
            
            # Record metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            grpc_latency.labels(method='GetSandwichDetections').observe(latency_ms)
            grpc_requests.labels(method='GetSandwichDetections', status='success').inc()
            
            return {'detections': detections}
            
        except Exception as e:
            logger.error(f"Error in GetSandwichDetections: {e}")
            grpc_requests.labels(method='GetSandwichDetections', status='error').inc()
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def StreamDetections(self, request, context):
        """Stream real-time detections"""
        active_connections.inc()
        
        try:
            # Subscribe to Redis pub/sub
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe('mev_detections')
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    detection = msgpack.unpackb(message['data'], raw=False)
                    
                    # Filter based on request criteria
                    if self._matches_filter(detection, request):
                        yield detection
                    
                    # Check if client is still connected
                    if context.is_active() is False:
                        break
        
        finally:
            active_connections.dec()
            await pubsub.unsubscribe('mev_detections')
    
    async def GetEntityBehavior(self, request, context):
        """Get entity behavioral analysis"""
        # Implement request coalescing for duplicate queries
        request_key = f"entity:{request.entity_address}:{request.date_range}"
        
        if request_key in self.pending_requests:
            # Wait for existing request
            return await self.pending_requests[request_key]
        
        # Create future for coalescing
        future = asyncio.Future()
        self.pending_requests[request_key] = future
        
        try:
            # Execute query
            result = await self._get_entity_behavior(request)
            future.set_result(result)
            return result
        
        except Exception as e:
            future.set_exception(e)
            raise
        
        finally:
            del self.pending_requests[request_key]
    
    async def GetMetrics(self, request, context):
        """Get system metrics"""
        query = """
            SELECT
                toStartOfMinute(now()) as minute,
                sum(rows_ingested) as rows_per_minute,
                sum(bytes_ingested) / 1024 / 1024 as mb_ingested,
                quantile(0.5)(p50_latency_us) / 1000 as p50_ms,
                quantile(0.95)(p95_latency_us) / 1000 as p95_ms,
                quantile(0.99)(p99_latency_us) / 1000 as p99_ms,
                sum(sandwiches_detected) as sandwiches,
                sum(arbitrage_detected) as arbitrage,
                sum(total_mev_extracted_lamports) / 1e9 as mev_extracted_sol
            FROM detection_metrics_5s
            WHERE timestamp >= now() - INTERVAL {window:UInt32} MINUTE
            GROUP BY minute
            ORDER BY minute DESC
            LIMIT 60
        """
        
        params = {'window': request.window_minutes or 60}
        
        result = await self.clickhouse_client.query(query, parameters=params)
        
        metrics = []
        for row in result.result_rows:
            metric = {
                'timestamp': row[0].isoformat(),
                'rows_per_minute': row[1],
                'mb_ingested': row[2],
                'p50_ms': row[3],
                'p95_ms': row[4],
                'p99_ms': row[5],
                'sandwiches': row[6],
                'arbitrage': row[7],
                'mev_sol': row[8]
            }
            metrics.append(metric)
        
        return {'metrics': metrics}
    
    async def _get_redis_cache(self, key: str) -> Optional[bytes]:
        """Get from Redis cache"""
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            logger.warning(f"Redis cache get failed: {e}")
            return None
    
    async def _set_redis_cache(self, key: str, value: any, ttl: int = 60):
        """Set Redis cache with TTL"""
        try:
            serialized = msgpack.packb(value, use_bin_type=True)
            compressed = lz4.frame.compress(serialized)
            await self.redis_client.setex(key, ttl, compressed)
        except Exception as e:
            logger.warning(f"Redis cache set failed: {e}")
    
    def _deserialize_response(self, data: bytes) -> dict:
        """Deserialize cached response"""
        decompressed = lz4.frame.decompress(data)
        return msgpack.unpackb(decompressed, raw=False)
    
    def _matches_filter(self, detection: dict, request) -> bool:
        """Check if detection matches request filter"""
        if request.min_confidence and detection.get('confidence', 0) < request.min_confidence:
            return False
        
        if request.detection_types and detection.get('type') not in request.detection_types:
            return False
        
        return True
    
    async def _get_entity_behavior(self, request) -> dict:
        """Get entity behavioral metrics"""
        query = """
            SELECT
                entity_address,
                sum(transaction_count) as total_transactions,
                uniqMerge(unique_programs) as unique_programs,
                sum(sandwich_attempts) as sandwich_attempts,
                sum(sandwich_successes) as sandwich_successes,
                sum(total_volume_lamports) / 1e9 as volume_sol,
                sum(total_profit_lamports) / 1e9 as profit_sol,
                avg(risk_score) as avg_risk_score,
                uniqMerge(unique_counterparties) as unique_counterparties
            FROM entity_behaviors_agg
            WHERE entity_address = {address:String}
              AND date >= today() - INTERVAL {days:UInt32} DAY
            GROUP BY entity_address
        """
        
        params = {
            'address': request.entity_address,
            'days': request.days or 30
        }
        
        result = await self.clickhouse_client.query(query, parameters=params)
        
        if result.result_rows:
            row = result.result_rows[0]
            return {
                'address': row[0],
                'total_transactions': row[1],
                'unique_programs': row[2],
                'sandwich_attempts': row[3],
                'sandwich_successes': row[4],
                'volume_sol': row[5],
                'profit_sol': row[6],
                'risk_score': row[7],
                'unique_counterparties': row[8]
            }
        
        return {}

class ArrowFlightService(flight.FlightServerBase):
    """Apache Arrow Flight service for ultra-fast data transfer"""
    
    def __init__(self, location, clickhouse_client):
        super().__init__(location)
        self.clickhouse_client = clickhouse_client
    
    def do_get(self, context, ticket):
        """Stream data using Arrow format"""
        query = ticket.ticket.decode()
        
        # Execute ClickHouse query
        result = self.clickhouse_client.query_arrow(query)
        
        # Stream Arrow batches
        table = pa.Table.from_batches([result])
        return flight.RecordBatchStream(table)
    
    def list_flights(self, context, criteria):
        """List available data streams"""
        flights = [
            flight.FlightInfo(
                pa.schema([
                    ('slot', pa.int64()),
                    ('signature', pa.binary()),
                    ('timestamp', pa.timestamp('ns'))
                ]),
                flight.FlightDescriptor.for_path(['shred_stream']),
                [],
                -1, -1
            )
        ]
        
        for f in flights:
            yield f

async def serve():
    """Start gRPC server with optimizations"""
    # Create server with performance options
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=100),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ('grpc.max_concurrent_streams', 1000),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_ping_strikes', 0),
        ],
        compression=grpc.Compression.Gzip
    )
    
    # Add service
    service = MEVDetectionService()
    # mev_detection_pb2_grpc.add_MEVDetectionServicer_to_server(service, server)
    
    # Enable reflection for debugging
    # SERVICE_NAMES = (
    #     mev_detection_pb2.DESCRIPTOR.services_by_name['MEVDetection'].full_name,
    #     reflection.SERVICE_NAME,
    # )
    # reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    # Add insecure port (use TLS in production)
    server.add_insecure_port('[::]:50051')
    
    # Start Arrow Flight server on separate port
    # flight_location = flight.Location.for_grpc_tcp("localhost", 50052)
    # flight_server = ArrowFlightService(flight_location, service.clickhouse_client)
    # flight_server.serve()
    
    logger.info("Starting gRPC server on port 50051")
    await server.start()
    
    # Start Prometheus metrics server
    prometheus_client.start_http_server(8001)
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(5)

if __name__ == '__main__':
    asyncio.run(serve())