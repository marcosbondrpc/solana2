"""
Elite FastAPI Service for Arbitrage Data
High-performance REST/WebSocket/GraphQL API with caching
"""

from fastapi import FastAPI, HTTPException, WebSocket, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from strawberry.fastapi import GraphQLRouter
import strawberry
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import asyncio
import aioredis
import ujson as json
import msgpack
import pandas as pd
import io
import logging
from contextlib import asynccontextmanager

from writer.clickhouse_writer import ClickHouseWriter, WriterConfig
from models.enhanced_models import Transaction, ArbitrageOpportunity, RiskMetrics, PerformanceMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class TransactionQuery(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    searcher_address: Optional[str] = None
    min_profit: Optional[float] = None
    mev_type: Optional[str] = None
    limit: int = Field(default=100, le=10000)
    offset: int = Field(default=0, ge=0)

class OpportunityQuery(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    opportunity_type: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0, le=100)
    executed_only: bool = False
    limit: int = Field(default=100, le=10000)

class AggregationQuery(BaseModel):
    interval: str = Field(default="1h", pattern="^(1m|5m|15m|1h|4h|1d|1w)$")
    start_time: datetime
    end_time: datetime
    metric: str = Field(default="profit", pattern="^(profit|volume|gas|roi|success_rate)$")
    group_by: Optional[str] = Field(None, pattern="^(searcher|mev_type|dex|hour|day)$")

class ExportRequest(BaseModel):
    table: str = Field(..., pattern="^(transactions|opportunities|snapshots|metrics)$")
    format: str = Field(default="parquet", pattern="^(csv|parquet|json|arrow)$")
    start_time: datetime
    end_time: datetime
    columns: Optional[List[str]] = None
    compression: Optional[str] = Field(None, pattern="^(gzip|lz4|snappy|zstd)$")
    chunk_size: int = Field(default=100000, le=1000000)

# GraphQL schema
@strawberry.type
class TransactionGQL:
    signature: str
    block_timestamp: str
    net_profit: float
    roi_percentage: float
    searcher_address: str
    mev_type: str

@strawberry.type
class Query:
    @strawberry.field
    async def transactions(
        self,
        limit: int = 100,
        min_profit: Optional[float] = None
    ) -> List[TransactionGQL]:
        # Implementation would query ClickHouse
        return []

# FastAPI app with lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.redis = await aioredis.create_redis_pool(
        'redis://localhost:6379',
        encoding='utf-8',
        minsize=5,
        maxsize=20
    )
    
    writer_config = WriterConfig()
    app.state.writer = ClickHouseWriter(writer_config)
    await app.state.writer.initialize()
    
    logger.info("API service started")
    
    yield
    
    # Shutdown
    app.state.redis.close()
    await app.state.redis.wait_closed()
    await app.state.writer.close()
    
    logger.info("API service stopped")

# Create FastAPI app
app = FastAPI(
    title="Elite Arbitrage Data API",
    description="High-performance API for arbitrage data access and ML training",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# GraphQL router
graphql_app = GraphQLRouter(
    schema=strawberry.Schema(query=Query),
)
app.include_router(graphql_app, prefix="/graphql")

# Cache decorator
def cache_key_wrapper(prefix: str, ttl: int = 60):
    def cache_decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{prefix}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached = await app.state.redis.get(cache_key)
            if cached:
                return msgpack.unpackb(cached, raw=False)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await app.state.redis.setex(
                cache_key,
                ttl,
                msgpack.packb(result, use_bin_type=True)
            )
            
            return result
        return wrapper
    return cache_decorator

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/stats")
@cache_key_wrapper("stats", ttl=10)
async def get_stats():
    """Get system statistics"""
    writer_stats = await app.state.writer.get_stats()
    
    # Get ClickHouse stats
    query = """
    SELECT
        count() as total_transactions,
        sum(net_profit) as total_profit,
        avg(roi_percentage) as avg_roi,
        max(net_profit) as max_profit,
        uniq(searcher_address) as unique_searchers
    FROM transactions
    WHERE block_timestamp > now() - INTERVAL 1 HOUR
    """
    
    ch_stats = await app.state.writer.execute_query(query)
    
    return {
        "writer": writer_stats,
        "database": ch_stats[0] if ch_stats else {},
        "cache_size": await app.state.redis.dbsize(),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/transactions/query")
async def query_transactions(
    query: TransactionQuery,
    _: str = Depends(RateLimiter(times=100, seconds=60))
):
    """Query transactions with filters"""
    
    # Build SQL query
    conditions = []
    params = {}
    
    if query.start_time:
        conditions.append("block_timestamp >= %(start_time)s")
        params['start_time'] = query.start_time
    
    if query.end_time:
        conditions.append("block_timestamp <= %(end_time)s")
        params['end_time'] = query.end_time
    
    if query.searcher_address:
        conditions.append("searcher_address = %(searcher)s")
        params['searcher'] = query.searcher_address
    
    if query.min_profit is not None:
        conditions.append("net_profit >= %(min_profit)s")
        params['min_profit'] = query.min_profit
    
    if query.mev_type:
        conditions.append("mev_type = %(mev_type)s")
        params['mev_type'] = query.mev_type
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    sql = f"""
    SELECT *
    FROM transactions
    WHERE {where_clause}
    ORDER BY block_timestamp DESC
    LIMIT {query.limit}
    OFFSET {query.offset}
    """
    
    results = await app.state.writer.execute_query(sql)
    
    return {
        "data": results,
        "count": len(results),
        "query": query.dict()
    }

@app.get("/transactions/stream")
async def stream_transactions(
    searcher: Optional[str] = None,
    min_profit: Optional[float] = None
):
    """Stream real-time transactions via Server-Sent Events"""
    
    async def event_generator():
        last_check = datetime.utcnow()
        
        while True:
            # Query new transactions
            sql = f"""
            SELECT *
            FROM transactions
            WHERE block_timestamp > %(last_check)s
            {"AND searcher_address = %(searcher)s" if searcher else ""}
            {"AND net_profit >= %(min_profit)s" if min_profit else ""}
            ORDER BY block_timestamp
            LIMIT 100
            """
            
            params = {'last_check': last_check}
            if searcher:
                params['searcher'] = searcher
            if min_profit:
                params['min_profit'] = min_profit
            
            results = await app.state.writer.execute_query(sql)
            
            for tx in results:
                yield f"data: {json.dumps(tx)}\n\n"
            
            last_check = datetime.utcnow()
            await asyncio.sleep(1)  # Poll every second
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.websocket("/ws/transactions")
async def websocket_transactions(websocket: WebSocket):
    """WebSocket endpoint for real-time transactions"""
    await websocket.accept()
    
    try:
        # Subscribe to Redis pub/sub
        channel = (await app.state.redis.subscribe('transactions:new'))[0]
        
        while True:
            # Wait for new message
            message = await channel.get()
            if message:
                data = msgpack.unpackb(message, raw=False)
                await websocket.send_json(data)
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.post("/opportunities/query")
@cache_key_wrapper("opportunities", ttl=30)
async def query_opportunities(query: OpportunityQuery):
    """Query arbitrage opportunities"""
    
    conditions = []
    
    if query.start_time:
        conditions.append(f"detected_at >= '{query.start_time}'")
    
    if query.end_time:
        conditions.append(f"detected_at <= '{query.end_time}'")
    
    if query.opportunity_type:
        conditions.append(f"opportunity_type = '{query.opportunity_type}'")
    
    if query.min_confidence is not None:
        conditions.append(f"confidence_score >= {query.min_confidence}")
    
    if query.executed_only:
        conditions.append("executed = 1")
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    sql = f"""
    SELECT *
    FROM arbitrage_opportunities
    WHERE {where_clause}
    ORDER BY minimum_profit DESC
    LIMIT {query.limit}
    """
    
    results = await app.state.writer.execute_query(sql)
    
    return {
        "opportunities": results,
        "count": len(results)
    }

@app.post("/aggregations/query")
async def query_aggregations(query: AggregationQuery):
    """Query aggregated metrics"""
    
    # Time bucket based on interval
    time_bucket = {
        "1m": "toStartOfMinute(block_timestamp)",
        "5m": "toStartOfFiveMinute(block_timestamp)",
        "15m": "toStartOfFifteenMinutes(block_timestamp)",
        "1h": "toStartOfHour(block_timestamp)",
        "4h": "toStartOfInterval(block_timestamp, INTERVAL 4 HOUR)",
        "1d": "toStartOfDay(block_timestamp)",
        "1w": "toStartOfWeek(block_timestamp)"
    }[query.interval]
    
    # Metric aggregation
    metric_agg = {
        "profit": "sum(net_profit) as value",
        "volume": "sum(amounts[1]) as value",
        "gas": "sum(gas_cost) as value",
        "roi": "avg(roi_percentage) as value",
        "success_rate": "countIf(status = 'success') / count() as value"
    }[query.metric]
    
    # Group by clause
    group_clause = ""
    if query.group_by:
        group_map = {
            "searcher": "searcher_address",
            "mev_type": "mev_type",
            "dex": "arrayJoin(dexes) as dex",
            "hour": "toHour(block_timestamp)",
            "day": "toDayOfWeek(block_timestamp)"
        }
        group_clause = f", {group_map[query.group_by]}"
    
    sql = f"""
    SELECT
        {time_bucket} as time,
        {metric_agg},
        count() as count
        {group_clause}
    FROM transactions
    WHERE block_timestamp BETWEEN %(start_time)s AND %(end_time)s
    GROUP BY time {", " + query.group_by if query.group_by else ""}
    ORDER BY time
    """
    
    results = await app.state.writer.execute_query(sql)
    
    return {
        "data": results,
        "interval": query.interval,
        "metric": query.metric
    }

@app.post("/export/data")
async def export_data(
    request: ExportRequest,
    background_tasks: BackgroundTasks
):
    """Export data in various formats"""
    
    # Build query
    table_map = {
        "transactions": "transactions",
        "opportunities": "arbitrage_opportunities",
        "snapshots": "market_snapshots",
        "metrics": "performance_metrics"
    }
    
    table = table_map[request.table]
    
    columns = "*" if not request.columns else ", ".join(request.columns)
    
    sql = f"""
    SELECT {columns}
    FROM {table}
    WHERE block_timestamp BETWEEN '{request.start_time}' AND '{request.end_time}'
    """
    
    # Generate export ID
    export_id = f"export_{datetime.utcnow().timestamp()}"
    
    # Start export in background
    background_tasks.add_task(
        process_export,
        export_id,
        sql,
        request.format,
        request.compression,
        request.chunk_size
    )
    
    return {
        "export_id": export_id,
        "status": "processing",
        "estimated_time_seconds": 30
    }

async def process_export(
    export_id: str,
    sql: str,
    format: str,
    compression: Optional[str],
    chunk_size: int
):
    """Process data export in background"""
    
    try:
        # Execute query
        data = await app.state.writer.execute_query(sql)
        
        df = pd.DataFrame(data)
        
        # Export based on format
        output_path = f"/tmp/{export_id}.{format}"
        
        if format == "csv":
            df.to_csv(output_path, index=False, compression=compression)
        elif format == "parquet":
            df.to_parquet(output_path, compression=compression or 'snappy')
        elif format == "json":
            df.to_json(output_path, orient='records', compression=compression)
        elif format == "arrow":
            import pyarrow as pa
            import pyarrow.feather as feather
            table = pa.Table.from_pandas(df)
            feather.write_feather(table, output_path, compression=compression or 'lz4')
        
        # Update export status in Redis
        await app.state.redis.setex(
            f"export:{export_id}",
            3600,  # 1 hour TTL
            json.dumps({
                "status": "completed",
                "path": output_path,
                "size": os.path.getsize(output_path),
                "rows": len(df)
            })
        )
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        await app.state.redis.setex(
            f"export:{export_id}",
            3600,
            json.dumps({"status": "failed", "error": str(e)})
        )

@app.get("/export/{export_id}/status")
async def get_export_status(export_id: str):
    """Check export status"""
    
    status = await app.state.redis.get(f"export:{export_id}")
    
    if not status:
        raise HTTPException(status_code=404, detail="Export not found")
    
    return json.loads(status)

@app.get("/export/{export_id}/download")
async def download_export(export_id: str):
    """Download exported file"""
    
    status = await app.state.redis.get(f"export:{export_id}")
    
    if not status:
        raise HTTPException(status_code=404, detail="Export not found")
    
    status_data = json.loads(status)
    
    if status_data['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Export not ready")
    
    return FileResponse(
        status_data['path'],
        media_type='application/octet-stream',
        filename=os.path.basename(status_data['path'])
    )

@app.get("/ml/features")
async def get_ml_features(
    start_time: datetime,
    end_time: datetime,
    feature_set: str = Query(default="full", pattern="^(basic|advanced|full)$")
):
    """Get ML-ready features"""
    
    # Define feature sets
    feature_columns = {
        "basic": [
            "net_profit", "roi_percentage", "hop_count",
            "slippage_percentage", "execution_time_ms"
        ],
        "advanced": [
            "net_profit", "roi_percentage", "hop_count",
            "slippage_percentage", "execution_time_ms",
            "market_volatility", "liquidity_score",
            "price_momentum", "volume_ratio"
        ],
        "full": "*"
    }
    
    columns = feature_columns[feature_set]
    columns_str = "*" if columns == "*" else ", ".join(columns)
    
    sql = f"""
    SELECT {columns_str}
    FROM transactions
    WHERE block_timestamp BETWEEN '{start_time}' AND '{end_time}'
        AND status = 'success'
    """
    
    data = await app.state.writer.execute_query(sql)
    
    return {
        "features": data,
        "count": len(data),
        "feature_set": feature_set
    }

@app.post("/ml/train-test-split")
async def create_train_test_split(
    start_time: datetime,
    end_time: datetime,
    test_size: float = 0.2,
    random_seed: int = 42
):
    """Create train/test split for ML"""
    
    sql = f"""
    SELECT *
    FROM transactions
    WHERE block_timestamp BETWEEN '{start_time}' AND '{end_time}'
        AND status = 'success'
    ORDER BY rand({random_seed})
    """
    
    data = await app.state.writer.execute_query(sql)
    
    split_index = int(len(data) * (1 - test_size))
    
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    # Save split info
    split_id = f"split_{datetime.utcnow().timestamp()}"
    
    await app.state.redis.setex(
        f"ml:split:{split_id}",
        86400,  # 24 hour TTL
        json.dumps({
            "train_size": len(train_data),
            "test_size": len(test_data),
            "test_ratio": test_size,
            "seed": random_seed
        })
    )
    
    return {
        "split_id": split_id,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "train_data": train_data[:100],  # Return sample
        "test_data": test_data[:100]  # Return sample
    }

# Performance monitoring endpoint
@app.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus-compatible metrics endpoint"""
    
    metrics = []
    
    # Get various metrics
    stats = await app.state.writer.get_stats()
    
    # Format as Prometheus metrics
    metrics.append(f"# HELP arbitrage_api_writes_total Total writes to database")
    metrics.append(f"# TYPE arbitrage_api_writes_total counter")
    metrics.append(f"arbitrage_api_writes_total {stats['total_written']}")
    
    metrics.append(f"# HELP arbitrage_api_write_errors_total Total write errors")
    metrics.append(f"# TYPE arbitrage_api_write_errors_total counter")
    metrics.append(f"arbitrage_api_write_errors_total {stats['failed_writes']}")
    
    metrics.append(f"# HELP arbitrage_api_buffer_size Current buffer size")
    metrics.append(f"# TYPE arbitrage_api_buffer_size gauge")
    metrics.append(f"arbitrage_api_buffer_size {stats.get('total_buffer_size', 0)}")
    
    return "\n".join(metrics)

if __name__ == "__main__":
    import uvicorn
    import os
    
    uvicorn.run(
        "fastapi_service:app",
        host="0.0.0.0",
        port=8080,
        workers=os.cpu_count(),
        loop="uvloop",
        log_level="info",
        access_log=True
    )