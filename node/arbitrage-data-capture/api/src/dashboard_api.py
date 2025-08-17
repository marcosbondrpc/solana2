from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json
import redis.asyncio as redis
from clickhouse_driver import Client as ClickHouseClient
import structlog
import uvicorn

logger = structlog.get_logger()

app = FastAPI(
    title="Arbitrage Detection Dashboard API",
    description="Real-time arbitrage monitoring and analytics",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connections
redis_client = None
clickhouse_client = None
websocket_manager = None

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Remove from all subscriptions
        for topic in self.subscriptions:
            if websocket in self.subscriptions[topic]:
                self.subscriptions[topic].remove(websocket)
                
    async def subscribe(self, websocket: WebSocket, topic: str):
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        if websocket not in self.subscriptions[topic]:
            self.subscriptions[topic].append(websocket)
            
    async def broadcast(self, message: dict, topic: str = None):
        if topic and topic in self.subscriptions:
            connections = self.subscriptions[topic]
        else:
            connections = self.active_connections
            
        for connection in connections:
            try:
                await connection.send_json(message)
            except:
                pass
                
class ArbitrageOpportunity(BaseModel):
    id: str
    timestamp: int
    token_a: str
    token_b: str
    dex_path: List[str]
    estimated_profit_usdc: float
    net_profit_usdc: float
    roi_percentage: float
    confidence_score: float
    risk_score: float
    opportunity_type: str
    
class SystemMetrics(BaseModel):
    detection_latency_ms: float
    transactions_processed: int
    opportunities_detected: int
    success_rate: float
    total_profit_captured: float
    active_monitors: int
    
class PerformanceStats(BaseModel):
    period: str
    total_opportunities: int
    profitable_opportunities: int
    total_volume_usdc: float
    total_profit_usdc: float
    average_roi: float
    success_rate: float
    top_pairs: List[Dict[str, Any]]
    top_dex_paths: List[Dict[str, Any]]

@app.on_event("startup")
async def startup():
    global redis_client, clickhouse_client, websocket_manager
    
    redis_client = await redis.from_url(
        "redis://localhost:6379",
        encoding='utf-8',
        decode_responses=True
    )
    
    clickhouse_client = ClickHouseClient(
        host='localhost',
        port=9000,
        database='solana_arbitrage'
    )
    
    websocket_manager = WebSocketManager()
    
    # Start background tasks
    asyncio.create_task(stream_opportunities())
    asyncio.create_task(update_metrics())
    
    logger.info("Dashboard API started")

@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()
    logger.info("Dashboard API shutdown")

@app.get("/")
async def root():
    return {
        "name": "Arbitrage Detection Dashboard API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "opportunities": "/api/opportunities",
            "metrics": "/api/metrics",
            "performance": "/api/performance",
            "websocket": "/ws"
        }
    }

@app.get("/api/opportunities", response_model=List[ArbitrageOpportunity])
async def get_opportunities(
    limit: int = Query(100, ge=1, le=1000),
    min_profit: float = Query(0, ge=0),
    max_risk: float = Query(100, ge=0, le=100),
    opportunity_type: Optional[str] = None
):
    """Get recent arbitrage opportunities"""
    
    where_clauses = ["1=1"]
    
    if min_profit > 0:
        where_clauses.append(f"net_profit_usdc >= {min_profit}")
        
    if max_risk < 100:
        where_clauses.append(f"risk_score <= {max_risk}")
        
    if opportunity_type:
        where_clauses.append(f"opportunity_type = '{opportunity_type}'")
        
    query = f"""
    SELECT 
        id,
        timestamp,
        token_a,
        token_b,
        dex_path,
        estimated_profit_usdc,
        net_profit_usdc,
        roi_percentage,
        confidence_score,
        risk_score,
        opportunity_type
    FROM arbitrage_opportunities
    WHERE {' AND '.join(where_clauses)}
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    
    results = clickhouse_client.execute(query)
    
    opportunities = []
    for row in results:
        opportunities.append(ArbitrageOpportunity(
            id=row[0],
            timestamp=row[1],
            token_a=row[2],
            token_b=row[3],
            dex_path=json.loads(row[4]) if isinstance(row[4], str) else row[4],
            estimated_profit_usdc=row[5],
            net_profit_usdc=row[6],
            roi_percentage=row[7],
            confidence_score=row[8],
            risk_score=row[9],
            opportunity_type=row[10]
        ))
        
    return opportunities

@app.get("/api/metrics", response_model=SystemMetrics)
async def get_metrics():
    """Get current system metrics"""
    
    # Get metrics from Redis cache
    metrics_data = await redis_client.get("system_metrics")
    
    if metrics_data:
        metrics = json.loads(metrics_data)
    else:
        # Calculate metrics from database
        metrics = await calculate_system_metrics()
        
    return SystemMetrics(**metrics)

@app.get("/api/performance/{period}", response_model=PerformanceStats)
async def get_performance(period: str):
    """Get performance statistics for a period (1h, 24h, 7d, 30d)"""
    
    period_map = {
        '1h': 'INTERVAL 1 HOUR',
        '24h': 'INTERVAL 24 HOUR',
        '7d': 'INTERVAL 7 DAY',
        '30d': 'INTERVAL 30 DAY'
    }
    
    if period not in period_map:
        raise HTTPException(status_code=400, detail="Invalid period")
        
    interval = period_map[period]
    
    # Main statistics
    stats_query = f"""
    SELECT 
        COUNT(*) as total_opportunities,
        SUM(CASE WHEN net_profit_usdc > 0 THEN 1 ELSE 0 END) as profitable,
        SUM(estimated_profit_usdc) as total_volume,
        SUM(net_profit_usdc) as total_profit,
        AVG(roi_percentage) as avg_roi,
        AVG(CASE WHEN executed = 1 AND net_profit_usdc > 0 THEN 1 ELSE 0 END) as success_rate
    FROM arbitrage_opportunities
    WHERE timestamp > now() - {interval}
    """
    
    stats_result = clickhouse_client.execute(stats_query)[0]
    
    # Top pairs
    pairs_query = f"""
    SELECT 
        concat(token_a, '/', token_b) as pair,
        COUNT(*) as count,
        SUM(net_profit_usdc) as total_profit
    FROM arbitrage_opportunities
    WHERE timestamp > now() - {interval}
    GROUP BY pair
    ORDER BY total_profit DESC
    LIMIT 10
    """
    
    pairs_result = clickhouse_client.execute(pairs_query)
    top_pairs = [
        {'pair': row[0], 'count': row[1], 'profit': row[2]}
        for row in pairs_result
    ]
    
    # Top DEX paths
    dex_query = f"""
    SELECT 
        arrayStringConcat(dex_path, '->') as path,
        COUNT(*) as count,
        SUM(net_profit_usdc) as total_profit
    FROM arbitrage_opportunities
    WHERE timestamp > now() - {interval}
    GROUP BY path
    ORDER BY total_profit DESC
    LIMIT 10
    """
    
    dex_result = clickhouse_client.execute(dex_query)
    top_dex_paths = [
        {'path': row[0], 'count': row[1], 'profit': row[2]}
        for row in dex_result
    ]
    
    return PerformanceStats(
        period=period,
        total_opportunities=stats_result[0] or 0,
        profitable_opportunities=stats_result[1] or 0,
        total_volume_usdc=stats_result[2] or 0.0,
        total_profit_usdc=stats_result[3] or 0.0,
        average_roi=stats_result[4] or 0.0,
        success_rate=stats_result[5] or 0.0,
        top_pairs=top_pairs,
        top_dex_paths=top_dex_paths
    )

@app.get("/api/risk-distribution")
async def get_risk_distribution():
    """Get risk score distribution"""
    
    query = """
    SELECT 
        floor(risk_score / 10) * 10 as risk_bucket,
        COUNT(*) as count,
        AVG(net_profit_usdc) as avg_profit
    FROM arbitrage_opportunities
    WHERE timestamp > now() - INTERVAL 24 HOUR
    GROUP BY risk_bucket
    ORDER BY risk_bucket
    """
    
    results = clickhouse_client.execute(query)
    
    distribution = [
        {
            'range': f"{row[0]}-{row[0]+10}",
            'count': row[1],
            'avg_profit': row[2]
        }
        for row in results
    ]
    
    return distribution

@app.get("/api/opportunity-types")
async def get_opportunity_types():
    """Get breakdown by opportunity type"""
    
    query = """
    SELECT 
        opportunity_type,
        COUNT(*) as count,
        AVG(net_profit_usdc) as avg_profit,
        AVG(roi_percentage) as avg_roi,
        AVG(risk_score) as avg_risk
    FROM arbitrage_opportunities
    WHERE timestamp > now() - INTERVAL 24 HOUR
    GROUP BY opportunity_type
    ORDER BY count DESC
    """
    
    results = clickhouse_client.execute(query)
    
    types = [
        {
            'type': row[0],
            'count': row[1],
            'avg_profit': row[2],
            'avg_roi': row[3],
            'avg_risk': row[4]
        }
        for row in results
    ]
    
    return types

@app.get("/api/hourly-stats")
async def get_hourly_stats(hours: int = Query(24, ge=1, le=168)):
    """Get hourly statistics"""
    
    query = f"""
    SELECT 
        toStartOfHour(timestamp) as hour,
        COUNT(*) as opportunities,
        SUM(net_profit_usdc) as total_profit,
        AVG(roi_percentage) as avg_roi
    FROM arbitrage_opportunities
    WHERE timestamp > now() - INTERVAL {hours} HOUR
    GROUP BY hour
    ORDER BY hour DESC
    """
    
    results = clickhouse_client.execute(query)
    
    stats = [
        {
            'hour': row[0].isoformat(),
            'opportunities': row[1],
            'total_profit': row[2],
            'avg_roi': row[3]
        }
        for row in results
    ]
    
    return stats

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            if data.get('action') == 'subscribe':
                topic = data.get('topic', 'opportunities')
                await websocket_manager.subscribe(websocket, topic)
                await websocket.send_json({
                    'type': 'subscription',
                    'topic': topic,
                    'status': 'subscribed'
                })
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

async def stream_opportunities():
    """Stream opportunities to WebSocket clients"""
    
    pubsub = redis_client.pubsub()
    await pubsub.subscribe('arbitrage_opportunities')
    
    async for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                opportunity = json.loads(message['data'])
                await websocket_manager.broadcast(
                    {
                        'type': 'opportunity',
                        'data': opportunity
                    },
                    topic='opportunities'
                )
            except Exception as e:
                logger.error(f"Error streaming opportunity: {e}")

async def update_metrics():
    """Periodically update system metrics"""
    
    while True:
        try:
            metrics = await calculate_system_metrics()
            
            # Store in Redis
            await redis_client.setex(
                "system_metrics",
                60,  # 60 second TTL
                json.dumps(metrics)
            )
            
            # Broadcast to WebSocket clients
            await websocket_manager.broadcast(
                {
                    'type': 'metrics',
                    'data': metrics
                },
                topic='metrics'
            )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            
        await asyncio.sleep(10)  # Update every 10 seconds

async def calculate_system_metrics() -> Dict[str, Any]:
    """Calculate current system metrics"""
    
    # Get latest metrics from ClickHouse
    query = """
    SELECT 
        AVG(execution_latency_ms) as avg_latency,
        COUNT(*) as tx_processed,
        SUM(CASE WHEN is_arbitrage = 1 THEN 1 ELSE 0 END) as opportunities,
        AVG(CASE WHEN executed = 1 AND net_profit_usdc > 0 THEN 1 ELSE 0 END) as success_rate,
        SUM(net_profit_usdc) as total_profit
    FROM arbitrage_opportunities
    WHERE timestamp > now() - INTERVAL 1 HOUR
    """
    
    result = clickhouse_client.execute(query)[0]
    
    # Get active monitors count from Redis
    active_monitors = await redis_client.get("active_monitors") or 6
    
    return {
        'detection_latency_ms': result[0] or 0.0,
        'transactions_processed': result[1] or 0,
        'opportunities_detected': result[2] or 0,
        'success_rate': result[3] or 0.0,
        'total_profit_captured': result[4] or 0.0,
        'active_monitors': int(active_monitors)
    }

@app.get("/api/labels/recent")
async def get_recent_labels(limit: int = Query(100, ge=1, le=1000)):
    """Get recently labeled transactions"""
    
    query = f"""
    SELECT 
        transaction_id,
        timestamp,
        is_arbitrage,
        confidence_score,
        arbitrage_type,
        profit_class,
        human_verified
    FROM arbitrage_labels
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    
    results = clickhouse_client.execute(query)
    
    labels = [
        {
            'transaction_id': row[0],
            'timestamp': row[1],
            'is_arbitrage': row[2],
            'confidence_score': row[3],
            'arbitrage_type': row[4],
            'profit_class': row[5],
            'human_verified': row[6]
        }
        for row in results
    ]
    
    return labels

@app.get("/api/model-performance")
async def get_model_performance():
    """Get ML model performance metrics"""
    
    # Get from Redis cache
    perf_data = await redis_client.get("model_performance")
    
    if perf_data:
        return json.loads(perf_data)
        
    # Default response
    return {
        'accuracy': 0.95,
        'precision': 0.93,
        'recall': 0.91,
        'f1_score': 0.92,
        'auc': 0.97,
        'last_updated': datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    try:
        # Check Redis
        await redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
        
    try:
        # Check ClickHouse
        clickhouse_client.execute("SELECT 1")
        clickhouse_status = "healthy"
    except:
        clickhouse_status = "unhealthy"
        
    return {
        'status': 'healthy' if redis_status == 'healthy' and clickhouse_status == 'healthy' else 'degraded',
        'services': {
            'redis': redis_status,
            'clickhouse': clickhouse_status
        },
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
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
                    "stream": "ext://sys.stderr",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )