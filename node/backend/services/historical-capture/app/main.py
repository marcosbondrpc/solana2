"""
FastAPI Historical Capture Service
High-performance backend for Solana blockchain data capture and MEV detection
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
import uuid
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .models import (
    CaptureRequest, ArbitrageConvertRequest, SandwichConvertRequest,
    JobMetadata, JobStatus, DatasetStats, HealthResponse
)
from .rpc import SolanaRPCClient
from .storage import ParquetStorage
from .capture import CaptureEngine
from .detection import MEVDetector
from .utils import PerformanceMonitor, RedisCache, measure_time, generate_correlation_id


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
rpc_client: Optional[SolanaRPCClient] = None
storage: Optional[ParquetStorage] = None
cache: Optional[RedisCache] = None
monitor: Optional[PerformanceMonitor] = None
active_jobs: Dict[str, JobMetadata] = {}
capture_tasks: Dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global rpc_client, storage, cache, monitor
    
    logger.info("Starting Historical Capture Service")
    
    # Initialize components
    rpc_client = SolanaRPCClient(
        endpoint="https://api.mainnet-beta.solana.com",
        max_connections=100
    )
    await rpc_client.connect()
    
    storage = ParquetStorage(base_path="./data")
    
    cache = RedisCache()
    try:
        await cache.connect()
    except Exception as e:
        logger.warning(f"Redis connection failed, running without cache: {e}")
        cache = None
        
    monitor = PerformanceMonitor()
    
    logger.info("Service initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Historical Capture Service")
    
    # Cancel all running jobs
    for task_id, task in capture_tasks.items():
        if not task.done():
            task.cancel()
            
    if rpc_client:
        await rpc_client.close()
    if cache:
        await cache.close()
        
    logger.info("Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Historical Capture Service",
    description="High-performance Solana blockchain data capture and MEV detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_correlation_id(request, call_next):
    """Add correlation ID to all requests"""
    correlation_id = generate_correlation_id()
    request.state.correlation_id = correlation_id
    
    start_time = datetime.utcnow()
    response = await call_next(request)
    
    # Add headers
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Process-Time"] = str(
        (datetime.utcnow() - start_time).total_seconds() * 1000
    )
    
    # Record metrics
    if monitor:
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        monitor.record_request(duration_ms, response.status_code < 400)
        
    return response


@app.post("/capture/start", response_model=JobMetadata)
@measure_time
async def start_capture(
    request: CaptureRequest,
    background_tasks: BackgroundTasks
) -> JobMetadata:
    """
    Start a new data capture job
    Returns job_id for tracking progress
    """
    logger.info(f"Starting capture job: {request.dict()}")
    
    # Create job metadata
    job = JobMetadata(
        config=request.dict(),
        status=JobStatus.RUNNING,
        started_at=datetime.utcnow()
    )
    
    # Store job
    active_jobs[job.job_id] = job
    if cache:
        await cache.set_job(job.job_id, job.dict())
        
    # Start capture in background
    async def run_capture():
        try:
            engine = CaptureEngine(
                rpc_client,
                storage,
                progress_callback=lambda j: update_job_progress(j)
            )
            
            stats = await engine.capture(request, job)
            
            # Update job on completion
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = stats
            
            # Write manifest
            await storage.write_manifest(job.job_id, request.dict(), stats)
            
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            logger.info(f"Job {job.job_id} cancelled")
        except Exception as e:
            job.status = JobStatus.FAILED
            job.errors.append(str(e))
            logger.error(f"Job {job.job_id} failed: {e}")
        finally:
            job.updated_at = datetime.utcnow()
            if cache:
                await cache.set_job(job.job_id, job.dict())
                
    # Create and store task
    task = asyncio.create_task(run_capture())
    capture_tasks[job.job_id] = task
    
    return job


@app.get("/jobs/{job_id}", response_model=JobMetadata)
async def get_job_status(job_id: str) -> JobMetadata:
    """Get current status and progress of a capture job"""
    
    # Check cache first
    if cache:
        cached_job = await cache.get_job(job_id)
        if cached_job:
            return JobMetadata(**cached_job)
            
    # Check active jobs
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return active_jobs[job_id]


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, str]:
    """Cancel a running capture job"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = active_jobs[job_id]
    
    if job.status != JobStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not running (status: {job.status})"
        )
        
    # Cancel the task
    if job_id in capture_tasks:
        task = capture_tasks[job_id]
        if not task.done():
            task.cancel()
            
    job.status = JobStatus.CANCELLED
    job.updated_at = datetime.utcnow()
    
    if cache:
        await cache.set_job(job_id, job.dict())
        
    return {"status": "cancelled", "job_id": job_id}


@app.post("/convert/arbitrage/start", response_model=JobMetadata)
@measure_time
async def start_arbitrage_detection(
    request: ArbitrageConvertRequest,
    background_tasks: BackgroundTasks
) -> JobMetadata:
    """
    Start arbitrage opportunity detection on captured data
    """
    logger.info(f"Starting arbitrage detection: {request.dict()}")
    
    job = JobMetadata(
        config=request.dict(),
        status=JobStatus.RUNNING,
        started_at=datetime.utcnow()
    )
    
    active_jobs[job.job_id] = job
    
    async def detect_arbitrage():
        try:
            detector = MEVDetector(min_profit_usd=request.min_profit_usd)
            
            # Load transactions from storage
            query = f"""
                SELECT * FROM transactions 
                WHERE slot IS NOT NULL
                ORDER BY slot
            """
            transactions = storage.query_with_duckdb(query)
            
            # Parse transactions
            parsed_txs = []
            for tx in transactions:
                try:
                    tx_data = {
                        "slot": tx["slot"],
                        "meta": json.loads(tx.get("err", "{}")),
                        "transaction": {
                            "signatures": [tx["signature"]],
                            "message": {}  # Would need full parsing
                        }
                    }
                    
                    # Add log messages
                    if tx.get("log_messages"):
                        tx_data["meta"]["logMessages"] = json.loads(tx["log_messages"])
                        
                    parsed_txs.append(tx_data)
                except:
                    continue
                    
            # Detect arbitrage
            opportunities = detector.detect_arbitrage(
                parsed_txs,
                max_slot_gap=request.max_slot_gap
            )
            
            # Save results
            if opportunities:
                await storage.write_arbitrage_opportunities(opportunities)
                
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = {
                "opportunities_found": len(opportunities),
                "total_profit_usd": sum(o["profit_usd"] for o in opportunities),
                "transactions_analyzed": len(parsed_txs)
            }
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.errors.append(str(e))
            logger.error(f"Arbitrage detection failed: {e}")
        finally:
            job.updated_at = datetime.utcnow()
            if cache:
                await cache.set_job(job.job_id, job.dict())
                
    task = asyncio.create_task(detect_arbitrage())
    capture_tasks[job.job_id] = task
    
    return job


@app.post("/convert/sandwich/start", response_model=JobMetadata)
@measure_time
async def start_sandwich_detection(
    request: SandwichConvertRequest,
    background_tasks: BackgroundTasks
) -> JobMetadata:
    """
    Start sandwich attack detection on captured data
    """
    logger.info(f"Starting sandwich detection: {request.dict()}")
    
    job = JobMetadata(
        config=request.dict(),
        status=JobStatus.RUNNING,
        started_at=datetime.utcnow()
    )
    
    active_jobs[job.job_id] = job
    
    async def detect_sandwiches():
        try:
            detector = MEVDetector(min_profit_usd=request.min_profit_usd)
            
            # Load transactions from storage
            query = f"""
                SELECT * FROM transactions 
                WHERE slot IS NOT NULL
                ORDER BY slot
            """
            transactions = storage.query_with_duckdb(query)
            
            # Parse and detect
            parsed_txs = []
            for tx in transactions:
                try:
                    tx_data = {
                        "slot": tx["slot"],
                        "meta": {},
                        "transaction": {
                            "signatures": [tx["signature"]],
                            "message": {}
                        }
                    }
                    
                    if tx.get("log_messages"):
                        tx_data["meta"]["logMessages"] = json.loads(tx["log_messages"])
                        
                    parsed_txs.append(tx_data)
                except:
                    continue
                    
            # Detect sandwich attacks
            attacks = detector.detect_sandwich_attacks(
                parsed_txs,
                max_slot_gap=request.max_slot_gap
            )
            
            # Save results
            if attacks:
                await storage.write_sandwich_attacks(attacks)
                
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = {
                "attacks_found": len(attacks),
                "total_profit_usd": sum(a["profit_usd"] for a in attacks),
                "total_victim_loss_usd": sum(a["victim_loss_usd"] for a in attacks),
                "transactions_analyzed": len(parsed_txs)
            }
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.errors.append(str(e))
            logger.error(f"Sandwich detection failed: {e}")
        finally:
            job.updated_at = datetime.utcnow()
            if cache:
                await cache.set_job(job.job_id, job.dict())
                
    task = asyncio.create_task(detect_sandwiches())
    capture_tasks[job.job_id] = task
    
    return job


@app.get("/datasets/stats", response_model=DatasetStats)
@measure_time
async def get_dataset_statistics() -> DatasetStats:
    """Get comprehensive statistics about captured datasets"""
    
    stats = await storage.get_dataset_stats()
    
    # Add MEV statistics
    try:
        # Count arbitrage opportunities
        arb_query = "SELECT COUNT(*) as count, SUM(profit_usd) as total FROM './data/labels/arbitrage.parquet'"
        arb_result = storage.query_with_duckdb(arb_query)
        if arb_result:
            stats["arbitrage_opportunities"] = arb_result[0]["count"]
            stats["total_mev_extracted_usd"] = arb_result[0]["total"] or 0
            
        # Count sandwich attacks
        sandwich_query = "SELECT COUNT(*) as count, SUM(profit_usd) as total FROM './data/labels/sandwich.parquet'"
        sandwich_result = storage.query_with_duckdb(sandwich_query)
        if sandwich_result:
            stats["sandwich_attacks"] = sandwich_result[0]["count"]
            stats["total_mev_extracted_usd"] += sandwich_result[0]["total"] or 0
            
    except:
        pass
        
    return DatasetStats(**stats)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health check endpoint"""
    
    health = HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=monitor.get_metrics()["uptime_seconds"] if monitor else 0,
        active_jobs=len([j for j in active_jobs.values() if j.status == JobStatus.RUNNING]),
        rpc_status="connected" if rpc_client else "disconnected",
        storage_available_gb=monitor.get_metrics()["storage_available_gb"] if monitor else 0,
        memory_usage_mb=monitor.get_metrics()["memory_usage_mb"] if monitor else 0,
        cpu_usage_percent=monitor.get_metrics()["cpu_usage_percent"] if monitor else 0
    )
    
    # Check component health
    if not rpc_client:
        health.status = "unhealthy"
        health.rpc_status = "disconnected"
    elif health.cpu_usage_percent > 90 or health.memory_usage_mb > 8000:
        health.status = "degraded"
        
    return health


@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    await websocket.accept()
    
    try:
        # Send initial job status
        if job_id in active_jobs:
            await websocket.send_json(active_jobs[job_id].dict())
            
        # Subscribe to updates
        while True:
            # Check for job updates every second
            await asyncio.sleep(1)
            
            if job_id in active_jobs:
                job = active_jobs[job_id]
                await websocket.send_json({
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "current_slot": job.current_slot,
                    "blocks_processed": job.blocks_processed,
                    "transactions_processed": job.transactions_processed
                })
                
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


async def update_job_progress(job: JobMetadata):
    """Update job progress and notify subscribers"""
    job.updated_at = datetime.utcnow()
    
    if cache:
        await cache.set_job(job.job_id, job.dict())
        await cache.publish_job_update(job.job_id, {
            "progress": job.progress,
            "current_slot": job.current_slot,
            "blocks_processed": job.blocks_processed
        })


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )