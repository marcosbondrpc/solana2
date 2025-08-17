"""
Dataset export endpoints
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from models.schemas import (
    DatasetExportRequest,
    DatasetExportResponse,
    JobStatus,
    UserRole
)
from security.auth import require_role, get_current_user, TokenData
from services.clickhouse_client import get_clickhouse_pool
from services.export_service import ExportService

router = APIRouter()

# Export service instance
export_service = ExportService()


@router.post("/export", response_model=DatasetExportResponse)
async def export_dataset(
    request: DatasetExportRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> DatasetExportResponse:
    """
    Export dataset to Parquet/Arrow format
    Requires ANALYST role or higher
    """
    
    # Validate query
    pool = await get_clickhouse_pool()
    is_valid, error = await pool.validate_query(request.query)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid query: {error}")
    
    # Apply role-based query limits
    from security.policy import get_query_limits
    limits = get_query_limits(current_user.role)
    
    if request.chunk_size > limits["max_rows"]:
        request.chunk_size = limits["max_rows"]
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Start export in background
    background_tasks.add_task(
        export_service.export_dataset,
        job_id=job_id,
        request=request,
        user_id=current_user.user_id
    )
    
    # Return immediate response
    return DatasetExportResponse(
        success=True,
        job_id=job_id,
        status=JobStatus.PENDING,
        format=request.format,
        message="Export job started"
    )


@router.get("/export/{job_id}/status", response_model=DatasetExportResponse)
async def get_export_status(
    job_id: str,
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> DatasetExportResponse:
    """Get export job status"""
    
    job_info = export_service.get_job_status(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if user owns the job
    if job_info["user_id"] != current_user.user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return DatasetExportResponse(
        success=True,
        job_id=job_id,
        status=job_info["status"],
        format=job_info["format"],
        estimated_rows=job_info.get("estimated_rows"),
        estimated_size_mb=job_info.get("estimated_size_mb"),
        download_url=job_info.get("download_url"),
        expires_at=job_info.get("expires_at"),
        message=job_info.get("message")
    )


@router.get("/export/{job_id}/download")
async def download_export(
    job_id: str,
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
):
    """Download exported dataset"""
    
    job_info = export_service.get_job_status(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if user owns the job
    if job_info["user_id"] != current_user.user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job_info["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Export not ready")
    
    file_path = job_info.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Export file not found")
    
    # Return file
    return FileResponse(
        path=file_path,
        filename=f"export_{job_id}.{job_info['format'].lower()}",
        media_type="application/octet-stream"
    )


@router.delete("/export/{job_id}")
async def cancel_export(
    job_id: str,
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> Dict[str, Any]:
    """Cancel export job"""
    
    job_info = export_service.get_job_status(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if user owns the job
    if job_info["user_id"] != current_user.user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job_info["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Job already finished")
    
    # Cancel job
    export_service.cancel_job(job_id)
    
    return {
        "success": True,
        "message": "Export job cancelled",
        "job_id": job_id
    }


@router.get("/templates")
async def get_export_templates(
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> Dict[str, Any]:
    """Get predefined export templates"""
    
    templates = {
        "arbitrage_opportunities": {
            "name": "Arbitrage Opportunities",
            "description": "Export detected arbitrage opportunities",
            "query": """
                SELECT 
                    slot,
                    tx_signature,
                    roi_pct,
                    est_profit,
                    legs,
                    dex_route,
                    tokens,
                    confidence,
                    detected_at
                FROM mev.arbitrage_alerts
                WHERE detected_at >= now() - INTERVAL 1 DAY
                ORDER BY detected_at DESC
            """,
            "format": "parquet"
        },
        "sandwich_attacks": {
            "name": "Sandwich Attacks",
            "description": "Export detected sandwich attacks",
            "query": """
                SELECT 
                    slot,
                    victim_tx,
                    front_tx,
                    back_tx,
                    victim_loss,
                    attacker_profit,
                    token_pair,
                    dex,
                    detected_at
                FROM mev.sandwich_alerts
                WHERE detected_at >= now() - INTERVAL 1 DAY
                ORDER BY detected_at DESC
            """,
            "format": "parquet"
        },
        "system_metrics": {
            "name": "System Performance Metrics",
            "description": "Export system performance metrics",
            "query": """
                SELECT 
                    timestamp,
                    latency_p50_ms,
                    latency_p99_ms,
                    bundle_land_rate,
                    ingestion_rate,
                    model_inference_us,
                    decision_dna_count
                FROM mev.system_metrics
                WHERE timestamp >= now() - INTERVAL 1 HOUR
                ORDER BY timestamp DESC
            """,
            "format": "parquet"
        },
        "thompson_bandit_stats": {
            "name": "Thompson Sampling Statistics",
            "description": "Export Thompson Sampling bandit statistics",
            "query": """
                SELECT 
                    timestamp,
                    arm_name,
                    alpha,
                    beta,
                    expected_value,
                    samples,
                    total_reward
                FROM mev.thompson_stats
                WHERE timestamp >= now() - INTERVAL 1 DAY
                ORDER BY timestamp DESC
            """,
            "format": "parquet"
        }
    }
    
    return {
        "success": True,
        "templates": templates
    }


@router.get("/recent")
async def get_recent_exports(
    limit: int = 10,
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> Dict[str, Any]:
    """Get user's recent exports"""
    
    recent_jobs = export_service.get_user_jobs(
        current_user.user_id,
        limit=limit
    )
    
    return {
        "success": True,
        "exports": recent_jobs
    }