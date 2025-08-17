"""
Dataset API: ClickHouse to Parquet/JSONL export
High-throughput data extraction with streaming support
"""

import os
import time
import json
import asyncio
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import clickhouse_connect
import pyarrow as pa
import pyarrow.parquet as pq

from .deps import User, get_current_user, require_permission, audit_log


router = APIRouter()

# Global state for job tracking
export_jobs: Dict[str, Dict[str, Any]] = {}
job_lock = asyncio.Lock()

# ClickHouse client
ch_client = None


def get_clickhouse_client():
    """Get or create ClickHouse client"""
    global ch_client
    if ch_client is None:
        ch_client = clickhouse_connect.get_client(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
            username=os.getenv("CLICKHOUSE_USER", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", ""),
            database=os.getenv("CLICKHOUSE_DATABASE", "mev"),
            settings={
                "max_block_size": 100000,
                "max_threads": 8,
                "max_memory_usage": 10000000000,  # 10GB
                "max_execution_time": 300,  # 5 minutes
                "send_progress_in_http_headers": 1
            }
        )
    return ch_client


class DatasetRequest(BaseModel):
    """Dataset export request"""
    dataset_type: str = Field(..., description="Type: mev_opportunities, arbitrage_opportunities, bundle_outcomes")
    start_time: datetime = Field(..., description="Start time for data range")
    end_time: datetime = Field(..., description="End time for data range")
    format: str = Field("parquet", description="Export format: parquet or jsonl")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")
    columns: Optional[List[str]] = Field(None, description="Specific columns to export")
    compression: str = Field("snappy", description="Compression: snappy, gzip, lz4, zstd, none")
    chunk_size: int = Field(100000, description="Rows per chunk for streaming")
    sample_rate: Optional[float] = Field(None, description="Sample rate (0.0-1.0) for large datasets")


class JobStatus(BaseModel):
    """Export job status"""
    job_id: str
    status: str
    progress: float
    rows_exported: int
    file_path: Optional[str]
    error: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]


async def export_dataset_task(job_id: str, request: DatasetRequest, user_id: str):
    """Background task to export dataset"""
    
    async with job_lock:
        if job_id not in export_jobs:
            return
        export_jobs[job_id]["status"] = "running"
        export_jobs[job_id]["started_at"] = datetime.utcnow()
    
    try:
        client = get_clickhouse_client()
        
        # Build query based on dataset type
        table_map = {
            "mev_opportunities": "mev_opportunities_typed",
            "arbitrage_opportunities": "arb_opportunities_typed",
            "bundle_outcomes": "bundle_outcomes_typed",
            "market_ticks": "market_ticks_typed",
            "metrics": "metrics_typed"
        }
        
        if request.dataset_type not in table_map:
            raise ValueError(f"Unknown dataset type: {request.dataset_type}")
        
        table = table_map[request.dataset_type]
        
        # Build SELECT clause
        if request.columns:
            columns = ", ".join(request.columns)
        else:
            columns = "*"
        
        # Build WHERE clause
        where_clauses = [
            f"timestamp >= '{request.start_time.isoformat()}'",
            f"timestamp <= '{request.end_time.isoformat()}'"
        ]
        
        # Add custom filters
        for key, value in request.filters.items():
            if isinstance(value, str):
                where_clauses.append(f"{key} = '{value}'")
            elif isinstance(value, (list, tuple)):
                values = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
                where_clauses.append(f"{key} IN ({values})")
            else:
                where_clauses.append(f"{key} = {value}")
        
        # Add sampling if requested
        sample_clause = ""
        if request.sample_rate and 0 < request.sample_rate < 1:
            sample_clause = f"SAMPLE {request.sample_rate}"
        
        # Build final query
        query = f"""
        SELECT {columns}
        FROM {table}
        WHERE {' AND '.join(where_clauses)}
        {sample_clause}
        ORDER BY timestamp
        """
        
        # Create output directory
        output_dir = Path(os.getenv("EXPORT_DIR", "/tmp/exports"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output file path
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{request.dataset_type}_{timestamp}_{job_id[:8]}.{request.format}"
        if request.compression != "none" and request.format == "jsonl":
            filename += f".{request.compression}"
        
        output_path = output_dir / filename
        
        # Execute query and export
        if request.format == "parquet":
            # Export to Parquet using PyArrow
            rows_exported = await export_to_parquet(
                client, query, output_path, request.compression,
                request.chunk_size, job_id
            )
        else:  # jsonl
            # Export to JSONL
            rows_exported = await export_to_jsonl(
                client, query, output_path, request.compression,
                request.chunk_size, job_id
            )
        
        # Update job status
        async with job_lock:
            export_jobs[job_id]["status"] = "completed"
            export_jobs[job_id]["completed_at"] = datetime.utcnow()
            export_jobs[job_id]["rows_exported"] = rows_exported
            export_jobs[job_id]["file_path"] = str(output_path)
            export_jobs[job_id]["file_size"] = output_path.stat().st_size
            export_jobs[job_id]["progress"] = 1.0
        
    except Exception as e:
        # Update job with error
        async with job_lock:
            export_jobs[job_id]["status"] = "failed"
            export_jobs[job_id]["error"] = str(e)
            export_jobs[job_id]["completed_at"] = datetime.utcnow()


async def export_to_parquet(
    client, query: str, output_path: Path,
    compression: str, chunk_size: int, job_id: str
) -> int:
    """Export query results to Parquet format"""
    
    total_rows = 0
    
    # Execute query with streaming
    result = client.query(query, settings={"max_block_size": chunk_size})
    
    # Get schema from first batch
    first_batch = result.first_block
    if not first_batch:
        raise ValueError("Query returned no results")
    
    # Convert to PyArrow table
    schema = pa.schema([
        (col[0], _clickhouse_to_arrow_type(col[1]))
        for col in result.column_types
    ])
    
    # Create Parquet writer
    writer = pq.ParquetWriter(
        output_path,
        schema,
        compression=compression if compression != "none" else None,
        use_dictionary=True,
        compression_level=3 if compression == "zstd" else None
    )
    
    # Process first batch
    table = pa.Table.from_pandas(first_batch.to_pandas())
    writer.write_table(table)
    total_rows += len(first_batch)
    
    # Process remaining batches
    for block in result.blocks:
        table = pa.Table.from_pandas(block.to_pandas())
        writer.write_table(table)
        total_rows += len(block)
        
        # Update progress
        async with job_lock:
            if job_id in export_jobs:
                export_jobs[job_id]["rows_exported"] = total_rows
    
    writer.close()
    return total_rows


async def export_to_jsonl(
    client, query: str, output_path: Path,
    compression: str, chunk_size: int, job_id: str
) -> int:
    """Export query results to JSONL format"""
    
    import gzip
    import lz4.frame
    import zstandard as zstd
    
    total_rows = 0
    
    # Open output file with compression if needed
    if compression == "gzip":
        file_obj = gzip.open(output_path, "wt", encoding="utf-8")
    elif compression == "lz4":
        file_obj = lz4.frame.open(output_path, "wt", encoding="utf-8")
    elif compression == "zstd":
        cctx = zstd.ZstdCompressor(level=3, threads=2)
        file_obj = zstd.open(output_path, "wt", encoding="utf-8", cctx=cctx)
    else:
        file_obj = open(output_path, "w", encoding="utf-8")
    
    try:
        # Execute query with streaming
        result = client.query(query, settings={"max_block_size": chunk_size})
        
        # Process blocks
        for block in result.blocks:
            for row in block.to_dict("records"):
                # Convert datetime objects to ISO format
                for key, value in row.items():
                    if isinstance(value, datetime):
                        row[key] = value.isoformat()
                
                # Write as JSON line
                json.dump(row, file_obj, default=str)
                file_obj.write("\n")
                total_rows += 1
            
            # Update progress
            async with job_lock:
                if job_id in export_jobs:
                    export_jobs[job_id]["rows_exported"] = total_rows
    
    finally:
        file_obj.close()
    
    return total_rows


def _clickhouse_to_arrow_type(ch_type: str):
    """Convert ClickHouse type to PyArrow type"""
    type_map = {
        "UInt8": pa.uint8(),
        "UInt16": pa.uint16(),
        "UInt32": pa.uint32(),
        "UInt64": pa.uint64(),
        "Int8": pa.int8(),
        "Int16": pa.int16(),
        "Int32": pa.int32(),
        "Int64": pa.int64(),
        "Float32": pa.float32(),
        "Float64": pa.float64(),
        "String": pa.string(),
        "DateTime": pa.timestamp("s"),
        "DateTime64": pa.timestamp("ns"),
        "Date": pa.date32(),
        "Bool": pa.bool_()
    }
    
    # Handle nullable types
    if ch_type.startswith("Nullable("):
        inner_type = ch_type[9:-1]
        return type_map.get(inner_type, pa.string())
    
    return type_map.get(ch_type, pa.string())


@router.post("/export", dependencies=[Depends(require_permission("datasets:write"))])
async def start_export(
    request: DatasetRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
) -> JobStatus:
    """Start dataset export job"""
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job
    async with job_lock:
        export_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "rows_exported": 0,
            "file_path": None,
            "error": None,
            "created_at": datetime.utcnow(),
            "completed_at": None,
            "user_id": user.id,
            "request": request.dict(),
            "metadata": {
                "dataset_type": request.dataset_type,
                "format": request.format,
                "compression": request.compression,
                "time_range": f"{request.start_time} to {request.end_time}"
            }
        }
    
    # Start background export
    background_tasks.add_task(
        export_dataset_task,
        job_id,
        request,
        user.id
    )
    
    # Audit log
    background_tasks.add_task(
        audit_log,
        "dataset_export",
        user,
        {
            "job_id": job_id,
            "dataset_type": request.dataset_type,
            "format": request.format
        }
    )
    
    return JobStatus(**export_jobs[job_id])


@router.get("/export/{job_id}", dependencies=[Depends(require_permission("datasets:read"))])
async def get_export_status(
    job_id: str,
    user: User = Depends(get_current_user)
) -> JobStatus:
    """Get export job status"""
    
    async with job_lock:
        if job_id not in export_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = export_jobs[job_id]
        
        # Check authorization
        if job["user_id"] != user.id and "admin" not in user.roles:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return JobStatus(**job)


@router.get("/export/{job_id}/download", dependencies=[Depends(require_permission("datasets:read"))])
async def download_export(
    job_id: str,
    user: User = Depends(get_current_user)
):
    """Download exported dataset"""
    
    async with job_lock:
        if job_id not in export_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = export_jobs[job_id]
        
        # Check authorization
        if job["user_id"] != user.id and "admin" not in user.roles:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Export not completed")
        
        if not job["file_path"] or not Path(job["file_path"]).exists():
            raise HTTPException(status_code=404, detail="Export file not found")
    
    # Return file
    return FileResponse(
        job["file_path"],
        media_type="application/octet-stream",
        filename=Path(job["file_path"]).name
    )


@router.get("/exports", dependencies=[Depends(require_permission("datasets:read"))])
async def list_exports(
    user: User = Depends(get_current_user),
    limit: int = Query(100, le=1000)
) -> List[JobStatus]:
    """List export jobs for current user"""
    
    async with job_lock:
        user_jobs = []
        for job in export_jobs.values():
            if job["user_id"] == user.id or "admin" in user.roles:
                user_jobs.append(JobStatus(**job))
        
        # Sort by created_at descending
        user_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return user_jobs[:limit]


@router.delete("/export/{job_id}", dependencies=[Depends(require_permission("datasets:write"))])
async def cancel_export(
    job_id: str,
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Cancel or delete export job"""
    
    async with job_lock:
        if job_id not in export_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = export_jobs[job_id]
        
        # Check authorization
        if job["user_id"] != user.id and "admin" not in user.roles:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete file if exists
        if job["file_path"] and Path(job["file_path"]).exists():
            Path(job["file_path"]).unlink()
        
        # Remove job
        del export_jobs[job_id]
    
    return {"status": "deleted", "job_id": job_id}


@router.get("/schema/{dataset_type}")
async def get_dataset_schema(
    dataset_type: str,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get schema for dataset type"""
    
    table_map = {
        "mev_opportunities": "mev_opportunities_typed",
        "arbitrage_opportunities": "arb_opportunities_typed",
        "bundle_outcomes": "bundle_outcomes_typed",
        "market_ticks": "market_ticks_typed",
        "metrics": "metrics_typed"
    }
    
    if dataset_type not in table_map:
        raise HTTPException(status_code=404, detail="Unknown dataset type")
    
    client = get_clickhouse_client()
    
    # Get table schema
    query = f"DESCRIBE TABLE {table_map[dataset_type]}"
    result = client.query(query)
    
    schema = []
    for row in result.result_rows:
        schema.append({
            "name": row[0],
            "type": row[1],
            "default": row[2] if len(row) > 2 else None,
            "comment": row[3] if len(row) > 3 else None
        })
    
    return {
        "dataset_type": dataset_type,
        "table": table_map[dataset_type],
        "columns": schema
    }