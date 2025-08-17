"""
Dataset export service for background processing
"""

import os
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from models.schemas import DatasetExportRequest, ExportFormat, JobStatus
from services.clickhouse_client import get_clickhouse_pool


class ExportService:
    """Background service for dataset exports"""
    
    def __init__(self):
        self.export_jobs: Dict[str, Dict] = {}
        self.export_dir = Path("/tmp/mev_exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_old_exports())
    
    async def export_dataset(
        self,
        job_id: str,
        request: DatasetExportRequest,
        user_id: str
    ):
        """Export dataset in background"""
        
        # Initialize job
        self.export_jobs[job_id] = {
            "job_id": job_id,
            "user_id": user_id,
            "status": JobStatus.RUNNING,
            "format": request.format,
            "started_at": datetime.now(),
            "request": request.dict()
        }
        
        try:
            # Get ClickHouse pool
            pool = await get_clickhouse_pool()
            
            # Apply time range filter if provided
            query = request.query
            if request.time_range:
                # Add time filter to query
                if "WHERE" in query.upper():
                    query += f" AND timestamp >= '{request.time_range.get('start')}'"
                    query += f" AND timestamp <= '{request.time_range.get('end')}'"
                else:
                    query += f" WHERE timestamp >= '{request.time_range.get('start')}'"
                    query += f" AND timestamp <= '{request.time_range.get('end')}'"
            
            # Get row count estimate
            count_query = f"SELECT count() FROM ({query}) AS subquery"
            count_result, _ = await pool.execute_query(count_query, use_cache=False)
            estimated_rows = count_result[0]["count()"] if count_result else 0
            
            self.export_jobs[job_id]["estimated_rows"] = estimated_rows
            
            # Determine file path and format
            file_extension = request.format.value.lower()
            file_path = self.export_dir / f"{job_id}.{file_extension}"
            
            if request.format == ExportFormat.PARQUET:
                # Export to Parquet
                export_result = await pool.export_to_parquet(
                    query=query,
                    output_path=str(file_path),
                    params=request.filters,
                    chunk_size=request.chunk_size,
                    compression=request.compression or "snappy"
                )
                
                self.export_jobs[job_id]["file_path"] = str(file_path)
                self.export_jobs[job_id]["file_size_bytes"] = export_result["file_size_bytes"]
                
            elif request.format == ExportFormat.ARROW:
                # Export to Arrow format
                await self._export_to_arrow(
                    pool=pool,
                    query=query,
                    file_path=file_path,
                    params=request.filters,
                    chunk_size=request.chunk_size
                )
                
                self.export_jobs[job_id]["file_path"] = str(file_path)
                self.export_jobs[job_id]["file_size_bytes"] = file_path.stat().st_size
                
            elif request.format == ExportFormat.CSV:
                # Export to CSV
                await self._export_to_csv(
                    pool=pool,
                    query=query,
                    file_path=file_path,
                    params=request.filters,
                    chunk_size=request.chunk_size
                )
                
                self.export_jobs[job_id]["file_path"] = str(file_path)
                self.export_jobs[job_id]["file_size_bytes"] = file_path.stat().st_size
                
            elif request.format == ExportFormat.JSON:
                # Export to JSON
                await self._export_to_json(
                    pool=pool,
                    query=query,
                    file_path=file_path,
                    params=request.filters,
                    chunk_size=request.chunk_size
                )
                
                self.export_jobs[job_id]["file_path"] = str(file_path)
                self.export_jobs[job_id]["file_size_bytes"] = file_path.stat().st_size
            
            # Add metadata if requested
            if request.include_metadata:
                metadata_path = self.export_dir / f"{job_id}_metadata.json"
                import json
                
                metadata = {
                    "job_id": job_id,
                    "query": query,
                    "format": request.format.value,
                    "compression": request.compression,
                    "rows_exported": estimated_rows,
                    "file_size_bytes": self.export_jobs[job_id].get("file_size_bytes"),
                    "created_at": datetime.now().isoformat(),
                    "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
                }
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            # Update job status
            self.export_jobs[job_id]["status"] = JobStatus.COMPLETED
            self.export_jobs[job_id]["completed_at"] = datetime.now()
            self.export_jobs[job_id]["expires_at"] = datetime.now() + timedelta(hours=24)
            self.export_jobs[job_id]["download_url"] = f"/datasets/export/{job_id}/download"
            self.export_jobs[job_id]["estimated_size_mb"] = (
                self.export_jobs[job_id]["file_size_bytes"] / 1024 / 1024
            )
            
        except Exception as e:
            self.export_jobs[job_id]["status"] = JobStatus.FAILED
            self.export_jobs[job_id]["error"] = str(e)
            self.export_jobs[job_id]["completed_at"] = datetime.now()
    
    async def _export_to_arrow(
        self,
        pool,
        query: str,
        file_path: Path,
        params: Optional[Dict],
        chunk_size: int
    ):
        """Export to Apache Arrow format"""
        
        # Get data in chunks
        offset = 0
        writer = None
        
        while True:
            chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
            data, _ = await pool.execute_query(
                chunk_query,
                params=params,
                use_cache=False
            )
            
            if not data:
                break
            
            # Convert to DataFrame then Arrow
            df = pd.DataFrame(data)
            table = pa.Table.from_pandas(df)
            
            # Write to Arrow file
            if writer is None:
                writer = pa.ipc.RecordBatchFileWriter(
                    str(file_path),
                    table.schema
                )
            
            writer.write_table(table)
            offset += chunk_size
        
        if writer:
            writer.close()
    
    async def _export_to_csv(
        self,
        pool,
        query: str,
        file_path: Path,
        params: Optional[Dict],
        chunk_size: int
    ):
        """Export to CSV format"""
        
        offset = 0
        first_chunk = True
        
        while True:
            chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
            data, _ = await pool.execute_query(
                chunk_query,
                params=params,
                use_cache=False
            )
            
            if not data:
                break
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Write to CSV
            if first_chunk:
                df.to_csv(file_path, index=False)
                first_chunk = False
            else:
                df.to_csv(file_path, mode='a', header=False, index=False)
            
            offset += chunk_size
    
    async def _export_to_json(
        self,
        pool,
        query: str,
        file_path: Path,
        params: Optional[Dict],
        chunk_size: int
    ):
        """Export to JSON format"""
        
        import json
        
        offset = 0
        all_data = []
        
        while True:
            chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
            data, _ = await pool.execute_query(
                chunk_query,
                params=params,
                use_cache=False
            )
            
            if not data:
                break
            
            all_data.extend(data)
            offset += chunk_size
        
        # Write JSON
        with open(file_path, "w") as f:
            json.dump(all_data, f, indent=2, default=str)
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get export job status"""
        return self.export_jobs.get(job_id)
    
    def get_user_jobs(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's recent export jobs"""
        
        user_jobs = [
            job for job in self.export_jobs.values()
            if job["user_id"] == user_id
        ]
        
        # Sort by started_at descending
        user_jobs.sort(key=lambda x: x["started_at"], reverse=True)
        
        return user_jobs[:limit]
    
    def cancel_job(self, job_id: str):
        """Cancel running export job"""
        
        if job_id in self.export_jobs:
            if self.export_jobs[job_id]["status"] == JobStatus.RUNNING:
                self.export_jobs[job_id]["status"] = JobStatus.CANCELLED
                self.export_jobs[job_id]["completed_at"] = datetime.now()
    
    async def _cleanup_old_exports(self):
        """Clean up expired export files"""
        
        while True:
            try:
                # Wait 1 hour between cleanups
                await asyncio.sleep(3600)
                
                now = datetime.now()
                
                # Check all jobs
                for job_id, job_info in list(self.export_jobs.items()):
                    expires_at = job_info.get("expires_at")
                    
                    if expires_at and expires_at < now:
                        # Delete file if exists
                        file_path = job_info.get("file_path")
                        if file_path and os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except Exception:
                                pass
                        
                        # Delete metadata file if exists
                        metadata_path = self.export_dir / f"{job_id}_metadata.json"
                        if metadata_path.exists():
                            try:
                                metadata_path.unlink()
                            except Exception:
                                pass
                        
                        # Remove from jobs dict
                        del self.export_jobs[job_id]
                
            except Exception as e:
                import logging
                logging.error(f"Cleanup error: {e}")