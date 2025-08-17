"""
Continuous Data Export Scheduler
Automated export of arbitrage data in multiple formats with compression and optimization
"""

import asyncio
import json
import os
import time
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import numpy as np
from clickhouse_driver.client import Client as ClickHouseClient
import aioredis
from aiokafka import AIOKafkaProducer
from prometheus_client import Counter, Gauge, Histogram
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import boto3
from google.cloud import storage as gcs
import lz4.frame
import zstandard as zstd

logger = structlog.get_logger()

# Metrics
exports_total = Counter('data_exports_total', 'Total data exports', ['format', 'status'])
export_duration = Histogram('export_duration_seconds', 'Export duration', ['format'])
export_size_bytes = Gauge('export_size_bytes', 'Export file size', ['format'])
export_rows_total = Counter('export_rows_total', 'Total rows exported', ['format'])

class ExportFormat(Enum):
    """Supported export formats"""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"  # JSON Lines
    ARROW = "arrow"
    HDF5 = "hdf5"
    TFRECORD = "tfrecord"
    FEATHER = "feather"
    ORC = "orc"
    AVRO = "avro"

class CompressionType(Enum):
    """Compression algorithms"""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"

class StorageBackend(Enum):
    """Storage backends for exports"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    HDFS = "hdfs"
    FTP = "ftp"

@dataclass
class ExportConfig:
    """Configuration for data export"""
    name: str
    query: str
    format: ExportFormat
    compression: CompressionType = CompressionType.SNAPPY
    storage: StorageBackend = StorageBackend.LOCAL
    storage_path: str = "/data/exports"
    schedule: str = "0 * * * *"  # Cron expression (hourly by default)
    retention_days: int = 30
    partition_by: Optional[List[str]] = None
    chunk_size: int = 100000
    parallel_exports: bool = True
    include_metadata: bool = True
    filters: Dict[str, Any] = field(default_factory=dict)
    transformations: List[str] = field(default_factory=list)
    notification_webhook: Optional[str] = None

@dataclass
class ExportJob:
    """Export job tracking"""
    job_id: str
    config: ExportConfig
    status: str  # scheduled, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    rows_exported: int = 0
    file_size_bytes: int = 0
    file_paths: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataExporter:
    """Handles data export in various formats"""
    
    def __init__(self, clickhouse_client: ClickHouseClient):
        self.clickhouse_client = clickhouse_client
        self.compression_handlers = {
            CompressionType.GZIP: self._compress_gzip,
            CompressionType.SNAPPY: self._compress_snappy,
            CompressionType.LZ4: self._compress_lz4,
            CompressionType.ZSTD: self._compress_zstd,
            CompressionType.BROTLI: self._compress_brotli
        }
    
    async def export(self, config: ExportConfig) -> ExportJob:
        """Execute data export based on configuration"""
        job = ExportJob(
            job_id=self._generate_job_id(config),
            config=config,
            status="running",
            start_time=datetime.utcnow()
        )
        
        try:
            # Execute query
            logger.info(f"Executing export query for {config.name}")
            data = await self._execute_query(config.query, config.chunk_size)
            
            if not data:
                job.status = "completed"
                job.end_time = datetime.utcnow()
                job.metadata['message'] = "No data to export"
                return job
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            job.rows_exported = len(df)
            
            # Apply transformations
            if config.transformations:
                df = await self._apply_transformations(df, config.transformations)
            
            # Apply filters
            if config.filters:
                df = self._apply_filters(df, config.filters)
            
            # Export based on format
            file_paths = await self._export_data(df, config)
            job.file_paths = file_paths
            
            # Calculate total file size
            job.file_size_bytes = sum(
                os.path.getsize(path) for path in file_paths 
                if os.path.exists(path)
            )
            
            # Upload to storage backend
            if config.storage != StorageBackend.LOCAL:
                await self._upload_to_storage(file_paths, config)
            
            # Update metrics
            exports_total.labels(format=config.format.value, status="success").inc()
            export_rows_total.labels(format=config.format.value).inc(job.rows_exported)
            export_size_bytes.labels(format=config.format.value).set(job.file_size_bytes)
            
            job.status = "completed"
            job.end_time = datetime.utcnow()
            
            logger.info(
                f"Export completed: {job.rows_exported} rows, "
                f"{job.file_size_bytes / (1024*1024):.2f} MB"
            )
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.utcnow()
            exports_total.labels(format=config.format.value, status="failed").inc()
            logger.error(f"Export failed: {e}")
        
        # Record duration
        if job.start_time and job.end_time:
            duration = (job.end_time - job.start_time).total_seconds()
            export_duration.labels(format=config.format.value).observe(duration)
        
        return job
    
    async def _execute_query(self, query: str, chunk_size: int) -> List[Dict]:
        """Execute ClickHouse query with chunking"""
        # Add LIMIT if not present for safety
        if 'LIMIT' not in query.upper():
            query += f" LIMIT {chunk_size * 10}"  # Max 10 chunks
        
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.clickhouse_client.execute,
            query,
            {'with_column_types': True}
        )
        
        if result:
            data, columns = result[0], result[1]
            column_names = [col[0] for col in columns]
            
            # Convert to list of dicts
            return [dict(zip(column_names, row)) for row in data]
        
        return []
    
    async def _apply_transformations(self, df: pd.DataFrame, 
                                    transformations: List[str]) -> pd.DataFrame:
        """Apply data transformations"""
        for transformation in transformations:
            if transformation == "normalize_timestamps":
                # Convert all timestamp columns to UTC
                for col in df.select_dtypes(include=['datetime64']).columns:
                    df[col] = pd.to_datetime(df[col], utc=True)
            
            elif transformation == "calculate_statistics":
                # Add statistical columns
                if 'net_profit' in df.columns:
                    df['profit_zscore'] = (df['net_profit'] - df['net_profit'].mean()) / df['net_profit'].std()
                    df['profit_percentile'] = df['net_profit'].rank(pct=True)
            
            elif transformation == "add_metadata":
                # Add export metadata
                df['export_timestamp'] = datetime.utcnow()
                df['export_version'] = "1.0.0"
            
            elif transformation == "optimize_dtypes":
                # Optimize data types for smaller file size
                df = self._optimize_dtypes(df)
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        return df
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to DataFrame"""
        for column, condition in filters.items():
            if column not in df.columns:
                continue
            
            if isinstance(condition, dict):
                # Complex condition
                if 'min' in condition:
                    df = df[df[column] >= condition['min']]
                if 'max' in condition:
                    df = df[df[column] <= condition['max']]
                if 'in' in condition:
                    df = df[df[column].isin(condition['in'])]
                if 'not_in' in condition:
                    df = df[~df[column].isin(condition['not_in'])]
            else:
                # Simple equality
                df = df[df[column] == condition]
        
        return df
    
    async def _export_data(self, df: pd.DataFrame, config: ExportConfig) -> List[str]:
        """Export DataFrame to specified format"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        base_path = Path(config.storage_path) / config.name / timestamp
        base_path.mkdir(parents=True, exist_ok=True)
        
        file_paths = []
        
        if config.format == ExportFormat.PARQUET:
            file_paths = await self._export_parquet(df, base_path, config)
        elif config.format == ExportFormat.CSV:
            file_paths = await self._export_csv(df, base_path, config)
        elif config.format == ExportFormat.JSON:
            file_paths = await self._export_json(df, base_path, config)
        elif config.format == ExportFormat.JSONL:
            file_paths = await self._export_jsonl(df, base_path, config)
        elif config.format == ExportFormat.ARROW:
            file_paths = await self._export_arrow(df, base_path, config)
        elif config.format == ExportFormat.HDF5:
            file_paths = await self._export_hdf5(df, base_path, config)
        elif config.format == ExportFormat.FEATHER:
            file_paths = await self._export_feather(df, base_path, config)
        elif config.format == ExportFormat.ORC:
            file_paths = await self._export_orc(df, base_path, config)
        elif config.format == ExportFormat.TFRECORD:
            file_paths = await self._export_tfrecord(df, base_path, config)
        else:
            raise ValueError(f"Unsupported format: {config.format}")
        
        # Apply compression if needed
        if config.compression != CompressionType.NONE:
            compressed_paths = []
            for path in file_paths:
                compressed_path = await self._compress_file(path, config.compression)
                compressed_paths.append(compressed_path)
                # Remove uncompressed file
                os.remove(path)
            file_paths = compressed_paths
        
        return file_paths
    
    async def _export_parquet(self, df: pd.DataFrame, base_path: Path, 
                             config: ExportConfig) -> List[str]:
        """Export to Parquet format"""
        file_path = base_path / f"data.parquet"
        
        # Parquet-specific optimizations
        table = pa.Table.from_pandas(df)
        
        # Apply partitioning if specified
        if config.partition_by:
            pq.write_to_dataset(
                table,
                root_path=str(base_path),
                partition_cols=config.partition_by,
                compression='snappy' if config.compression == CompressionType.NONE else None
            )
            # Return all partition files
            return [str(p) for p in base_path.rglob("*.parquet")]
        else:
            pq.write_table(
                table,
                str(file_path),
                compression='snappy' if config.compression == CompressionType.NONE else None,
                use_dictionary=True,
                use_deprecated_int96_timestamps=False
            )
            return [str(file_path)]
    
    async def _export_csv(self, df: pd.DataFrame, base_path: Path, 
                         config: ExportConfig) -> List[str]:
        """Export to CSV format"""
        file_paths = []
        
        if config.chunk_size and len(df) > config.chunk_size:
            # Split into chunks
            num_chunks = (len(df) + config.chunk_size - 1) // config.chunk_size
            
            for i in range(num_chunks):
                start_idx = i * config.chunk_size
                end_idx = min((i + 1) * config.chunk_size, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
                
                file_path = base_path / f"data_part_{i:04d}.csv"
                chunk_df.to_csv(str(file_path), index=False)
                file_paths.append(str(file_path))
        else:
            file_path = base_path / "data.csv"
            df.to_csv(str(file_path), index=False)
            file_paths.append(str(file_path))
        
        return file_paths
    
    async def _export_json(self, df: pd.DataFrame, base_path: Path, 
                          config: ExportConfig) -> List[str]:
        """Export to JSON format"""
        file_path = base_path / "data.json"
        
        # Convert datetime columns to ISO format
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        df.to_json(
            str(file_path),
            orient='records',
            date_format='iso',
            indent=2 if config.include_metadata else None
        )
        
        return [str(file_path)]
    
    async def _export_jsonl(self, df: pd.DataFrame, base_path: Path, 
                           config: ExportConfig) -> List[str]:
        """Export to JSON Lines format"""
        file_path = base_path / "data.jsonl"
        
        # Write each row as a JSON object on a new line
        with open(file_path, 'w') as f:
            for _, row in df.iterrows():
                json_row = row.to_json(date_format='iso')
                f.write(json_row + '\n')
        
        return [str(file_path)]
    
    async def _export_arrow(self, df: pd.DataFrame, base_path: Path, 
                           config: ExportConfig) -> List[str]:
        """Export to Apache Arrow format"""
        file_path = base_path / "data.arrow"
        
        table = pa.Table.from_pandas(df)
        
        with pa.OSFile(str(file_path), 'wb') as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write(table)
        
        return [str(file_path)]
    
    async def _export_hdf5(self, df: pd.DataFrame, base_path: Path, 
                          config: ExportConfig) -> List[str]:
        """Export to HDF5 format"""
        file_path = base_path / "data.h5"
        
        # Use compression if specified
        complib = 'zlib' if config.compression != CompressionType.NONE else None
        
        df.to_hdf(
            str(file_path),
            key='data',
            mode='w',
            complib=complib,
            complevel=9 if complib else 0
        )
        
        # Add metadata if requested
        if config.include_metadata:
            with h5py.File(str(file_path), 'a') as f:
                f.attrs['export_timestamp'] = datetime.utcnow().isoformat()
                f.attrs['rows'] = len(df)
                f.attrs['columns'] = list(df.columns)
        
        return [str(file_path)]
    
    async def _export_feather(self, df: pd.DataFrame, base_path: Path, 
                             config: ExportConfig) -> List[str]:
        """Export to Feather format"""
        file_path = base_path / "data.feather"
        
        # Feather format with compression
        compression = 'lz4' if config.compression in [CompressionType.NONE, CompressionType.LZ4] else 'uncompressed'
        df.to_feather(str(file_path), compression=compression)
        
        return [str(file_path)]
    
    async def _export_orc(self, df: pd.DataFrame, base_path: Path, 
                         config: ExportConfig) -> List[str]:
        """Export to ORC format"""
        file_path = base_path / "data.orc"
        
        # ORC requires pyarrow
        table = pa.Table.from_pandas(df)
        
        # ORC writer with compression
        compression = 'snappy' if config.compression == CompressionType.SNAPPY else 'uncompressed'
        
        with pa.OSFile(str(file_path), 'wb') as sink:
            with pa.orc.ORCWriter(sink, compression=compression) as writer:
                writer.write(table)
        
        return [str(file_path)]
    
    async def _export_tfrecord(self, df: pd.DataFrame, base_path: Path, 
                              config: ExportConfig) -> List[str]:
        """Export to TensorFlow TFRecord format"""
        import tensorflow as tf
        
        file_path = base_path / "data.tfrecord"
        
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        
        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        
        with tf.io.TFRecordWriter(str(file_path)) as writer:
            for _, row in df.iterrows():
                feature = {}
                
                for col, value in row.items():
                    if pd.isna(value):
                        continue
                    
                    if isinstance(value, (int, np.integer)):
                        feature[col] = _int64_feature(int(value))
                    elif isinstance(value, (float, np.floating)):
                        feature[col] = _float_feature(float(value))
                    else:
                        feature[col] = _bytes_feature(str(value).encode())
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        
        return [str(file_path)]
    
    async def _compress_file(self, file_path: str, compression: CompressionType) -> str:
        """Compress file using specified algorithm"""
        if compression == CompressionType.NONE:
            return file_path
        
        handler = self.compression_handlers.get(compression)
        if handler:
            return await handler(file_path)
        
        return file_path
    
    async def _compress_gzip(self, file_path: str) -> str:
        """Compress using gzip"""
        import gzip
        
        output_path = f"{file_path}.gz"
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return output_path
    
    async def _compress_snappy(self, file_path: str) -> str:
        """Compress using Snappy"""
        import snappy
        
        output_path = f"{file_path}.snappy"
        
        with open(file_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(snappy.compress(f_in.read()))
        
        return output_path
    
    async def _compress_lz4(self, file_path: str) -> str:
        """Compress using LZ4"""
        output_path = f"{file_path}.lz4"
        
        with open(file_path, 'rb') as f_in:
            with lz4.frame.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return output_path
    
    async def _compress_zstd(self, file_path: str) -> str:
        """Compress using Zstandard"""
        output_path = f"{file_path}.zst"
        
        cctx = zstd.ZstdCompressor(level=3)
        
        with open(file_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(cctx.compress(f_in.read()))
        
        return output_path
    
    async def _compress_brotli(self, file_path: str) -> str:
        """Compress using Brotli"""
        import brotli
        
        output_path = f"{file_path}.br"
        
        with open(file_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(brotli.compress(f_in.read()))
        
        return output_path
    
    async def _upload_to_storage(self, file_paths: List[str], config: ExportConfig):
        """Upload files to configured storage backend"""
        if config.storage == StorageBackend.S3:
            await self._upload_to_s3(file_paths, config)
        elif config.storage == StorageBackend.GCS:
            await self._upload_to_gcs(file_paths, config)
        # Add other storage backends as needed
    
    async def _upload_to_s3(self, file_paths: List[str], config: ExportConfig):
        """Upload files to Amazon S3"""
        s3_client = boto3.client('s3')
        bucket = os.environ.get('S3_BUCKET', 'arbitrage-exports')
        
        for file_path in file_paths:
            key = f"{config.name}/{os.path.basename(file_path)}"
            
            s3_client.upload_file(
                file_path,
                bucket,
                key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'StorageClass': 'INTELLIGENT_TIERING'
                }
            )
            
            logger.info(f"Uploaded {file_path} to s3://{bucket}/{key}")
    
    async def _upload_to_gcs(self, file_paths: List[str], config: ExportConfig):
        """Upload files to Google Cloud Storage"""
        client = gcs.Client()
        bucket = client.bucket(os.environ.get('GCS_BUCKET', 'arbitrage-exports'))
        
        for file_path in file_paths:
            blob_name = f"{config.name}/{os.path.basename(file_path)}"
            blob = bucket.blob(blob_name)
            
            blob.upload_from_filename(file_path)
            
            logger.info(f"Uploaded {file_path} to gs://{bucket.name}/{blob_name}")
    
    def _generate_job_id(self, config: ExportConfig) -> str:
        """Generate unique job ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        config_hash = hashlib.md5(f"{config.name}{config.query}".encode()).hexdigest()[:8]
        return f"{config.name}_{timestamp}_{config_hash}"

class ContinuousExportScheduler:
    """Manages continuous data export scheduling"""
    
    def __init__(self, clickhouse_client: ClickHouseClient):
        self.clickhouse_client = clickhouse_client
        self.exporter = DataExporter(clickhouse_client)
        self.scheduler = AsyncIOScheduler()
        self.export_configs: Dict[str, ExportConfig] = {}
        self.job_history: Dict[str, List[ExportJob]] = {}
        self.redis_client = None
        self.kafka_producer = None
        
        # Default export configurations
        self._setup_default_exports()
    
    def _setup_default_exports(self):
        """Setup default export configurations"""
        
        # Hourly arbitrage transactions export
        self.export_configs['hourly_arbitrage'] = ExportConfig(
            name="hourly_arbitrage",
            query="""
                SELECT * FROM arbitrage_transactions 
                WHERE block_timestamp >= now() - INTERVAL 1 HOUR
                ORDER BY block_timestamp DESC
            """,
            format=ExportFormat.PARQUET,
            compression=CompressionType.SNAPPY,
            schedule="0 * * * *",  # Every hour
            partition_by=["toDate(block_timestamp)"],
            retention_days=30
        )
        
        # Daily market snapshots export
        self.export_configs['daily_snapshots'] = ExportConfig(
            name="daily_snapshots",
            query="""
                SELECT * FROM market_snapshots 
                WHERE snapshot_time >= today()
                ORDER BY snapshot_time DESC
            """,
            format=ExportFormat.PARQUET,
            compression=CompressionType.ZSTD,
            schedule="0 0 * * *",  # Daily at midnight
            retention_days=90
        )
        
        # Weekly ML training data export
        self.export_configs['ml_training_data'] = ExportConfig(
            name="ml_training_data",
            query="""
                SELECT 
                    t.*,
                    r.frontrun_probability,
                    r.slippage_risk,
                    r.overall_risk_level
                FROM arbitrage_transactions t
                LEFT JOIN risk_metrics r ON t.signature = r.transaction_signature
                WHERE t.block_timestamp >= now() - INTERVAL 7 DAY
            """,
            format=ExportFormat.TFRECORD,
            compression=CompressionType.NONE,
            schedule="0 0 * * 0",  # Weekly on Sunday
            transformations=["normalize_timestamps", "calculate_statistics", "optimize_dtypes"],
            retention_days=180
        )
        
        # Real-time high-value arbitrage export
        self.export_configs['high_value_realtime'] = ExportConfig(
            name="high_value_realtime",
            query="""
                SELECT * FROM arbitrage_transactions 
                WHERE net_profit > 10000000
                AND block_timestamp >= now() - INTERVAL 15 MINUTE
            """,
            format=ExportFormat.JSONL,
            compression=CompressionType.LZ4,
            schedule="*/15 * * * *",  # Every 15 minutes
            retention_days=7,
            notification_webhook="http://alerts-webhook:8080/high-value"
        )
    
    async def start(self):
        """Start the continuous export scheduler"""
        logger.info("Starting continuous export scheduler...")
        
        # Initialize connections
        await self._initialize_connections()
        
        # Schedule all exports
        for config_name, config in self.export_configs.items():
            self._schedule_export(config)
        
        # Start scheduler
        self.scheduler.start()
        
        # Start cleanup task
        self.scheduler.add_job(
            self._cleanup_old_exports,
            trigger=CronTrigger(hour=2, minute=0),  # Daily at 2 AM
            id='cleanup_old_exports',
            name='Cleanup old exports'
        )
        
        # Start monitoring task
        asyncio.create_task(self._monitor_exports())
        
        logger.info(f"Scheduled {len(self.export_configs)} export jobs")
    
    async def _initialize_connections(self):
        """Initialize external connections"""
        try:
            self.redis_client = await aioredis.create_redis_pool(
                'redis://redis:6390',
                minsize=2,
                maxsize=10
            )
            
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers='kafka:9092',
                value_serializer=lambda v: json.dumps(v).encode()
            )
            await self.kafka_producer.start()
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
    
    def _schedule_export(self, config: ExportConfig):
        """Schedule an export job"""
        trigger = CronTrigger.from_crontab(config.schedule)
        
        self.scheduler.add_job(
            self._execute_export,
            trigger=trigger,
            args=[config],
            id=f"export_{config.name}",
            name=f"Export {config.name}",
            misfire_grace_time=300  # 5 minutes grace period
        )
        
        logger.info(f"Scheduled export '{config.name}' with schedule: {config.schedule}")
    
    async def _execute_export(self, config: ExportConfig):
        """Execute a scheduled export"""
        logger.info(f"Starting export: {config.name}")
        
        try:
            # Execute export
            job = await self.exporter.export(config)
            
            # Store job history
            if config.name not in self.job_history:
                self.job_history[config.name] = []
            
            self.job_history[config.name].append(job)
            
            # Keep only last 100 jobs
            if len(self.job_history[config.name]) > 100:
                self.job_history[config.name] = self.job_history[config.name][-100:]
            
            # Send notification if configured
            if config.notification_webhook and job.status == "completed":
                await self._send_notification(config.notification_webhook, job)
            
            # Store job metadata in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    f"export:job:{job.job_id}",
                    86400,  # 24 hour TTL
                    json.dumps({
                        'job_id': job.job_id,
                        'status': job.status,
                        'rows_exported': job.rows_exported,
                        'file_size_bytes': job.file_size_bytes,
                        'file_paths': job.file_paths,
                        'start_time': job.start_time.isoformat() if job.start_time else None,
                        'end_time': job.end_time.isoformat() if job.end_time else None
                    })
                )
            
            # Send metrics to Kafka
            if self.kafka_producer:
                await self.kafka_producer.send(
                    'export-metrics',
                    {
                        'job_id': job.job_id,
                        'config_name': config.name,
                        'status': job.status,
                        'rows_exported': job.rows_exported,
                        'file_size_bytes': job.file_size_bytes,
                        'duration_seconds': (job.end_time - job.start_time).total_seconds() if job.start_time and job.end_time else 0
                    }
                )
            
        except Exception as e:
            logger.error(f"Export '{config.name}' failed: {e}")
    
    async def _send_notification(self, webhook_url: str, job: ExportJob):
        """Send export completion notification"""
        import httpx
        
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    webhook_url,
                    json={
                        'job_id': job.job_id,
                        'status': job.status,
                        'rows_exported': job.rows_exported,
                        'file_size_mb': job.file_size_bytes / (1024 * 1024),
                        'file_paths': job.file_paths,
                        'timestamp': datetime.utcnow().isoformat()
                    },
                    timeout=10.0
                )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    async def _cleanup_old_exports(self):
        """Clean up old export files based on retention policy"""
        logger.info("Starting cleanup of old exports...")
        
        for config_name, config in self.export_configs.items():
            try:
                base_path = Path(config.storage_path) / config.name
                
                if not base_path.exists():
                    continue
                
                cutoff_date = datetime.utcnow() - timedelta(days=config.retention_days)
                
                # Find old directories
                for export_dir in base_path.iterdir():
                    if export_dir.is_dir():
                        # Parse timestamp from directory name
                        try:
                            dir_timestamp = datetime.strptime(export_dir.name, '%Y%m%d_%H%M%S')
                            
                            if dir_timestamp < cutoff_date:
                                # Remove old export
                                shutil.rmtree(export_dir)
                                logger.info(f"Removed old export: {export_dir}")
                        except ValueError:
                            # Skip directories that don't match timestamp format
                            continue
                
            except Exception as e:
                logger.error(f"Cleanup failed for {config_name}: {e}")
    
    async def _monitor_exports(self):
        """Monitor export jobs and alert on failures"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for config_name, jobs in self.job_history.items():
                    if not jobs:
                        continue
                    
                    # Check recent failures
                    recent_jobs = jobs[-10:]  # Last 10 jobs
                    failures = sum(1 for j in recent_jobs if j.status == "failed")
                    
                    if failures > 3:
                        logger.warning(f"High failure rate for {config_name}: {failures}/10 jobs failed")
                        
                        # Send alert
                        if self.kafka_producer:
                            await self.kafka_producer.send(
                                'system-alerts',
                                {
                                    'type': 'export_failure',
                                    'config': config_name,
                                    'failure_rate': failures / 10,
                                    'timestamp': datetime.utcnow().isoformat()
                                }
                            )
                
            except Exception as e:
                logger.error(f"Export monitoring error: {e}")
    
    async def add_export(self, config: ExportConfig):
        """Add a new export configuration"""
        self.export_configs[config.name] = config
        self._schedule_export(config)
        logger.info(f"Added new export: {config.name}")
    
    async def remove_export(self, config_name: str):
        """Remove an export configuration"""
        if config_name in self.export_configs:
            del self.export_configs[config_name]
            self.scheduler.remove_job(f"export_{config_name}")
            logger.info(f"Removed export: {config_name}")
    
    async def get_export_status(self, config_name: str) -> Dict[str, Any]:
        """Get status of export jobs"""
        if config_name not in self.job_history:
            return {'error': 'Export not found'}
        
        jobs = self.job_history[config_name]
        
        if not jobs:
            return {'message': 'No jobs executed yet'}
        
        recent_job = jobs[-1]
        
        return {
            'config_name': config_name,
            'last_run': recent_job.start_time.isoformat() if recent_job.start_time else None,
            'status': recent_job.status,
            'rows_exported': recent_job.rows_exported,
            'file_size_mb': recent_job.file_size_bytes / (1024 * 1024),
            'total_runs': len(jobs),
            'success_rate': sum(1 for j in jobs if j.status == "completed") / len(jobs) * 100
        }
    
    async def shutdown(self):
        """Shutdown the scheduler"""
        logger.info("Shutting down export scheduler...")
        
        self.scheduler.shutdown()
        
        if self.kafka_producer:
            await self.kafka_producer.stop()
        
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        
        logger.info("Export scheduler shutdown complete")

async def main():
    """Main entry point for continuous export"""
    
    # Initialize ClickHouse client
    clickhouse_client = ClickHouseClient(
        host='clickhouse',
        port=9000,
        database='solana_arbitrage'
    )
    
    # Create scheduler
    scheduler = ContinuousExportScheduler(clickhouse_client)
    
    # Start scheduler
    await scheduler.start()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(60)
            
            # Print status periodically
            for config_name in scheduler.export_configs:
                status = await scheduler.get_export_status(config_name)
                if 'last_run' in status:
                    logger.info(f"Export {config_name}: {status['status']}, Success rate: {status.get('success_rate', 0):.1f}%")
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(main())