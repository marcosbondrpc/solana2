"""
ClickHouse client with connection pooling and query optimization
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseError
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import json
from contextlib import asynccontextmanager


class ClickHousePool:
    """Connection pool for ClickHouse"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "mev",
        user: str = "default",
        password: str = "",
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        self.config = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "settings": {
                "max_execution_time": 300,
                "max_memory_usage": 10737418240,  # 10GB
                "max_bytes_before_external_group_by": 5368709120,  # 5GB
                "max_threads": 8,
                "distributed_product_mode": "global",
            }
        }
        
        self.pool: List[Client] = []
        self.available: asyncio.Queue = asyncio.Queue()
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.created_connections = 0
        self.lock = asyncio.Lock()
        
        # Query cache
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(seconds=60)
    
    async def initialize(self):
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            conn = Client(**self.config)
            self.pool.append(conn)
            await self.available.put(conn)
            self.created_connections += 1
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool"""
        conn = None
        created_new = False
        
        try:
            # Try to get from pool
            try:
                conn = await asyncio.wait_for(self.available.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Create overflow connection if allowed
                async with self.lock:
                    if self.created_connections < self.pool_size + self.max_overflow:
                        conn = Client(**self.config)
                        self.created_connections += 1
                        created_new = True
                    else:
                        # Wait for connection
                        conn = await self.available.get()
            
            yield conn
            
        finally:
            if conn and not created_new:
                # Return to pool
                await self.available.put(conn)
            elif conn and created_new:
                # Close overflow connection
                conn.disconnect()
                async with self.lock:
                    self.created_connections -= 1
    
    def _get_query_hash(self, query: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for query"""
        cache_key = f"{query}:{json.dumps(params, sort_keys=True) if params else ''}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict] = None,
        with_column_types: bool = True,
        use_cache: bool = True,
        cache_ttl: Optional[timedelta] = None
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Execute query with retry logic and caching"""
        
        # Check cache
        if use_cache:
            query_hash = self._get_query_hash(query, params)
            if query_hash in self.query_cache:
                cached_result, cached_time = self.query_cache[query_hash]
                ttl = cache_ttl or self.cache_ttl
                if datetime.now() - cached_time < ttl:
                    return cached_result
        
        async with self.get_connection() as conn:
            try:
                # Execute query in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                if with_column_types:
                    result = await loop.run_in_executor(
                        None,
                        conn.execute,
                        query,
                        params,
                        True
                    )
                    
                    # Parse results
                    data, columns_with_types = result
                    columns = [{"name": col[0], "type": str(col[1])} for col in columns_with_types]
                    
                    # Convert to list of dicts
                    rows = []
                    for row in data:
                        rows.append(dict(zip([col["name"] for col in columns], row)))
                    
                    stats = {
                        "rows_read": conn.last_query.progress.rows,
                        "bytes_read": conn.last_query.progress.bytes,
                        "elapsed_ms": conn.last_query.elapsed * 1000
                    }
                    
                    result = (rows, {"columns": columns, "stats": stats})
                    
                else:
                    data = await loop.run_in_executor(
                        None,
                        conn.execute,
                        query,
                        params
                    )
                    result = (data, {"stats": {}})
                
                # Cache result
                if use_cache:
                    self.query_cache[query_hash] = (result, datetime.now())
                
                return result
                
            except ClickHouseError as e:
                raise Exception(f"ClickHouse query failed: {str(e)}")
    
    async def export_to_parquet(
        self,
        query: str,
        output_path: str,
        params: Optional[Dict] = None,
        chunk_size: int = 100000,
        compression: str = "snappy"
    ) -> Dict[str, Any]:
        """Export query results to Parquet file"""
        
        total_rows = 0
        file_size = 0
        
        async with self.get_connection() as conn:
            # Get total count first
            count_query = f"SELECT count() FROM ({query}) AS subquery"
            loop = asyncio.get_event_loop()
            count_result = await loop.run_in_executor(
                None,
                conn.execute,
                count_query,
                params
            )
            total_count = count_result[0][0] if count_result else 0
            
            # Stream results in chunks
            writer = None
            offset = 0
            
            while offset < total_count:
                chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
                
                # Execute chunk query
                chunk_data, metadata = await self.execute_query(
                    chunk_query,
                    params,
                    with_column_types=True,
                    use_cache=False
                )
                
                if not chunk_data:
                    break
                
                # Convert to pandas DataFrame
                df = pd.DataFrame(chunk_data)
                
                # Convert to PyArrow table
                table = pa.Table.from_pandas(df)
                
                # Write to Parquet
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression=compression
                    )
                
                writer.write_table(table)
                
                total_rows += len(chunk_data)
                offset += chunk_size
            
            if writer:
                writer.close()
                
                # Get file size
                import os
                file_size = os.path.getsize(output_path)
        
        return {
            "rows_exported": total_rows,
            "file_size_bytes": file_size,
            "file_path": output_path,
            "compression": compression
        }
    
    async def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get statistics for a table"""
        
        queries = {
            "row_count": f"SELECT count() FROM {table_name}",
            "size_bytes": f"SELECT sum(bytes) FROM system.parts WHERE table = '{table_name}'",
            "partitions": f"SELECT count(distinct partition) FROM system.parts WHERE table = '{table_name}'",
            "columns": f"SELECT name, type FROM system.columns WHERE table = '{table_name}'"
        }
        
        stats = {}
        for key, query in queries.items():
            try:
                result, _ = await self.execute_query(query, use_cache=False)
                if key == "columns":
                    stats[key] = result
                else:
                    stats[key] = result[0] if result else None
            except Exception:
                stats[key] = None
        
        return stats
    
    async def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate query syntax without executing"""
        
        # Add EXPLAIN to validate without execution
        explain_query = f"EXPLAIN {query}"
        
        try:
            async with self.get_connection() as conn:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    conn.execute,
                    explain_query
                )
            return True, None
        except Exception as e:
            return False, str(e)
    
    async def close(self):
        """Close all connections in pool"""
        while not self.available.empty():
            try:
                conn = await self.available.get_nowait()
                conn.disconnect()
            except asyncio.QueueEmpty:
                break
        
        for conn in self.pool:
            try:
                conn.disconnect()
            except Exception:
                pass


# Global connection pool instance
clickhouse_pool: Optional[ClickHousePool] = None


async def initialize_clickhouse(
    host: str = "localhost",
    port: int = 8123,
    database: str = "mev",
    user: str = "default",
    password: str = ""
):
    """Initialize global ClickHouse connection pool"""
    global clickhouse_pool
    
    clickhouse_pool = ClickHousePool(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    
    await clickhouse_pool.initialize()


async def get_clickhouse_pool() -> ClickHousePool:
    """Get global ClickHouse pool instance"""
    if clickhouse_pool is None:
        await initialize_clickhouse()
    return clickhouse_pool