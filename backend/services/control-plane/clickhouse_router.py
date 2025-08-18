"""
ClickHouse Router: Ultra-High-Performance Database API
Institutional-grade analytics with nanosecond precision
"""

import os
import time
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import clickhouse_connect

from deps import User, get_current_user, require_permission
from .clickhouse_queries import MEVQueries, MarketDataQueries


router = APIRouter()

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
                "max_execution_time": 30,  # 30 seconds
                "send_progress_in_http_headers": 1,
                "enable_http_compression": 1
            }
        )
    return ch_client


class QueryRequest(BaseModel):
    """Custom query request"""
    query: str = Field(..., description="ClickHouse SQL query")
    format: str = Field("JSON", description="Output format: JSON, CSV, Parquet")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Query settings")


class TableInfo(BaseModel):
    """Table information model"""
    name: str
    engine: str
    total_rows: int
    total_bytes: int
    columns: List[Dict[str, str]]


@router.get("/query", dependencies=[Depends(require_permission("clickhouse:read"))])
async def execute_query(
    query: str = Query(..., description="ClickHouse query"),
    format: str = Query("JSON", description="Output format"),
    limit: int = Query(10000, le=100000),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Execute custom ClickHouse query"""
    
    # Security check - prevent dangerous queries
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'INSERT', 'UPDATE', 'ALTER']
    query_upper = query.upper()
    
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            raise HTTPException(
                status_code=403, 
                detail=f"Query contains forbidden keyword: {keyword}"
            )
    
    # Add limit if not present
    if 'LIMIT' not in query_upper:
        query = f"{query.rstrip(';')} LIMIT {limit}"
    
    client = get_clickhouse_client()
    
    try:
        start_time = time.perf_counter()
        result = client.query(query)
        execution_time = (time.perf_counter() - start_time) * 1000
        
        if format.upper() == "JSON":
            data = result.result_rows
            columns = [col[0] for col in result.column_types]
            
            # Convert to list of dicts
            rows = []
            for row in data:
                row_dict = {}
                for i, value in enumerate(row):
                    if isinstance(value, datetime):
                        value = value.isoformat()
                    row_dict[columns[i]] = value
                rows.append(row_dict)
            
            return {
                "data": rows,
                "columns": columns,
                "rows_returned": len(rows),
                "execution_time_ms": execution_time,
                "query": query
            }
        else:
            return {
                "data": result.result_rows,
                "execution_time_ms": execution_time,
                "format": format
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query failed: {str(e)}")


@router.post("/query", dependencies=[Depends(require_permission("clickhouse:read"))])
async def execute_query_post(
    request: QueryRequest,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Execute query via POST with advanced options"""
    
    # Security validation
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'INSERT', 'UPDATE', 'ALTER']
    query_upper = request.query.upper()
    
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            raise HTTPException(
                status_code=403, 
                detail=f"Query contains forbidden keyword: {keyword}"
            )
    
    client = get_clickhouse_client()
    
    try:
        start_time = time.perf_counter()
        
        # Apply custom settings
        if request.settings:
            client.command(f"SET {', '.join([f'{k}={v}' for k, v in request.settings.items()])}")
        
        result = client.query(request.query)
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "data": result.result_rows,
            "columns": [col[0] for col in result.column_types],
            "rows_returned": len(result.result_rows),
            "execution_time_ms": execution_time,
            "format": request.format
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query failed: {str(e)}")


@router.get("/opportunities/recent", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_recent_opportunities(
    limit: int = Query(1000, le=10000),
    time_window_minutes: int = Query(5, le=60),
    user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get recent MEV opportunities"""
    
    try:
        opportunities = await MEVQueries.get_recent_opportunities(limit, time_window_minutes)
        return opportunities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch opportunities: {str(e)}")


@router.get("/stats/execution", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_execution_stats(
    time_window_hours: int = Query(24, le=168),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get execution statistics"""
    
    try:
        stats = await MEVQueries.get_execution_stats(time_window_hours)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.get("/bandits/performance", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_bandit_performance(
    bandit_type: str = Query("all"),
    user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get Thompson Sampling bandit performance"""
    
    try:
        performance = await MEVQueries.get_bandit_performance(bandit_type)
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch bandit performance: {str(e)}")


@router.get("/pools/liquidity", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_pool_liquidity(
    pools: Optional[str] = Query(None, description="Comma-separated pool addresses"),
    user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get pool liquidity data"""
    
    try:
        pool_list = pools.split(',') if pools else None
        liquidity = await MEVQueries.get_pool_liquidity(pool_list)
        return liquidity
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch liquidity: {str(e)}")


@router.get("/bundles/land-rate", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_bundle_land_rate(
    time_window_hours: int = Query(24, le=168),
    region: Optional[str] = Query(None),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get Jito bundle landing statistics"""
    
    try:
        land_rate = await MEVQueries.get_bundle_land_rate(time_window_hours, region)
        return land_rate
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch land rate: {str(e)}")


@router.get("/decision/lineage/{decision_dna}", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_decision_lineage(
    decision_dna: str,
    user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Trace decision lineage through DNA fingerprints"""
    
    try:
        lineage = await MEVQueries.get_decision_lineage(decision_dna)
        return lineage
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch lineage: {str(e)}")


@router.get("/heatmap/performance", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_performance_heatmap(
    granularity_minutes: int = Query(5, ge=1, le=60),
    lookback_hours: int = Query(24, le=168),
    user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get performance heatmap data"""
    
    try:
        heatmap = await MEVQueries.get_performance_heatmap(granularity_minutes, lookback_hours)
        return heatmap
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch heatmap: {str(e)}")


@router.get("/arbitrage/cycles", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_arbitrage_cycles(
    min_profit_ratio: float = Query(1.001, ge=1.0),
    max_hops: int = Query(4, ge=2, le=10),
    user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Find arbitrage cycles"""
    
    try:
        cycles = await MEVQueries.get_arbitrage_cycles(min_profit_ratio, max_hops)
        return cycles
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch cycles: {str(e)}")


@router.get("/market/price-impact", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_price_impact(
    pool_address: str = Query(...),
    amount_in: float = Query(..., gt=0),
    token_in: str = Query(...),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Calculate price impact for a trade"""
    
    try:
        impact = await MarketDataQueries.get_price_impact(pool_address, amount_in, token_in)
        return impact
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate price impact: {str(e)}")


@router.get("/market/mempool", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_mempool_stats(
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current mempool statistics"""
    
    try:
        stats = await MarketDataQueries.get_mempool_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch mempool stats: {str(e)}")


@router.get("/tables", dependencies=[Depends(require_permission("clickhouse:read"))])
async def list_tables(
    database: str = Query("mev"),
    user: User = Depends(get_current_user)
) -> List[TableInfo]:
    """List all tables in database"""
    
    client = get_clickhouse_client()
    
    try:
        # Get table list
        tables_query = f"""
        SELECT 
            name,
            engine,
            total_rows,
            total_bytes
        FROM system.tables 
        WHERE database = '{database}'
        ORDER BY name
        """
        
        result = client.query(tables_query)
        tables = []
        
        for row in result.result_rows:
            table_name = row[0]
            
            # Get column information
            columns_query = f"""
            SELECT name, type
            FROM system.columns
            WHERE database = '{database}' AND table = '{table_name}'
            ORDER BY position
            """
            
            columns_result = client.query(columns_query)
            columns = [{"name": col[0], "type": col[1]} for col in columns_result.result_rows]
            
            tables.append(TableInfo(
                name=table_name,
                engine=row[1],
                total_rows=row[2],
                total_bytes=row[3],
                columns=columns
            ))
        
        return tables
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")


@router.get("/schema/{table_name}", dependencies=[Depends(require_permission("clickhouse:read"))])
async def get_table_schema(
    table_name: str,
    database: str = Query("mev"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed table schema"""
    
    client = get_clickhouse_client()
    
    try:
        schema_query = f"""
        SELECT 
            name,
            type,
            default_kind,
            default_expression,
            comment,
            codec_expression,
            is_in_partition_key,
            is_in_sorting_key,
            is_in_primary_key,
            is_in_sampling_key
        FROM system.columns
        WHERE database = '{database}' AND table = '{table_name}'
        ORDER BY position
        """
        
        result = client.query(schema_query)
        
        if not result.result_rows:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found")
        
        columns = []
        for row in result.result_rows:
            columns.append({
                "name": row[0],
                "type": row[1],
                "default_kind": row[2],
                "default_expression": row[3],
                "comment": row[4],
                "codec": row[5],
                "partition_key": row[6],
                "sorting_key": row[7],
                "primary_key": row[8],
                "sampling_key": row[9]
            })
        
        # Get table info
        table_query = f"""
        SELECT 
            engine,
            partition_key,
            sorting_key,
            primary_key,
            sampling_key,
            total_rows,
            total_bytes
        FROM system.tables
        WHERE database = '{database}' AND name = '{table_name}'
        """
        
        table_result = client.query(table_query)
        table_info = table_result.result_rows[0] if table_result.result_rows else None
        
        return {
            "table": table_name,
            "database": database,
            "engine": table_info[0] if table_info else None,
            "partition_key": table_info[1] if table_info else None,
            "sorting_key": table_info[2] if table_info else None,
            "primary_key": table_info[3] if table_info else None,
            "sampling_key": table_info[4] if table_info else None,
            "total_rows": table_info[5] if table_info else 0,
            "total_bytes": table_info[6] if table_info else 0,
            "columns": columns
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")


@router.get("/sample/{table_name}", dependencies=[Depends(require_permission("clickhouse:read"))])
async def sample_table_data(
    table_name: str,
    database: str = Query("mev"),
    limit: int = Query(100, le=1000),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get sample data from table"""
    
    client = get_clickhouse_client()
    
    try:
        sample_query = f"""
        SELECT *
        FROM {database}.{table_name}
        ORDER BY RAND()
        LIMIT {limit}
        """
        
        start_time = time.perf_counter()
        result = client.query(sample_query)
        execution_time = (time.perf_counter() - start_time) * 1000
        
        columns = [col[0] for col in result.column_types]
        
        # Convert to list of dicts
        rows = []
        for row in result.result_rows:
            row_dict = {}
            for i, value in enumerate(row):
                if isinstance(value, datetime):
                    value = value.isoformat()
                row_dict[columns[i]] = value
            rows.append(row_dict)
        
        return {
            "table": table_name,
            "database": database,
            "columns": columns,
            "sample_data": rows,
            "rows_returned": len(rows),
            "execution_time_ms": execution_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sample table: {str(e)}")