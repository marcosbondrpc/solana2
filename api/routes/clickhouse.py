"""
ClickHouse query endpoints - READ ONLY
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException

from models.schemas import (
    ClickHouseQueryRequest,
    ClickHouseQueryResponse,
    UserRole
)
from security.auth import require_role, get_current_user, TokenData, check_rate_limit
from services.clickhouse_client import get_clickhouse_pool

router = APIRouter()

# Whitelisted table patterns for security
ALLOWED_TABLES = [
    "mev.arbitrage_alerts",
    "mev.sandwich_alerts",
    "mev.mev_opportunities",
    "mev.system_metrics",
    "mev.thompson_stats",
    "mev.model_deployments",
    "mev.decision_lineage",
    "mev.control_acks",
    "system.query_log",
    "system.metrics",
    "system.parts"
]

# Query templates for common operations
QUERY_TEMPLATES = {
    "recent_arbitrage": """
        SELECT * FROM mev.arbitrage_alerts 
        WHERE detected_at >= now() - INTERVAL {hours} HOUR 
        ORDER BY detected_at DESC 
        LIMIT {limit}
    """,
    "top_profitable": """
        SELECT 
            date(detected_at) as date,
            count() as opportunities,
            avg(roi_pct) as avg_roi,
            max(est_profit) as max_profit,
            sum(est_profit) as total_profit
        FROM mev.arbitrage_alerts
        WHERE detected_at >= now() - INTERVAL {days} DAY
        GROUP BY date
        ORDER BY date DESC
    """,
    "sandwich_summary": """
        SELECT 
            dex,
            count() as attacks,
            avg(victim_loss) as avg_loss,
            sum(victim_loss) as total_loss,
            avg(attacker_profit) as avg_profit
        FROM mev.sandwich_alerts
        WHERE detected_at >= now() - INTERVAL {hours} HOUR
        GROUP BY dex
        ORDER BY attacks DESC
    """,
    "system_performance": """
        SELECT 
            toStartOfMinute(timestamp) as minute,
            avg(latency_p50_ms) as p50_ms,
            avg(latency_p99_ms) as p99_ms,
            avg(bundle_land_rate) as land_rate,
            avg(ingestion_rate) as ingestion_rate
        FROM mev.system_metrics
        WHERE timestamp >= now() - INTERVAL {minutes} MINUTE
        GROUP BY minute
        ORDER BY minute DESC
    """
}


def validate_table_access(query: str) -> bool:
    """Validate that query only accesses allowed tables"""
    query_lower = query.lower()
    
    # Check if any allowed table is referenced
    has_allowed = any(
        table.lower() in query_lower 
        for table in ALLOWED_TABLES
    )
    
    # Check for system tables that might be dangerous
    dangerous_patterns = [
        "system.users",
        "system.grants",
        "system.passwords",
        "system.settings"
    ]
    
    has_dangerous = any(
        pattern in query_lower 
        for pattern in dangerous_patterns
    )
    
    return has_allowed and not has_dangerous


@router.post("/query", response_model=ClickHouseQueryResponse)
async def execute_query(
    request: ClickHouseQueryRequest,
    current_user: TokenData = Depends(check_rate_limit)
) -> ClickHouseQueryResponse:
    """
    Execute read-only ClickHouse query
    Requires ANALYST role or higher
    """
    
    # Additional role check
    if current_user.role not in [UserRole.ANALYST, UserRole.OPERATOR, UserRole.ML_ENGINEER, UserRole.ADMIN]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Validate table access
    if not validate_table_access(request.query):
        raise HTTPException(
            status_code=403,
            detail="Query accesses restricted tables"
        )
    
    # Apply role-based limits
    from security.policy import get_query_limits
    limits = get_query_limits(current_user.role)
    
    if request.max_rows > limits["max_rows"]:
        request.max_rows = limits["max_rows"]
    
    if request.timeout_seconds > limits["timeout_seconds"]:
        request.timeout_seconds = limits["timeout_seconds"]
    
    # Execute query
    pool = await get_clickhouse_pool()
    
    try:
        # Add LIMIT if not present
        query = request.query.strip()
        if "limit" not in query.lower():
            query = f"{query} LIMIT {request.max_rows}"
        
        # Execute with timeout
        import asyncio
        data, metadata = await asyncio.wait_for(
            pool.execute_query(
                query,
                params=request.parameters,
                with_column_types=True,
                use_cache=request.query_type != "aggregate"
            ),
            timeout=request.timeout_seconds
        )
        
        # Build response
        response = ClickHouseQueryResponse(
            success=True,
            data=data,
            columns=metadata.get("columns", []),
            row_count=len(data),
            execution_time_ms=metadata.get("stats", {}).get("elapsed_ms", 0),
            bytes_read=metadata.get("stats", {}).get("bytes_read"),
            rows_read=metadata.get("stats", {}).get("rows_read"),
            cache_hit=False  # Will be updated when cache is implemented
        )
        
        return response
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Query timeout ({request.timeout_seconds}s)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_query_templates(
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> Dict[str, Any]:
    """Get available query templates"""
    
    return {
        "success": True,
        "templates": {
            name: {
                "query": template,
                "parameters": _extract_parameters(template),
                "description": _get_template_description(name)
            }
            for name, template in QUERY_TEMPLATES.items()
        }
    }


@router.post("/template/{template_name}")
async def execute_template(
    template_name: str,
    parameters: Dict[str, Any],
    current_user: TokenData = Depends(check_rate_limit)
) -> ClickHouseQueryResponse:
    """Execute a query template"""
    
    if template_name not in QUERY_TEMPLATES:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Format template with parameters
    try:
        query = QUERY_TEMPLATES[template_name].format(**parameters)
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing parameter: {e}"
        )
    
    # Execute as regular query
    request = ClickHouseQueryRequest(
        query=query,
        query_type="select"
    )
    
    return await execute_query(request, current_user)


@router.get("/tables")
async def list_tables(
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> Dict[str, Any]:
    """List available tables with metadata"""
    
    pool = await get_clickhouse_pool()
    
    tables_info = []
    for table in ALLOWED_TABLES:
        try:
            stats = await pool.get_table_stats(table)
            tables_info.append({
                "name": table,
                "row_count": stats.get("row_count"),
                "size_bytes": stats.get("size_bytes"),
                "partitions": stats.get("partitions"),
                "columns": stats.get("columns", [])
            })
        except Exception:
            continue
    
    return {
        "success": True,
        "tables": tables_info
    }


@router.get("/table/{table_name}/schema")
async def get_table_schema(
    table_name: str,
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> Dict[str, Any]:
    """Get table schema"""
    
    full_table = f"mev.{table_name}"
    if full_table not in ALLOWED_TABLES:
        raise HTTPException(status_code=403, detail="Table access not allowed")
    
    pool = await get_clickhouse_pool()
    
    query = f"""
        SELECT 
            name,
            type,
            default_type,
            default_expression,
            comment
        FROM system.columns
        WHERE database = 'mev' AND table = '{table_name}'
        ORDER BY position
    """
    
    data, _ = await pool.execute_query(query, use_cache=True)
    
    return {
        "success": True,
        "table": table_name,
        "columns": data
    }


@router.get("/table/{table_name}/sample")
async def get_table_sample(
    table_name: str,
    limit: int = 10,
    current_user: TokenData = Depends(require_role(UserRole.ANALYST))
) -> Dict[str, Any]:
    """Get sample data from table"""
    
    full_table = f"mev.{table_name}"
    if full_table not in ALLOWED_TABLES:
        raise HTTPException(status_code=403, detail="Table access not allowed")
    
    if limit > 100:
        limit = 100
    
    pool = await get_clickhouse_pool()
    
    query = f"SELECT * FROM {full_table} LIMIT {limit}"
    
    data, metadata = await pool.execute_query(query, use_cache=True)
    
    return {
        "success": True,
        "table": table_name,
        "data": data,
        "columns": metadata.get("columns", [])
    }


def _extract_parameters(template: str) -> List[str]:
    """Extract parameters from query template"""
    import re
    return list(set(re.findall(r'\{(\w+)\}', template)))


def _get_template_description(name: str) -> str:
    """Get description for template"""
    descriptions = {
        "recent_arbitrage": "Get recent arbitrage opportunities",
        "top_profitable": "Get most profitable opportunities by day",
        "sandwich_summary": "Get sandwich attack summary by DEX",
        "system_performance": "Get system performance metrics"
    }
    return descriptions.get(name, "Query template")