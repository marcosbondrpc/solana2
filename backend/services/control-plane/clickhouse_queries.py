"""
ClickHouse Query Module - Ultra-High-Performance Data Retrieval
Optimized queries for sub-millisecond latency at scale
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseError
import asynch
from asynch import connect

# ClickHouse connection pool
_ch_pool = None
_ch_lock = asyncio.Lock()


async def get_clickhouse_pool():
    """Get or create async ClickHouse connection pool"""
    global _ch_pool
    
    async with _ch_lock:
        if _ch_pool is None:
            _ch_pool = await connect(
                host=os.getenv("CLICKHOUSE_HOST", "localhost"),
                port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
                database=os.getenv("CLICKHOUSE_DATABASE", "mev"),
                user=os.getenv("CLICKHOUSE_USER", "default"),
                password=os.getenv("CLICKHOUSE_PASSWORD", ""),
                # Performance settings
                settings={
                    "max_threads": 8,
                    "max_memory_usage": 10000000000,  # 10GB
                    "max_block_size": 65536,
                    "max_execution_time": 10,
                    "enable_http_compression": 1,
                    "distributed_product_mode": "global",
                    "compile_expressions": 1,
                    "min_count_to_compile_expression": 3,
                }
            )
    
    return _ch_pool


class MEVQueries:
    """High-performance queries for MEV data"""
    
    @staticmethod
    async def get_recent_opportunities(
        limit: int = 1000,
        time_window_minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent MEV opportunities from ClickHouse"""
        
        pool = await get_clickhouse_pool()
        
        query = """
        SELECT 
            opportunity_id,
            opportunity_type,
            profit_estimate,
            confidence,
            gas_estimate,
            timestamp,
            route,
            decision_dna,
            toFloat64(profit_estimate - gas_estimate) as net_profit
        FROM mev.opportunities
        WHERE timestamp >= now() - INTERVAL {minutes} MINUTE
        ORDER BY net_profit DESC
        LIMIT {limit}
        FORMAT JSON
        """
        
        async with pool.cursor() as cursor:
            await cursor.execute(
                query.format(minutes=time_window_minutes, limit=limit)
            )
            result = await cursor.fetchall()
            
        return result
    
    @staticmethod
    async def get_execution_stats(
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get execution statistics with percentiles"""
        
        pool = await get_clickhouse_pool()
        
        query = """
        SELECT
            count() as total_executions,
            countIf(status = 'success') as successful,
            countIf(status = 'failed') as failed,
            avg(profit_actual) as avg_profit,
            sum(profit_actual) as total_profit,
            quantile(0.5)(latency_ms) as p50_latency,
            quantile(0.95)(latency_ms) as p95_latency,
            quantile(0.99)(latency_ms) as p99_latency,
            min(latency_ms) as min_latency,
            max(latency_ms) as max_latency,
            uniqExact(opportunity_type) as unique_types,
            avgIf(profit_actual, status = 'success') as avg_profit_success
        FROM mev.executions
        WHERE timestamp >= now() - INTERVAL {hours} HOUR
        FORMAT JSON
        """
        
        async with pool.cursor() as cursor:
            await cursor.execute(query.format(hours=time_window_hours))
            result = await cursor.fetchone()
            
        return result
    
    @staticmethod
    async def get_bandit_performance(
        bandit_type: str = "all"
    ) -> List[Dict[str, Any]]:
        """Get Thompson Sampling bandit performance metrics"""
        
        pool = await get_clickhouse_pool()
        
        if bandit_type == "all":
            type_filter = "1=1"
        else:
            type_filter = f"bandit_type = '{bandit_type}'"
        
        query = f"""
        SELECT
            bandit_type,
            arm_id,
            count() as total_pulls,
            avg(reward) as avg_reward,
            sum(reward) as total_reward,
            stddevPop(reward) as reward_stddev,
            max(timestamp) as last_update,
            quantile(0.25)(reward) as q25_reward,
            quantile(0.75)(reward) as q75_reward
        FROM mev.bandit_results
        WHERE {type_filter}
            AND timestamp >= now() - INTERVAL 7 DAY
        GROUP BY bandit_type, arm_id
        ORDER BY bandit_type, avg_reward DESC
        FORMAT JSON
        """
        
        async with pool.cursor() as cursor:
            await cursor.execute(query)
            result = await cursor.fetchall()
            
        return result
    
    @staticmethod
    async def get_pool_liquidity(
        pools: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get current pool liquidity and volume data"""
        
        pool = await get_clickhouse_pool()
        
        if pools:
            pool_filter = f"pool_address IN ({','.join(['%s'] * len(pools))})"
            params = pools
        else:
            pool_filter = "1=1"
            params = []
        
        query = f"""
        SELECT
            pool_address,
            token_a,
            token_b,
            last_value(reserve_a) as current_reserve_a,
            last_value(reserve_b) as current_reserve_b,
            sum(volume_24h) as total_volume_24h,
            avg(fee_rate) as avg_fee_rate,
            count() as update_count,
            max(timestamp) as last_update
        FROM mev.pool_states
        WHERE {pool_filter}
            AND timestamp >= now() - INTERVAL 1 HOUR
        GROUP BY pool_address, token_a, token_b
        ORDER BY total_volume_24h DESC
        FORMAT JSON
        """
        
        async with pool.cursor() as cursor:
            await cursor.execute(query, params)
            result = await cursor.fetchall()
            
        return result
    
    @staticmethod
    async def get_bundle_land_rate(
        time_window_hours: int = 24,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get Jito bundle landing statistics"""
        
        pool = await get_clickhouse_pool()
        
        region_filter = f"AND region = '{region}'" if region else ""
        
        query = f"""
        SELECT
            count() as total_bundles,
            countIf(landed = 1) as landed_bundles,
            countIf(landed = 1) / count() as land_rate,
            avg(tip_lamports) as avg_tip,
            quantile(0.5)(tip_lamports) as median_tip,
            max(tip_lamports) as max_tip,
            avgIf(profit_actual, landed = 1) as avg_profit_landed,
            sumIf(profit_actual, landed = 1) as total_profit_landed
        FROM mev.bundle_submissions
        WHERE timestamp >= now() - INTERVAL {time_window_hours} HOUR
            {region_filter}
        FORMAT JSON
        """
        
        async with pool.cursor() as cursor:
            await cursor.execute(query)
            result = await cursor.fetchone()
            
        return result
    
    @staticmethod
    async def get_decision_lineage(
        decision_dna: str
    ) -> List[Dict[str, Any]]:
        """Trace decision lineage through DNA fingerprints"""
        
        pool = await get_clickhouse_pool()
        
        query = """
        WITH RECURSIVE lineage AS (
            SELECT 
                decision_dna,
                parent_dna,
                module,
                action,
                timestamp,
                metadata,
                1 as depth
            FROM mev.decision_lineage
            WHERE decision_dna = %s
            
            UNION ALL
            
            SELECT
                dl.decision_dna,
                dl.parent_dna,
                dl.module,
                dl.action,
                dl.timestamp,
                dl.metadata,
                l.depth + 1
            FROM mev.decision_lineage dl
            JOIN lineage l ON dl.decision_dna = l.parent_dna
            WHERE l.depth < 10
        )
        SELECT * FROM lineage
        ORDER BY depth, timestamp
        FORMAT JSON
        """
        
        async with pool.cursor() as cursor:
            await cursor.execute(query, (decision_dna,))
            result = await cursor.fetchall()
            
        return result
    
    @staticmethod
    async def get_performance_heatmap(
        granularity_minutes: int = 5,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get performance heatmap data for visualization"""
        
        pool = await get_clickhouse_pool()
        
        query = """
        SELECT
            toStartOfInterval(timestamp, INTERVAL {granularity} MINUTE) as time_bucket,
            opportunity_type,
            count() as opportunity_count,
            avg(profit_estimate) as avg_profit_estimate,
            countIf(executed = 1) as executed_count,
            avgIf(profit_actual, executed = 1) as avg_profit_actual,
            avgIf(latency_ms, executed = 1) as avg_latency,
            maxIf(latency_ms, executed = 1) as max_latency
        FROM mev.opportunity_lifecycle
        WHERE timestamp >= now() - INTERVAL {hours} HOUR
        GROUP BY time_bucket, opportunity_type
        ORDER BY time_bucket DESC, opportunity_count DESC
        FORMAT JSON
        """
        
        async with pool.cursor() as cursor:
            await cursor.execute(
                query.format(
                    granularity=granularity_minutes,
                    hours=lookback_hours
                )
            )
            result = await cursor.fetchall()
            
        return result
    
    @staticmethod
    async def insert_opportunity(
        opportunity: Dict[str, Any]
    ) -> bool:
        """Insert new opportunity into ClickHouse"""
        
        pool = await get_clickhouse_pool()
        
        query = """
        INSERT INTO mev.opportunities (
            opportunity_id,
            opportunity_type,
            profit_estimate,
            confidence,
            gas_estimate,
            timestamp,
            route,
            decision_dna,
            metadata
        ) VALUES
        """
        
        try:
            async with pool.cursor() as cursor:
                await cursor.execute(
                    query,
                    (
                        opportunity["id"],
                        opportunity["type"],
                        opportunity["profit_estimate"],
                        opportunity["confidence"],
                        opportunity["gas_estimate"],
                        datetime.utcnow(),
                        json.dumps(opportunity["route"]),
                        opportunity["dna_fingerprint"],
                        json.dumps(opportunity.get("metadata", {}))
                    )
                )
            return True
        except ClickHouseError as e:
            print(f"Failed to insert opportunity: {e}")
            return False
    
    @staticmethod
    async def insert_execution(
        execution: Dict[str, Any]
    ) -> bool:
        """Insert execution result into ClickHouse"""
        
        pool = await get_clickhouse_pool()
        
        query = """
        INSERT INTO mev.executions (
            execution_id,
            opportunity_id,
            status,
            profit_actual,
            gas_used,
            latency_ms,
            strategy,
            timestamp,
            metadata
        ) VALUES
        """
        
        try:
            async with pool.cursor() as cursor:
                await cursor.execute(
                    query,
                    (
                        execution["id"],
                        execution["opportunity_id"],
                        execution["status"],
                        execution.get("profit_actual", 0),
                        execution.get("gas_used", 0),
                        execution.get("latency_ms", 0),
                        execution.get("strategy", ""),
                        datetime.utcnow(),
                        json.dumps(execution.get("metadata", {}))
                    )
                )
            return True
        except ClickHouseError as e:
            print(f"Failed to insert execution: {e}")
            return False
    
    @staticmethod
    async def get_arbitrage_cycles(
        min_profit_ratio: float = 1.001,
        max_hops: int = 4
    ) -> List[Dict[str, Any]]:
        """Find arbitrage cycles using graph analysis"""
        
        pool = await get_clickhouse_pool()
        
        query = """
        WITH cycles AS (
            SELECT
                arrayJoin(route) as hop,
                opportunity_id,
                profit_estimate,
                count() OVER (PARTITION BY opportunity_id) as hop_count
            FROM mev.opportunities
            WHERE opportunity_type = 'arbitrage'
                AND timestamp >= now() - INTERVAL 1 HOUR
                AND profit_estimate / gas_estimate > {min_ratio}
        )
        SELECT
            opportunity_id,
            groupArray(hop) as cycle_route,
            max(profit_estimate) as max_profit,
            hop_count
        FROM cycles
        WHERE hop_count <= {max_hops}
        GROUP BY opportunity_id, hop_count
        ORDER BY max_profit DESC
        LIMIT 100
        FORMAT JSON
        """
        
        async with pool.cursor() as cursor:
            await cursor.execute(
                query.format(min_ratio=min_profit_ratio, max_hops=max_hops)
            )
            result = await cursor.fetchall()
            
        return result


class MarketDataQueries:
    """Queries for market data and analytics"""
    
    @staticmethod
    async def get_price_impact(
        pool_address: str,
        amount_in: float,
        token_in: str
    ) -> Dict[str, Any]:
        """Calculate price impact for a given trade"""
        
        pool = await get_clickhouse_pool()
        
        query = """
        SELECT
            pool_address,
            token_a,
            token_b,
            last_value(reserve_a) as reserve_a,
            last_value(reserve_b) as reserve_b,
            avg(fee_rate) as fee_rate
        FROM mev.pool_states
        WHERE pool_address = %s
            AND timestamp >= now() - INTERVAL 1 MINUTE
        GROUP BY pool_address, token_a, token_b
        LIMIT 1
        FORMAT JSON
        """
        
        async with pool.cursor() as cursor:
            await cursor.execute(query, (pool_address,))
            pool_data = await cursor.fetchone()
            
        if not pool_data:
            return {"error": "Pool not found"}
        
        # Calculate price impact
        if token_in == pool_data["token_a"]:
            reserve_in = pool_data["reserve_a"]
            reserve_out = pool_data["reserve_b"]
        else:
            reserve_in = pool_data["reserve_b"]
            reserve_out = pool_data["reserve_a"]
        
        amount_out = (amount_in * reserve_out) / (reserve_in + amount_in)
        spot_price = reserve_out / reserve_in
        execution_price = amount_out / amount_in
        price_impact = abs(1 - execution_price / spot_price)
        
        return {
            "pool_address": pool_address,
            "amount_in": amount_in,
            "amount_out": amount_out,
            "spot_price": spot_price,
            "execution_price": execution_price,
            "price_impact": price_impact,
            "fee_rate": pool_data["fee_rate"]
        }
    
    @staticmethod
    async def get_mempool_stats() -> Dict[str, Any]:
        """Get current mempool statistics"""
        
        pool = await get_clickhouse_pool()
        
        query = """
        SELECT
            count() as pending_transactions,
            avg(priority_fee) as avg_priority_fee,
            quantile(0.5)(priority_fee) as median_priority_fee,
            quantile(0.95)(priority_fee) as p95_priority_fee,
            max(priority_fee) as max_priority_fee,
            countIf(transaction_type = 'swap') as swap_count,
            countIf(transaction_type = 'add_liquidity') as add_liquidity_count,
            countIf(transaction_type = 'remove_liquidity') as remove_liquidity_count,
            uniqExact(sender) as unique_senders
        FROM mev.mempool
        WHERE timestamp >= now() - INTERVAL 10 SECOND
        FORMAT JSON
        """
        
        async with pool.cursor() as cursor:
            await cursor.execute(query)
            result = await cursor.fetchone()
            
        return result


# Optimized table creation queries
CLICKHOUSE_TABLES = """
-- MEV Opportunities table with optimized storage
CREATE TABLE IF NOT EXISTS mev.opportunities (
    opportunity_id String,
    opportunity_type LowCardinality(String),
    profit_estimate Float64,
    confidence Float32,
    gas_estimate Float64,
    timestamp DateTime64(3),
    route String CODEC(ZSTD(3)),
    decision_dna FixedString(64),
    metadata String CODEC(ZSTD(3)),
    
    INDEX idx_profit profit_estimate TYPE minmax GRANULARITY 1,
    INDEX idx_type opportunity_type TYPE set(100) GRANULARITY 1,
    INDEX idx_dna decision_dna TYPE bloom_filter(0.01) GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (opportunity_type, timestamp, opportunity_id)
TTL timestamp + INTERVAL 30 DAY TO DISK 's3'
SETTINGS index_granularity = 8192;

-- Executions table with columnar compression
CREATE TABLE IF NOT EXISTS mev.executions (
    execution_id String,
    opportunity_id String,
    status LowCardinality(String),
    profit_actual Float64,
    gas_used Float64,
    latency_ms Float32,
    strategy LowCardinality(String),
    timestamp DateTime64(3),
    metadata String CODEC(ZSTD(3)),
    
    INDEX idx_status status TYPE set(10) GRANULARITY 1,
    INDEX idx_latency latency_ms TYPE minmax GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (status, timestamp, execution_id)
TTL timestamp + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- Bandit results for Thompson Sampling
CREATE TABLE IF NOT EXISTS mev.bandit_results (
    bandit_type LowCardinality(String),
    arm_id UInt8,
    reward Float32,
    timestamp DateTime64(3),
    context String CODEC(ZSTD(3))
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (bandit_type, arm_id, timestamp)
TTL timestamp + INTERVAL 7 DAY
SETTINGS index_granularity = 8192;

-- Bundle submissions tracking
CREATE TABLE IF NOT EXISTS mev.bundle_submissions (
    bundle_id String,
    tip_lamports UInt64,
    landed UInt8,
    profit_actual Float64,
    region LowCardinality(String),
    timestamp DateTime64(3),
    transactions Array(String),
    
    INDEX idx_landed landed TYPE set(2) GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (landed, timestamp, bundle_id)
TTL timestamp + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- Decision lineage for DNA tracking
CREATE TABLE IF NOT EXISTS mev.decision_lineage (
    decision_dna FixedString(64),
    parent_dna Nullable(FixedString(64)),
    module LowCardinality(String),
    action LowCardinality(String),
    timestamp DateTime64(3),
    metadata String CODEC(ZSTD(3)),
    
    INDEX idx_dna decision_dna TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_parent parent_dna TYPE bloom_filter(0.01) GRANULARITY 1
) ENGINE = MergeTree()
ORDER BY (decision_dna, timestamp)
TTL timestamp + INTERVAL 180 DAY
SETTINGS index_granularity = 8192;
"""

import os