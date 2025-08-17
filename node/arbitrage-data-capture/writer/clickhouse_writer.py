"""
Ultra-High-Performance ClickHouse Writer
Bulk inserts with connection pooling and retry logic
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
import numpy as np

from clickhouse_driver import Client
from aiochclient import ChClient
import aiohttp
from asyncio import Queue, QueueFull
import msgpack
import ujson as json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WriterConfig:
    """Configuration for ClickHouse writer"""
    hosts: List[str] = None
    database: str = "arbitrage_mainnet"
    user: str = "default"
    password: str = ""
    
    # Performance settings
    batch_size: int = 10000
    max_batch_wait_ms: int = 100
    compression: str = "lz4"
    async_insert: bool = True
    
    # Connection pooling
    pool_size: int = 20
    max_retries: int = 3
    retry_delay_ms: int = 100
    
    # Buffer settings
    buffer_size: int = 1000000  # 1M records
    flush_interval_ms: int = 1000
    
    def __post_init__(self):
        if not self.hosts:
            self.hosts = ["clickhouse1:9000", "clickhouse2:9000", "clickhouse3:9000"]

class ClickHouseWriter:
    """High-performance writer with batching and compression"""
    
    def __init__(self, config: WriterConfig):
        self.config = config
        self.clients = []
        self.async_client = None
        self.write_buffer = {
            'transactions': [],
            'arbitrage_opportunities': [],
            'market_snapshots': [],
            'risk_metrics': [],
            'performance_metrics': []
        }
        self.buffer_locks = {
            table: asyncio.Lock() for table in self.write_buffer.keys()
        }
        self.flush_tasks = {}
        self.stats = {
            'total_written': 0,
            'total_batches': 0,
            'failed_writes': 0,
            'retry_count': 0,
            'avg_batch_time_ms': 0
        }
    
    async def initialize(self):
        """Initialize connection pool"""
        # Create sync clients for bulk operations
        for host in self.config.hosts:
            client = Client(
                host=host.split(':')[0],
                port=int(host.split(':')[1]) if ':' in host else 9000,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                compression=self.config.compression,
                settings={
                    'async_insert': 1 if self.config.async_insert else 0,
                    'wait_for_async_insert': 0,
                    'async_insert_max_data_size': 10485760,  # 10MB
                    'async_insert_busy_timeout_ms': 100,
                    'max_insert_block_size': 1048576,
                    'insert_quorum': 2,
                    'insert_quorum_parallel': 1,
                }
            )
            self.clients.append(client)
        
        # Create async client for real-time operations
        self.async_client = ChClient(
            session=aiohttp.ClientSession(),
            url=f"http://{self.config.hosts[0].replace('9000', '8123')}",
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            compress_response=True
        )
        
        # Start flush tasks for each table
        for table in self.write_buffer.keys():
            self.flush_tasks[table] = asyncio.create_task(
                self._auto_flush_loop(table)
            )
        
        logger.info(f"ClickHouse writer initialized with {len(self.clients)} clients")
    
    async def write_transaction(self, transaction: Dict[str, Any]):
        """Write transaction to buffer"""
        await self._add_to_buffer('transactions', transaction)
    
    async def write_opportunity(self, opportunity: Dict[str, Any]):
        """Write arbitrage opportunity to buffer"""
        await self._add_to_buffer('arbitrage_opportunities', opportunity)
    
    async def write_market_snapshot(self, snapshot: Dict[str, Any]):
        """Write market snapshot to buffer"""
        await self._add_to_buffer('market_snapshots', snapshot)
    
    async def write_risk_metrics(self, metrics: Dict[str, Any]):
        """Write risk metrics to buffer"""
        await self._add_to_buffer('risk_metrics', metrics)
    
    async def _add_to_buffer(self, table: str, data: Dict[str, Any]):
        """Add data to write buffer with overflow protection"""
        async with self.buffer_locks[table]:
            self.write_buffer[table].append(data)
            
            # Flush if buffer is full
            if len(self.write_buffer[table]) >= self.config.batch_size:
                asyncio.create_task(self._flush_table(table))
    
    async def _auto_flush_loop(self, table: str):
        """Automatically flush buffer at intervals"""
        while True:
            await asyncio.sleep(self.config.flush_interval_ms / 1000)
            
            async with self.buffer_locks[table]:
                if self.write_buffer[table]:
                    asyncio.create_task(self._flush_table(table))
    
    async def _flush_table(self, table: str):
        """Flush buffer for specific table"""
        async with self.buffer_locks[table]:
            if not self.write_buffer[table]:
                return
            
            # Get data and clear buffer
            data = self.write_buffer[table]
            self.write_buffer[table] = []
        
        # Write batch with retry
        success = await self._write_batch_with_retry(table, data)
        
        if not success:
            # Re-add to buffer if failed
            async with self.buffer_locks[table]:
                self.write_buffer[table].extend(data)
    
    async def _write_batch_with_retry(self, table: str, data: List[Dict]) -> bool:
        """Write batch with exponential backoff retry"""
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                # Choose client round-robin
                client = self.clients[attempt % len(self.clients)]
                
                # Prepare batch insert
                success = await self._execute_batch_insert(client, table, data)
                
                if success:
                    # Update stats
                    batch_time = (time.time() - start_time) * 1000
                    self.stats['total_written'] += len(data)
                    self.stats['total_batches'] += 1
                    self.stats['avg_batch_time_ms'] = (
                        (self.stats['avg_batch_time_ms'] * (self.stats['total_batches'] - 1) + batch_time) /
                        self.stats['total_batches']
                    )
                    
                    logger.info(f"Written {len(data)} records to {table} in {batch_time:.2f}ms")
                    return True
                
            except Exception as e:
                logger.error(f"Write attempt {attempt + 1} failed for {table}: {e}")
                self.stats['retry_count'] += 1
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(
                        self.config.retry_delay_ms * (2 ** attempt) / 1000
                    )
        
        self.stats['failed_writes'] += len(data)
        logger.error(f"Failed to write {len(data)} records to {table} after {self.config.max_retries} attempts")
        return False
    
    async def _execute_batch_insert(self, client: Client, table: str, data: List[Dict]) -> bool:
        """Execute batch insert with optimal performance"""
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def insert_batch():
            # Prepare column data
            columns = self._prepare_columns(table, data)
            
            # Execute insert
            query = f"INSERT INTO {table} VALUES"
            client.execute(query, columns, types_check=False)
            
            return True
        
        result = await loop.run_in_executor(None, insert_batch)
        return result
    
    def _prepare_columns(self, table: str, data: List[Dict]) -> List[tuple]:
        """Prepare data for columnar insert"""
        
        if table == 'transactions':
            return self._prepare_transaction_columns(data)
        elif table == 'arbitrage_opportunities':
            return self._prepare_opportunity_columns(data)
        elif table == 'market_snapshots':
            return self._prepare_snapshot_columns(data)
        elif table == 'risk_metrics':
            return self._prepare_risk_columns(data)
        elif table == 'performance_metrics':
            return self._prepare_performance_columns(data)
        else:
            raise ValueError(f"Unknown table: {table}")
    
    def _prepare_transaction_columns(self, data: List[Dict]) -> List[tuple]:
        """Prepare transaction data for insert"""
        rows = []
        
        for tx in data:
            row = (
                tx['signature'],
                tx['block_height'],
                datetime.fromisoformat(tx['block_timestamp']),
                tx['slot'],
                tx.get('fee', 0),
                tx.get('compute_units_used', 0),
                tx.get('priority_fee', 0),
                tx.get('lamports_per_signature', 0),
                tx.get('is_mev_transaction', False),
                tx.get('mev_type', 'arbitrage'),
                tx.get('bundle_id', ''),
                tx['searcher_address'],
                tx.get('profit_amount', 0),
                tx.get('profit_token', ''),
                tx.get('gas_cost', 0),
                tx.get('net_profit', 0),
                tx.get('roi_percentage', 0.0),
                tx.get('dex_count', 0),
                tx.get('hop_count', 0),
                tx.get('path_hash', ''),
                tx.get('dexes', []),
                tx.get('tokens', []),
                tx.get('amounts', []),
                tx.get('slippage_percentage', 0.0),
                tx.get('impermanent_loss', 0.0),
                tx.get('max_drawdown', 0.0),
                tx.get('sharpe_ratio', 0.0),
                tx.get('volatility_score', 0.0),
                tx.get('market_volatility', 0.0),
                tx.get('liquidity_depth', 0),
                tx.get('spread_basis_points', 0),
                tx.get('volume_24h', 0),
                tx.get('execution_time_ms', 0),
                tx.get('simulation_time_ms', 0),
                tx.get('mempool_time_ms', 0),
                tx.get('confirmation_time_ms', 0),
                tx.get('price_momentum', 0.0),
                tx.get('volume_ratio', 0.0),
                tx.get('liquidity_score', 0.0),
                tx.get('market_impact', 0.0),
                tx.get('cross_dex_correlation', 0.0),
                tx.get('program_ids', []),
                tx.get('instruction_count', 0),
                tx.get('cross_program_invocations', 0),
                tx.get('error_code', None),
                tx.get('status', 'success'),
                datetime.utcnow()
            )
            rows.append(row)
        
        return rows
    
    def _prepare_opportunity_columns(self, data: List[Dict]) -> List[tuple]:
        """Prepare opportunity data for insert"""
        rows = []
        
        for opp in data:
            row = (
                opp['opportunity_id'],
                datetime.fromisoformat(opp['detected_at']),
                opp['block_height'],
                opp['opportunity_type'],
                opp['input_token'],
                opp['output_token'],
                opp['input_amount'],
                opp['expected_output'],
                opp['minimum_profit'],
                opp['path_json'],
                opp['dex_sequence'],
                opp['pool_addresses'],
                opp.get('pool_reserves', []),
                opp.get('pool_fees', []),
                opp.get('price_impacts', []),
                opp.get('executed', False),
                opp.get('execution_tx', None),
                opp.get('actual_profit', None),
                opp.get('execution_latency_ms', None),
                opp.get('competing_txs', 0),
                opp.get('frontrun_attempts', 0),
                opp.get('backrun_success', False),
                opp['confidence_score'],
                opp['risk_score'],
                opp['profitability_score']
            )
            rows.append(row)
        
        return rows
    
    def _prepare_snapshot_columns(self, data: List[Dict]) -> List[tuple]:
        """Prepare market snapshot data for insert"""
        rows = []
        
        for snap in data:
            row = (
                datetime.fromisoformat(snap['snapshot_time']),
                snap['dex'],
                snap['pool_address'],
                snap['reserve0'],
                snap['reserve1'],
                snap['total_liquidity'],
                snap['price'],
                snap.get('price_change_1m', 0.0),
                snap.get('price_change_5m', 0.0),
                snap.get('price_change_1h', 0.0),
                snap.get('volume_1m', 0),
                snap.get('volume_5m', 0),
                snap.get('volume_1h', 0),
                snap.get('trade_count_1m', 0),
                snap.get('bid_liquidity', []),
                snap.get('ask_liquidity', []),
                snap.get('spread_bps', 0)
            )
            rows.append(row)
        
        return rows
    
    def _prepare_risk_columns(self, data: List[Dict]) -> List[tuple]:
        """Prepare risk metrics data for insert"""
        # Implementation similar to above
        return []
    
    def _prepare_performance_columns(self, data: List[Dict]) -> List[tuple]:
        """Prepare performance metrics data for insert"""
        # Implementation similar to above
        return []
    
    async def execute_query(self, query: str) -> List[Dict]:
        """Execute read query asynchronously"""
        try:
            result = await self.async_client.execute(query)
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    async def get_stats(self) -> Dict:
        """Get writer statistics"""
        buffer_sizes = {
            table: len(buffer) 
            for table, buffer in self.write_buffer.items()
        }
        
        return {
            **self.stats,
            'buffer_sizes': buffer_sizes,
            'total_buffer_size': sum(buffer_sizes.values()),
            'clients_active': len(self.clients)
        }
    
    async def flush_all(self):
        """Force flush all buffers"""
        tasks = []
        for table in self.write_buffer.keys():
            tasks.append(self._flush_table(table))
        
        await asyncio.gather(*tasks)
        logger.info("All buffers flushed")
    
    async def close(self):
        """Gracefully shutdown writer"""
        # Cancel flush tasks
        for task in self.flush_tasks.values():
            task.cancel()
        
        # Final flush
        await self.flush_all()
        
        # Close connections
        for client in self.clients:
            client.disconnect()
        
        if self.async_client:
            await self.async_client.close()
        
        logger.info(f"Writer closed. Final stats: {self.stats}")

class BulkLoader:
    """Specialized bulk loader for initial data import"""
    
    def __init__(self, config: WriterConfig):
        self.config = config
        self.client = None
    
    async def initialize(self):
        """Initialize bulk loader"""
        self.client = Client(
            host=self.config.hosts[0].split(':')[0],
            database=self.config.database,
            settings={
                'max_insert_block_size': 10485760,  # 10MB blocks
                'max_threads': 16,
                'max_memory_usage': 10737418240,  # 10GB
            }
        )
    
    async def load_csv(self, table: str, csv_path: str, format: str = 'CSVWithNames'):
        """Load CSV file directly into ClickHouse"""
        query = f"""
        INSERT INTO {table}
        FORMAT {format}
        """
        
        with open(csv_path, 'rb') as f:
            self.client.execute(query, f)
        
        logger.info(f"Loaded CSV {csv_path} into {table}")
    
    async def load_parquet(self, table: str, parquet_path: str):
        """Load Parquet file into ClickHouse"""
        query = f"""
        INSERT INTO {table}
        FORMAT Parquet
        """
        
        with open(parquet_path, 'rb') as f:
            self.client.execute(query, f)
        
        logger.info(f"Loaded Parquet {parquet_path} into {table}")

# Optimized writer for different data types
class OptimizedWriter:
    """Specialized writers for different data patterns"""
    
    def __init__(self, config: WriterConfig):
        self.config = config
        self.writer = ClickHouseWriter(config)
        self.transaction_buffer = []
        self.snapshot_buffer = []
    
    async def initialize(self):
        """Initialize optimized writer"""
        await self.writer.initialize()
    
    async def write_high_frequency_data(self, data: List[Dict]):
        """Optimized for high-frequency market data"""
        # Use memory buffer for ultra-high frequency
        self.snapshot_buffer.extend(data)
        
        if len(self.snapshot_buffer) >= 100000:
            # Compress and write
            compressed = self._compress_snapshots(self.snapshot_buffer)
            await self.writer._write_batch_with_retry(
                'market_snapshots',
                compressed
            )
            self.snapshot_buffer = []
    
    def _compress_snapshots(self, snapshots: List[Dict]) -> List[Dict]:
        """Compress market snapshots"""
        # Group by pool and aggregate
        compressed = {}
        
        for snap in snapshots:
            key = f"{snap['dex']}:{snap['pool_address']}"
            if key not in compressed:
                compressed[key] = snap
            else:
                # Aggregate (keep latest)
                compressed[key] = snap
        
        return list(compressed.values())