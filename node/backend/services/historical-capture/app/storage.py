"""
High-performance Parquet storage with partitioning
Optimized for time-series blockchain data
"""

import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
import asyncio
from pathlib import Path
import aiofiles
import duckdb
from concurrent.futures import ThreadPoolExecutor


class ParquetStorage:
    """
    Production-grade storage engine for blockchain data
    with daily partitioning and streaming writes
    """
    
    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Define schemas for each data type
        self.block_schema = pa.schema([
            pa.field("slot", pa.int64()),
            pa.field("block_height", pa.int64()),
            pa.field("block_time", pa.int64()),
            pa.field("parent_slot", pa.int64()),
            pa.field("previous_blockhash", pa.string()),
            pa.field("blockhash", pa.string()),
            pa.field("rewards", pa.string()),  # JSON string
            pa.field("transaction_count", pa.int32()),
            pa.field("year", pa.int32()),
            pa.field("month", pa.int32()),
            pa.field("day", pa.int32())
        ])
        
        self.transaction_schema = pa.schema([
            pa.field("signature", pa.string()),
            pa.field("slot", pa.int64()),
            pa.field("block_time", pa.int64()),
            pa.field("err", pa.string()),  # JSON string
            pa.field("fee", pa.int64()),
            pa.field("pre_balances", pa.string()),  # JSON array
            pa.field("post_balances", pa.string()),  # JSON array
            pa.field("pre_token_balances", pa.string()),  # JSON
            pa.field("post_token_balances", pa.string()),  # JSON
            pa.field("log_messages", pa.string()),  # JSON array
            pa.field("compute_units_consumed", pa.int64()),
            pa.field("loaded_addresses", pa.string()),  # JSON
            pa.field("year", pa.int32()),
            pa.field("month", pa.int32()),
            pa.field("day", pa.int32())
        ])
        
        self.log_schema = pa.schema([
            pa.field("signature", pa.string()),
            pa.field("slot", pa.int64()),
            pa.field("instruction_index", pa.int32()),
            pa.field("inner_index", pa.int32()),
            pa.field("program_id", pa.string()),
            pa.field("log_message", pa.string()),
            pa.field("year", pa.int32()),
            pa.field("month", pa.int32()),
            pa.field("day", pa.int32())
        ])
        
        self.swap_schema = pa.schema([
            pa.field("signature", pa.string()),
            pa.field("slot", pa.int64()),
            pa.field("instruction_index", pa.int32()),
            pa.field("program_id", pa.string()),
            pa.field("pool_address", pa.string()),
            pa.field("user", pa.string()),
            pa.field("token_in", pa.string()),
            pa.field("token_out", pa.string()),
            pa.field("amount_in", pa.int64()),
            pa.field("amount_out", pa.int64()),
            pa.field("price", pa.float64()),
            pa.field("timestamp", pa.int64())
        ])
        
        self.arbitrage_schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("start_slot", pa.int64()),
            pa.field("end_slot", pa.int64()),
            pa.field("transactions", pa.string()),  # JSON array
            pa.field("path", pa.string()),  # JSON array
            pa.field("tokens", pa.string()),  # JSON array
            pa.field("profit_token", pa.string()),
            pa.field("profit_amount", pa.int64()),
            pa.field("profit_usd", pa.float64()),
            pa.field("gas_used", pa.int64()),
            pa.field("net_profit_usd", pa.float64()),
            pa.field("detected_at", pa.timestamp('us'))
        ])
        
        self.sandwich_schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("front_tx", pa.string()),
            pa.field("victim_tx", pa.string()),
            pa.field("back_tx", pa.string()),
            pa.field("slot", pa.int64()),
            pa.field("pool_address", pa.string()),
            pa.field("attacker", pa.string()),
            pa.field("victim", pa.string()),
            pa.field("profit_token", pa.string()),
            pa.field("profit_amount", pa.int64()),
            pa.field("profit_usd", pa.float64()),
            pa.field("victim_loss_usd", pa.float64()),
            pa.field("detected_at", pa.timestamp('us'))
        ])
        
    def _get_partition_path(
        self,
        data_type: str,
        year: int,
        month: int,
        day: int
    ) -> Path:
        """Generate partition path"""
        return self.base_path / "raw" / data_type / f"year={year}" / f"month={month:02d}" / f"day={day:02d}"
        
    async def write_blocks_batch(
        self,
        blocks: List[Dict[str, Any]],
        partition_date: date
    ) -> str:
        """Write batch of blocks to Parquet with partitioning"""
        if not blocks:
            return ""
            
        # Convert to table format
        records = []
        for block in blocks:
            if block is None:
                continue
                
            record = {
                "slot": block.get("parentSlot", 0),
                "block_height": block.get("blockHeight"),
                "block_time": block.get("blockTime"),
                "parent_slot": block.get("parentSlot", 0),
                "previous_blockhash": block.get("previousBlockhash", ""),
                "blockhash": block.get("blockhash", ""),
                "rewards": json.dumps(block.get("rewards", [])),
                "transaction_count": len(block.get("transactions", [])),
                "year": partition_date.year,
                "month": partition_date.month,
                "day": partition_date.day
            }
            records.append(record)
            
        if not records:
            return ""
            
        # Create partition directory
        partition_path = self._get_partition_path(
            "blocks",
            partition_date.year,
            partition_date.month,
            partition_date.day
        )
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Write to Parquet
        table = pa.Table.from_pylist(records, schema=self.block_schema)
        filename = partition_path / f"part-{blocks[0]['parentSlot']}-{blocks[-1]['parentSlot']}.parquet"
        
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            pq.write_table,
            table,
            str(filename),
            "snappy"
        )
        
        return str(filename)
        
    async def write_transactions_batch(
        self,
        transactions: List[Tuple[str, Dict[str, Any]]],
        partition_date: date
    ) -> str:
        """Write batch of transactions to Parquet"""
        if not transactions:
            return ""
            
        records = []
        for sig, tx in transactions:
            if tx is None:
                continue
                
            meta = tx.get("meta", {})
            record = {
                "signature": sig,
                "slot": tx.get("slot", 0),
                "block_time": tx.get("blockTime"),
                "err": json.dumps(meta.get("err")) if meta.get("err") else None,
                "fee": meta.get("fee", 0),
                "pre_balances": json.dumps(meta.get("preBalances", [])),
                "post_balances": json.dumps(meta.get("postBalances", [])),
                "pre_token_balances": json.dumps(meta.get("preTokenBalances", [])),
                "post_token_balances": json.dumps(meta.get("postTokenBalances", [])),
                "log_messages": json.dumps(meta.get("logMessages", [])),
                "compute_units_consumed": meta.get("computeUnitsConsumed"),
                "loaded_addresses": json.dumps(meta.get("loadedAddresses", {})),
                "year": partition_date.year,
                "month": partition_date.month,
                "day": partition_date.day
            }
            records.append(record)
            
        if not records:
            return ""
            
        partition_path = self._get_partition_path(
            "transactions",
            partition_date.year,
            partition_date.month,
            partition_date.day
        )
        partition_path.mkdir(parents=True, exist_ok=True)
        
        table = pa.Table.from_pylist(records, schema=self.transaction_schema)
        filename = partition_path / f"part-{transactions[0][1]['slot']}.parquet"
        
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            pq.write_table,
            table,
            str(filename),
            "snappy"
        )
        
        return str(filename)
        
    async def write_logs_batch(
        self,
        logs: List[Dict[str, Any]],
        partition_date: date
    ) -> str:
        """Write batch of logs to Parquet"""
        if not logs:
            return ""
            
        partition_path = self._get_partition_path(
            "logs",
            partition_date.year,
            partition_date.month,
            partition_date.day
        )
        partition_path.mkdir(parents=True, exist_ok=True)
        
        table = pa.Table.from_pylist(logs, schema=self.log_schema)
        filename = partition_path / f"part-{logs[0]['slot']}.parquet"
        
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            pq.write_table,
            table,
            str(filename),
            "snappy"
        )
        
        return str(filename)
        
    async def write_arbitrage_opportunities(
        self,
        opportunities: List[Dict[str, Any]]
    ) -> str:
        """Write detected arbitrage opportunities"""
        if not opportunities:
            return ""
            
        labels_path = self.base_path / "labels"
        labels_path.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime objects
        for opp in opportunities:
            if isinstance(opp.get("detected_at"), datetime):
                opp["detected_at"] = opp["detected_at"]
            if isinstance(opp.get("transactions"), list):
                opp["transactions"] = json.dumps(opp["transactions"])
            if isinstance(opp.get("path"), list):
                opp["path"] = json.dumps(opp["path"])
            if isinstance(opp.get("tokens"), list):
                opp["tokens"] = json.dumps(opp["tokens"])
                
        table = pa.Table.from_pylist(opportunities, schema=self.arbitrage_schema)
        filename = labels_path / "arbitrage.parquet"
        
        # Append if file exists
        if filename.exists():
            existing_table = pq.read_table(str(filename))
            table = pa.concat_tables([existing_table, table])
            
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            pq.write_table,
            table,
            str(filename),
            "snappy"
        )
        
        return str(filename)
        
    async def write_sandwich_attacks(
        self,
        attacks: List[Dict[str, Any]]
    ) -> str:
        """Write detected sandwich attacks"""
        if not attacks:
            return ""
            
        labels_path = self.base_path / "labels"
        labels_path.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime objects
        for attack in attacks:
            if isinstance(attack.get("detected_at"), datetime):
                attack["detected_at"] = attack["detected_at"]
                
        table = pa.Table.from_pylist(attacks, schema=self.sandwich_schema)
        filename = labels_path / "sandwich.parquet"
        
        # Append if file exists
        if filename.exists():
            existing_table = pq.read_table(str(filename))
            table = pa.concat_tables([existing_table, table])
            
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            pq.write_table,
            table,
            str(filename),
            "snappy"
        )
        
        return str(filename)
        
    async def write_manifest(
        self,
        job_id: str,
        config: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> str:
        """Write job manifest for reproducibility"""
        manifest_path = self.base_path / "manifests"
        manifest_path.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "job_id": job_id,
            "created_at": datetime.utcnow().isoformat(),
            "config": config,
            "stats": stats,
            "version": "1.0.0"
        }
        
        filename = manifest_path / f"job_{job_id}.json"
        async with aiofiles.open(filename, 'w') as f:
            await f.write(json.dumps(manifest, indent=2, default=str))
            
        return str(filename)
        
    def query_with_duckdb(self, query: str) -> List[Dict[str, Any]]:
        """Execute analytical query using DuckDB for fast performance"""
        conn = duckdb.connect(":memory:")
        
        # Register Parquet files
        blocks_pattern = str(self.base_path / "raw" / "blocks" / "**" / "*.parquet")
        tx_pattern = str(self.base_path / "raw" / "transactions" / "**" / "*.parquet")
        
        conn.execute(f"CREATE VIEW blocks AS SELECT * FROM read_parquet('{blocks_pattern}')")
        conn.execute(f"CREATE VIEW transactions AS SELECT * FROM read_parquet('{tx_pattern}')")
        
        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]
        
        return [dict(zip(columns, row)) for row in result]
        
    async def get_dataset_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics"""
        stats = {
            "total_blocks": 0,
            "total_transactions": 0,
            "total_logs": 0,
            "storage_size_mb": 0.0,
            "partitions": []
        }
        
        # Count files and calculate sizes
        for data_type in ["blocks", "transactions", "logs"]:
            type_path = self.base_path / "raw" / data_type
            if type_path.exists():
                for parquet_file in type_path.rglob("*.parquet"):
                    stats["storage_size_mb"] += parquet_file.stat().st_size / (1024 * 1024)
                    
                    # Add partition info
                    partition = str(parquet_file.parent.relative_to(type_path))
                    if partition not in stats["partitions"]:
                        stats["partitions"].append(partition)
                        
        # Use DuckDB for fast counting
        try:
            conn = duckdb.connect(":memory:")
            
            blocks_pattern = str(self.base_path / "raw" / "blocks" / "**" / "*.parquet")
            if Path(blocks_pattern).parent.exists():
                stats["total_blocks"] = conn.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{blocks_pattern}')"
                ).fetchone()[0]
                
            tx_pattern = str(self.base_path / "raw" / "transactions" / "**" / "*.parquet")
            if Path(tx_pattern).parent.exists():
                stats["total_transactions"] = conn.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{tx_pattern}')"
                ).fetchone()[0]
                
        except:
            pass
            
        return stats