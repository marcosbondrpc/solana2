"""
High-performance data capture engine
Concurrent block fetching with optimal batching
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, date, timedelta
import time
from collections import deque
import logging
from functools import partial

from .rpc import SolanaRPCClient
from .storage import ParquetStorage
from .models import CaptureRequest, JobMetadata, JobStatus, Granularity


logger = logging.getLogger(__name__)


class CaptureEngine:
    """
    Production-grade capture engine with concurrent fetching,
    adaptive batching, and fault tolerance
    """
    
    def __init__(
        self,
        rpc_client: SolanaRPCClient,
        storage: ParquetStorage,
        progress_callback: Optional[Callable] = None
    ):
        self.rpc = rpc_client
        self.storage = storage
        self.progress_callback = progress_callback
        self.cancelled = False
        self.stats = {
            "blocks_fetched": 0,
            "transactions_fetched": 0,
            "logs_extracted": 0,
            "errors": [],
            "start_time": None,
            "end_time": None
        }
        
    async def capture(
        self,
        request: CaptureRequest,
        job: JobMetadata
    ) -> Dict[str, Any]:
        """
        Main capture orchestration with optimal concurrency
        """
        self.stats["start_time"] = time.time()
        self.cancelled = False
        
        try:
            # Convert dates to slots using binary search
            start_ts = int(datetime.combine(request.start, datetime.min.time()).timestamp())
            end_ts = int(datetime.combine(request.end, datetime.max.time()).timestamp())
            
            logger.info(f"Finding slots for date range {request.start} to {request.end}")
            
            start_slot, _ = await self.rpc.find_slot_for_timestamp(start_ts)
            end_slot, _ = await self.rpc.find_slot_for_timestamp(end_ts)
            
            logger.info(f"Capturing slots {start_slot} to {end_slot}")
            
            # Update job metadata
            job.total_slots = end_slot - start_slot
            job.current_slot = start_slot
            
            # Process based on granularity
            if request.granularity == Granularity.DAY:
                await self._capture_by_day(request, job, start_slot, end_slot)
            elif request.granularity == Granularity.MONTH:
                await self._capture_by_month(request, job, start_slot, end_slot)
            else:  # YEAR
                await self._capture_by_year(request, job, start_slot, end_slot)
                
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            self.stats["errors"].append(str(e))
            job.status = JobStatus.FAILED
            raise
        finally:
            self.stats["end_time"] = time.time()
            
        return self.stats
        
    async def _capture_by_day(
        self,
        request: CaptureRequest,
        job: JobMetadata,
        start_slot: int,
        end_slot: int
    ):
        """Capture data with daily partitioning"""
        current_date = request.start
        
        while current_date <= request.end and not self.cancelled:
            # Get slot range for this day
            day_start_ts = int(datetime.combine(current_date, datetime.min.time()).timestamp())
            day_end_ts = int(datetime.combine(current_date, datetime.max.time()).timestamp())
            
            day_start_slot, _ = await self.rpc.find_slot_for_timestamp(day_start_ts)
            day_end_slot, _ = await self.rpc.find_slot_for_timestamp(day_end_ts)
            
            # Ensure within bounds
            day_start_slot = max(day_start_slot, start_slot)
            day_end_slot = min(day_end_slot, end_slot)
            
            logger.info(f"Processing {current_date}: slots {day_start_slot} to {day_end_slot}")
            
            # Process this day's data
            await self._process_slot_range(
                request,
                job,
                day_start_slot,
                day_end_slot,
                current_date
            )
            
            # Move to next day
            current_date += timedelta(days=1)
            
    async def _capture_by_month(
        self,
        request: CaptureRequest,
        job: JobMetadata,
        start_slot: int,
        end_slot: int
    ):
        """Capture data with monthly partitioning"""
        current_date = request.start.replace(day=1)
        
        while current_date <= request.end and not self.cancelled:
            # Calculate month boundaries
            if current_date.month == 12:
                next_month = current_date.replace(year=current_date.year + 1, month=1)
            else:
                next_month = current_date.replace(month=current_date.month + 1)
                
            month_end = next_month - timedelta(days=1)
            
            # Get slot range for this month
            month_start_ts = int(datetime.combine(current_date, datetime.min.time()).timestamp())
            month_end_ts = int(datetime.combine(month_end, datetime.max.time()).timestamp())
            
            month_start_slot, _ = await self.rpc.find_slot_for_timestamp(month_start_ts)
            month_end_slot, _ = await self.rpc.find_slot_for_timestamp(month_end_ts)
            
            # Process month's data day by day for proper partitioning
            day = current_date
            while day <= month_end and day <= request.end:
                await self._process_slot_range(
                    request,
                    job,
                    max(month_start_slot, start_slot),
                    min(month_end_slot, end_slot),
                    day
                )
                day += timedelta(days=1)
                
            current_date = next_month
            
    async def _capture_by_year(
        self,
        request: CaptureRequest,
        job: JobMetadata,
        start_slot: int,
        end_slot: int
    ):
        """Capture data with yearly partitioning"""
        # Similar to month but with year boundaries
        current_year = request.start.year
        
        while current_year <= request.end.year and not self.cancelled:
            year_start = date(current_year, 1, 1)
            year_end = date(current_year, 12, 31)
            
            # Process year's data month by month
            for month in range(1, 13):
                if self.cancelled:
                    break
                    
                month_date = date(current_year, month, 1)
                if month_date < request.start or month_date > request.end:
                    continue
                    
                await self._capture_by_month(
                    request,
                    job,
                    start_slot,
                    end_slot
                )
                
            current_year += 1
            
    async def _process_slot_range(
        self,
        request: CaptureRequest,
        job: JobMetadata,
        start_slot: int,
        end_slot: int,
        partition_date: date
    ):
        """
        Process a range of slots with optimal batching and concurrency
        """
        # Get list of confirmed blocks in range
        blocks_to_fetch = []
        current = start_slot
        
        while current <= end_slot and not self.cancelled:
            # Get next batch of confirmed blocks
            confirmed = await self.rpc.get_blocks_with_limit(
                current,
                min(1000, end_slot - current + 1)
            )
            
            if not confirmed:
                current += 1000
                continue
                
            blocks_to_fetch.extend(confirmed)
            current = confirmed[-1] + 1
            
        logger.info(f"Found {len(blocks_to_fetch)} confirmed blocks to fetch")
        
        # Process in batches with concurrency control
        batch_size = request.block_batch
        semaphore = asyncio.Semaphore(10)  # Max concurrent batches
        
        tasks = []
        for i in range(0, len(blocks_to_fetch), batch_size):
            if self.cancelled:
                break
                
            batch = blocks_to_fetch[i:i + batch_size]
            task = self._process_block_batch(
                request,
                job,
                batch,
                partition_date,
                semaphore
            )
            tasks.append(task)
            
        # Execute all batches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle errors
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                self.stats["errors"].append(str(result))
                
    async def _process_block_batch(
        self,
        request: CaptureRequest,
        job: JobMetadata,
        slots: List[int],
        partition_date: date,
        semaphore: asyncio.Semaphore
    ):
        """
        Process a batch of blocks with transaction extraction
        """
        async with semaphore:
            try:
                # Fetch blocks in parallel
                blocks = await self.rpc.get_blocks_batch(
                    slots,
                    encoding="jsonParsed" if request.json_parsed else "base64",
                    max_supported_transaction_version=request.max_tx_version
                )
                
                # Filter out None results
                valid_blocks = [b for b in blocks if b is not None]
                
                if not valid_blocks:
                    return
                    
                # Extract transactions and logs
                all_transactions = []
                all_logs = []
                
                for block in valid_blocks:
                    if not block:
                        continue
                        
                    # Update stats
                    self.stats["blocks_fetched"] += 1
                    job.blocks_processed += 1
                    
                    # Extract transactions if requested
                    if request.include_transactions and block.get("transactions"):
                        for tx_wrapper in block["transactions"]:
                            tx = tx_wrapper.get("transaction")
                            meta = tx_wrapper.get("meta")
                            
                            if not tx or not meta:
                                continue
                                
                            # Check if transaction involves target programs
                            if request.programs:
                                involved = False
                                for inst in tx.get("message", {}).get("instructions", []):
                                    if inst.get("programId") in request.programs:
                                        involved = True
                                        break
                                        
                                if not involved:
                                    continue
                                    
                            # Store transaction
                            sig = tx.get("signatures", [""])[0]
                            all_transactions.append((sig, {
                                "slot": block["parentSlot"],
                                "blockTime": block.get("blockTime"),
                                "transaction": tx,
                                "meta": meta
                            }))
                            
                            self.stats["transactions_fetched"] += 1
                            job.transactions_processed += 1
                            
                            # Extract logs if requested
                            if request.include_logs and meta.get("logMessages"):
                                for log_idx, log_msg in enumerate(meta["logMessages"]):
                                    all_logs.append({
                                        "signature": sig,
                                        "slot": block["parentSlot"],
                                        "instruction_index": log_idx,
                                        "inner_index": 0,
                                        "program_id": "",  # Extract from instruction
                                        "log_message": log_msg,
                                        "year": partition_date.year,
                                        "month": partition_date.month,
                                        "day": partition_date.day
                                    })
                                    self.stats["logs_extracted"] += 1
                                    
                # Write to storage
                if request.include_blocks and valid_blocks:
                    await self.storage.write_blocks_batch(valid_blocks, partition_date)
                    
                if request.include_transactions and all_transactions:
                    await self.storage.write_transactions_batch(all_transactions, partition_date)
                    
                if request.include_logs and all_logs:
                    await self.storage.write_logs_batch(all_logs, partition_date)
                    
                # Update progress
                if job.total_slots:
                    job.progress = (job.blocks_processed / job.total_slots) * 100
                    job.current_slot = slots[-1] if slots else job.current_slot
                    
                if self.progress_callback:
                    await self.progress_callback(job)
                    
            except Exception as e:
                logger.error(f"Error processing batch {slots[0]}-{slots[-1]}: {e}")
                raise
                
    def cancel(self):
        """Cancel ongoing capture"""
        self.cancelled = True
        logger.info("Capture cancelled by user")