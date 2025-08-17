"""
High-performance Solana RPC client with connection pooling
Optimized for massive parallel block fetching
"""

import asyncio
import aiohttp
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import backoff
import json
from functools import lru_cache
import time


class SolanaRPCClient:
    """
    Production-grade RPC client with connection pooling,
    retry logic, and optimized batch processing
    """
    
    def __init__(
        self,
        endpoint: str = "https://api.mainnet-beta.solana.com",
        max_connections: int = 100,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.endpoint = endpoint
        self.max_connections = max_connections
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_id = 0
        self._slot_cache = {}  # LRU cache for slot lookups
        self._block_time_cache = {}
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def connect(self):
        """Initialize connection pool"""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            force_close=True,
            enable_cleanup_closed=True
        )
        timeout_config = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_config,
            headers={"Content-Type": "application/json"}
        )
        
    async def close(self):
        """Clean up connections"""
        if self.session:
            await self.session.close()
            
    def _get_request_id(self) -> str:
        """Generate unique request ID"""
        self._request_id += 1
        return str(self._request_id)
        
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def _make_request(self, method: str, params: List[Any]) -> Any:
        """Execute RPC request with retry logic"""
        payload = {
            "jsonrpc": "2.0",
            "id": self._get_request_id(),
            "method": method,
            "params": params
        }
        
        async with self.session.post(self.endpoint, json=payload) as response:
            result = await response.json()
            if "error" in result:
                raise Exception(f"RPC error: {result['error']}")
            return result.get("result")
            
    async def _batch_request(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Execute batch RPC request for maximum efficiency"""
        async with self.session.post(self.endpoint, json=requests) as response:
            results = await response.json()
            return [r.get("result") for r in results]
            
    @lru_cache(maxsize=10000)
    async def get_slot(self) -> int:
        """Get current slot with caching"""
        return await self._make_request("getSlot", [])
        
    async def get_block_time(self, slot: int) -> Optional[int]:
        """Get block time for slot with caching"""
        if slot in self._block_time_cache:
            return self._block_time_cache[slot]
            
        result = await self._make_request("getBlockTime", [slot])
        self._block_time_cache[slot] = result
        return result
        
    async def get_block(
        self,
        slot: int,
        encoding: str = "jsonParsed",
        transaction_details: str = "full",
        max_supported_transaction_version: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Fetch block with full transaction details"""
        params = [
            slot,
            {
                "encoding": encoding,
                "transactionDetails": transaction_details,
                "maxSupportedTransactionVersion": max_supported_transaction_version,
                "rewards": True
            }
        ]
        return await self._make_request("getBlock", params)
        
    async def get_blocks_batch(
        self,
        slots: List[int],
        encoding: str = "jsonParsed",
        max_supported_transaction_version: int = 0
    ) -> List[Optional[Dict[str, Any]]]:
        """Fetch multiple blocks in parallel for maximum throughput"""
        requests = []
        for slot in slots:
            requests.append({
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "getBlock",
                "params": [
                    slot,
                    {
                        "encoding": encoding,
                        "transactionDetails": "full",
                        "maxSupportedTransactionVersion": max_supported_transaction_version,
                        "rewards": True
                    }
                ]
            })
        return await self._batch_request(requests)
        
    async def get_blocks_with_limit(
        self,
        start_slot: int,
        limit: int = 500000
    ) -> List[int]:
        """Get confirmed blocks within range"""
        return await self._make_request(
            "getBlocksWithLimit",
            [start_slot, limit]
        )
        
    async def find_slot_for_timestamp(
        self,
        target_timestamp: int,
        tolerance: int = 60
    ) -> Tuple[int, int]:
        """
        Binary search to find slot for given timestamp.
        Returns (slot, actual_timestamp)
        """
        current_slot = await self.get_slot()
        
        # Binary search boundaries
        left_slot = 0
        right_slot = current_slot
        
        while left_slot < right_slot:
            mid_slot = (left_slot + right_slot) // 2
            
            # Try to get block time, handle missing slots
            block_time = None
            search_range = 100  # Search nearby slots if exact one is missing
            
            for offset in range(-search_range, search_range + 1):
                test_slot = mid_slot + offset
                if test_slot < 0 or test_slot > current_slot:
                    continue
                    
                try:
                    block_time = await self.get_block_time(test_slot)
                    if block_time:
                        mid_slot = test_slot
                        break
                except:
                    continue
                    
            if not block_time:
                # If we can't find any block in range, adjust search
                if mid_slot < current_slot // 2:
                    left_slot = mid_slot + search_range
                else:
                    right_slot = mid_slot - search_range
                continue
                
            # Check if we're close enough
            if abs(block_time - target_timestamp) <= tolerance:
                return mid_slot, block_time
                
            if block_time < target_timestamp:
                left_slot = mid_slot + 1
            else:
                right_slot = mid_slot - 1
                
        # Return closest found
        final_time = await self.get_block_time(left_slot)
        return left_slot, final_time or target_timestamp
        
    async def get_signatures_for_address(
        self,
        address: str,
        limit: int = 1000,
        before: Optional[str] = None,
        until: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get transaction signatures for address"""
        params = [
            address,
            {
                "limit": limit,
                "before": before,
                "until": until
            }
        ]
        return await self._make_request("getSignaturesForAddress", params)
        
    async def get_transaction(
        self,
        signature: str,
        encoding: str = "jsonParsed",
        max_supported_transaction_version: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Fetch single transaction"""
        params = [
            signature,
            {
                "encoding": encoding,
                "maxSupportedTransactionVersion": max_supported_transaction_version
            }
        ]
        return await self._make_request("getTransaction", params)
        
    async def get_transactions_batch(
        self,
        signatures: List[str],
        encoding: str = "jsonParsed",
        max_supported_transaction_version: int = 0
    ) -> List[Optional[Dict[str, Any]]]:
        """Fetch multiple transactions in batch"""
        requests = []
        for sig in signatures:
            requests.append({
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "getTransaction",
                "params": [
                    sig,
                    {
                        "encoding": encoding,
                        "maxSupportedTransactionVersion": max_supported_transaction_version
                    }
                ]
            })
        return await self._batch_request(requests)
        
    async def get_program_accounts(
        self,
        program_id: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        encoding: str = "base64"
    ) -> List[Dict[str, Any]]:
        """Get all accounts owned by program"""
        params = [
            program_id,
            {
                "encoding": encoding,
                "filters": filters or []
            }
        ]
        return await self._make_request("getProgramAccounts", params)
        
    def estimate_slots_per_day(self) -> int:
        """Estimate number of slots per day (roughly 2 slots/second)"""
        return 2 * 60 * 60 * 24  # ~172,800 slots per day
        
    def estimate_slots_for_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Estimate total slots in date range"""
        days = (end_date - start_date).days + 1
        return days * self.estimate_slots_per_day()