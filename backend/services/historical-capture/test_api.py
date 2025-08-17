#!/usr/bin/env python3
"""
Test script for Historical Capture API
Verifies all endpoints are working correctly
"""

import asyncio
import aiohttp
import json
from datetime import date, datetime


async def test_api():
    """Test all API endpoints"""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        print("Testing Historical Capture API...")
        print("-" * 50)
        
        # Test health endpoint
        print("\n1. Testing /health endpoint...")
        async with session.get(f"{base_url}/health") as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✓ Health check passed: {data['status']}")
                print(f"  - Version: {data['version']}")
                print(f"  - RPC Status: {data['rpc_status']}")
            else:
                print(f"✗ Health check failed: {resp.status}")
                
        # Test dataset stats
        print("\n2. Testing /datasets/stats endpoint...")
        async with session.get(f"{base_url}/datasets/stats") as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✓ Dataset stats retrieved")
                print(f"  - Total blocks: {data.get('total_blocks', 0)}")
                print(f"  - Total transactions: {data.get('total_transactions', 0)}")
            else:
                print(f"✗ Dataset stats failed: {resp.status}")
                
        # Test capture start (small test range)
        print("\n3. Testing /capture/start endpoint...")
        capture_request = {
            "granularity": "day",
            "start": "2025-01-10",
            "end": "2025-01-10",
            "source": "rpc",
            "programs": ["675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"],  # Raydium
            "include_blocks": True,
            "include_transactions": True,
            "include_logs": True,
            "out_uri": "./data",
            "block_batch": 32,
            "json_parsed": True,
            "max_tx_version": 0
        }
        
        async with session.post(
            f"{base_url}/capture/start",
            json=capture_request
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                job_id = data['job_id']
                print(f"✓ Capture job started: {job_id}")
                print(f"  - Status: {data['status']}")
                
                # Wait a bit and check status
                await asyncio.sleep(2)
                
                # Test job status endpoint
                print(f"\n4. Testing /jobs/{job_id} endpoint...")
                async with session.get(f"{base_url}/jobs/{job_id}") as status_resp:
                    if status_resp.status == 200:
                        status_data = await status_resp.json()
                        print(f"✓ Job status retrieved")
                        print(f"  - Status: {status_data['status']}")
                        print(f"  - Progress: {status_data['progress']:.1f}%")
                        print(f"  - Blocks processed: {status_data['blocks_processed']}")
                    else:
                        print(f"✗ Job status failed: {status_resp.status}")
                        
                # Test job cancel
                print(f"\n5. Testing /jobs/{job_id}/cancel endpoint...")
                async with session.post(f"{base_url}/jobs/{job_id}/cancel") as cancel_resp:
                    if cancel_resp.status == 200:
                        cancel_data = await cancel_resp.json()
                        print(f"✓ Job cancelled successfully")
                    else:
                        print(f"✗ Job cancel failed: {cancel_resp.status}")
                        
            else:
                print(f"✗ Capture start failed: {resp.status}")
                error = await resp.text()
                print(f"  Error: {error}")
                
        # Test arbitrage detection
        print("\n6. Testing /convert/arbitrage/start endpoint...")
        arb_request = {
            "raw_uri": "./data/raw",
            "out_uri": "./data/labels",
            "min_profit_usd": 1.0,
            "max_slot_gap": 3
        }
        
        async with session.post(
            f"{base_url}/convert/arbitrage/start",
            json=arb_request
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✓ Arbitrage detection started: {data['job_id']}")
            else:
                print(f"✗ Arbitrage detection failed: {resp.status}")
                
        # Test sandwich detection
        print("\n7. Testing /convert/sandwich/start endpoint...")
        sandwich_request = {
            "raw_uri": "./data/raw",
            "out_uri": "./data/labels",
            "min_profit_usd": 1.0,
            "max_slot_gap": 2
        }
        
        async with session.post(
            f"{base_url}/convert/sandwich/start",
            json=sandwich_request
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✓ Sandwich detection started: {data['job_id']}")
            else:
                print(f"✗ Sandwich detection failed: {resp.status}")
                
        print("\n" + "=" * 50)
        print("API testing complete!")


if __name__ == "__main__":
    asyncio.run(test_api())