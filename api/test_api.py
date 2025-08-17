#!/usr/bin/env python3
"""
Test script for MEV API endpoints
"""

import asyncio
import httpx
import json
from datetime import datetime


BASE_URL = "http://localhost:8000"


async def test_health():
    """Test health endpoint"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"✅ Health Check: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200


async def test_metrics():
    """Test metrics endpoint"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/metrics")
        print(f"✅ Metrics: {response.status_code}")
        print(f"Response length: {len(response.text)} bytes")
        return response.status_code == 200


async def test_docs():
    """Test API documentation"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/docs")
        print(f"✅ API Docs: {response.status_code}")
        return response.status_code == 200


async def test_clickhouse_tables():
    """Test ClickHouse tables endpoint"""
    async with httpx.AsyncClient() as client:
        # This will fail without auth in production
        try:
            response = await client.get(f"{BASE_URL}/clickhouse/tables")
            print(f"ℹ️  ClickHouse Tables: {response.status_code}")
            if response.status_code == 401:
                print("   (Authentication required - expected)")
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False


async def test_query_templates():
    """Test query templates endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/clickhouse/templates")
            print(f"ℹ️  Query Templates: {response.status_code}")
            if response.status_code == 401:
                print("   (Authentication required - expected)")
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False


async def test_websocket():
    """Test WebSocket endpoint"""
    import websockets
    
    try:
        uri = "ws://localhost:8000/realtime/ws"
        async with websockets.connect(uri) as websocket:
            # Receive welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            print(f"✅ WebSocket Connected: {welcome_data['type']}")
            
            # Subscribe to topics
            await websocket.send(json.dumps({
                "type": "subscribe",
                "topics": ["node.health", "system.metrics"]
            }))
            
            # Receive subscription confirmation
            confirmation = await websocket.recv()
            conf_data = json.loads(confirmation)
            print(f"   Subscribed to: {conf_data.get('topics', [])}")
            
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            
            # Receive pong
            pong = await websocket.recv()
            pong_data = json.loads(pong)
            print(f"   Ping/Pong: {pong_data['type']}")
            
            return True
            
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
        return False


async def main():
    """Run all tests"""
    print("=" * 50)
    print("MEV API Test Suite")
    print("=" * 50)
    print(f"Testing API at: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    print("-" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Metrics", test_metrics),
        ("API Documentation", test_docs),
        ("ClickHouse Tables", test_clickhouse_tables),
        ("Query Templates", test_query_templates),
        ("WebSocket", test_websocket)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🧪 Testing: {name}")
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((name, False))
        print("-" * 30)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("-" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)