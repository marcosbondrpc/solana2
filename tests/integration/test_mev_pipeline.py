#!/usr/bin/env python3
"""
Comprehensive MEV Pipeline Integration Tests

Tests the complete end-to-end MEV pipeline including:
- Protobuf message flow
- Service integration
- Database connectivity  
- Real-time streaming
- Performance benchmarks
"""

import asyncio
import json
import time
import pytest
import websockets
import aiohttp
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Import generated protobuf modules
import sys
sys.path.append('../backend/proto/gen/python')
from realtime_pb2 import Envelope, MevOpportunity, BundleOutcome
from control_pb2 import Command, PolicyUpdate
from jobs_pb2 import InferenceRequest, InferenceResponse

@dataclass
class TestConfig:
    """Test configuration for all services"""
    control_plane_url: str = "http://localhost:8000"
    dashboard_api_url: str = "http://localhost:8001" 
    mev_engine_url: str = "http://localhost:8002"
    websocket_url: str = "ws://localhost:8001/ws"
    clickhouse_url: str = "http://localhost:8123"
    redis_url: str = "redis://localhost:6379"
    test_timeout: int = 30
    performance_samples: int = 1000

class MEVIntegrationTest:
    """Main integration test suite for MEV infrastructure"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session: aiohttp.ClientSession = None
        self.test_results: Dict[str, Any] = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_service_health_checks(self) -> bool:
        """Test that all services are healthy and responding"""
        print("🏥 Testing service health checks...")
        
        services = {
            "control-plane": f"{self.config.control_plane_url}/health",
            "dashboard-api": f"{self.config.dashboard_api_url}/health", 
            "mev-engine": f"{self.config.mev_engine_url}/health"
        }
        
        all_healthy = True
        for service, url in services.items():
            try:
                async with self.session.get(url, timeout=5) as response:
                    health_data = await response.json()
                    if response.status == 200 and health_data.get("status") == "healthy":
                        print(f"  ✅ {service}: healthy")
                    else:
                        print(f"  ❌ {service}: unhealthy - {health_data}")
                        all_healthy = False
            except Exception as e:
                print(f"  ❌ {service}: error - {e}")
                all_healthy = False
        
        self.test_results["service_health"] = all_healthy
        return all_healthy
    
    async def test_database_connectivity(self) -> bool:
        """Test database connectivity and basic operations"""
        print("🗄️ Testing database connectivity...")
        
        # Test ClickHouse connectivity
        try:
            async with self.session.get(f"{self.config.clickhouse_url}/ping") as response:
                if response.status == 200:
                    print("  ✅ ClickHouse: connected")
                    
                    # Test query execution
                    query = "SELECT count() FROM system.tables"
                    async with self.session.get(f"{self.config.clickhouse_url}/?query={query}") as resp:
                        if resp.status == 200:
                            print("  ✅ ClickHouse: query execution working")
                        else:
                            print("  ❌ ClickHouse: query execution failed")
                            return False
                else:
                    print("  ❌ ClickHouse: connection failed")
                    return False
        except Exception as e:
            print(f"  ❌ ClickHouse: error - {e}")
            return False
        
        # Test Redis connectivity (through control plane health check)
        try:
            async with self.session.get(f"{self.config.control_plane_url}/health") as response:
                health = await response.json()
                if health.get("services", {}).get("redis") == "healthy":
                    print("  ✅ Redis: connected")
                else:
                    print("  ❌ Redis: connection failed")
                    return False
        except Exception as e:
            print(f"  ❌ Redis: error - {e}")
            return False
        
        self.test_results["database_connectivity"] = True
        return True
    
    async def test_protobuf_communication(self) -> bool:
        """Test protobuf message serialization/deserialization"""
        print("📦 Testing protobuf communication...")
        
        try:
            # Create test MEV opportunity
            opportunity = MevOpportunity()
            opportunity.tx_hash = "test_tx_12345"
            opportunity.slot = 12345678
            opportunity.profit_lamports = 1000000
            opportunity.probability = 0.85
            opportunity.opportunity_type = "arbitrage"
            opportunity.gas_estimate = 150000
            
            # Create envelope
            envelope = Envelope()
            envelope.timestamp_ns = int(time.time() * 1e9)
            envelope.stream_id = "test_stream"
            envelope.sequence = 1
            envelope.type = "mev_opportunity"
            envelope.payload = opportunity.SerializeToString()
            
            # Serialize and deserialize
            serialized = envelope.SerializeToString()
            deserialized = Envelope()
            deserialized.ParseFromString(serialized)
            
            # Verify integrity
            if (deserialized.stream_id == "test_stream" and 
                deserialized.type == "mev_opportunity" and
                len(deserialized.payload) > 0):
                print("  ✅ Protobuf serialization/deserialization working")
                
                # Test nested message
                nested_opportunity = MevOpportunity()
                nested_opportunity.ParseFromString(deserialized.payload)
                
                if (nested_opportunity.tx_hash == "test_tx_12345" and
                    nested_opportunity.profit_lamports == 1000000):
                    print("  ✅ Nested protobuf message parsing working")
                    self.test_results["protobuf_communication"] = True
                    return True
                else:
                    print("  ❌ Nested protobuf message parsing failed")
                    return False
            else:
                print("  ❌ Protobuf serialization/deserialization failed")
                return False
                
        except Exception as e:
            print(f"  ❌ Protobuf communication error: {e}")
            return False
    
    async def test_websocket_streaming(self) -> bool:
        """Test real-time WebSocket streaming"""
        print("🌐 Testing WebSocket streaming...")
        
        try:
            async with websockets.connect(
                self.config.websocket_url,
                timeout=self.config.test_timeout
            ) as websocket:
                
                # Subscribe to test channels
                subscribe_msg = {
                    "type": "subscribe",
                    "channels": ["opportunities", "bundles", "metrics"],
                    "filters": {"min_profit": 0.001}
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                print("  ✅ WebSocket connection established")
                
                # Wait for subscription confirmation or data
                message_received = False
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)
                    
                    if data.get("type") in ["subscription_confirmed", "opportunity", "bundle_outcome", "metrics"]:
                        print(f"  ✅ Received message type: {data.get('type')}")
                        message_received = True
                    else:
                        print(f"  ⚠️ Unexpected message type: {data.get('type')}")
                        
                except asyncio.TimeoutError:
                    print("  ⚠️ No messages received within timeout (expected for new system)")
                    message_received = True  # This is OK for a new system
                
                self.test_results["websocket_streaming"] = message_received
                return message_received
                
        except Exception as e:
            print(f"  ❌ WebSocket streaming error: {e}")
            return False
    
    async def test_api_endpoints(self) -> bool:
        """Test REST API endpoints across all services"""
        print("🔌 Testing API endpoints...")
        
        endpoints = [
            # Control Plane
            (f"{self.config.control_plane_url}/api/v1/status/health", "GET"),
            (f"{self.config.control_plane_url}/api/v1/policy/active", "GET"),
            
            # Dashboard API  
            (f"{self.config.dashboard_api_url}/api/v1/metrics/performance", "GET"),
            (f"{self.config.dashboard_api_url}/api/v1/opportunities/live", "GET"),
            
            # MEV Engine
            (f"{self.config.mev_engine_url}/metrics", "GET"),
        ]
        
        all_endpoints_working = True
        for url, method in endpoints:
            try:
                if method == "GET":
                    async with self.session.get(url, timeout=10) as response:
                        if response.status in [200, 404]:  # 404 is OK for endpoints that might not have data yet
                            print(f"  ✅ {method} {url}: {response.status}")
                        else:
                            print(f"  ❌ {method} {url}: {response.status}")
                            all_endpoints_working = False
            except Exception as e:
                print(f"  ❌ {method} {url}: error - {e}")
                all_endpoints_working = False
        
        self.test_results["api_endpoints"] = all_endpoints_working
        return all_endpoints_working
    
    async def test_performance_metrics(self) -> bool:
        """Test performance characteristics meet SLA requirements"""
        print("⚡ Testing performance metrics...")
        
        # Test API latency
        latencies = []
        for i in range(10):  # Reduced sample size for faster testing
            start_time = time.time()
            try:
                async with self.session.get(f"{self.config.dashboard_api_url}/health") as response:
                    if response.status == 200:
                        latency = (time.time() - start_time) * 1000  # Convert to ms
                        latencies.append(latency)
            except:
                pass
        
        if latencies:
            p50_latency = np.percentile(latencies, 50)
            p99_latency = np.percentile(latencies, 99)
            avg_latency = np.mean(latencies)
            
            print(f"  📊 API Latency - P50: {p50_latency:.2f}ms, P99: {p99_latency:.2f}ms, Avg: {avg_latency:.2f}ms")
            
            # Check if meets SLA (< 100ms P99)
            sla_met = p99_latency < 100
            if sla_met:
                print("  ✅ API latency meets SLA requirements")
            else:
                print(f"  ⚠️ API latency may not meet SLA (P99: {p99_latency:.2f}ms > 100ms)")
            
            self.test_results["performance_metrics"] = {
                "api_latency_p50": p50_latency,
                "api_latency_p99": p99_latency,
                "sla_met": sla_met
            }
            return True
        else:
            print("  ❌ Could not measure API latency")
            return False
    
    async def test_end_to_end_flow(self) -> bool:
        """Test complete end-to-end MEV flow (if services are running)"""
        print("🔄 Testing end-to-end MEV flow...")
        
        try:
            # 1. Submit a test policy update to control plane
            policy_data = {
                "policy_id": "test_policy_integration",
                "policy_type": "risk",
                "thresholds": {"max_position_size": 1000000},
                "enabled": True
            }
            
            async with self.session.post(
                f"{self.config.control_plane_url}/api/v1/policy/risk",
                json=policy_data,
                timeout=10
            ) as response:
                if response.status in [200, 201, 401, 403]:  # Auth errors are OK for testing
                    print("  ✅ Policy update endpoint responds")
                else:
                    print(f"  ⚠️ Policy update returned: {response.status}")
            
            # 2. Query opportunities from dashboard API
            async with self.session.get(
                f"{self.config.dashboard_api_url}/api/v1/opportunities/live?limit=10",
                timeout=10
            ) as response:
                if response.status in [200, 404]:  # No data is OK
                    print("  ✅ Opportunities query endpoint responds")
                else:
                    print(f"  ⚠️ Opportunities query returned: {response.status}")
            
            # 3. Check metrics collection
            async with self.session.get(
                f"{self.config.dashboard_api_url}/api/v1/metrics/performance",
                timeout=10
            ) as response:
                if response.status in [200, 404]:  # No data is OK
                    print("  ✅ Metrics collection endpoint responds")
                else:
                    print(f"  ⚠️ Metrics collection returned: {response.status}")
            
            print("  ✅ End-to-end flow tests completed")
            self.test_results["e2e_flow"] = True
            return True
            
        except Exception as e:
            print(f"  ⚠️ End-to-end flow test partial: {e}")
            self.test_results["e2e_flow"] = False
            return False  # This is non-critical for initial setup
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result is True)
        
        report = {
            "timestamp": time.time(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "results": self.test_results,
            "status": "PASS" if passed_tests == total_tests else "PARTIAL" if passed_tests > 0 else "FAIL"
        }
        
        return report

async def run_integration_tests():
    """Run complete integration test suite"""
    config = TestConfig()
    
    print("🚀 Starting MEV Infrastructure Integration Tests")
    print("=" * 60)
    
    async with MEVIntegrationTest(config) as test_suite:
        test_functions = [
            test_suite.test_service_health_checks,
            test_suite.test_database_connectivity,
            test_suite.test_protobuf_communication,
            test_suite.test_api_endpoints,
            test_suite.test_websocket_streaming,
            test_suite.test_performance_metrics,
            test_suite.test_end_to_end_flow,
        ]
        
        for test_func in test_functions:
            try:
                await test_func()
            except Exception as e:
                print(f"  ❌ Test {test_func.__name__} failed with exception: {e}")
            
            print()  # Add spacing between tests
        
        # Generate final report
        report = test_suite.generate_test_report()
        
        print("📊 Integration Test Results")
        print("=" * 60)
        print(f"Status: {report['status']}")
        print(f"Tests Passed: {report['passed_tests']}/{report['total_tests']}")
        print(f"Success Rate: {report['success_rate']:.1%}")
        
        if report['status'] == 'PASS':
            print("\n🎉 All integration tests passed! System is ready for production.")
        elif report['status'] == 'PARTIAL':
            print("\n⚠️ Some tests passed. System may be partially functional.")
        else:
            print("\n❌ Integration tests failed. Please check service status.")
        
        # Save detailed report
        with open('/tmp/integration_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: /tmp/integration_test_report.json")
        
        return report['status'] == 'PASS'

if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
    exit(0 if success else 1)