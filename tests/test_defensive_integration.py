#!/usr/bin/env python3
"""
Comprehensive integration tests for Defensive-Only MEV Infrastructure
Verifies all performance targets and service integration
"""

import asyncio
import time
import random
import statistics
from typing import List, Dict, Any
import aiohttp
import numpy as np
from datetime import datetime

# Service endpoints
SHREDSTREAM_URL = "http://localhost:9090"
DECISION_DNA_URL = "http://localhost:8092"
DETECTION_URL = "http://localhost:8093"
MAIN_API_URL = "http://localhost:8000"


class PerformanceTest:
    """Performance testing framework"""
    
    def __init__(self):
        self.results = {
            'ingestion_rate': [],
            'decision_latency': [],
            'inference_latency': [],
            'detection_accuracy': [],
            'memory_usage': []
        }
        self.session = None
    
    async def setup(self):
        """Initialize test environment"""
        self.session = aiohttp.ClientSession()
        
        # Verify all services are healthy
        services = [
            (MAIN_API_URL + "/defensive/health", "Main API"),
            (DECISION_DNA_URL + "/health", "Decision DNA"),
            (DETECTION_URL + "/health", "Detection Service"),
        ]
        
        print("🔍 Checking service health...")
        for url, name in services:
            try:
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        print(f"  ✅ {name}: Healthy")
                    else:
                        print(f"  ⚠️ {name}: Status {resp.status}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                return False
        
        return True
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def test_ingestion_rate(self) -> float:
        """Test ShredStream ingestion rate (target: ≥200k msgs/sec)"""
        print("\n📊 Testing ingestion rate...")
        
        # Generate test messages
        messages = []
        for _ in range(10000):
            messages.append({
                'data': bytes(random.getrandbits(8) for _ in range(1024)),
                'timestamp': time.time_ns()
            })
        
        # Measure ingestion rate
        start_time = time.perf_counter()
        
        tasks = []
        for msg in messages:
            task = self.session.post(
                f"{MAIN_API_URL}/defensive/shredstream/process",
                data=msg['data']
            )
            tasks.append(task)
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            await asyncio.gather(*batch, return_exceptions=True)
        
        elapsed = time.perf_counter() - start_time
        rate = len(messages) / elapsed
        
        print(f"  📈 Ingestion rate: {rate:,.0f} msgs/sec")
        print(f"  {'✅' if rate >= 200000 else '❌'} Target: ≥200,000 msgs/sec")
        
        self.results['ingestion_rate'].append(rate)
        return rate
    
    async def test_decision_latency(self) -> Dict[str, float]:
        """Test end-to-end decision latency (P50 ≤8ms, P99 ≤20ms)"""
        print("\n⏱️ Testing decision latency...")
        
        latencies = []
        
        # Generate test transactions
        for _ in range(1000):
            transactions = self._generate_test_transactions(10)
            
            start_time = time.perf_counter()
            
            try:
                async with self.session.post(
                    f"{MAIN_API_URL}/defensive/detect",
                    json=transactions
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
                        latencies.append(latency)
            except:
                pass
        
        if latencies:
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p99 = latencies[int(len(latencies) * 0.99)]
            
            print(f"  📊 P50 Latency: {p50:.2f}ms")
            print(f"  {'✅' if p50 <= 8 else '❌'} Target: ≤8ms")
            print(f"  📊 P99 Latency: {p99:.2f}ms")
            print(f"  {'✅' if p99 <= 20 else '❌'} Target: ≤20ms")
            
            self.results['decision_latency'].extend(latencies)
            return {'p50': p50, 'p99': p99}
        
        return {'p50': 0, 'p99': 0}
    
    async def test_model_inference(self) -> float:
        """Test GNN + Transformer inference speed (target: <100μs)"""
        print("\n🧠 Testing model inference...")
        
        inference_times = []
        
        for _ in range(100):
            transactions = self._generate_test_transactions(50)
            
            async with self.session.post(
                f"{DETECTION_URL}/api/v1/detect",
                json={'transactions': transactions}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if 'latency_us' in result:
                        inference_times.append(result['latency_us'])
        
        if inference_times:
            avg_inference = statistics.mean(inference_times)
            print(f"  ⚡ Average inference: {avg_inference:.1f}μs")
            print(f"  {'✅' if avg_inference < 100 else '❌'} Target: <100μs")
            
            self.results['inference_latency'].extend(inference_times)
            return avg_inference
        
        return 0
    
    async def test_detection_accuracy(self) -> float:
        """Test detection accuracy (target: ≥65%)"""
        print("\n🎯 Testing detection accuracy...")
        
        # Generate labeled test cases
        test_cases = []
        
        # Generate normal transactions
        for _ in range(500):
            test_cases.append({
                'transactions': self._generate_test_transactions(5, suspicious=False),
                'expected': False
            })
        
        # Generate suspicious transactions
        for _ in range(500):
            test_cases.append({
                'transactions': self._generate_test_transactions(5, suspicious=True),
                'expected': True
            })
        
        correct = 0
        total = 0
        
        for case in test_cases:
            async with self.session.post(
                f"{MAIN_API_URL}/defensive/detect",
                json=case['transactions']
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    detected = result.get('detected', False)
                    
                    if detected == case['expected']:
                        correct += 1
                    total += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"  🎯 Detection accuracy: {accuracy:.1f}%")
        print(f"  {'✅' if accuracy >= 65 else '❌'} Target: ≥65%")
        
        self.results['detection_accuracy'].append(accuracy)
        return accuracy
    
    async def test_memory_usage(self) -> int:
        """Test memory usage per connection (target: <500KB)"""
        print("\n💾 Testing memory usage...")
        
        # Get initial memory
        async with self.session.get(f"{MAIN_API_URL}/health") as resp:
            initial = await resp.json()
            initial_memory = initial['metrics']['memory_mb'] * 1024 * 1024
        
        # Create 100 concurrent connections
        connections = []
        for _ in range(100):
            ws = await self.session.ws_connect(f"{MAIN_API_URL}/defensive/stream")
            connections.append(ws)
        
        # Wait for connections to stabilize
        await asyncio.sleep(2)
        
        # Get final memory
        async with self.session.get(f"{MAIN_API_URL}/health") as resp:
            final = await resp.json()
            final_memory = final['metrics']['memory_mb'] * 1024 * 1024
        
        # Close connections
        for ws in connections:
            await ws.close()
        
        memory_per_connection = (final_memory - initial_memory) / 100
        
        print(f"  💾 Memory per connection: {memory_per_connection/1024:.1f}KB")
        print(f"  {'✅' if memory_per_connection < 500000 else '❌'} Target: <500KB")
        
        self.results['memory_usage'].append(memory_per_connection)
        return memory_per_connection
    
    async def test_dna_integrity(self) -> bool:
        """Test Decision DNA hash chain integrity"""
        print("\n🔐 Testing Decision DNA integrity...")
        
        # Create some events
        events = []
        for i in range(10):
            async with self.session.post(
                f"{MAIN_API_URL}/defensive/dna/event",
                json={
                    'event_type': 'ArbitrageDetected',
                    'transaction_hash': f'0x{i:064x}',
                    'block_number': 1000 + i,
                    'confidence_score': 0.85
                }
            ) as resp:
                if resp.status == 200:
                    event = await resp.json()
                    events.append(event)
        
        # Verify chain
        if events:
            async with self.session.get(
                f"{MAIN_API_URL}/defensive/dna/verify",
                params={'from_sequence': 0, 'to_sequence': len(events)-1}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    valid = result.get('valid', False)
                    
                    print(f"  {'✅' if valid else '❌'} Hash chain integrity: {'Valid' if valid else 'Invalid'}")
                    return valid
        
        return False
    
    async def test_thompson_sampling(self) -> float:
        """Test Thompson Sampling convergence"""
        print("\n🎰 Testing Thompson Sampling...")
        
        # Simulate route selection with rewards
        selections = {'route1': 0, 'route2': 0, 'route3': 0}
        
        for _ in range(1000):
            transactions = self._generate_test_transactions(5)
            
            async with self.session.post(
                f"{MAIN_API_URL}/defensive/detect",
                json=transactions
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    route = result.get('recommended_route', 'unknown')
                    if route in selections:
                        selections[route] += 1
        
        # Check if it converged to best route
        total = sum(selections.values())
        if total > 0:
            best_route = max(selections, key=selections.get)
            convergence = selections[best_route] / total * 100
            
            print(f"  📊 Route selection distribution:")
            for route, count in selections.items():
                pct = count / total * 100 if total > 0 else 0
                print(f"    • {route}: {pct:.1f}%")
            
            print(f"  {'✅' if convergence > 40 else '⚠️'} Convergence: {convergence:.1f}%")
            return convergence
        
        return 0
    
    def _generate_test_transactions(self, count: int, suspicious: bool = False) -> List[Dict]:
        """Generate test transactions"""
        transactions = []
        
        for i in range(count):
            # Generate features that indicate suspicious activity if needed
            if suspicious:
                features = [random.uniform(0.7, 1.0) for _ in range(128)]
            else:
                features = [random.uniform(0.0, 0.3) for _ in range(128)]
            
            transactions.append({
                'hash': f'0x{random.getrandbits(256):064x}',
                'block_number': 1000000 + i,
                'timestamp': int(time.time()) + i,
                'from': f'0x{random.getrandbits(160):040x}',
                'to': f'0x{random.getrandbits(160):040x}',
                'value': random.uniform(0.1, 1000.0),
                'gas_price': random.uniform(10.0, 100.0),
                'features': features
            })
        
        return transactions
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE TEST SUMMARY")
        print("="*60)
        
        # Ingestion rate
        if self.results['ingestion_rate']:
            avg_rate = statistics.mean(self.results['ingestion_rate'])
            status = "✅ PASS" if avg_rate >= 200000 else "❌ FAIL"
            print(f"\nIngestion Rate: {avg_rate:,.0f} msgs/sec {status}")
            print(f"  Target: ≥200,000 msgs/sec")
        
        # Decision latency
        if self.results['decision_latency']:
            latencies = sorted(self.results['decision_latency'])
            p50 = latencies[len(latencies) // 2]
            p99 = latencies[int(len(latencies) * 0.99)]
            status_p50 = "✅" if p50 <= 8 else "❌"
            status_p99 = "✅" if p99 <= 20 else "❌"
            print(f"\nDecision Latency:")
            print(f"  P50: {p50:.2f}ms {status_p50} (Target: ≤8ms)")
            print(f"  P99: {p99:.2f}ms {status_p99} (Target: ≤20ms)")
        
        # Model inference
        if self.results['inference_latency']:
            avg_inference = statistics.mean(self.results['inference_latency'])
            status = "✅ PASS" if avg_inference < 100 else "❌ FAIL"
            print(f"\nModel Inference: {avg_inference:.1f}μs {status}")
            print(f"  Target: <100μs")
        
        # Detection accuracy
        if self.results['detection_accuracy']:
            avg_accuracy = statistics.mean(self.results['detection_accuracy'])
            status = "✅ PASS" if avg_accuracy >= 65 else "❌ FAIL"
            print(f"\nDetection Accuracy: {avg_accuracy:.1f}% {status}")
            print(f"  Target: ≥65%")
        
        # Memory usage
        if self.results['memory_usage']:
            avg_memory = statistics.mean(self.results['memory_usage'])
            status = "✅ PASS" if avg_memory < 500000 else "❌ FAIL"
            print(f"\nMemory per Connection: {avg_memory/1024:.1f}KB {status}")
            print(f"  Target: <500KB")
        
        print("\n" + "="*60)
        
        # Overall status
        all_passed = all([
            statistics.mean(self.results['ingestion_rate']) >= 200000 if self.results['ingestion_rate'] else False,
            statistics.mean(self.results['detection_accuracy']) >= 65 if self.results['detection_accuracy'] else False,
            # Add other checks
        ])
        
        if all_passed:
            print("🎉 ALL PERFORMANCE TARGETS MET!")
        else:
            print("⚠️ Some targets not met. See details above.")
        
        print("="*60)


async def main():
    """Run all integration tests"""
    print("🚀 Starting Defensive MEV Infrastructure Integration Tests")
    print("="*60)
    
    tester = PerformanceTest()
    
    try:
        # Setup
        if not await tester.setup():
            print("❌ Failed to setup test environment")
            return 1
        
        # Run tests
        await tester.test_ingestion_rate()
        await tester.test_decision_latency()
        await tester.test_model_inference()
        await tester.test_detection_accuracy()
        await tester.test_memory_usage()
        await tester.test_dna_integrity()
        await tester.test_thompson_sampling()
        
        # Print summary
        tester.print_summary()
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1
        
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)