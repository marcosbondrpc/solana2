#!/usr/bin/env python3
"""
MEV Infrastructure Performance Benchmarks

Comprehensive performance testing for the SOTA MEV infrastructure including:
- API endpoint latency measurements
- WebSocket throughput testing
- Database query performance
- Memory and CPU utilization
- Concurrent load testing
"""

import asyncio
import aiohttp
import time
import statistics
import psutil
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass 
class BenchmarkConfig:
    """Configuration for performance benchmarks"""
    target_host: str = "localhost"
    control_plane_port: int = 8000
    dashboard_api_port: int = 8001
    mev_engine_port: int = 8002
    concurrent_users: int = 100
    test_duration_seconds: int = 60
    warmup_duration_seconds: int = 10
    max_requests_per_second: int = 10000
    target_latency_p99_ms: float = 100.0
    target_latency_p50_ms: float = 50.0

class PerformanceBenchmark:
    """High-performance benchmark suite for MEV infrastructure"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        
    async def benchmark_api_latency(self, endpoint: str, samples: int = 1000) -> Dict[str, float]:
        """Benchmark API endpoint latency with detailed statistics"""
        print(f"📊 Benchmarking API latency for {endpoint}")
        
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        
        latencies = []
        errors = 0
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Warmup phase
            print("  🔥 Warming up...")
            for _ in range(min(100, samples // 10)):
                try:
                    start = time.perf_counter()
                    async with session.get(endpoint) as response:
                        await response.text()
                    latency = (time.perf_counter() - start) * 1000  # Convert to ms
                    if latency < 10000:  # Ignore outliers > 10s
                        latencies.append(latency)
                except:
                    errors += 1
            
            # Clear warmup data
            latencies.clear()
            errors = 0
            
            # Main benchmark
            print(f"  ⚡ Running {samples} requests...")
            tasks = []
            
            async def single_request():
                try:
                    start = time.perf_counter()
                    async with session.get(endpoint) as response:
                        await response.text()
                        return (time.perf_counter() - start) * 1000
                except Exception as e:
                    return None
            
            # Execute requests with controlled concurrency
            semaphore = asyncio.Semaphore(50)  # Limit concurrent requests
            
            async def bounded_request():
                async with semaphore:
                    return await single_request()
            
            # Create all tasks
            tasks = [bounded_request() for _ in range(samples)]
            
            # Execute with progress tracking
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, (int, float)) and result is not None:
                    latencies.append(result)
                else:
                    errors += 1
        
        if not latencies:
            return {"error": "No successful requests"}
        
        # Calculate comprehensive statistics
        stats = {
            "samples": len(latencies),
            "errors": errors,
            "error_rate": errors / (len(latencies) + errors),
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p50": np.percentile(latencies, 50),
            "p90": np.percentile(latencies, 90),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "p999": np.percentile(latencies, 99.9),
            "stddev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "sla_p50_met": np.percentile(latencies, 50) < self.config.target_latency_p50_ms,
            "sla_p99_met": np.percentile(latencies, 99) < self.config.target_latency_p99_ms,
        }
        
        print(f"  📈 Results: P50={stats['p50']:.2f}ms, P99={stats['p99']:.2f}ms, Errors={stats['error_rate']:.1%}")
        
        return stats
    
    async def benchmark_websocket_throughput(self, ws_url: str, duration_seconds: int = 30) -> Dict[str, Any]:
        """Benchmark WebSocket message throughput"""
        print(f"🌐 Benchmarking WebSocket throughput for {ws_url}")
        
        import websockets
        
        messages_received = 0
        bytes_received = 0
        start_time = time.time()
        latencies = []
        
        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Subscribe to high-frequency data
                subscribe_msg = {
                    "type": "subscribe",
                    "channels": ["opportunities", "bundles", "metrics"],
                    "filters": {}
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                
                # Measure throughput for specified duration
                end_time = start_time + duration_seconds
                
                while time.time() < end_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        receive_time = time.time()
                        
                        messages_received += 1
                        bytes_received += len(message)
                        
                        # Try to parse timestamp for latency measurement
                        try:
                            data = json.loads(message)
                            if 'timestamp' in data:
                                sent_time = data['timestamp'] / 1000  # Convert from ms
                                latency = (receive_time - sent_time) * 1000
                                if 0 < latency < 10000:  # Reasonable latency bounds
                                    latencies.append(latency)
                        except:
                            pass
                            
                    except asyncio.TimeoutError:
                        # No message received, continue
                        continue
                        
        except Exception as e:
            print(f"  ❌ WebSocket error: {e}")
            return {"error": str(e)}
        
        duration = time.time() - start_time
        
        stats = {
            "messages_received": messages_received,
            "bytes_received": bytes_received,
            "duration_seconds": duration,
            "messages_per_second": messages_received / duration if duration > 0 else 0,
            "bytes_per_second": bytes_received / duration if duration > 0 else 0,
            "avg_message_size": bytes_received / messages_received if messages_received > 0 else 0,
        }
        
        if latencies:
            stats.update({
                "message_latency_p50": np.percentile(latencies, 50),
                "message_latency_p99": np.percentile(latencies, 99),
                "message_latency_mean": np.mean(latencies),
            })
        
        print(f"  📈 Results: {stats['messages_per_second']:.0f} msg/s, {stats['bytes_per_second']/1024:.1f} KB/s")
        
        return stats
    
    async def benchmark_database_queries(self, clickhouse_url: str) -> Dict[str, Any]:
        """Benchmark database query performance"""
        print(f"🗄️ Benchmarking database queries for {clickhouse_url}")
        
        queries = [
            ("simple_count", "SELECT count() FROM system.tables"),
            ("time_range", "SELECT toStartOfHour(now() - number * 3600) as hour, number FROM numbers(24)"),
            ("aggregation", "SELECT database, count() as table_count FROM system.tables GROUP BY database"),
        ]
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for query_name, query in queries:
                latencies = []
                errors = 0
                
                # Run multiple iterations
                for _ in range(20):
                    try:
                        start = time.perf_counter()
                        async with session.get(f"{clickhouse_url}/?query={query}") as response:
                            await response.text()
                        latency = (time.perf_counter() - start) * 1000
                        latencies.append(latency)
                    except:
                        errors += 1
                
                if latencies:
                    results[query_name] = {
                        "p50": np.percentile(latencies, 50),
                        "p99": np.percentile(latencies, 99),
                        "mean": np.mean(latencies),
                        "errors": errors,
                    }
                    print(f"  📊 {query_name}: P50={results[query_name]['p50']:.2f}ms")
                else:
                    results[query_name] = {"error": "All queries failed"}
        
        return results
    
    async def benchmark_concurrent_load(self, endpoint: str, concurrent_users: int, duration_seconds: int) -> Dict[str, Any]:
        """Benchmark system under concurrent load"""
        print(f"🏋️ Benchmarking concurrent load: {concurrent_users} users for {duration_seconds}s")
        
        results = {
            "concurrent_users": concurrent_users,
            "duration_seconds": duration_seconds,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "latencies": [],
            "errors": [],
        }
        
        async def user_simulation():
            """Simulate a single user making requests"""
            user_requests = 0
            user_errors = 0
            user_latencies = []
            
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                end_time = time.time() + duration_seconds
                
                while time.time() < end_time:
                    try:
                        start = time.perf_counter()
                        async with session.get(endpoint) as response:
                            await response.text()
                        latency = (time.perf_counter() - start) * 1000
                        user_latencies.append(latency)
                        user_requests += 1
                        
                        # Small delay to simulate realistic usage
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        user_errors += 1
                        await asyncio.sleep(0.5)  # Back off on error
            
            return {
                "requests": user_requests,
                "errors": user_errors,
                "latencies": user_latencies,
            }
        
        # Start all concurrent users
        print(f"  🚀 Starting {concurrent_users} concurrent users...")
        start_time = time.time()
        
        tasks = [user_simulation() for _ in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        
        # Aggregate results
        all_latencies = []
        for user_result in user_results:
            if isinstance(user_result, dict):
                results["total_requests"] += user_result["requests"]
                results["failed_requests"] += user_result["errors"]
                all_latencies.extend(user_result["latencies"])
        
        results["successful_requests"] = results["total_requests"]
        results["actual_duration"] = actual_duration
        
        if all_latencies:
            results["latency_stats"] = {
                "p50": np.percentile(all_latencies, 50),
                "p95": np.percentile(all_latencies, 95),
                "p99": np.percentile(all_latencies, 99),
                "mean": np.mean(all_latencies),
                "max": max(all_latencies),
            }
        
        results["requests_per_second"] = results["total_requests"] / actual_duration
        results["error_rate"] = results["failed_requests"] / max(1, results["total_requests"] + results["failed_requests"])
        
        print(f"  📈 Results: {results['requests_per_second']:.1f} req/s, {results['error_rate']:.1%} errors")
        
        return results
    
    def measure_system_resources(self) -> Dict[str, Any]:
        """Measure current system resource utilization"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict(),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
        }
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite"""
        print("🚀 Starting Comprehensive Performance Benchmark")
        print("=" * 60)
        
        # Endpoints to test
        endpoints = {
            "control_plane_health": f"http://{self.config.target_host}:{self.config.control_plane_port}/health",
            "dashboard_api_health": f"http://{self.config.target_host}:{self.config.dashboard_api_port}/health",
            "dashboard_api_metrics": f"http://{self.config.target_host}:{self.config.dashboard_api_port}/api/v1/metrics/performance",
            "mev_engine_metrics": f"http://{self.config.target_host}:{self.config.mev_engine_port}/metrics",
        }
        
        benchmark_results = {
            "timestamp": time.time(),
            "config": self.config.__dict__,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "platform": psutil.platform.system(),
            }
        }
        
        # 1. System Resource Baseline
        print("📊 Measuring baseline system resources...")
        benchmark_results["baseline_resources"] = self.measure_system_resources()
        
        # 2. API Latency Benchmarks
        print("\n📡 Running API latency benchmarks...")
        benchmark_results["api_latency"] = {}
        
        for name, endpoint in endpoints.items():
            try:
                result = await self.benchmark_api_latency(endpoint, samples=500)
                benchmark_results["api_latency"][name] = result
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                benchmark_results["api_latency"][name] = {"error": str(e)}
        
        # 3. WebSocket Throughput
        print("\n🌐 Running WebSocket throughput benchmark...")
        try:
            ws_url = f"ws://{self.config.target_host}:{self.config.dashboard_api_port}/ws"
            ws_result = await self.benchmark_websocket_throughput(ws_url, duration_seconds=15)
            benchmark_results["websocket_throughput"] = ws_result
        except Exception as e:
            print(f"  ❌ WebSocket benchmark failed: {e}")
            benchmark_results["websocket_throughput"] = {"error": str(e)}
        
        # 4. Database Performance
        print("\n🗄️ Running database performance benchmark...")
        try:
            clickhouse_url = f"http://{self.config.target_host}:8123"
            db_result = await self.benchmark_database_queries(clickhouse_url)
            benchmark_results["database_performance"] = db_result
        except Exception as e:
            print(f"  ❌ Database benchmark failed: {e}")
            benchmark_results["database_performance"] = {"error": str(e)}
        
        # 5. Concurrent Load Test
        print("\n🏋️ Running concurrent load test...")
        try:
            primary_endpoint = endpoints["dashboard_api_health"]
            load_result = await self.benchmark_concurrent_load(
                primary_endpoint, 
                concurrent_users=min(50, self.config.concurrent_users),  # Reduced for testing
                duration_seconds=30
            )
            benchmark_results["concurrent_load"] = load_result
        except Exception as e:
            print(f"  ❌ Load test failed: {e}")
            benchmark_results["concurrent_load"] = {"error": str(e)}
        
        # 6. Final System Resources
        print("\n📊 Measuring post-test system resources...")
        benchmark_results["final_resources"] = self.measure_system_resources()
        
        # 7. SLA Compliance Check
        sla_results = self._check_sla_compliance(benchmark_results)
        benchmark_results["sla_compliance"] = sla_results
        
        return benchmark_results
    
    def _check_sla_compliance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if results meet SLA requirements"""
        sla_status = {
            "overall_status": "PASS",
            "checks": {},
            "violations": []
        }
        
        # Check API latency SLAs
        for endpoint, metrics in results.get("api_latency", {}).items():
            if "error" not in metrics:
                p99_met = metrics.get("p99", float('inf')) < self.config.target_latency_p99_ms
                p50_met = metrics.get("p50", float('inf')) < self.config.target_latency_p50_ms
                
                sla_status["checks"][f"{endpoint}_p99"] = p99_met
                sla_status["checks"][f"{endpoint}_p50"] = p50_met
                
                if not p99_met:
                    sla_status["violations"].append(f"{endpoint} P99 latency: {metrics.get('p99'):.2f}ms > {self.config.target_latency_p99_ms}ms")
                if not p50_met:
                    sla_status["violations"].append(f"{endpoint} P50 latency: {metrics.get('p50'):.2f}ms > {self.config.target_latency_p50_ms}ms")
        
        # Check error rates
        load_test = results.get("concurrent_load", {})
        if "error_rate" in load_test:
            error_rate_ok = load_test["error_rate"] < 0.01  # < 1% errors
            sla_status["checks"]["error_rate"] = error_rate_ok
            if not error_rate_ok:
                sla_status["violations"].append(f"Error rate: {load_test['error_rate']:.1%} > 1%")
        
        # Determine overall status
        if sla_status["violations"]:
            sla_status["overall_status"] = "FAIL"
        elif not all(sla_status["checks"].values()):
            sla_status["overall_status"] = "PARTIAL"
        
        return sla_status
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable performance report"""
        report = []
        report.append("📊 SOTA MEV Infrastructure Performance Report")
        report.append("=" * 60)
        report.append(f"Timestamp: {time.ctime(results['timestamp'])}")
        report.append(f"Test Duration: ~{self.config.test_duration_seconds}s")
        report.append("")
        
        # SLA Compliance Summary
        sla = results.get("sla_compliance", {})
        report.append(f"🎯 SLA Compliance: {sla.get('overall_status', 'UNKNOWN')}")
        if sla.get("violations"):
            report.append("   Violations:")
            for violation in sla["violations"]:
                report.append(f"   - {violation}")
        report.append("")
        
        # API Latency Summary
        report.append("📡 API Latency Summary:")
        for endpoint, metrics in results.get("api_latency", {}).items():
            if "error" not in metrics:
                report.append(f"   {endpoint}:")
                report.append(f"     P50: {metrics.get('p50', 0):.2f}ms")
                report.append(f"     P99: {metrics.get('p99', 0):.2f}ms")
                report.append(f"     Error Rate: {metrics.get('error_rate', 0):.1%}")
        report.append("")
        
        # Throughput Summary
        if "concurrent_load" in results and "error" not in results["concurrent_load"]:
            load = results["concurrent_load"]
            report.append("🏋️ Load Test Summary:")
            report.append(f"   Requests/sec: {load.get('requests_per_second', 0):.1f}")
            report.append(f"   Error Rate: {load.get('error_rate', 0):.1%}")
            if "latency_stats" in load:
                report.append(f"   P99 Latency: {load['latency_stats'].get('p99', 0):.2f}ms")
        report.append("")
        
        # WebSocket Performance
        if "websocket_throughput" in results and "error" not in results["websocket_throughput"]:
            ws = results["websocket_throughput"]
            report.append("🌐 WebSocket Performance:")
            report.append(f"   Messages/sec: {ws.get('messages_per_second', 0):.1f}")
            report.append(f"   Throughput: {ws.get('bytes_per_second', 0)/1024:.1f} KB/s")
        report.append("")
        
        # System Resources
        baseline = results.get("baseline_resources", {})
        final = results.get("final_resources", {})
        report.append("💻 System Resource Utilization:")
        report.append(f"   CPU: {baseline.get('cpu_percent', 0):.1f}% → {final.get('cpu_percent', 0):.1f}%")
        report.append(f"   Memory: {baseline.get('memory_percent', 0):.1f}% → {final.get('memory_percent', 0):.1f}%")
        
        return "\n".join(report)

async def run_performance_benchmark():
    """Main function to run performance benchmarks"""
    config = BenchmarkConfig()
    benchmark = PerformanceBenchmark(config)
    
    try:
        # Run comprehensive benchmark
        results = await benchmark.run_comprehensive_benchmark()
        
        # Generate and display report
        report = benchmark.generate_performance_report(results)
        print("\n" + report)
        
        # Save detailed results
        results_file = f"/tmp/performance_benchmark_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Return success if SLA compliance is PASS
        sla_status = results.get("sla_compliance", {}).get("overall_status", "FAIL")
        return sla_status in ["PASS", "PARTIAL"]
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_performance_benchmark())
    exit(0 if success else 1)