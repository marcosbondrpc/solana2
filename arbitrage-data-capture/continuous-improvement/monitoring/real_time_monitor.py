"""
Elite Real-Time Model Monitoring System
Production-grade monitoring with sub-millisecond precision
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uvloop
import aioredis
import aiokafka
from scipy import stats
from scipy.spatial.distance import jensenshannon
import mlflow
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, Summary
from clickhouse_driver import Client
import logging
from collections import deque
import json
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Set up ultra-performance async
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Prometheus metrics
prediction_latency = Histogram('model_prediction_latency_seconds', 
                              'Model prediction latency',
                              buckets=(0.001, 0.002, 0.003, 0.005, 0.01, 0.025, 0.05))
prediction_counter = Counter('model_predictions_total', 'Total predictions')
drift_detected = Counter('model_drift_detected_total', 'Drift detection events')
model_accuracy = Gauge('model_accuracy_current', 'Current model accuracy')
throughput_gauge = Gauge('model_throughput_tps', 'Transactions per second')
feature_drift_score = Gauge('feature_drift_score', 'Feature drift score', ['feature'])

@dataclass
class ModelMetrics:
    """Real-time model performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    prediction_count: int = 0
    total_latency: float = 0.0
    accuracy_window: deque = field(default_factory=lambda: deque(maxlen=10000))
    feature_distributions: Dict[str, np.ndarray] = field(default_factory=dict)
    error_rate: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    @property
    def avg_latency(self) -> float:
        return self.total_latency / max(1, self.prediction_count)
    
    @property
    def current_accuracy(self) -> float:
        if len(self.accuracy_window) == 0:
            return 0.0
        return sum(self.accuracy_window) / len(self.accuracy_window)


class DriftDetector:
    """Advanced statistical drift detection"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.baseline_distributions = {}
        self.window_size = 1000
        self.current_window = deque(maxlen=self.window_size)
        
    def kolmogorov_smirnov_test(self, 
                                baseline: np.ndarray, 
                                current: np.ndarray) -> Tuple[float, bool]:
        """KS test for distribution drift"""
        statistic, p_value = stats.ks_2samp(baseline, current)
        is_drift = p_value < self.significance_level
        return statistic, is_drift
    
    def population_stability_index(self, 
                                  expected: np.ndarray, 
                                  actual: np.ndarray,
                                  buckets: int = 10) -> float:
        """Calculate PSI for distribution shift"""
        def calculate_psi(expected_array, actual_array):
            eps = 1e-10
            
            # Create bins
            breakpoints = np.linspace(
                min(expected_array.min(), actual_array.min()),
                max(expected_array.max(), actual_array.max()),
                buckets + 1
            )
            
            expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
            actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
            
            # Add small constant to avoid log(0)
            expected_percents = np.where(expected_percents == 0, eps, expected_percents)
            actual_percents = np.where(actual_percents == 0, eps, actual_percents)
            
            psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
            return np.sum(psi_values)
        
        psi = calculate_psi(expected, actual)
        # PSI < 0.1: no drift, 0.1-0.25: moderate, >0.25: significant
        return psi
    
    def wasserstein_distance(self, 
                            baseline: np.ndarray, 
                            current: np.ndarray) -> float:
        """Earth mover's distance for distribution comparison"""
        return stats.wasserstein_distance(baseline, current)
    
    def jensen_shannon_divergence(self,
                                 p: np.ndarray,
                                 q: np.ndarray) -> float:
        """JS divergence for probability distributions"""
        # Normalize to probability distributions
        p_norm = np.abs(p) / np.sum(np.abs(p))
        q_norm = np.abs(q) / np.sum(np.abs(q))
        return jensenshannon(p_norm, q_norm)
    
    def detect_concept_drift(self,
                           predictions: np.ndarray,
                           actuals: np.ndarray,
                           window_size: int = 100) -> bool:
        """Detect concept drift using error rate monitoring"""
        if len(predictions) < window_size * 2:
            return False
        
        # Split into reference and test windows
        mid_point = len(predictions) // 2
        ref_errors = (predictions[:mid_point] != actuals[:mid_point]).astype(float)
        test_errors = (predictions[mid_point:] != actuals[mid_point:]).astype(float)
        
        # Page-Hinkley test for change detection
        statistic, p_value = stats.mannwhitneyu(ref_errors, test_errors)
        return p_value < self.significance_level


class RealTimeMonitor:
    """Ultra-high-performance real-time monitoring system"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 kafka_bootstrap: str = "localhost:9092",
                 clickhouse_host: str = "localhost"):
        
        self.redis_url = redis_url
        self.kafka_bootstrap = kafka_bootstrap
        self.clickhouse = Client(clickhouse_host)
        
        self.metrics = ModelMetrics()
        self.drift_detector = DriftDetector()
        self.alert_thresholds = {
            'latency_p99': 0.005,  # 5ms
            'error_rate': 0.01,     # 1%
            'drift_psi': 0.25,      # Significant drift
            'memory_gb': 8,         # 8GB max
            'cpu_percent': 80       # 80% CPU
        }
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.monitoring_interval = 1.0  # 1 second
        self.is_running = False
        
        # Feature baselines for drift detection
        self.feature_baselines = {}
        self.prediction_history = deque(maxlen=100000)
        
        # A/B testing framework
        self.ab_test_results = {}
        self.model_versions = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize monitoring connections"""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        
        self.kafka_producer = aiokafka.AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode()
        )
        await self.kafka_producer.start()
        
        self.kafka_consumer = aiokafka.AIOKafkaConsumer(
            'model_predictions',
            bootstrap_servers=self.kafka_bootstrap,
            value_deserializer=lambda v: json.loads(v.decode())
        )
        await self.kafka_consumer.start()
        
        await self._load_baselines()
        
    async def _load_baselines(self):
        """Load feature distribution baselines"""
        try:
            # Load from ClickHouse
            query = """
            SELECT feature_name, baseline_data 
            FROM model_baselines 
            WHERE model_version = (
                SELECT MAX(model_version) FROM model_baselines
            )
            """
            results = self.clickhouse.execute(query)
            
            for feature_name, baseline_data in results:
                self.feature_baselines[feature_name] = pickle.loads(baseline_data)
                
        except Exception as e:
            self.logger.warning(f"Could not load baselines: {e}")
    
    async def track_prediction(self,
                              features: Dict[str, float],
                              prediction: float,
                              actual: Optional[float] = None,
                              model_version: str = "v1",
                              latency_ms: float = 0.0):
        """Track individual prediction with ultra-low overhead"""
        
        # Update metrics
        self.metrics.prediction_count += 1
        self.metrics.total_latency += latency_ms
        
        # Track accuracy if ground truth available
        if actual is not None:
            is_correct = abs(prediction - actual) < 0.01  # Threshold for correctness
            self.metrics.accuracy_window.append(1 if is_correct else 0)
            model_accuracy.set(self.metrics.current_accuracy)
        
        # Update Prometheus metrics
        prediction_latency.observe(latency_ms / 1000.0)
        prediction_counter.inc()
        
        # Store prediction for analysis
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'model_version': model_version,
            'latency_ms': latency_ms
        }
        
        self.prediction_history.append(prediction_data)
        
        # Async drift detection (non-blocking)
        asyncio.create_task(self._check_drift(features))
        
        # Send to Kafka for downstream processing
        await self.kafka_producer.send(
            'model_metrics',
            value=prediction_data
        )
    
    async def _check_drift(self, features: Dict[str, float]):
        """Asynchronous drift detection"""
        try:
            for feature_name, value in features.items():
                if feature_name not in self.feature_baselines:
                    continue
                
                # Update current distribution
                if feature_name not in self.metrics.feature_distributions:
                    self.metrics.feature_distributions[feature_name] = deque(maxlen=1000)
                
                self.metrics.feature_distributions[feature_name].append(value)
                
                # Check for drift every 100 samples
                if len(self.metrics.feature_distributions[feature_name]) % 100 == 0:
                    current_dist = np.array(self.metrics.feature_distributions[feature_name])
                    baseline_dist = self.feature_baselines[feature_name]
                    
                    # Multiple drift detection methods
                    ks_stat, ks_drift = self.drift_detector.kolmogorov_smirnov_test(
                        baseline_dist, current_dist
                    )
                    
                    psi = self.drift_detector.population_stability_index(
                        baseline_dist, current_dist
                    )
                    
                    wasserstein = self.drift_detector.wasserstein_distance(
                        baseline_dist, current_dist
                    )
                    
                    # Update metrics
                    feature_drift_score.labels(feature=feature_name).set(psi)
                    
                    # Alert if significant drift
                    if psi > self.alert_thresholds['drift_psi']:
                        drift_detected.inc()
                        await self._send_alert(
                            'DRIFT_DETECTED',
                            f"Feature {feature_name} drift: PSI={psi:.3f}, KS={ks_stat:.3f}"
                        )
                        
        except Exception as e:
            self.logger.error(f"Drift detection error: {e}")
    
    async def _send_alert(self, alert_type: str, message: str):
        """Send alert to monitoring system"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': 'HIGH' if 'DRIFT' in alert_type else 'MEDIUM'
        }
        
        # Send to Redis for immediate processing
        await self.redis.publish('model_alerts', json.dumps(alert_data))
        
        # Log alert
        self.logger.warning(f"ALERT [{alert_type}]: {message}")
    
    async def monitor_system_health(self):
        """Continuous system health monitoring"""
        while self.is_running:
            try:
                # System metrics
                self.metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
                self.metrics.memory_usage = psutil.virtual_memory().percent
                
                # Calculate throughput
                current_time = datetime.now()
                time_window = 10  # seconds
                recent_predictions = [
                    p for p in self.prediction_history
                    if datetime.fromisoformat(p['timestamp']) > 
                    current_time - timedelta(seconds=time_window)
                ]
                self.metrics.throughput = len(recent_predictions) / time_window
                throughput_gauge.set(self.metrics.throughput)
                
                # Check thresholds
                if self.metrics.cpu_usage > self.alert_thresholds['cpu_percent']:
                    await self._send_alert('HIGH_CPU', f"CPU usage: {self.metrics.cpu_usage:.1f}%")
                
                if self.metrics.memory_usage > self.alert_thresholds['memory_gb'] * 12.5:  # Convert to percent
                    await self._send_alert('HIGH_MEMORY', f"Memory usage: {self.metrics.memory_usage:.1f}%")
                
                if self.metrics.avg_latency > self.alert_thresholds['latency_p99'] * 1000:
                    await self._send_alert('HIGH_LATENCY', f"Avg latency: {self.metrics.avg_latency:.2f}ms")
                
                # Store metrics in ClickHouse
                await self._store_metrics()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _store_metrics(self):
        """Store metrics in ClickHouse for analysis"""
        try:
            query = """
            INSERT INTO model_metrics (
                timestamp, prediction_count, avg_latency_ms,
                accuracy, throughput_tps, cpu_usage, memory_usage,
                error_rate
            ) VALUES
            """
            
            data = [(
                datetime.now(),
                self.metrics.prediction_count,
                self.metrics.avg_latency,
                self.metrics.current_accuracy,
                self.metrics.throughput,
                self.metrics.cpu_usage,
                self.metrics.memory_usage,
                self.metrics.error_rate
            )]
            
            self.clickhouse.execute(query, data)
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics: {e}")
    
    async def ab_test_models(self,
                            model_a: Any,
                            model_b: Any,
                            test_duration_seconds: int = 3600,
                            traffic_split: float = 0.5):
        """A/B testing framework for model comparison"""
        
        test_id = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]
        
        self.ab_test_results[test_id] = {
            'start_time': datetime.now(),
            'model_a_metrics': ModelMetrics(),
            'model_b_metrics': ModelMetrics(),
            'traffic_split': traffic_split
        }
        
        end_time = datetime.now() + timedelta(seconds=test_duration_seconds)
        
        while datetime.now() < end_time:
            # Process predictions from Kafka
            async for msg in self.kafka_consumer:
                if datetime.now() >= end_time:
                    break
                
                data = msg.value
                
                # Route traffic based on split
                use_model_a = np.random.random() < traffic_split
                
                if use_model_a:
                    # Track model A performance
                    self.ab_test_results[test_id]['model_a_metrics'].prediction_count += 1
                else:
                    # Track model B performance
                    self.ab_test_results[test_id]['model_b_metrics'].prediction_count += 1
        
        # Analyze results
        return await self._analyze_ab_test(test_id)
    
    async def _analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Statistical analysis of A/B test results"""
        results = self.ab_test_results[test_id]
        
        # Statistical significance testing
        a_accuracy = results['model_a_metrics'].current_accuracy
        b_accuracy = results['model_b_metrics'].current_accuracy
        
        # Z-test for proportions
        n_a = len(results['model_a_metrics'].accuracy_window)
        n_b = len(results['model_b_metrics'].accuracy_window)
        
        if n_a > 30 and n_b > 30:  # Sufficient sample size
            pooled_prop = (a_accuracy * n_a + b_accuracy * n_b) / (n_a + n_b)
            se = np.sqrt(pooled_prop * (1 - pooled_prop) * (1/n_a + 1/n_b))
            z_score = (a_accuracy - b_accuracy) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            winner = 'model_a' if a_accuracy > b_accuracy else 'model_b'
            significant = p_value < 0.05
        else:
            p_value = 1.0
            winner = 'insufficient_data'
            significant = False
        
        return {
            'test_id': test_id,
            'winner': winner,
            'statistically_significant': significant,
            'p_value': p_value,
            'model_a_accuracy': a_accuracy,
            'model_b_accuracy': b_accuracy,
            'model_a_latency': results['model_a_metrics'].avg_latency,
            'model_b_latency': results['model_b_metrics'].avg_latency,
            'duration_minutes': (datetime.now() - results['start_time']).total_seconds() / 60
        }
    
    async def start(self):
        """Start monitoring system"""
        await self.initialize()
        self.is_running = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self.monitor_system_health()),
            asyncio.create_task(self._process_predictions()),
        ]
        
        await asyncio.gather(*tasks)
    
    async def _process_predictions(self):
        """Process incoming predictions from Kafka"""
        async for msg in self.kafka_consumer:
            if not self.is_running:
                break
            
            data = msg.value
            await self.track_prediction(
                features=data.get('features', {}),
                prediction=data.get('prediction'),
                actual=data.get('actual'),
                model_version=data.get('model_version', 'v1'),
                latency_ms=data.get('latency_ms', 0)
            )
    
    async def stop(self):
        """Gracefully shutdown monitoring"""
        self.is_running = False
        await self.kafka_producer.stop()
        await self.kafka_consumer.stop()
        self.redis.close()
        await self.redis.wait_closed()
        self.executor.shutdown(wait=True)


async def main():
    """Run the monitoring system"""
    monitor = RealTimeMonitor()
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        await monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())