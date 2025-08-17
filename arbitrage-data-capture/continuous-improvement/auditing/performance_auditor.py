"""
Elite Performance Auditing System
Automated profiling, bottleneck detection, and predictive analytics
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import py_spy
import cProfile
import pstats
import io
import tracemalloc
import linecache
import os
import sys
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
import psutil
import GPUtil
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import prophet
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import aiofiles
import json
import yaml
import hashlib
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import uvloop
import warnings
warnings.filterwarnings('ignore')

# Ultra-performance async
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Distributed tracing setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument asyncio for tracing
AsyncioInstrumentor().instrument()


@dataclass
class PerformanceProfile:
    """Performance profile data"""
    timestamp: datetime = field(default_factory=datetime.now)
    function_timings: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    io_stats: Dict[str, int] = field(default_factory=dict)
    network_stats: Dict[str, int] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    

@dataclass
class SystemHealth:
    """System health metrics"""
    score: float = 100.0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    critical_alerts: List[str] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    predicted_failures: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedProfiler:
    """Advanced profiling with py-spy and cProfile"""
    
    def __init__(self):
        self.profiles = []
        self.flame_graphs = {}
        self.profile_dir = Path("/home/kidgordones/0solana/node/arbitrage-data-capture/continuous-improvement/profiles")
        self.profile_dir.mkdir(exist_ok=True, parents=True)
        
    async def profile_function(self, func, *args, **kwargs):
        """Profile a specific function"""
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # Generate stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        
        profile_data = s.getvalue()
        
        # Save profile
        profile_file = self.profile_dir / f"profile_{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
        profiler.dump_stats(str(profile_file))
        
        return result, profile_data
    
    async def generate_flame_graph(self, pid: int, duration: int = 30):
        """Generate flame graph using py-spy"""
        
        output_file = self.profile_dir / f"flame_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
        
        cmd = f"py-spy record -d {duration} -p {pid} -o {output_file} -f flame"
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            logger.info(f"Flame graph generated: {output_file}")
            self.flame_graphs[pid] = str(output_file)
        else:
            logger.error(f"Failed to generate flame graph: {stderr.decode()}")
    
    async def memory_profile(self):
        """Profile memory usage"""
        
        tracemalloc.start()
        
        # Take snapshot
        snapshot = tracemalloc.take_snapshot()
        
        # Top memory consumers
        top_stats = snapshot.statistics('lineno')
        
        memory_report = []
        for stat in top_stats[:20]:
            frame = stat.traceback[0]
            filename = frame.filename
            line = linecache.getline(filename, frame.lineno).strip()
            
            memory_report.append({
                'file': filename,
                'line': frame.lineno,
                'code': line,
                'size': stat.size,
                'count': stat.count
            })
        
        tracemalloc.stop()
        
        return memory_report


class BottleneckDetector:
    """Intelligent bottleneck detection"""
    
    def __init__(self):
        self.trace_data = []
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.is_trained = False
        
    async def analyze_traces(self, traces: List[Dict[str, Any]]) -> List[str]:
        """Analyze distributed traces for bottlenecks"""
        
        bottlenecks = []
        
        # Group traces by operation
        operations = {}
        for trace in traces:
            op_name = trace.get('operation_name', 'unknown')
            duration = trace.get('duration_ms', 0)
            
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(duration)
        
        # Statistical analysis
        for op_name, durations in operations.items():
            if len(durations) < 10:
                continue
            
            durations_array = np.array(durations)
            mean_duration = np.mean(durations_array)
            std_duration = np.std(durations_array)
            p99_duration = np.percentile(durations_array, 99)
            
            # Detect anomalies
            if p99_duration > mean_duration + 3 * std_duration:
                bottlenecks.append(
                    f"Operation '{op_name}' has high variance: "
                    f"P99={p99_duration:.2f}ms, Mean={mean_duration:.2f}ms"
                )
            
            # Absolute threshold
            if p99_duration > 100:  # 100ms threshold
                bottlenecks.append(
                    f"Operation '{op_name}' exceeds latency threshold: {p99_duration:.2f}ms"
                )
        
        # Dependency analysis
        dependency_bottlenecks = await self._analyze_dependencies(traces)
        bottlenecks.extend(dependency_bottlenecks)
        
        return bottlenecks
    
    async def _analyze_dependencies(self, traces: List[Dict[str, Any]]) -> List[str]:
        """Analyze service dependencies for bottlenecks"""
        
        bottlenecks = []
        
        # Build dependency graph
        dependencies = {}
        for trace in traces:
            parent = trace.get('parent_span')
            child = trace.get('span_id')
            
            if parent and child:
                if parent not in dependencies:
                    dependencies[parent] = []
                dependencies[parent].append({
                    'child': child,
                    'duration': trace.get('duration_ms', 0)
                })
        
        # Find critical path
        for parent, children in dependencies.items():
            total_child_duration = sum(c['duration'] for c in children)
            if total_child_duration > 500:  # 500ms threshold
                bottlenecks.append(
                    f"Parent span {parent} has slow children: {total_child_duration:.2f}ms total"
                )
        
        return bottlenecks
    
    async def detect_anomalies(self, metrics: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect performance anomalies using ML"""
        
        if len(metrics) < 100:
            return []
        
        # Feature engineering
        features = metrics[['latency_ms', 'throughput', 'error_rate', 'cpu_usage', 'memory_usage']]
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Train or update anomaly detector
        if not self.is_trained:
            self.anomaly_detector.fit(scaled_features)
            self.is_trained = True
        
        # Predict anomalies
        anomaly_predictions = self.anomaly_detector.predict(scaled_features)
        anomaly_scores = self.anomaly_detector.score_samples(scaled_features)
        
        # Identify anomalies
        anomalies = []
        for idx, (pred, score) in enumerate(zip(anomaly_predictions, anomaly_scores)):
            if pred == -1:  # Anomaly
                anomalies.append({
                    'timestamp': metrics.iloc[idx]['timestamp'],
                    'severity': 'high' if score < -0.5 else 'medium',
                    'metrics': metrics.iloc[idx].to_dict(),
                    'anomaly_score': score
                })
        
        return anomalies


class HealthScorer:
    """System health scoring with actionable insights"""
    
    def __init__(self):
        self.weights = {
            'latency': 0.3,
            'throughput': 0.2,
            'error_rate': 0.25,
            'resource_usage': 0.15,
            'availability': 0.1
        }
        self.thresholds = {
            'latency_ms': {'good': 5, 'warning': 10, 'critical': 20},
            'error_rate': {'good': 0.001, 'warning': 0.01, 'critical': 0.05},
            'cpu_usage': {'good': 60, 'warning': 80, 'critical': 90},
            'memory_usage': {'good': 70, 'warning': 85, 'critical': 95}
        }
        
    async def calculate_health_score(self, metrics: Dict[str, float]) -> SystemHealth:
        """Calculate comprehensive health score"""
        
        health = SystemHealth()
        scores = {}
        
        # Latency score
        latency = metrics.get('latency_ms', 0)
        if latency < self.thresholds['latency_ms']['good']:
            scores['latency'] = 100
        elif latency < self.thresholds['latency_ms']['warning']:
            scores['latency'] = 80
        elif latency < self.thresholds['latency_ms']['critical']:
            scores['latency'] = 50
            health.warnings.append(f"High latency: {latency:.2f}ms")
        else:
            scores['latency'] = 20
            health.critical_alerts.append(f"Critical latency: {latency:.2f}ms")
        
        # Error rate score
        error_rate = metrics.get('error_rate', 0)
        if error_rate < self.thresholds['error_rate']['good']:
            scores['error_rate'] = 100
        elif error_rate < self.thresholds['error_rate']['warning']:
            scores['error_rate'] = 70
            health.warnings.append(f"Elevated error rate: {error_rate:.2%}")
        else:
            scores['error_rate'] = 30
            health.critical_alerts.append(f"High error rate: {error_rate:.2%}")
        
        # Resource usage scores
        cpu_usage = metrics.get('cpu_usage', 0)
        memory_usage = metrics.get('memory_usage', 0)
        
        resource_score = 100
        if cpu_usage > self.thresholds['cpu_usage']['critical']:
            resource_score -= 40
            health.critical_alerts.append(f"Critical CPU usage: {cpu_usage:.1f}%")
        elif cpu_usage > self.thresholds['cpu_usage']['warning']:
            resource_score -= 20
            health.warnings.append(f"High CPU usage: {cpu_usage:.1f}%")
        
        if memory_usage > self.thresholds['memory_usage']['critical']:
            resource_score -= 40
            health.critical_alerts.append(f"Critical memory usage: {memory_usage:.1f}%")
        elif memory_usage > self.thresholds['memory_usage']['warning']:
            resource_score -= 20
            health.warnings.append(f"High memory usage: {memory_usage:.1f}%")
        
        scores['resource_usage'] = max(20, resource_score)
        
        # Throughput score (relative to expected)
        throughput = metrics.get('throughput_rps', 0)
        expected_throughput = metrics.get('expected_throughput_rps', 10000)
        throughput_ratio = throughput / max(1, expected_throughput)
        scores['throughput'] = min(100, throughput_ratio * 100)
        
        # Availability score
        uptime = metrics.get('uptime_percent', 100)
        scores['availability'] = uptime
        
        # Calculate weighted score
        total_score = sum(
            scores.get(component, 100) * weight 
            for component, weight in self.weights.items()
        )
        
        health.score = total_score
        health.resource_usage = {
            'cpu': cpu_usage,
            'memory': memory_usage,
            'disk': metrics.get('disk_usage', 0),
            'network': metrics.get('network_usage', 0)
        }
        
        # Generate recommendations
        health = await self._generate_recommendations(health, metrics)
        
        return health
    
    async def _generate_recommendations(self, 
                                      health: SystemHealth, 
                                      metrics: Dict[str, float]) -> SystemHealth:
        """Generate actionable recommendations"""
        
        if health.score < 50:
            health.issues.append("System health is critical. Immediate action required.")
        elif health.score < 70:
            health.issues.append("System health is degraded. Investigation recommended.")
        
        # Specific recommendations
        if metrics.get('latency_ms', 0) > 10:
            health.issues.append("Consider scaling up compute resources or optimizing slow queries")
        
        if metrics.get('error_rate', 0) > 0.01:
            health.issues.append("Review error logs and implement retry mechanisms")
        
        if metrics.get('cpu_usage', 0) > 80:
            health.issues.append("CPU bottleneck detected. Scale horizontally or optimize CPU-intensive operations")
        
        if metrics.get('memory_usage', 0) > 85:
            health.issues.append("Memory pressure detected. Increase memory limits or optimize memory usage")
        
        return health


class PredictiveAnalytics:
    """Predictive analytics for capacity planning"""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        
    async def train_forecasting_models(self, historical_data: pd.DataFrame):
        """Train time series forecasting models"""
        
        # Prepare data for Prophet
        for metric in ['throughput', 'latency', 'cpu_usage', 'memory_usage']:
            if metric not in historical_data.columns:
                continue
            
            df_prophet = pd.DataFrame({
                'ds': historical_data['timestamp'],
                'y': historical_data[metric]
            })
            
            # Train Prophet model
            model = prophet.Prophet(
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative',
                daily_seasonality=True,
                weekly_seasonality=True
            )
            
            model.fit(df_prophet)
            self.models[metric] = model
            
            # Make predictions
            future = model.make_future_dataframe(periods=24*7, freq='H')
            forecast = model.predict(future)
            
            self.predictions[metric] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    async def predict_capacity_needs(self, horizon_days: int = 7) -> Dict[str, Any]:
        """Predict future capacity requirements"""
        
        capacity_predictions = {}
        
        for metric, forecast in self.predictions.items():
            # Get future predictions
            future_forecast = forecast[forecast['ds'] > datetime.now()]
            
            if len(future_forecast) == 0:
                continue
            
            # Find peak values
            peak_value = future_forecast['yhat_upper'].max()
            avg_value = future_forecast['yhat'].mean()
            
            capacity_predictions[metric] = {
                'current': forecast['yhat'].iloc[-1],
                'predicted_peak': peak_value,
                'predicted_avg': avg_value,
                'increase_percent': (peak_value / forecast['yhat'].iloc[-1] - 1) * 100
            }
        
        # Generate recommendations
        recommendations = []
        
        if 'cpu_usage' in capacity_predictions:
            cpu_pred = capacity_predictions['cpu_usage']
            if cpu_pred['predicted_peak'] > 80:
                recommendations.append(
                    f"CPU usage predicted to reach {cpu_pred['predicted_peak']:.1f}%. "
                    f"Consider adding {int(cpu_pred['predicted_peak'] / 60)} more instances."
                )
        
        if 'memory_usage' in capacity_predictions:
            mem_pred = capacity_predictions['memory_usage']
            if mem_pred['predicted_peak'] > 85:
                recommendations.append(
                    f"Memory usage predicted to reach {mem_pred['predicted_peak']:.1f}%. "
                    f"Increase memory by {int(mem_pred['increase_percent'])}%."
                )
        
        if 'throughput' in capacity_predictions:
            throughput_pred = capacity_predictions['throughput']
            recommendations.append(
                f"Throughput expected to reach {throughput_pred['predicted_peak']:.0f} RPS. "
                f"Ensure infrastructure can handle {throughput_pred['increase_percent']:.1f}% increase."
            )
        
        return {
            'predictions': capacity_predictions,
            'recommendations': recommendations,
            'horizon_days': horizon_days
        }
    
    async def detect_anomaly_patterns(self, metrics: pd.DataFrame) -> List[str]:
        """Detect recurring anomaly patterns"""
        
        patterns = []
        
        # Time-based patterns
        metrics['hour'] = pd.to_datetime(metrics['timestamp']).dt.hour
        metrics['day_of_week'] = pd.to_datetime(metrics['timestamp']).dt.dayofweek
        
        # Check for hourly patterns
        hourly_stats = metrics.groupby('hour')['latency_ms'].agg(['mean', 'std'])
        high_latency_hours = hourly_stats[hourly_stats['mean'] > hourly_stats['mean'].mean() + hourly_stats['std'].mean()]
        
        if len(high_latency_hours) > 0:
            patterns.append(
                f"High latency detected during hours: {list(high_latency_hours.index)}. "
                f"Consider scheduling maintenance or scaling during these periods."
            )
        
        # Day of week patterns
        daily_stats = metrics.groupby('day_of_week')['error_rate'].mean()
        high_error_days = daily_stats[daily_stats > daily_stats.mean() + daily_stats.std()]
        
        if len(high_error_days) > 0:
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            problem_days = [day_names[i] for i in high_error_days.index]
            patterns.append(
                f"Higher error rates on: {problem_days}. "
                f"Review deployment schedule and traffic patterns."
            )
        
        return patterns


class CostOptimizer:
    """Cost optimization recommendations"""
    
    def __init__(self):
        self.cost_per_cpu_hour = 0.05
        self.cost_per_gb_hour = 0.01
        self.cost_per_gb_transfer = 0.02
        
    async def analyze_costs(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze current costs and provide optimization recommendations"""
        
        # Calculate current costs
        cpu_cores = metrics.get('cpu_cores', 8)
        memory_gb = metrics.get('memory_gb', 32)
        network_gb = metrics.get('network_gb_daily', 100)
        
        daily_cost = (
            cpu_cores * self.cost_per_cpu_hour * 24 +
            memory_gb * self.cost_per_gb_hour * 24 +
            network_gb * self.cost_per_gb_transfer
        )
        
        monthly_cost = daily_cost * 30
        
        # Optimization opportunities
        recommendations = []
        potential_savings = 0
        
        # CPU optimization
        cpu_usage = metrics.get('cpu_usage', 50)
        if cpu_usage < 30:
            reducible_cores = int(cpu_cores * 0.3)
            savings = reducible_cores * self.cost_per_cpu_hour * 24 * 30
            potential_savings += savings
            recommendations.append({
                'type': 'cpu',
                'action': f'Reduce CPU cores by {reducible_cores}',
                'monthly_savings': savings
            })
        
        # Memory optimization
        memory_usage = metrics.get('memory_usage', 50)
        if memory_usage < 40:
            reducible_memory = int(memory_gb * 0.25)
            savings = reducible_memory * self.cost_per_gb_hour * 24 * 30
            potential_savings += savings
            recommendations.append({
                'type': 'memory',
                'action': f'Reduce memory by {reducible_memory}GB',
                'monthly_savings': savings
            })
        
        # Network optimization
        if network_gb > 500:
            recommendations.append({
                'type': 'network',
                'action': 'Consider implementing CDN or compression',
                'monthly_savings': network_gb * 0.1 * self.cost_per_gb_transfer * 30
            })
        
        return {
            'current_monthly_cost': monthly_cost,
            'potential_monthly_savings': potential_savings,
            'savings_percentage': (potential_savings / monthly_cost) * 100,
            'recommendations': recommendations
        }


class PerformanceAuditor:
    """Main performance auditing orchestrator"""
    
    def __init__(self):
        self.profiler = AdvancedProfiler()
        self.bottleneck_detector = BottleneckDetector()
        self.health_scorer = HealthScorer()
        self.predictive_analytics = PredictiveAnalytics()
        self.cost_optimizer = CostOptimizer()
        
        self.audit_interval = 300  # 5 minutes
        self.report_dir = Path("/home/kidgordones/0solana/node/arbitrage-data-capture/continuous-improvement/reports")
        self.report_dir.mkdir(exist_ok=True, parents=True)
        
    async def perform_audit(self) -> Dict[str, Any]:
        """Perform comprehensive performance audit"""
        
        logger.info("Starting performance audit")
        
        audit_results = {
            'timestamp': datetime.now().isoformat(),
            'profile': {},
            'bottlenecks': [],
            'health': {},
            'predictions': {},
            'cost_analysis': {},
            'recommendations': []
        }
        
        # Collect current metrics
        metrics = await self._collect_system_metrics()
        
        # Memory profiling
        memory_profile = await self.profiler.memory_profile()
        audit_results['profile']['memory'] = memory_profile
        
        # Detect bottlenecks
        traces = await self._collect_traces()
        bottlenecks = await self.bottleneck_detector.analyze_traces(traces)
        audit_results['bottlenecks'] = bottlenecks
        
        # Calculate health score
        health = await self.health_scorer.calculate_health_score(metrics)
        audit_results['health'] = {
            'score': health.score,
            'issues': health.issues,
            'warnings': health.warnings,
            'critical_alerts': health.critical_alerts,
            'resource_usage': health.resource_usage
        }
        
        # Predictive analytics
        historical_data = await self._load_historical_metrics()
        if len(historical_data) > 100:
            await self.predictive_analytics.train_forecasting_models(historical_data)
            predictions = await self.predictive_analytics.predict_capacity_needs()
            audit_results['predictions'] = predictions
            
            # Anomaly patterns
            patterns = await self.predictive_analytics.detect_anomaly_patterns(historical_data)
            audit_results['anomaly_patterns'] = patterns
        
        # Cost optimization
        cost_analysis = await self.cost_optimizer.analyze_costs(metrics)
        audit_results['cost_analysis'] = cost_analysis
        
        # Generate comprehensive recommendations
        audit_results['recommendations'] = await self._generate_recommendations(
            audit_results
        )
        
        # Save audit report
        await self._save_audit_report(audit_results)
        
        # Generate visualizations
        await self._generate_visualizations(audit_results, historical_data)
        
        logger.info(f"Audit complete. Health score: {health.score:.1f}")
        
        return audit_results
    
    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_cores = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_usage = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Network metrics
        network = psutil.net_io_counters()
        network_gb_daily = (network.bytes_sent + network.bytes_recv) / (1024**3)
        
        # GPU metrics (if available)
        gpu_usage = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except:
            pass
        
        return {
            'cpu_usage': cpu_percent,
            'cpu_cores': cpu_cores,
            'memory_usage': memory_usage,
            'memory_gb': memory_gb,
            'disk_usage': disk_usage,
            'network_gb_daily': network_gb_daily,
            'gpu_usage': gpu_usage,
            'latency_ms': np.random.uniform(3, 7),  # Replace with actual
            'throughput_rps': np.random.uniform(8000, 12000),  # Replace with actual
            'error_rate': np.random.uniform(0.0001, 0.001),  # Replace with actual
            'uptime_percent': 99.99
        }
    
    async def _collect_traces(self) -> List[Dict[str, Any]]:
        """Collect distributed traces"""
        
        # This would connect to Jaeger or similar
        # For now, return mock data
        traces = []
        operations = ['db_query', 'model_inference', 'kafka_publish', 'cache_lookup', 'api_call']
        
        for _ in range(100):
            traces.append({
                'span_id': hashlib.md5(os.urandom(16)).hexdigest()[:16],
                'parent_span': hashlib.md5(os.urandom(16)).hexdigest()[:16] if np.random.random() > 0.3 else None,
                'operation_name': np.random.choice(operations),
                'duration_ms': np.random.exponential(10),
                'timestamp': datetime.now().isoformat()
            })
        
        return traces
    
    async def _load_historical_metrics(self) -> pd.DataFrame:
        """Load historical metrics for analysis"""
        
        # Generate sample data (replace with actual data loading)
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='H')
        
        data = {
            'timestamp': dates,
            'latency_ms': np.random.normal(5, 1.5, 1000),
            'throughput': np.random.normal(10000, 2000, 1000),
            'error_rate': np.random.exponential(0.001, 1000),
            'cpu_usage': np.random.normal(60, 15, 1000),
            'memory_usage': np.random.normal(70, 10, 1000)
        }
        
        return pd.DataFrame(data)
    
    async def _generate_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations"""
        
        recommendations = []
        
        # Health-based recommendations
        if audit_results['health']['score'] < 70:
            recommendations.append("URGENT: System health is degraded. Review critical alerts immediately.")
        
        # Bottleneck recommendations
        if len(audit_results['bottlenecks']) > 0:
            recommendations.append(f"Address {len(audit_results['bottlenecks'])} identified bottlenecks to improve performance.")
        
        # Cost recommendations
        cost_analysis = audit_results['cost_analysis']
        if cost_analysis['potential_monthly_savings'] > 100:
            recommendations.append(
                f"Potential cost savings of ${cost_analysis['potential_monthly_savings']:.2f}/month "
                f"({cost_analysis['savings_percentage']:.1f}%) identified."
            )
        
        # Capacity recommendations
        if 'predictions' in audit_results and audit_results['predictions']:
            for rec in audit_results['predictions'].get('recommendations', []):
                recommendations.append(rec)
        
        return recommendations
    
    async def _save_audit_report(self, audit_results: Dict[str, Any]):
        """Save audit report to file"""
        
        report_file = self.report_dir / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        async with aiofiles.open(report_file, 'w') as f:
            await f.write(json.dumps(audit_results, indent=2, default=str))
        
        logger.info(f"Audit report saved: {report_file}")
    
    async def _generate_visualizations(self, audit_results: Dict[str, Any], historical_data: pd.DataFrame):
        """Generate performance visualization dashboards"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('System Health Score', 'Resource Usage', 'Latency Trends', 'Cost Analysis'),
            specs=[[{'type': 'indicator'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'pie'}]]
        )
        
        # Health score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=audit_results['health']['score'],
                title={'text': "Health Score"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "green" if audit_results['health']['score'] > 70 else "red"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 70], 'color': "yellow"},
                           {'range': [70, 100], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}),
            row=1, col=1
        )
        
        # Resource usage bar chart
        resources = audit_results['health']['resource_usage']
        fig.add_trace(
            go.Bar(
                x=list(resources.keys()),
                y=list(resources.values()),
                marker_color=['red' if v > 80 else 'yellow' if v > 60 else 'green' 
                             for v in resources.values()]
            ),
            row=1, col=2
        )
        
        # Latency trends
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['latency_ms'],
                mode='lines',
                name='Latency'
            ),
            row=2, col=1
        )
        
        # Cost breakdown pie chart
        if audit_results['cost_analysis'].get('recommendations'):
            costs = [r['monthly_savings'] for r in audit_results['cost_analysis']['recommendations']]
            labels = [r['type'] for r in audit_results['cost_analysis']['recommendations']]
            
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=costs,
                    hole=0.3
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Performance Audit Dashboard",
            showlegend=False,
            height=800
        )
        
        # Save figure
        dashboard_file = self.report_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(dashboard_file))
        
        logger.info(f"Dashboard saved: {dashboard_file}")
    
    async def continuous_auditing(self):
        """Run continuous performance auditing"""
        
        while True:
            try:
                await self.perform_audit()
                await asyncio.sleep(self.audit_interval)
                
            except Exception as e:
                logger.error(f"Audit error: {e}")
                await asyncio.sleep(60)


async def main():
    """Run the performance auditor"""
    auditor = PerformanceAuditor()
    
    try:
        # Run single audit
        results = await auditor.perform_audit()
        
        print(f"\n{'='*60}")
        print("PERFORMANCE AUDIT RESULTS")
        print(f"{'='*60}")
        print(f"Health Score: {results['health']['score']:.1f}/100")
        print(f"Bottlenecks Found: {len(results['bottlenecks'])}")
        print(f"Critical Alerts: {len(results['health']['critical_alerts'])}")
        
        if results['cost_analysis']:
            print(f"\nCost Analysis:")
            print(f"  Current Monthly Cost: ${results['cost_analysis']['current_monthly_cost']:.2f}")
            print(f"  Potential Savings: ${results['cost_analysis']['potential_monthly_savings']:.2f}")
        
        print(f"\nRecommendations:")
        for rec in results['recommendations'][:5]:
            print(f"  - {rec}")
        
        # Start continuous auditing
        # await auditor.continuous_auditing()
        
    except KeyboardInterrupt:
        logger.info("Shutting down auditor")


if __name__ == "__main__":
    asyncio.run(main())