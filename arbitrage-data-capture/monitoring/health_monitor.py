"""
Health Monitoring and Alerting System
Real-time health checks, anomaly detection, and intelligent alerting
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import statistics

import numpy as np
from scipy import stats
import aioredis
from aiokafka import AIOKafkaProducer
import httpx
from prometheus_client import Counter, Gauge, Histogram, Summary
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Prometheus metrics
health_checks_total = Counter('health_checks_total', 'Total health checks performed', ['service', 'status'])
health_check_duration = Histogram('health_check_duration_seconds', 'Health check duration', ['service'])
alerts_triggered = Counter('alerts_triggered_total', 'Total alerts triggered', ['severity', 'service'])
anomalies_detected = Counter('anomalies_detected_total', 'Total anomalies detected', ['service', 'type'])
service_uptime = Gauge('service_uptime_seconds', 'Service uptime in seconds', ['service'])
error_rate = Gauge('service_error_rate', 'Service error rate', ['service'])
latency_p99 = Gauge('service_latency_p99_ms', 'Service P99 latency', ['service'])

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics to monitor"""
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    QUEUE_DEPTH = "queue_depth"
    CONNECTION_POOL = "connection_pool"

@dataclass
class ServiceHealth:
    """Health status of a service"""
    service_name: str
    status: HealthStatus
    last_check: datetime
    uptime_seconds: float = 0
    error_rate: float = 0
    avg_latency_ms: float = 0
    p99_latency_ms: float = 0
    throughput_ops: float = 0
    cpu_usage: float = 0
    memory_usage: float = 0
    active_connections: int = 0
    queue_depth: int = 0
    consecutive_failures: int = 0
    health_score: float = 100.0
    issues: List[str] = field(default_factory=list)
    metrics_history: Dict[str, deque] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert information"""
    id: str
    service: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_type: Optional[MetricType] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    duration_seconds: Optional[float] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    notifications_sent: List[str] = field(default_factory=list)

@dataclass
class HealthCheckConfig:
    """Configuration for health checks"""
    service_name: str
    endpoint: str
    interval_seconds: int = 30
    timeout_seconds: int = 5
    failure_threshold: int = 3
    recovery_threshold: int = 2
    
    # Thresholds for alerts
    latency_warning_ms: float = 500
    latency_critical_ms: float = 1000
    error_rate_warning: float = 0.01  # 1%
    error_rate_critical: float = 0.05  # 5%
    cpu_warning: float = 80
    cpu_critical: float = 95
    memory_warning: float = 80
    memory_critical: float = 95
    
    # Advanced monitoring
    enable_anomaly_detection: bool = True
    enable_predictive_alerts: bool = True

class HealthMonitor:
    """
    Comprehensive health monitoring system with anomaly detection
    and intelligent alerting capabilities
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        self.configs: Dict[str, HealthCheckConfig] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Connections
        self.redis_client = None
        self.kafka_producer = None
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # Monitoring tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        
        # Anomaly detection models
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Alert deduplication
        self.alert_fingerprints: Set[str] = set()
        self.alert_cooldown: Dict[str, datetime] = {}
        
        # Webhook endpoints for notifications
        self.webhook_endpoints = {
            AlertSeverity.INFO: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.ERROR: ["http://alertmanager:9093/api/v1/alerts"],
            AlertSeverity.CRITICAL: ["http://pagerduty-webhook:8080/alert"]
        }
        
        self._initialize_default_services()
    
    def _initialize_default_services(self):
        """Initialize monitoring for default services"""
        default_services = [
            HealthCheckConfig(
                service_name="clickhouse",
                endpoint="http://clickhouse:8123/ping",
                interval_seconds=15,
                latency_warning_ms=100,
                latency_critical_ms=500
            ),
            HealthCheckConfig(
                service_name="redis",
                endpoint="redis://redis:6379",
                interval_seconds=10,
                latency_warning_ms=50,
                latency_critical_ms=200
            ),
            HealthCheckConfig(
                service_name="kafka",
                endpoint="kafka:9092",
                interval_seconds=20,
                error_rate_warning=0.001,
                error_rate_critical=0.01
            ),
            HealthCheckConfig(
                service_name="arbitrage-detector",
                endpoint="http://arbitrage-detector:8080/health",
                interval_seconds=30,
                latency_warning_ms=1000,
                latency_critical_ms=3000
            ),
            HealthCheckConfig(
                service_name="ml-pipeline",
                endpoint="http://ml-pipeline:8082/health",
                interval_seconds=60,
                cpu_warning=70,
                cpu_critical=90
            ),
            HealthCheckConfig(
                service_name="dashboard-api",
                endpoint="http://dashboard-api:8000/health",
                interval_seconds=30,
                latency_warning_ms=200,
                latency_critical_ms=1000
            )
        ]
        
        for config in default_services:
            self.configs[config.service_name] = config
            self.services[config.service_name] = ServiceHealth(
                service_name=config.service_name,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.utcnow(),
                metrics_history={
                    'latency': deque(maxlen=100),
                    'error_rate': deque(maxlen=100),
                    'cpu_usage': deque(maxlen=100),
                    'memory_usage': deque(maxlen=100)
                }
            )
    
    async def start(self):
        """Start the health monitoring system"""
        logger.info("Starting Health Monitor...")
        self.is_running = True
        
        # Initialize connections
        await self._initialize_connections()
        
        # Start monitoring tasks for each service
        for service_name in self.configs:
            task = asyncio.create_task(self._monitor_service(service_name))
            self.monitoring_tasks[service_name] = task
        
        # Start anomaly detection task
        asyncio.create_task(self._anomaly_detection_loop())
        
        # Start alert processing task
        asyncio.create_task(self._alert_processing_loop())
        
        # Start metrics aggregation task
        asyncio.create_task(self._metrics_aggregation_loop())
        
        logger.info("Health Monitor started successfully")
    
    async def _initialize_connections(self):
        """Initialize external connections"""
        try:
            # Redis connection
            self.redis_client = await aioredis.create_redis_pool(
                'redis://redis:6379',
                minsize=2,
                maxsize=10
            )
            
            # Kafka producer for alerts
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers='kafka:9092',
                value_serializer=lambda v: json.dumps(v).encode()
            )
            await self.kafka_producer.start()
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
    
    async def _monitor_service(self, service_name: str):
        """Monitor a single service continuously"""
        config = self.configs[service_name]
        health = self.services[service_name]
        
        while self.is_running:
            try:
                # Perform health check
                start_time = time.time()
                check_result = await self._perform_health_check(config)
                duration = time.time() - start_time
                
                # Update metrics
                health_check_duration.labels(service=service_name).observe(duration)
                
                # Process health check result
                await self._process_health_check_result(
                    service_name,
                    check_result,
                    duration * 1000  # Convert to ms
                )
                
                # Sleep until next check
                await asyncio.sleep(config.interval_seconds)
                
            except Exception as e:
                logger.error(f"Error monitoring {service_name}: {e}")
                health.consecutive_failures += 1
                await asyncio.sleep(config.interval_seconds)
    
    async def _perform_health_check(self, config: HealthCheckConfig) -> Dict[str, Any]:
        """Perform health check for a service"""
        result = {
            'success': False,
            'latency_ms': 0,
            'status_code': None,
            'error': None,
            'metrics': {}
        }
        
        try:
            if config.endpoint.startswith('http'):
                # HTTP health check
                start = time.time()
                response = await self.http_client.get(
                    config.endpoint,
                    timeout=config.timeout_seconds
                )
                latency = (time.time() - start) * 1000
                
                result['success'] = response.status_code == 200
                result['latency_ms'] = latency
                result['status_code'] = response.status_code
                
                # Try to parse metrics from response
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if isinstance(data, dict):
                            result['metrics'] = data.get('metrics', {})
                    except:
                        pass
            
            elif config.endpoint.startswith('redis://'):
                # Redis health check
                if self.redis_client:
                    start = time.time()
                    await self.redis_client.ping()
                    latency = (time.time() - start) * 1000
                    
                    result['success'] = True
                    result['latency_ms'] = latency
                    
                    # Get Redis info
                    info = await self.redis_client.info()
                    result['metrics'] = {
                        'connected_clients': info.get('connected_clients', 0),
                        'used_memory': info.get('used_memory', 0),
                        'ops_per_sec': info.get('instantaneous_ops_per_sec', 0)
                    }
            
            elif 'kafka' in config.endpoint:
                # Kafka health check (simplified)
                # In production, use admin client to check cluster health
                result['success'] = True
                result['latency_ms'] = 10  # Placeholder
                
        except Exception as e:
            result['error'] = str(e)
            logger.debug(f"Health check failed for {config.endpoint}: {e}")
        
        return result
    
    async def _process_health_check_result(self, service_name: str, 
                                          result: Dict[str, Any], 
                                          duration_ms: float):
        """Process health check result and update service health"""
        config = self.configs[service_name]
        health = self.services[service_name]
        
        # Update basic metrics
        health.last_check = datetime.utcnow()
        health.avg_latency_ms = result.get('latency_ms', 0)
        
        # Update metrics history
        health.metrics_history['latency'].append(result.get('latency_ms', 0))
        
        if result['success']:
            health.consecutive_failures = 0
            health_checks_total.labels(service=service_name, status='success').inc()
            
            # Update metrics from response
            metrics = result.get('metrics', {})
            if metrics:
                health.cpu_usage = metrics.get('cpu_usage', health.cpu_usage)
                health.memory_usage = metrics.get('memory_usage', health.memory_usage)
                health.throughput_ops = metrics.get('throughput', health.throughput_ops)
                health.active_connections = metrics.get('connections', health.active_connections)
                health.queue_depth = metrics.get('queue_depth', health.queue_depth)
                
                # Update history
                health.metrics_history['cpu_usage'].append(health.cpu_usage)
                health.metrics_history['memory_usage'].append(health.memory_usage)
            
            # Calculate P99 latency from history
            if len(health.metrics_history['latency']) > 10:
                health.p99_latency_ms = np.percentile(
                    list(health.metrics_history['latency']), 
                    99
                )
                latency_p99.labels(service=service_name).set(health.p99_latency_ms)
            
            # Determine health status
            health.issues = []
            
            if health.avg_latency_ms > config.latency_critical_ms:
                health.status = HealthStatus.CRITICAL
                health.issues.append(f"High latency: {health.avg_latency_ms:.0f}ms")
            elif health.avg_latency_ms > config.latency_warning_ms:
                health.status = HealthStatus.DEGRADED
                health.issues.append(f"Elevated latency: {health.avg_latency_ms:.0f}ms")
            elif health.cpu_usage > config.cpu_critical:
                health.status = HealthStatus.CRITICAL
                health.issues.append(f"Critical CPU usage: {health.cpu_usage:.1f}%")
            elif health.cpu_usage > config.cpu_warning:
                health.status = HealthStatus.DEGRADED
                health.issues.append(f"High CPU usage: {health.cpu_usage:.1f}%")
            elif health.memory_usage > config.memory_critical:
                health.status = HealthStatus.CRITICAL
                health.issues.append(f"Critical memory usage: {health.memory_usage:.1f}%")
            elif health.memory_usage > config.memory_warning:
                health.status = HealthStatus.DEGRADED
                health.issues.append(f"High memory usage: {health.memory_usage:.1f}%")
            else:
                health.status = HealthStatus.HEALTHY
            
            # Calculate health score (0-100)
            health.health_score = self._calculate_health_score(health, config)
            
        else:
            health.consecutive_failures += 1
            health_checks_total.labels(service=service_name, status='failure').inc()
            
            if result.get('error'):
                health.issues.append(f"Health check error: {result['error']}")
            
            # Determine status based on consecutive failures
            if health.consecutive_failures >= config.failure_threshold:
                health.status = HealthStatus.UNHEALTHY
                
                # Trigger alert
                await self._create_alert(
                    service=service_name,
                    severity=AlertSeverity.ERROR,
                    message=f"Service {service_name} is unhealthy after {health.consecutive_failures} consecutive failures",
                    metric_type=MetricType.ERROR_RATE,
                    current_value=health.consecutive_failures,
                    threshold=config.failure_threshold
                )
            elif health.consecutive_failures > 1:
                health.status = HealthStatus.DEGRADED
        
        # Update uptime
        if health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
            health.uptime_seconds += config.interval_seconds
            service_uptime.labels(service=service_name).set(health.uptime_seconds)
        
        # Store health status in Redis
        await self._store_health_status(service_name, health)
        
        # Check for threshold violations
        await self._check_thresholds(service_name, health, config)
    
    def _calculate_health_score(self, health: ServiceHealth, 
                               config: HealthCheckConfig) -> float:
        """Calculate overall health score for a service"""
        score = 100.0
        
        # Latency impact (0-30 points)
        if health.avg_latency_ms > 0:
            latency_ratio = health.avg_latency_ms / config.latency_critical_ms
            latency_penalty = min(30, latency_ratio * 30)
            score -= latency_penalty
        
        # Error rate impact (0-30 points)
        if health.error_rate > 0:
            error_ratio = health.error_rate / config.error_rate_critical
            error_penalty = min(30, error_ratio * 30)
            score -= error_penalty
        
        # Resource usage impact (0-20 points)
        if health.cpu_usage > config.cpu_warning:
            cpu_ratio = (health.cpu_usage - config.cpu_warning) / (100 - config.cpu_warning)
            score -= min(10, cpu_ratio * 10)
        
        if health.memory_usage > config.memory_warning:
            mem_ratio = (health.memory_usage - config.memory_warning) / (100 - config.memory_warning)
            score -= min(10, mem_ratio * 10)
        
        # Consecutive failures impact (0-20 points)
        if health.consecutive_failures > 0:
            failure_penalty = min(20, health.consecutive_failures * 5)
            score -= failure_penalty
        
        return max(0, score)
    
    async def _check_thresholds(self, service_name: str, 
                               health: ServiceHealth,
                               config: HealthCheckConfig):
        """Check if any thresholds are violated and create alerts"""
        
        # Latency threshold
        if health.avg_latency_ms > config.latency_critical_ms:
            await self._create_alert(
                service=service_name,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical latency for {service_name}: {health.avg_latency_ms:.0f}ms",
                metric_type=MetricType.LATENCY,
                current_value=health.avg_latency_ms,
                threshold=config.latency_critical_ms
            )
        elif health.avg_latency_ms > config.latency_warning_ms:
            await self._create_alert(
                service=service_name,
                severity=AlertSeverity.WARNING,
                message=f"High latency for {service_name}: {health.avg_latency_ms:.0f}ms",
                metric_type=MetricType.LATENCY,
                current_value=health.avg_latency_ms,
                threshold=config.latency_warning_ms
            )
        
        # CPU threshold
        if health.cpu_usage > config.cpu_critical:
            await self._create_alert(
                service=service_name,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical CPU usage for {service_name}: {health.cpu_usage:.1f}%",
                metric_type=MetricType.RESOURCE_USAGE,
                current_value=health.cpu_usage,
                threshold=config.cpu_critical
            )
        
        # Memory threshold
        if health.memory_usage > config.memory_critical:
            await self._create_alert(
                service=service_name,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical memory usage for {service_name}: {health.memory_usage:.1f}%",
                metric_type=MetricType.RESOURCE_USAGE,
                current_value=health.memory_usage,
                threshold=config.memory_critical
            )
    
    async def _create_alert(self, service: str, severity: AlertSeverity,
                          message: str, metric_type: Optional[MetricType] = None,
                          current_value: Optional[float] = None,
                          threshold: Optional[float] = None):
        """Create and process an alert"""
        
        # Create fingerprint for deduplication
        fingerprint = f"{service}:{severity.value}:{metric_type.value if metric_type else 'general'}"
        
        # Check cooldown
        if fingerprint in self.alert_cooldown:
            cooldown_until = self.alert_cooldown[fingerprint]
            if datetime.utcnow() < cooldown_until:
                return  # Skip alert due to cooldown
        
        # Create alert
        alert = Alert(
            id=f"{service}_{int(time.time())}",
            service=service,
            severity=severity,
            message=message,
            timestamp=datetime.utcnow(),
            metric_type=metric_type,
            current_value=current_value,
            threshold=threshold
        )
        
        # Store alert
        self.alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Set cooldown (5 minutes for warnings, 15 for critical)
        cooldown_minutes = 15 if severity == AlertSeverity.CRITICAL else 5
        self.alert_cooldown[fingerprint] = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
        
        # Update metrics
        alerts_triggered.labels(severity=severity.value, service=service).inc()
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        # Store in Redis for other services
        await self._store_alert(alert)
        
        logger.warning(f"Alert created: [{severity.value}] {message}")
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through various channels"""
        
        # Get webhook endpoints for this severity
        endpoints = self.webhook_endpoints.get(alert.severity, [])
        
        for endpoint in endpoints:
            try:
                # Prepare alert payload
                payload = {
                    'alert': {
                        'id': alert.id,
                        'service': alert.service,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'metric_type': alert.metric_type.value if alert.metric_type else None,
                        'current_value': alert.current_value,
                        'threshold': alert.threshold
                    }
                }
                
                # Send webhook
                response = await self.http_client.post(
                    endpoint,
                    json=payload,
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    alert.notifications_sent.append(endpoint)
                    logger.info(f"Alert notification sent to {endpoint}")
                
            except Exception as e:
                logger.error(f"Failed to send alert to {endpoint}: {e}")
        
        # Send to Kafka for processing by other services
        if self.kafka_producer:
            try:
                await self.kafka_producer.send(
                    'system-alerts',
                    {
                        'alert_id': alert.id,
                        'service': alert.service,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'metric_type': alert.metric_type.value if alert.metric_type else None,
                        'current_value': alert.current_value,
                        'threshold': alert.threshold
                    }
                )
            except Exception as e:
                logger.error(f"Failed to send alert to Kafka: {e}")
    
    async def _anomaly_detection_loop(self):
        """Continuous anomaly detection using ML models"""
        
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                for service_name, health in self.services.items():
                    config = self.configs[service_name]
                    
                    if not config.enable_anomaly_detection:
                        continue
                    
                    # Check if we have enough data
                    if len(health.metrics_history['latency']) < 30:
                        continue
                    
                    # Prepare features for anomaly detection
                    features = self._prepare_anomaly_features(health)
                    
                    if features is not None:
                        # Detect anomalies
                        anomalies = await self._detect_anomalies(
                            service_name,
                            features
                        )
                        
                        if anomalies:
                            for anomaly_type, confidence in anomalies:
                                anomalies_detected.labels(
                                    service=service_name,
                                    type=anomaly_type
                                ).inc()
                                
                                await self._create_alert(
                                    service=service_name,
                                    severity=AlertSeverity.WARNING,
                                    message=f"Anomaly detected in {service_name}: {anomaly_type} (confidence: {confidence:.2f})"
                                )
                
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
    
    def _prepare_anomaly_features(self, health: ServiceHealth) -> Optional[np.ndarray]:
        """Prepare features for anomaly detection"""
        try:
            features = []
            
            # Latency features
            latencies = list(health.metrics_history['latency'])
            if latencies:
                features.extend([
                    np.mean(latencies),
                    np.std(latencies),
                    np.percentile(latencies, 95),
                    np.percentile(latencies, 99)
                ])
            
            # CPU features
            cpu_usage = list(health.metrics_history['cpu_usage'])
            if cpu_usage:
                features.extend([
                    np.mean(cpu_usage),
                    np.std(cpu_usage),
                    np.max(cpu_usage)
                ])
            
            # Memory features
            memory_usage = list(health.metrics_history['memory_usage'])
            if memory_usage:
                features.extend([
                    np.mean(memory_usage),
                    np.std(memory_usage),
                    np.max(memory_usage)
                ])
            
            # Additional features
            features.extend([
                health.error_rate,
                health.throughput_ops,
                health.active_connections,
                health.queue_depth
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing anomaly features: {e}")
            return None
    
    async def _detect_anomalies(self, service_name: str, 
                               features: np.ndarray) -> List[Tuple[str, float]]:
        """Detect anomalies using Isolation Forest"""
        anomalies = []
        
        try:
            # Initialize model if not exists
            if service_name not in self.anomaly_detectors:
                self.anomaly_detectors[service_name] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                self.scalers[service_name] = StandardScaler()
                
                # Need to fit with some initial data
                # In production, load pre-trained models
                return anomalies
            
            # Scale features
            scaler = self.scalers[service_name]
            features_scaled = scaler.transform(features)
            
            # Predict anomaly
            detector = self.anomaly_detectors[service_name]
            anomaly_score = detector.decision_function(features_scaled)[0]
            is_anomaly = detector.predict(features_scaled)[0] == -1
            
            if is_anomaly:
                confidence = abs(anomaly_score)
                
                # Determine anomaly type based on feature analysis
                if features[0, 0] > np.mean(features[0, :4]) * 2:  # High latency
                    anomalies.append(("latency_spike", confidence))
                
                if features[0, 4] > np.mean(features[0, 4:7]) * 1.5:  # High CPU
                    anomalies.append(("cpu_spike", confidence))
                
                if features[0, 7] > np.mean(features[0, 7:10]) * 1.5:  # High memory
                    anomalies.append(("memory_spike", confidence))
                
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    async def _alert_processing_loop(self):
        """Process and manage alerts"""
        
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Process every 30 seconds
                
                # Check for resolved alerts
                for alert_id, alert in list(self.alerts.items()):
                    if alert.resolved:
                        continue
                    
                    # Check if condition is resolved
                    health = self.services.get(alert.service)
                    if health and health.status == HealthStatus.HEALTHY:
                        alert.resolved = True
                        alert.resolution_time = datetime.utcnow()
                        
                        duration = (alert.resolution_time - alert.timestamp).total_seconds()
                        
                        logger.info(f"Alert {alert_id} resolved after {duration:.0f} seconds")
                        
                        # Send resolution notification
                        await self._send_resolution_notification(alert)
                
                # Clean up old resolved alerts
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                for alert_id in list(self.alerts.keys()):
                    alert = self.alerts[alert_id]
                    if alert.resolved and alert.resolution_time < cutoff_time:
                        del self.alerts[alert_id]
                
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
    
    async def _send_resolution_notification(self, alert: Alert):
        """Send notification that alert has been resolved"""
        if self.kafka_producer:
            try:
                await self.kafka_producer.send(
                    'system-alerts',
                    {
                        'alert_id': alert.id,
                        'event': 'resolved',
                        'service': alert.service,
                        'severity': alert.severity.value,
                        'resolution_time': alert.resolution_time.isoformat(),
                        'duration_seconds': (alert.resolution_time - alert.timestamp).total_seconds()
                    }
                )
            except Exception as e:
                logger.error(f"Failed to send resolution notification: {e}")
    
    async def _metrics_aggregation_loop(self):
        """Aggregate and store metrics periodically"""
        
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Aggregate every minute
                
                # Aggregate metrics for all services
                aggregated_metrics = {}
                
                for service_name, health in self.services.items():
                    aggregated_metrics[service_name] = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'status': health.status.value,
                        'health_score': health.health_score,
                        'uptime_seconds': health.uptime_seconds,
                        'avg_latency_ms': health.avg_latency_ms,
                        'p99_latency_ms': health.p99_latency_ms,
                        'error_rate': health.error_rate,
                        'cpu_usage': health.cpu_usage,
                        'memory_usage': health.memory_usage,
                        'throughput_ops': health.throughput_ops,
                        'active_alerts': len([
                            a for a in self.alerts.values()
                            if a.service == service_name and not a.resolved
                        ])
                    }
                
                # Store in Redis
                if self.redis_client:
                    await self.redis_client.setex(
                        'health:metrics:latest',
                        300,  # 5 minute TTL
                        json.dumps(aggregated_metrics)
                    )
                
                # Send to Kafka for long-term storage
                if self.kafka_producer:
                    await self.kafka_producer.send(
                        'health-metrics',
                        aggregated_metrics
                    )
                
            except Exception as e:
                logger.error(f"Error in metrics aggregation: {e}")
    
    async def _store_health_status(self, service_name: str, health: ServiceHealth):
        """Store health status in Redis"""
        if not self.redis_client:
            return
        
        try:
            health_data = {
                'service': service_name,
                'status': health.status.value,
                'health_score': health.health_score,
                'last_check': health.last_check.isoformat(),
                'uptime_seconds': health.uptime_seconds,
                'issues': health.issues,
                'metrics': {
                    'latency_ms': health.avg_latency_ms,
                    'p99_latency_ms': health.p99_latency_ms,
                    'error_rate': health.error_rate,
                    'cpu_usage': health.cpu_usage,
                    'memory_usage': health.memory_usage,
                    'throughput_ops': health.throughput_ops
                }
            }
            
            await self.redis_client.setex(
                f'health:service:{service_name}',
                60,  # 1 minute TTL
                json.dumps(health_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to store health status: {e}")
    
    async def _store_alert(self, alert: Alert):
        """Store alert in Redis"""
        if not self.redis_client:
            return
        
        try:
            alert_data = {
                'id': alert.id,
                'service': alert.service,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metric_type': alert.metric_type.value if alert.metric_type else None,
                'current_value': alert.current_value,
                'threshold': alert.threshold,
                'resolved': alert.resolved
            }
            
            # Store in sorted set by timestamp
            await self.redis_client.zadd(
                'health:alerts:active',
                alert.timestamp.timestamp(),
                json.dumps(alert_data)
            )
            
            # Trim to keep only last 1000 alerts
            await self.redis_client.zremrangebyrank('health:alerts:active', 0, -1001)
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': HealthStatus.HEALTHY.value,
            'services': {},
            'active_alerts': [],
            'system_health_score': 100.0
        }
        
        unhealthy_count = 0
        degraded_count = 0
        total_health_score = 0
        
        for service_name, health in self.services.items():
            summary['services'][service_name] = {
                'status': health.status.value,
                'health_score': health.health_score,
                'uptime_seconds': health.uptime_seconds,
                'issues': health.issues
            }
            
            if health.status == HealthStatus.UNHEALTHY:
                unhealthy_count += 1
            elif health.status == HealthStatus.DEGRADED:
                degraded_count += 1
            
            total_health_score += health.health_score
        
        # Calculate overall status
        if unhealthy_count > 0:
            summary['overall_status'] = HealthStatus.UNHEALTHY.value
        elif degraded_count > 0:
            summary['overall_status'] = HealthStatus.DEGRADED.value
        
        # Calculate system health score
        if self.services:
            summary['system_health_score'] = total_health_score / len(self.services)
        
        # Add active alerts
        for alert in self.alerts.values():
            if not alert.resolved:
                summary['active_alerts'].append({
                    'service': alert.service,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'duration_seconds': (datetime.utcnow() - alert.timestamp).total_seconds()
                })
        
        return summary
    
    async def shutdown(self):
        """Shutdown the health monitor"""
        logger.info("Shutting down Health Monitor...")
        self.is_running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Close connections
        if self.kafka_producer:
            await self.kafka_producer.stop()
        
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        
        await self.http_client.aclose()
        
        logger.info("Health Monitor shutdown complete")

async def main():
    """Main entry point for health monitoring"""
    monitor = HealthMonitor()
    
    try:
        await monitor.start()
        
        # Keep running
        while True:
            # Print health summary periodically
            summary = await monitor.get_system_health_summary()
            logger.info(f"System health: {summary['overall_status']}, Score: {summary['system_health_score']:.1f}")
            
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await monitor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())