"""
Master Orchestrator Service for Solana Arbitrage Data Capture System
Production-grade orchestration with health monitoring, failover, and distributed coordination
"""

import asyncio
import json
import time
import signal
import sys
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from contextlib import asynccontextmanager

import uvloop
import aioredis
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from clickhouse_driver.client import Client as ClickHouseClient
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import psutil
import httpx

# Use uvloop for maximum performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
orchestrator_health = Gauge('orchestrator_health_status', 'Overall orchestrator health (0=unhealthy, 1=healthy)')
component_status = Gauge('component_status', 'Component health status', ['component'])
orchestration_latency = Histogram('orchestration_latency_seconds', 'Orchestration operation latency')
component_restarts = Counter('component_restarts_total', 'Number of component restarts', ['component'])
data_pipeline_throughput = Gauge('data_pipeline_throughput_ops', 'Data pipeline throughput (ops/sec)')
system_resource_usage = Gauge('system_resource_usage_percent', 'System resource usage', ['resource'])
orchestration_errors = Counter('orchestration_errors_total', 'Total orchestration errors', ['error_type'])
scheduled_tasks_executed = Counter('scheduled_tasks_executed_total', 'Number of scheduled tasks executed', ['task'])

class ComponentState(Enum):
    """Component lifecycle states"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RESTARTING = "restarting"
    STOPPED = "stopped"
    FAILED = "failed"

class ComponentPriority(Enum):
    """Component priority levels for startup/shutdown ordering"""
    CRITICAL = 1  # Infrastructure (DB, Cache, Queue)
    HIGH = 2      # Core services (Data capture, Processing)
    MEDIUM = 3    # Analysis services (ML, Risk)
    LOW = 4       # Optional services (Export, Monitoring)

@dataclass
class ComponentConfig:
    """Configuration for managed components"""
    name: str
    service_type: str
    priority: ComponentPriority
    health_check_url: Optional[str] = None
    health_check_interval: int = 30  # seconds
    restart_policy: str = "on-failure"  # always, on-failure, never
    max_restarts: int = 5
    restart_delay: int = 10  # seconds
    dependencies: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    startup_timeout: int = 60  # seconds
    shutdown_timeout: int = 30  # seconds
    resource_limits: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentHealth:
    """Health status of a component"""
    component: str
    state: ComponentState
    last_check: datetime
    consecutive_failures: int = 0
    restart_count: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

class MasterOrchestrator:
    """
    Master orchestrator for the entire arbitrage data capture system.
    Manages component lifecycle, health monitoring, and coordination.
    """
    
    def __init__(self, config_path: str = "orchestrator_config.json"):
        self.config_path = config_path
        self.components: Dict[str, ComponentConfig] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.component_processes: Dict[str, Any] = {}
        
        # Core services
        self.redis_client = None
        self.kafka_producer = None
        self.clickhouse_client = None
        self.scheduler = AsyncIOScheduler()
        
        # Thread pools for different workloads
        self.io_executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="io")
        self.cpu_executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
        # State management
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.startup_complete = asyncio.Event()
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.start_time = None
        self.metrics_buffer = []
        
        # Circuit breakers for components
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load orchestrator configuration"""
        # Default configuration for all components
        self.components = {
            # Infrastructure
            "clickhouse": ComponentConfig(
                name="clickhouse",
                service_type="database",
                priority=ComponentPriority.CRITICAL,
                health_check_url="http://clickhouse:8123/ping",
                restart_policy="always",
                resource_limits={"memory": "8GB", "cpu": 4}
            ),
            "redis": ComponentConfig(
                name="redis",
                service_type="cache",
                priority=ComponentPriority.CRITICAL,
                health_check_url="redis://redis:6390",
                restart_policy="always",
                resource_limits={"memory": "2GB", "cpu": 2}
            ),
            "kafka": ComponentConfig(
                name="kafka",
                service_type="queue",
                priority=ComponentPriority.CRITICAL,
                health_check_url="kafka:9092",
                restart_policy="always",
                dependencies=["zookeeper"],
                resource_limits={"memory": "4GB", "cpu": 2}
            ),
            
            # Core Services
            "arbitrage-detector": ComponentConfig(
                name="arbitrage-detector",
                service_type="detector",
                priority=ComponentPriority.HIGH,
                health_check_url="http://arbitrage-detector:8080/health",
                restart_policy="on-failure",
                dependencies=["clickhouse", "redis", "kafka"],
                resource_limits={"memory": "4GB", "cpu": 4}
            ),
            "labeling-service": ComponentConfig(
                name="labeling-service",
                service_type="processor",
                priority=ComponentPriority.HIGH,
                health_check_url="http://labeling-service:8081/health",
                restart_policy="on-failure",
                dependencies=["clickhouse", "kafka"],
                resource_limits={"memory": "2GB", "cpu": 2}
            ),
            "ml-pipeline": ComponentConfig(
                name="ml-pipeline",
                service_type="ml",
                priority=ComponentPriority.MEDIUM,
                health_check_url="http://ml-pipeline:8082/health",
                restart_policy="on-failure",
                dependencies=["clickhouse", "redis"],
                resource_limits={"memory": "6GB", "cpu": 4}
            ),
            
            # API and UI
            "dashboard-api": ComponentConfig(
                name="dashboard-api",
                service_type="api",
                priority=ComponentPriority.MEDIUM,
                health_check_url="http://dashboard-api:8000/health",
                restart_policy="on-failure",
                dependencies=["clickhouse", "redis"],
                resource_limits={"memory": "1GB", "cpu": 2}
            ),
            
            # Monitoring
            "prometheus": ComponentConfig(
                name="prometheus",
                service_type="monitoring",
                priority=ComponentPriority.LOW,
                health_check_url="http://prometheus:9090/-/healthy",
                restart_policy="always",
                resource_limits={"memory": "2GB", "cpu": 1}
            ),
            "grafana": ComponentConfig(
                name="grafana",
                service_type="monitoring",
                priority=ComponentPriority.LOW,
                health_check_url="http://grafana:3000/api/health",
                restart_policy="on-failure",
                dependencies=["prometheus"],
                resource_limits={"memory": "1GB", "cpu": 1}
            )
        }
        
        # Initialize health tracking
        for component_name, config in self.components.items():
            self.component_health[component_name] = ComponentHealth(
                component=component_name,
                state=ComponentState.STOPPED,
                last_check=datetime.utcnow()
            )
            
            # Initialize circuit breaker
            self.circuit_breakers[component_name] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=60,
                expected_exception=Exception
            )
    
    async def start(self):
        """Start the orchestrator and all managed components"""
        logger.info("Starting Master Orchestrator...")
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        # Start Prometheus metrics server
        start_http_server(9091)
        
        # Initialize core connections
        await self._initialize_core_services()
        
        # Start components in priority order
        await self._start_components_by_priority()
        
        # Setup scheduled tasks
        self._setup_scheduled_tasks()
        
        # Start health monitoring
        await self._start_health_monitoring()
        
        # Start the scheduler
        self.scheduler.start()
        
        # Signal startup complete
        self.startup_complete.set()
        orchestrator_health.set(1)
        
        logger.info("Master Orchestrator started successfully")
        
        # Main orchestration loop
        await self._orchestration_loop()
    
    async def _initialize_core_services(self):
        """Initialize connections to core services"""
        try:
            # Redis connection
            self.redis_client = await aioredis.create_redis_pool(
                'redis://redis:6390',
                minsize=5,
                maxsize=20
            )
            
            # Kafka producer
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers='kafka:9092',
                value_serializer=lambda v: json.dumps(v).encode()
            )
            await self.kafka_producer.start()
            
            # ClickHouse client (synchronous, use in executor)
            self.clickhouse_client = ClickHouseClient(
                host='clickhouse',
                port=9000,
                database='solana_arbitrage'
            )
            
            logger.info("Core services initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize core services: {e}")
            orchestration_errors.labels(error_type="initialization").inc()
            raise
    
    async def _start_components_by_priority(self):
        """Start components in priority order with dependency resolution"""
        # Group components by priority
        priority_groups = {}
        for component_name, config in self.components.items():
            priority = config.priority.value
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(component_name)
        
        # Start each priority group
        for priority in sorted(priority_groups.keys()):
            components = priority_groups[priority]
            logger.info(f"Starting priority {priority} components: {components}")
            
            # Start components in parallel within the same priority
            tasks = []
            for component_name in components:
                if await self._check_dependencies(component_name):
                    task = asyncio.create_task(self._start_component(component_name))
                    tasks.append(task)
                else:
                    logger.warning(f"Skipping {component_name} due to unmet dependencies")
            
            # Wait for all components in this priority to start
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Small delay between priority groups
            await asyncio.sleep(2)
    
    async def _check_dependencies(self, component_name: str) -> bool:
        """Check if all dependencies for a component are healthy"""
        config = self.components[component_name]
        
        for dependency in config.dependencies:
            if dependency not in self.component_health:
                logger.warning(f"Unknown dependency {dependency} for {component_name}")
                return False
            
            health = self.component_health[dependency]
            if health.state not in [ComponentState.HEALTHY, ComponentState.DEGRADED]:
                logger.warning(f"Dependency {dependency} is not healthy for {component_name}")
                return False
        
        return True
    
    async def _start_component(self, component_name: str):
        """Start a single component with timeout and error handling"""
        config = self.components[component_name]
        health = self.component_health[component_name]
        
        try:
            logger.info(f"Starting component: {component_name}")
            health.state = ComponentState.INITIALIZING
            component_status.labels(component=component_name).set(0.5)
            
            # Simulate component startup (in production, this would use docker/k8s API)
            # For now, we just check if the service is reachable
            start_time = time.time()
            
            while time.time() - start_time < config.startup_timeout:
                if await self._check_component_health(component_name):
                    health.state = ComponentState.HEALTHY
                    health.last_check = datetime.utcnow()
                    health.consecutive_failures = 0
                    component_status.labels(component=component_name).set(1)
                    logger.info(f"Component {component_name} started successfully")
                    return
                
                await asyncio.sleep(2)
            
            # Timeout reached
            raise TimeoutError(f"Component {component_name} failed to start within {config.startup_timeout}s")
            
        except Exception as e:
            logger.error(f"Failed to start component {component_name}: {e}")
            health.state = ComponentState.FAILED
            health.error_message = str(e)
            component_status.labels(component=component_name).set(0)
            orchestration_errors.labels(error_type="component_start").inc()
    
    async def _check_component_health(self, component_name: str) -> bool:
        """Check health of a specific component"""
        config = self.components[component_name]
        
        if not config.health_check_url:
            # No health check configured, assume healthy
            return True
        
        try:
            if config.service_type == "database":
                # ClickHouse health check
                if "clickhouse" in component_name:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(config.health_check_url, timeout=5.0)
                        return response.status_code == 200
            
            elif config.service_type == "cache":
                # Redis health check
                if "redis" in component_name:
                    # Simple ping check
                    if self.redis_client:
                        await self.redis_client.ping()
                        return True
            
            elif config.service_type == "queue":
                # Kafka health check
                if "kafka" in component_name:
                    # Check if Kafka is reachable (simplified)
                    return True  # Assume healthy if no exception
            
            else:
                # HTTP health check for services
                async with httpx.AsyncClient() as client:
                    response = await client.get(config.health_check_url, timeout=5.0)
                    return response.status_code == 200
                    
        except Exception as e:
            logger.debug(f"Health check failed for {component_name}: {e}")
            return False
        
        return False
    
    async def _start_health_monitoring(self):
        """Start health monitoring tasks for all components"""
        for component_name in self.components:
            task = asyncio.create_task(self._monitor_component_health(component_name))
            self.health_check_tasks[component_name] = task
    
    async def _monitor_component_health(self, component_name: str):
        """Continuously monitor health of a component"""
        config = self.components[component_name]
        health = self.component_health[component_name]
        
        while self.is_running:
            try:
                await asyncio.sleep(config.health_check_interval)
                
                # Use circuit breaker
                async with self.circuit_breakers[component_name]:
                    is_healthy = await self._check_component_health(component_name)
                
                if is_healthy:
                    if health.state != ComponentState.HEALTHY:
                        logger.info(f"Component {component_name} recovered")
                    health.state = ComponentState.HEALTHY
                    health.consecutive_failures = 0
                    health.error_message = None
                    component_status.labels(component=component_name).set(1)
                else:
                    health.consecutive_failures += 1
                    
                    if health.consecutive_failures >= 3:
                        health.state = ComponentState.UNHEALTHY
                        component_status.labels(component=component_name).set(0)
                        
                        # Trigger restart if policy allows
                        if config.restart_policy in ["always", "on-failure"]:
                            if health.restart_count < config.max_restarts:
                                await self._restart_component(component_name)
                    else:
                        health.state = ComponentState.DEGRADED
                        component_status.labels(component=component_name).set(0.5)
                
                health.last_check = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Error monitoring {component_name}: {e}")
                orchestration_errors.labels(error_type="health_check").inc()
    
    async def _restart_component(self, component_name: str):
        """Restart a failed component"""
        config = self.components[component_name]
        health = self.component_health[component_name]
        
        logger.warning(f"Restarting component {component_name} (attempt {health.restart_count + 1}/{config.max_restarts})")
        
        health.state = ComponentState.RESTARTING
        health.restart_count += 1
        component_restarts.labels(component=component_name).inc()
        
        # Wait before restart
        await asyncio.sleep(config.restart_delay)
        
        # Attempt to start the component
        await self._start_component(component_name)
    
    def _setup_scheduled_tasks(self):
        """Setup scheduled tasks for maintenance and optimization"""
        
        # Data export every hour
        self.scheduler.add_job(
            self._export_data,
            trigger=IntervalTrigger(hours=1),
            id='data_export',
            name='Hourly data export',
            misfire_grace_time=300
        )
        
        # Database optimization daily at 3 AM
        self.scheduler.add_job(
            self._optimize_database,
            trigger=CronTrigger(hour=3, minute=0),
            id='db_optimization',
            name='Daily database optimization'
        )
        
        # Cleanup old data weekly
        self.scheduler.add_job(
            self._cleanup_old_data,
            trigger=CronTrigger(day_of_week='sun', hour=2, minute=0),
            id='data_cleanup',
            name='Weekly data cleanup'
        )
        
        # System metrics collection every minute
        self.scheduler.add_job(
            self._collect_system_metrics,
            trigger=IntervalTrigger(seconds=60),
            id='metrics_collection',
            name='System metrics collection'
        )
        
        # Performance report generation every 6 hours
        self.scheduler.add_job(
            self._generate_performance_report,
            trigger=IntervalTrigger(hours=6),
            id='performance_report',
            name='Performance report generation'
        )
    
    async def _orchestration_loop(self):
        """Main orchestration loop"""
        logger.info("Entering main orchestration loop")
        
        while self.is_running:
            try:
                # Check overall system health
                system_healthy = await self._check_system_health()
                orchestrator_health.set(1 if system_healthy else 0)
                
                # Process any pending orchestration tasks
                await self._process_orchestration_queue()
                
                # Update resource usage metrics
                await self._update_resource_metrics()
                
                # Check for any alerts
                await self._process_alerts()
                
                # Sleep for a short interval
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                orchestration_errors.labels(error_type="orchestration_loop").inc()
                await asyncio.sleep(10)
    
    async def _check_system_health(self) -> bool:
        """Check overall system health"""
        unhealthy_components = []
        degraded_components = []
        
        for component_name, health in self.component_health.items():
            if health.state == ComponentState.UNHEALTHY:
                unhealthy_components.append(component_name)
            elif health.state == ComponentState.DEGRADED:
                degraded_components.append(component_name)
        
        if unhealthy_components:
            logger.warning(f"Unhealthy components: {unhealthy_components}")
            
            # Check if any critical components are unhealthy
            critical_unhealthy = any(
                self.components[comp].priority == ComponentPriority.CRITICAL
                for comp in unhealthy_components
            )
            
            if critical_unhealthy:
                logger.error("Critical components unhealthy, system is not operational")
                return False
        
        if degraded_components:
            logger.info(f"Degraded components: {degraded_components}")
        
        return len(unhealthy_components) == 0
    
    async def _process_orchestration_queue(self):
        """Process any pending orchestration tasks from Redis queue"""
        if not self.redis_client:
            return
        
        try:
            # Check for orchestration commands
            task_data = await self.redis_client.lpop('orchestration:tasks')
            if task_data:
                task = json.loads(task_data)
                await self._handle_orchestration_task(task)
                
        except Exception as e:
            logger.error(f"Error processing orchestration queue: {e}")
    
    async def _handle_orchestration_task(self, task: Dict[str, Any]):
        """Handle an orchestration task"""
        task_type = task.get('type')
        
        with orchestration_latency.time():
            if task_type == 'restart_component':
                component = task.get('component')
                if component in self.components:
                    await self._restart_component(component)
            
            elif task_type == 'scale_component':
                component = task.get('component')
                replicas = task.get('replicas', 1)
                await self._scale_component(component, replicas)
            
            elif task_type == 'update_config':
                await self._update_configuration(task.get('config', {}))
            
            elif task_type == 'trigger_backup':
                await self._trigger_backup()
            
            else:
                logger.warning(f"Unknown orchestration task type: {task_type}")
    
    async def _scale_component(self, component_name: str, replicas: int):
        """Scale a component to specified number of replicas"""
        logger.info(f"Scaling {component_name} to {replicas} replicas")
        # In production, this would interact with Kubernetes or Docker Swarm
        # For now, just log the action
        pass
    
    async def _update_configuration(self, new_config: Dict[str, Any]):
        """Update orchestrator configuration"""
        logger.info("Updating orchestrator configuration")
        # Merge new configuration
        # In production, this would reload configuration and apply changes
        pass
    
    async def _trigger_backup(self):
        """Trigger system backup"""
        logger.info("Triggering system backup")
        scheduled_tasks_executed.labels(task="backup").inc()
        
        # Backup ClickHouse data
        await self._backup_clickhouse()
        
        # Backup Redis state
        await self._backup_redis()
        
        # Backup configuration
        await self._backup_configuration()
    
    async def _update_resource_metrics(self):
        """Update system resource usage metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            system_resource_usage.labels(resource="cpu").set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            system_resource_usage.labels(resource="memory").set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            system_resource_usage.labels(resource="disk").set(disk.percent)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            bytes_sent = net_io.bytes_sent
            bytes_recv = net_io.bytes_recv
            
            # Calculate throughput (simplified)
            if hasattr(self, '_last_net_io'):
                sent_rate = (bytes_sent - self._last_net_io[0]) / 60  # bytes per second
                recv_rate = (bytes_recv - self._last_net_io[1]) / 60
                system_resource_usage.labels(resource="network_send").set(sent_rate)
                system_resource_usage.labels(resource="network_recv").set(recv_rate)
            
            self._last_net_io = (bytes_sent, bytes_recv)
            
        except Exception as e:
            logger.error(f"Error updating resource metrics: {e}")
    
    async def _process_alerts(self):
        """Process and handle system alerts"""
        if not self.redis_client:
            return
        
        try:
            # Check for alerts in Redis
            alerts = await self.redis_client.lrange('system:alerts', 0, -1)
            
            for alert_data in alerts:
                alert = json.loads(alert_data)
                await self._handle_alert(alert)
            
            # Clear processed alerts
            if alerts:
                await self.redis_client.delete('system:alerts')
                
        except Exception as e:
            logger.error(f"Error processing alerts: {e}")
    
    async def _handle_alert(self, alert: Dict[str, Any]):
        """Handle a system alert"""
        severity = alert.get('severity', 'info')
        message = alert.get('message', '')
        component = alert.get('component', 'unknown')
        
        logger.warning(f"Alert [{severity}] from {component}: {message}")
        
        # Take action based on severity
        if severity == 'critical':
            # Send notification (webhook, email, etc.)
            await self._send_critical_alert(alert)
        
        elif severity == 'warning':
            # Log and monitor
            pass
    
    async def _send_critical_alert(self, alert: Dict[str, Any]):
        """Send critical alert notification"""
        # In production, integrate with PagerDuty, Slack, etc.
        logger.critical(f"CRITICAL ALERT: {alert}")
    
    # Scheduled task implementations
    async def _export_data(self):
        """Export data to various formats"""
        logger.info("Starting scheduled data export")
        scheduled_tasks_executed.labels(task="data_export").inc()
        
        try:
            # Trigger export through Kafka
            await self.kafka_producer.send(
                'orchestration-commands',
                {
                    'command': 'export_data',
                    'timestamp': datetime.utcnow().isoformat(),
                    'formats': ['parquet', 'csv', 'json']
                }
            )
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            orchestration_errors.labels(error_type="scheduled_task").inc()
    
    async def _optimize_database(self):
        """Optimize database tables and indices"""
        logger.info("Starting database optimization")
        scheduled_tasks_executed.labels(task="db_optimization").inc()
        
        try:
            # Run optimization queries in executor
            await asyncio.get_event_loop().run_in_executor(
                self.io_executor,
                self._run_clickhouse_optimization
            )
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            orchestration_errors.labels(error_type="scheduled_task").inc()
    
    def _run_clickhouse_optimization(self):
        """Run ClickHouse optimization queries"""
        if not self.clickhouse_client:
            return
        
        optimization_queries = [
            "OPTIMIZE TABLE arbitrage_transactions FINAL",
            "OPTIMIZE TABLE arbitrage_opportunities FINAL",
            "OPTIMIZE TABLE market_snapshots PARTITION tuple() FINAL",
            "SYSTEM DROP MARK CACHE",
            "SYSTEM DROP UNCOMPRESSED CACHE"
        ]
        
        for query in optimization_queries:
            try:
                self.clickhouse_client.execute(query)
                logger.info(f"Executed: {query}")
            except Exception as e:
                logger.error(f"Optimization query failed: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data based on retention policies"""
        logger.info("Starting data cleanup")
        scheduled_tasks_executed.labels(task="data_cleanup").inc()
        
        try:
            # Define retention periods
            retention_policies = {
                'market_snapshots': 7,      # 7 days
                'arbitrage_transactions': 30,  # 30 days
                'arbitrage_opportunities': 14, # 14 days
                'system_logs': 7             # 7 days
            }
            
            for table, days in retention_policies.items():
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                query = f"ALTER TABLE {table} DELETE WHERE timestamp < '{cutoff_date.isoformat()}'"
                
                await asyncio.get_event_loop().run_in_executor(
                    self.io_executor,
                    self.clickhouse_client.execute,
                    query
                )
                
                logger.info(f"Cleaned up {table} older than {days} days")
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            orchestration_errors.labels(error_type="scheduled_task").inc()
    
    async def _collect_system_metrics(self):
        """Collect and store system metrics"""
        try:
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'component_health': {
                    name: health.state.value
                    for name, health in self.component_health.items()
                },
                'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds()
            }
            
            # Store in Redis for dashboard
            await self.redis_client.setex(
                'system:metrics:latest',
                60,  # TTL 60 seconds
                json.dumps(metrics)
            )
            
            # Send to Kafka for long-term storage
            await self.kafka_producer.send('system-metrics', metrics)
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    async def _generate_performance_report(self):
        """Generate system performance report"""
        logger.info("Generating performance report")
        scheduled_tasks_executed.labels(task="performance_report").inc()
        
        try:
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'uptime_hours': (datetime.utcnow() - self.start_time).total_seconds() / 3600,
                'component_availability': {},
                'error_rates': {},
                'throughput_metrics': {}
            }
            
            # Calculate component availability
            for component_name, health in self.component_health.items():
                if health.state == ComponentState.HEALTHY:
                    availability = 100.0
                elif health.state == ComponentState.DEGRADED:
                    availability = 75.0
                else:
                    availability = 0.0
                
                report['component_availability'][component_name] = availability
            
            # Store report
            await self.redis_client.lpush(
                'performance:reports',
                json.dumps(report)
            )
            
            # Trim to keep only last 100 reports
            await self.redis_client.ltrim('performance:reports', 0, 99)
            
            logger.info("Performance report generated successfully")
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            orchestration_errors.labels(error_type="scheduled_task").inc()
    
    async def _backup_clickhouse(self):
        """Backup ClickHouse data"""
        logger.info("Backing up ClickHouse data")
        # In production, use clickhouse-backup tool
        pass
    
    async def _backup_redis(self):
        """Backup Redis data"""
        logger.info("Backing up Redis data")
        if self.redis_client:
            await self.redis_client.bgsave()
    
    async def _backup_configuration(self):
        """Backup system configuration"""
        logger.info("Backing up configuration")
        # In production, backup to S3 or similar
        pass
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down Master Orchestrator...")
        self.is_running = False
        
        # Stop scheduler
        self.scheduler.shutdown(wait=False)
        
        # Cancel health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Stop components in reverse priority order
        await self._stop_components_by_priority()
        
        # Close connections
        if self.kafka_producer:
            await self.kafka_producer.stop()
        
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        
        # Shutdown executors
        self.io_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)
        
        orchestrator_health.set(0)
        logger.info("Master Orchestrator shutdown complete")
    
    async def _stop_components_by_priority(self):
        """Stop components in reverse priority order"""
        # Group components by priority
        priority_groups = {}
        for component_name, config in self.components.items():
            priority = config.priority.value
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(component_name)
        
        # Stop each priority group in reverse order
        for priority in sorted(priority_groups.keys(), reverse=True):
            components = priority_groups[priority]
            logger.info(f"Stopping priority {priority} components: {components}")
            
            tasks = []
            for component_name in components:
                task = asyncio.create_task(self._stop_component(component_name))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _stop_component(self, component_name: str):
        """Stop a single component"""
        config = self.components[component_name]
        health = self.component_health[component_name]
        
        logger.info(f"Stopping component: {component_name}")
        health.state = ComponentState.STOPPED
        component_status.labels(component=component_name).set(0)
        
        # In production, this would interact with container orchestration
        # to actually stop the component

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    async def __aenter__(self):
        if self.state == 'open':
            if self.last_failure_time:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'half-open'
                    self.failure_count = 0
                else:
                    raise Exception("Circuit breaker is open")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            if self.state == 'half-open':
                self.state = 'closed'
            self.failure_count = 0
        elif issubclass(exc_type, self.expected_exception):
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        
        return False  # Don't suppress exceptions

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    asyncio.create_task(orchestrator.shutdown())
    sys.exit(0)

# Global orchestrator instance
orchestrator = None

async def main():
    """Main entry point"""
    global orchestrator
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start orchestrator
    orchestrator = MasterOrchestrator()
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())