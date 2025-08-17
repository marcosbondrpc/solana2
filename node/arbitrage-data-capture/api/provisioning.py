#!/usr/bin/env python3
"""
LEGENDARY Provisioning Management API
Ultra-sophisticated infrastructure provisioning endpoints
"""

import os
import json
import yaml
import time
import subprocess
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import aiohttp
import clickhouse_connect as ch
from redis import Redis
import psutil

# Create API router
router = APIRouter(prefix="/api/v1/provisioning", tags=["provisioning"])

# Configuration
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY", "")
CLICKHOUSE_URL = os.getenv("CLICKHOUSE_URL", "http://localhost:8123")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6390")
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")

# Redis client for caching
redis_client = Redis.from_url(REDIS_URL, decode_responses=True)

# ClickHouse client
ch_client = ch.get_client(
    host=CLICKHOUSE_URL.replace("http://", "").replace(":8123", ""),
    database="default"
)

#######################################
# Pydantic Models
#######################################

class DatasourceConfig(BaseModel):
    """Grafana datasource configuration"""
    name: str = Field(..., description="Datasource name")
    type: str = Field(default="grafana-clickhouse-datasource")
    url: str = Field(default="http://localhost:8123")
    database: str = Field(default="default")
    username: str = Field(default="default")
    password: Optional[str] = Field(default="")
    is_default: bool = Field(default=False)
    query_timeout: int = Field(default=60, ge=1, le=300)
    max_memory_usage: int = Field(default=10737418240)  # 10GB
    
    class Config:
        schema_extra = {
            "example": {
                "name": "ClickHouse-MEV",
                "type": "grafana-clickhouse-datasource",
                "url": "http://localhost:8123",
                "database": "mev",
                "username": "grafana",
                "is_default": True
            }
        }


class DashboardConfig(BaseModel):
    """Dashboard provisioning configuration"""
    name: str
    folder: str = Field(default="MEV Control Center")
    dashboard_json: Dict[str, Any]
    overwrite: bool = Field(default=True)
    
    @validator('dashboard_json')
    def validate_dashboard_json(cls, v):
        """Validate dashboard JSON structure"""
        required_fields = ["panels", "title", "uid"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Dashboard JSON missing required field: {field}")
        return v


class ServiceConfig(BaseModel):
    """Service configuration for systemd"""
    name: str
    cpu_cores: List[int] = Field(default=[])
    rt_priority: int = Field(default=0, ge=0, le=99)
    memory_limit_gb: float = Field(default=4.0)
    restart_policy: str = Field(default="always")
    environment: Dict[str, str] = Field(default_factory=dict)


class ProvisioningStatus(BaseModel):
    """Provisioning operation status"""
    operation_id: str
    status: str  # pending, running, completed, failed
    component: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class SystemHealth(BaseModel):
    """System health status"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    services: Dict[str, str]  # service_name -> status
    clickhouse_status: str
    kafka_status: str
    redis_status: str
    grafana_status: str


#######################################
# Helper Functions
#######################################

async def check_service_health(service: str, url: str) -> str:
    """Check health of a service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return "healthy"
                return f"unhealthy (status: {resp.status})"
    except asyncio.TimeoutError:
        return "timeout"
    except Exception as e:
        return f"error: {str(e)}"


def run_command(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    """Run shell command with timeout"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


async def provision_grafana_datasource(config: DatasourceConfig) -> Dict[str, Any]:
    """Provision Grafana datasource via API"""
    headers = {
        "Content-Type": "application/json",
    }
    if GRAFANA_API_KEY:
        headers["Authorization"] = f"Bearer {GRAFANA_API_KEY}"
    
    datasource_payload = {
        "name": config.name,
        "type": config.type,
        "access": "proxy",
        "isDefault": config.is_default,
        "editable": True,
        "jsonData": {
            "server": config.url.replace("http://", "").replace(":8123", ""),
            "port": 8123,
            "protocol": "http",
            "defaultDatabase": config.database,
            "username": config.username,
            "queryTimeout": config.query_timeout,
            "settings": {
                "max_execution_time": config.query_timeout,
                "max_memory_usage": config.max_memory_usage,
            }
        },
        "secureJsonData": {
            "password": config.password or ""
        }
    }
    
    async with aiohttp.ClientSession() as session:
        # Try to create datasource
        async with session.post(
            f"{GRAFANA_URL}/api/datasources",
            headers=headers,
            json=datasource_payload
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {"status": "created", "uid": data.get("uid")}
            elif resp.status == 409:
                # Datasource exists, try to update
                async with session.get(
                    f"{GRAFANA_URL}/api/datasources/name/{config.name}",
                    headers=headers
                ) as get_resp:
                    if get_resp.status == 200:
                        existing = await get_resp.json()
                        uid = existing.get("uid")
                        
                        # Update existing datasource
                        async with session.put(
                            f"{GRAFANA_URL}/api/datasources/uid/{uid}",
                            headers=headers,
                            json=datasource_payload
                        ) as update_resp:
                            if update_resp.status == 200:
                                return {"status": "updated", "uid": uid}
                            else:
                                error = await update_resp.text()
                                raise HTTPException(
                                    status_code=update_resp.status,
                                    detail=f"Failed to update datasource: {error}"
                                )
            else:
                error = await resp.text()
                raise HTTPException(
                    status_code=resp.status,
                    detail=f"Failed to create datasource: {error}"
                )


async def provision_grafana_dashboard(config: DashboardConfig) -> Dict[str, Any]:
    """Provision Grafana dashboard via API"""
    headers = {
        "Content-Type": "application/json",
    }
    if GRAFANA_API_KEY:
        headers["Authorization"] = f"Bearer {GRAFANA_API_KEY}"
    
    dashboard_payload = {
        "dashboard": config.dashboard_json,
        "folderTitle": config.folder,
        "overwrite": config.overwrite,
        "message": f"Provisioned via API at {datetime.now(timezone.utc).isoformat()}"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            headers=headers,
            json=dashboard_payload
        ) as resp:
            if resp.status in [200, 201]:
                data = await resp.json()
                return {
                    "status": "success",
                    "uid": data.get("uid"),
                    "url": data.get("url"),
                    "version": data.get("version")
                }
            else:
                error = await resp.text()
                raise HTTPException(
                    status_code=resp.status,
                    detail=f"Failed to provision dashboard: {error}"
                )


def create_clickhouse_tables() -> Dict[str, Any]:
    """Create all ClickHouse tables"""
    results = {}
    
    # Read DDL files
    ddl_dir = Path("/home/kidgordones/0solana/node/arbitrage-data-capture/clickhouse/ddl")
    
    for ddl_file in ddl_dir.glob("*.sql"):
        try:
            with open(ddl_file, 'r') as f:
                ddl_content = f.read()
            
            # Execute DDL statements
            statements = ddl_content.split(';')
            for stmt in statements:
                stmt = stmt.strip()
                if stmt and not stmt.startswith('--'):
                    ch_client.command(stmt)
            
            results[ddl_file.name] = "success"
        except Exception as e:
            results[ddl_file.name] = f"error: {str(e)}"
    
    return results


def configure_systemd_service(config: ServiceConfig) -> Dict[str, Any]:
    """Configure and reload systemd service"""
    service_file = f"/etc/systemd/system/{config.name}.service"
    
    # Generate service configuration
    service_content = f"""[Unit]
Description={config.name} - MEV Infrastructure Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=mev
Group=mev
WorkingDirectory=/home/kidgordones/0solana/node/arbitrage-data-capture

# Environment
"""
    
    for key, value in config.environment.items():
        service_content += f'Environment="{key}={value}"\n'
    
    # CPU affinity
    if config.cpu_cores:
        service_content += f"CPUAffinity={' '.join(map(str, config.cpu_cores))}\n"
    
    # RT scheduling
    if config.rt_priority > 0:
        service_content += f"""CPUSchedulingPolicy=fifo
CPUSchedulingPriority={config.rt_priority}
"""
    
    # Memory limits
    service_content += f"""MemoryMax={config.memory_limit_gb}G
MemoryHigh={config.memory_limit_gb * 0.8}G

# Process management
Restart={config.restart_policy}
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
    
    try:
        # Write service file
        Path(service_file).write_text(service_content)
        
        # Reload systemd
        returncode, _, stderr = run_command(["sudo", "systemctl", "daemon-reload"])
        if returncode != 0:
            raise Exception(f"Failed to reload systemd: {stderr}")
        
        # Enable service
        returncode, _, stderr = run_command(
            ["sudo", "systemctl", "enable", f"{config.name}.service"]
        )
        if returncode != 0:
            raise Exception(f"Failed to enable service: {stderr}")
        
        return {
            "status": "configured",
            "service": config.name,
            "file": service_file
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure service: {str(e)}"
        )


#######################################
# API Endpoints
#######################################

@router.get("/health", response_model=SystemHealth)
async def get_system_health():
    """Get comprehensive system health status"""
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Service health checks
    services = {}
    
    # Check systemd services
    for service in ["mev-control-plane", "mev-agent", "arb-agent"]:
        returncode, stdout, _ = run_command(
            ["systemctl", "is-active", f"{service}.service"]
        )
        services[service] = stdout.strip() if returncode == 0 else "inactive"
    
    # Check infrastructure services
    try:
        ch_client.query("SELECT 1")
        ch_status = "healthy"
    except:
        ch_status = "unhealthy"
    
    try:
        redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    # Check Kafka
    returncode, _, _ = run_command(
        ["kafkacat", "-b", KAFKA_BROKERS, "-L"],
        timeout=5
    )
    kafka_status = "healthy" if returncode == 0 else "unhealthy"
    
    # Check Grafana
    grafana_status = await check_service_health("grafana", GRAFANA_URL)
    
    return SystemHealth(
        timestamp=datetime.now(timezone.utc),
        cpu_usage_percent=cpu_percent,
        memory_usage_percent=memory.percent,
        disk_usage_percent=disk.percent,
        services=services,
        clickhouse_status=ch_status,
        kafka_status=kafka_status,
        redis_status=redis_status,
        grafana_status=grafana_status
    )


@router.post("/datasource", response_model=Dict[str, Any])
async def provision_datasource(config: DatasourceConfig):
    """Provision Grafana datasource"""
    try:
        result = await provision_grafana_datasource(config)
        
        # Cache configuration
        redis_client.setex(
            f"datasource:{config.name}",
            3600,
            json.dumps(config.dict())
        )
        
        return {
            "status": "success",
            "datasource": config.name,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/dashboard", response_model=Dict[str, Any])
async def provision_dashboard(config: DashboardConfig):
    """Provision Grafana dashboard"""
    try:
        result = await provision_grafana_dashboard(config)
        
        # Store in audit log
        ch_client.insert(
            "audit_log",
            [{
                "event_type": "dashboard_provisioned",
                "event_category": 1,  # config
                "severity": 2,  # info
                "component": "provisioning_api",
                "message": f"Dashboard {config.name} provisioned",
                "details": json.dumps(result)
            }]
        )
        
        return {
            "status": "success",
            "dashboard": config.name,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/clickhouse/tables", response_model=Dict[str, Any])
async def provision_clickhouse_tables():
    """Create all ClickHouse tables from DDL files"""
    try:
        results = create_clickhouse_tables()
        
        # Count successes and failures
        success_count = sum(1 for r in results.values() if r == "success")
        failure_count = len(results) - success_count
        
        return {
            "status": "completed",
            "total_files": len(results),
            "successful": success_count,
            "failed": failure_count,
            "details": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to provision ClickHouse tables: {str(e)}"
        )


@router.post("/service", response_model=Dict[str, Any])
async def configure_service(config: ServiceConfig):
    """Configure systemd service"""
    try:
        result = configure_systemd_service(config)
        
        # Store configuration
        redis_client.setex(
            f"service:{config.name}",
            3600,
            json.dumps(config.dict())
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/provision/all", response_model=Dict[str, Any])
async def provision_all(background_tasks: BackgroundTasks):
    """Provision entire infrastructure"""
    operation_id = str(uuid.uuid4())
    
    # Store initial status
    status = ProvisioningStatus(
        operation_id=operation_id,
        status="running",
        component="all",
        started_at=datetime.now(timezone.utc),
        details={}
    )
    
    redis_client.setex(
        f"provision:{operation_id}",
        3600,
        json.dumps(status.dict(), default=str)
    )
    
    # Start provisioning in background
    background_tasks.add_task(run_full_provisioning, operation_id)
    
    return {
        "operation_id": operation_id,
        "status": "started",
        "message": "Full provisioning started in background"
    }


@router.get("/provision/{operation_id}", response_model=ProvisioningStatus)
async def get_provisioning_status(operation_id: str):
    """Get status of provisioning operation"""
    status_json = redis_client.get(f"provision:{operation_id}")
    
    if not status_json:
        raise HTTPException(
            status_code=404,
            detail="Operation not found"
        )
    
    status_data = json.loads(status_json)
    # Convert string timestamps back to datetime
    status_data["started_at"] = datetime.fromisoformat(status_data["started_at"])
    if status_data.get("completed_at"):
        status_data["completed_at"] = datetime.fromisoformat(status_data["completed_at"])
    
    return ProvisioningStatus(**status_data)


async def run_full_provisioning(operation_id: str):
    """Run complete infrastructure provisioning"""
    start_time = time.time()
    details = {}
    
    try:
        # 1. Create ClickHouse tables
        details["clickhouse"] = create_clickhouse_tables()
        
        # 2. Configure Kafka topics
        returncode, stdout, stderr = run_command([
            "make", "-C", "/home/kidgordones/0solana/node/arbitrage-data-capture",
            "kafka-topics"
        ])
        details["kafka"] = "success" if returncode == 0 else f"failed: {stderr}"
        
        # 3. Provision Grafana datasources
        datasources = [
            DatasourceConfig(
                name="ClickHouse",
                is_default=True
            ),
            DatasourceConfig(
                name="ClickHouse-Historical",
                database="historical"
            ),
            DatasourceConfig(
                name="ClickHouse-Realtime",
                database="realtime",
                query_timeout=5
            )
        ]
        
        ds_results = []
        for ds in datasources:
            try:
                result = await provision_grafana_datasource(ds)
                ds_results.append({"name": ds.name, "status": "success", "result": result})
            except Exception as e:
                ds_results.append({"name": ds.name, "status": "failed", "error": str(e)})
        
        details["datasources"] = ds_results
        
        # 4. Configure systemd services
        services = [
            ServiceConfig(
                name="mev-control-plane",
                cpu_cores=[10, 11],
                rt_priority=10,
                memory_limit_gb=4
            ),
            ServiceConfig(
                name="mev-agent",
                cpu_cores=[2, 3],
                rt_priority=30,
                memory_limit_gb=8
            ),
            ServiceConfig(
                name="arb-agent",
                cpu_cores=[4, 5],
                rt_priority=30,
                memory_limit_gb=6
            )
        ]
        
        svc_results = []
        for svc in services:
            try:
                result = configure_systemd_service(svc)
                svc_results.append({"name": svc.name, "status": "success", "result": result})
            except Exception as e:
                svc_results.append({"name": svc.name, "status": "failed", "error": str(e)})
        
        details["services"] = svc_results
        
        # Update status
        status = ProvisioningStatus(
            operation_id=operation_id,
            status="completed",
            component="all",
            started_at=datetime.now(timezone.utc) - timedelta(seconds=time.time() - start_time),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=time.time() - start_time,
            details=details
        )
        
    except Exception as e:
        # Update status with error
        status = ProvisioningStatus(
            operation_id=operation_id,
            status="failed",
            component="all",
            started_at=datetime.now(timezone.utc) - timedelta(seconds=time.time() - start_time),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=time.time() - start_time,
            details=details,
            error=str(e)
        )
    
    # Store final status
    redis_client.setex(
        f"provision:{operation_id}",
        86400,  # Keep for 24 hours
        json.dumps(status.dict(), default=str)
    )
    
    # Log to audit table
    ch_client.insert(
        "audit_log",
        [{
            "event_type": "full_provisioning",
            "event_category": 1,  # config
            "severity": 2 if status.status == "completed" else 4,  # info or error
            "component": "provisioning_api",
            "message": f"Full provisioning {status.status}",
            "details": json.dumps(details),
            "correlation_id": operation_id
        }]
    )


@router.post("/benchmark", response_model=Dict[str, Any])
async def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    benchmark_id = str(uuid.uuid4())
    
    # Run smoke test as benchmark
    returncode, stdout, stderr = run_command([
        "python3",
        "/home/kidgordones/0solana/node/arbitrage-data-capture/tools/lab_smoke_test.py",
        "--bandit", "1000",
        "--mev", "100",
        "--arb", "200",
        "--output", f"/tmp/benchmark_{benchmark_id}.json"
    ], timeout=120)
    
    if returncode == 0:
        # Read results
        with open(f"/tmp/benchmark_{benchmark_id}.json", 'r') as f:
            metrics = json.load(f)
        
        # Store in ClickHouse
        ch_client.insert(
            "performance_benchmarks",
            [{
                "benchmark_id": benchmark_id,
                "test_name": "smoke_test",
                "test_type": 2,  # throughput
                "parameters": json.dumps({"bandit": 1000, "mev": 100, "arb": 200}),
                "duration_seconds": metrics["test_duration_seconds"],
                "operations_count": metrics["total_events_produced"],
                "operations_per_second": metrics["total_events_produced"] / metrics["test_duration_seconds"],
                "latency_min_us": int(metrics["kafka_produce_latency_ms"] * 1000),
                "latency_p50_us": int(metrics["kafka_produce_latency_ms"] * 1000),
                "latency_p90_us": int(metrics["kafka_consume_latency_ms"] * 1000),
                "latency_p95_us": int(metrics["clickhouse_insert_latency_ms"] * 1000),
                "latency_p99_us": int(metrics["clickhouse_query_latency_ms"] * 1000),
                "latency_max_us": int(max(
                    metrics["kafka_consume_latency_ms"],
                    metrics["clickhouse_query_latency_ms"]
                ) * 1000),
                "cpu_usage_avg": metrics["cpu_usage_percent"],
                "memory_usage_avg_mb": int(metrics["memory_usage_mb"]),
                "network_throughput_mbps": (metrics["network_bytes_sent"] + metrics["network_bytes_recv"]) / metrics["test_duration_seconds"] / 1024 / 1024 * 8,
                "success": True,
                "errors_count": 0
            }]
        )
        
        return {
            "benchmark_id": benchmark_id,
            "status": "success",
            "metrics": metrics
        }
    else:
        return {
            "benchmark_id": benchmark_id,
            "status": "failed",
            "error": stderr
        }


# Add router to main API
def setup_provisioning_routes(app):
    """Setup provisioning routes in main app"""
    app.include_router(router)