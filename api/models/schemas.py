"""
Pydantic models for API request/response validation
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import numpy as np


# Enums
class UserRole(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    OPERATOR = "operator"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"


class ExportFormat(str, Enum):
    PARQUET = "parquet"
    ARROW = "arrow"
    CSV = "csv"
    JSON = "json"


class QueryType(str, Enum):
    SELECT = "select"
    AGGREGATE = "aggregate"
    TIME_SERIES = "time_series"


class ModelStatus(str, Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    FAILED = "failed"


class DeploymentMode(str, Enum):
    SHADOW = "shadow"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Base Models
class BaseResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class PaginationParams(BaseModel):
    offset: int = Field(0, ge=0)
    limit: int = Field(100, ge=1, le=10000)
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "desc"


# Health Check Models
class DependencyStatus(BaseModel):
    name: str
    status: Literal["healthy", "degraded", "down"]
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseResponse):
    status: Literal["healthy", "degraded", "critical"]
    version: str
    uptime_seconds: float
    dependencies: List[DependencyStatus]
    metrics: Dict[str, float]


# Dataset Export Models
class DatasetExportRequest(BaseModel):
    query: str = Field(..., description="ClickHouse SQL query")
    format: ExportFormat = ExportFormat.PARQUET
    compression: Optional[Literal["gzip", "snappy", "zstd", "lz4"]] = "snappy"
    filters: Optional[Dict[str, Any]] = None
    time_range: Optional[Dict[str, datetime]] = None
    chunk_size: int = Field(100000, ge=1000, le=10000000)
    include_metadata: bool = True
    
    @validator("query")
    def validate_query_safe(cls, v):
        # Basic SQL injection prevention
        forbidden = ["DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "CREATE", "TRUNCATE"]
        if any(word in v.upper() for word in forbidden):
            raise ValueError("Write operations not allowed")
        return v


class DatasetExportResponse(BaseResponse):
    job_id: str
    status: JobStatus
    format: ExportFormat
    estimated_rows: Optional[int] = None
    estimated_size_mb: Optional[float] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None


# ClickHouse Query Models
class ClickHouseQueryRequest(BaseModel):
    query: str
    query_type: QueryType = QueryType.SELECT
    parameters: Optional[Dict[str, Any]] = None
    timeout_seconds: int = Field(30, ge=1, le=300)
    max_rows: int = Field(10000, ge=1, le=1000000)
    include_stats: bool = False
    
    @validator("query")
    def validate_read_only(cls, v):
        # Whitelist of allowed operations
        allowed_starts = ["SELECT", "WITH", "SHOW", "DESCRIBE", "EXISTS"]
        query_upper = v.strip().upper()
        if not any(query_upper.startswith(op) for op in allowed_starts):
            raise ValueError("Only read operations allowed")
        return v


class ClickHouseQueryResponse(BaseResponse):
    data: List[Dict[str, Any]]
    columns: List[Dict[str, str]]
    row_count: int
    execution_time_ms: float
    bytes_read: Optional[int] = None
    rows_read: Optional[int] = None
    cache_hit: bool = False


# ML Training Models
class TrainingRequest(BaseModel):
    model_type: Literal["xgboost", "lightgbm", "random_forest", "neural_net"]
    dataset_query: str
    features: List[str]
    target: str
    validation_split: float = Field(0.2, ge=0.1, le=0.5)
    hyperparameters: Optional[Dict[str, Any]] = None
    optimize_for: Literal["accuracy", "latency", "f1", "roc_auc"] = "accuracy"
    max_training_time_minutes: int = Field(60, ge=1, le=1440)
    use_pgo: bool = True
    compile_treelite: bool = True


class TrainingResponse(BaseResponse):
    job_id: str
    model_id: str
    status: ModelStatus
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    progress_percent: float = 0.0
    metrics: Optional[Dict[str, float]] = None


class ModelInfo(BaseModel):
    model_id: str
    version: str
    model_type: str
    status: ModelStatus
    created_at: datetime
    trained_by: str
    accuracy: Optional[float] = None
    latency_p99_us: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = {}


class ModelDeployRequest(BaseModel):
    model_id: str
    mode: DeploymentMode = DeploymentMode.SHADOW
    canary_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    auto_rollback: bool = True
    slo_thresholds: Optional[Dict[str, float]] = None


# Control Plane Models
class KillSwitchRequest(BaseModel):
    target: Literal["all", "mev", "arbitrage", "sandwich", "liquidation"]
    reason: str
    duration_seconds: Optional[int] = Field(None, ge=1, le=86400)
    force: bool = False
    signature: str = Field(..., description="Ed25519 signature")
    public_key: str = Field(..., description="Public key for verification")


class KillSwitchResponse(BaseResponse):
    activated: bool
    target: str
    expires_at: Optional[datetime] = None
    ack_hash: str
    multisig_status: Optional[Dict[str, bool]] = None


class AuditLogEntry(BaseModel):
    id: str
    timestamp: datetime
    user_id: str
    role: UserRole
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    success: bool
    error_message: Optional[str] = None


# WebSocket Models
class WebSocketMessage(BaseModel):
    seq: int
    topic: str
    payload: bytes
    ts_ns: int
    node_id: str


class WebSocketSubscription(BaseModel):
    topics: List[str]
    batch_window_ms: int = Field(15, ge=1, le=1000)
    max_batch_size: int = Field(256, ge=1, le=10000)
    compression: bool = True


# MEV Detection Models (Defensive Only)
class MEVAlert(BaseModel):
    alert_id: str
    alert_type: Literal["arbitrage", "sandwich", "liquidation", "jit"]
    slot: int
    detected_at: datetime
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_profit: float
    gas_estimate: int
    route: List[str]
    metadata: Dict[str, Any]
    decision_dna: Optional[str] = None


class ArbitrageDetection(BaseModel):
    tx_signature: str
    slot: int
    roi_pct: float
    est_profit: float
    legs: int
    dex_route: List[str]
    tokens: List[str]
    confidence: float


class SandwichDetection(BaseModel):
    victim_tx: str
    front_tx: str
    back_tx: str
    slot: int
    victim_loss: float
    attacker_profit: float
    token_pair: str
    dex: str


# System Metrics Models
class SystemMetrics(BaseModel):
    latency_p50_ms: float
    latency_p99_ms: float
    bundle_land_rate: float
    ingestion_rate: int
    model_inference_us: float
    decision_dna_count: int
    timestamp: datetime


class ThompsonBanditStats(BaseModel):
    arms: List[Dict[str, Any]]
    total_reward: float
    total_samples: int
    timestamp: datetime


# Authentication Models
class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class UserInfo(BaseModel):
    user_id: str
    username: str
    role: UserRole
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None