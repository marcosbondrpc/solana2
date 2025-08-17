"""
Pydantic models for request/response validation
Optimized for high-throughput data processing
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid


class Granularity(str, Enum):
    DAY = "day"
    MONTH = "month"
    YEAR = "year"


class DataSource(str, Enum):
    RPC = "rpc"
    BIGTABLE = "bigtable"
    ARCHIVE = "archive"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CaptureRequest(BaseModel):
    """High-performance capture configuration"""
    granularity: Granularity
    start: date
    end: date
    source: DataSource = DataSource.RPC
    programs: List[str] = Field(default_factory=list)
    include_blocks: bool = True
    include_transactions: bool = True
    include_logs: bool = True
    out_uri: str = "./data"
    block_batch: int = Field(64, ge=1, le=256)
    json_parsed: bool = True
    max_tx_version: int = Field(0, ge=0)
    
    @validator('end')
    def validate_date_range(cls, v, values):
        if 'start' in values and v < values['start']:
            raise ValueError('end date must be after start date')
        return v
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat()
        }


class ArbitrageConvertRequest(BaseModel):
    """Configuration for arbitrage detection"""
    raw_uri: str = "./data/raw"
    out_uri: str = "./data/labels"
    min_profit_usd: float = Field(1.0, ge=0)
    max_slot_gap: int = Field(3, ge=1, le=100)
    programs: Optional[List[str]] = None
    include_failed: bool = False


class SandwichConvertRequest(BaseModel):
    """Configuration for sandwich attack detection"""
    raw_uri: str = "./data/raw"
    out_uri: str = "./data/labels"
    min_profit_usd: float = Field(1.0, ge=0)
    max_slot_gap: int = Field(2, ge=1, le=10)
    victim_slippage_threshold: float = Field(0.01, ge=0, le=1)
    programs: Optional[List[str]] = None


class JobMetadata(BaseModel):
    """Job tracking metadata"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = Field(0.0, ge=0, le=100)
    current_slot: Optional[int] = None
    total_slots: Optional[int] = None
    blocks_processed: int = 0
    transactions_processed: int = 0
    errors: List[str] = Field(default_factory=list)
    config: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None


class DatasetStats(BaseModel):
    """Analytics for captured datasets"""
    total_blocks: int = 0
    total_transactions: int = 0
    total_logs: int = 0
    date_range: Optional[Dict[str, date]] = None
    programs_captured: List[str] = Field(default_factory=list)
    storage_size_mb: float = 0.0
    arbitrage_opportunities: int = 0
    sandwich_attacks: int = 0
    total_mev_extracted_usd: float = 0.0
    capture_duration_seconds: Optional[float] = None
    avg_block_time_ms: Optional[float] = None
    partitions: List[str] = Field(default_factory=list)


class BlockData(BaseModel):
    """Optimized block representation"""
    slot: int
    block_height: Optional[int]
    block_time: Optional[int]
    parent_slot: int
    previous_blockhash: str
    blockhash: str
    rewards: Optional[List[Dict[str, Any]]]
    transaction_count: int
    
    class Config:
        orm_mode = True


class TransactionData(BaseModel):
    """Optimized transaction representation"""
    signature: str
    slot: int
    block_time: Optional[int]
    err: Optional[Dict[str, Any]]
    fee: int
    pre_balances: List[int]
    post_balances: List[int]
    pre_token_balances: Optional[List[Dict[str, Any]]]
    post_token_balances: Optional[List[Dict[str, Any]]]
    log_messages: Optional[List[str]]
    compute_units_consumed: Optional[int]
    loaded_addresses: Optional[Dict[str, List[str]]]
    
    class Config:
        orm_mode = True


class SwapEvent(BaseModel):
    """Normalized swap event for MEV detection"""
    signature: str
    slot: int
    instruction_index: int
    program_id: str
    pool_address: str
    user: str
    token_in: str
    token_out: str
    amount_in: int
    amount_out: int
    price: float
    timestamp: Optional[int]


class ArbitrageOpportunity(BaseModel):
    """Detected arbitrage opportunity"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_slot: int
    end_slot: int
    transactions: List[str]
    path: List[str]  # Pool addresses in order
    tokens: List[str]  # Token addresses in path
    profit_token: str
    profit_amount: int
    profit_usd: float
    gas_used: int
    net_profit_usd: float
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class SandwichAttack(BaseModel):
    """Detected sandwich attack"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    front_tx: str
    victim_tx: str
    back_tx: str
    slot: int
    pool_address: str
    attacker: str
    victim: str
    profit_token: str
    profit_amount: int
    profit_usd: float
    victim_loss_usd: float
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Service health status"""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: float
    active_jobs: int
    rpc_status: str
    storage_available_gb: float
    memory_usage_mb: float
    cpu_usage_percent: float