"""
High-Performance Data Models for Arbitrage System
Optimized for both in-memory processing and database persistence
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal
from datetime import datetime
from enum import Enum
import uuid
import msgpack
import json
from pydantic import BaseModel, Field, validator
import numpy as np

class ArbitrageType(str, Enum):
    """Types of arbitrage opportunities"""
    TRIANGULAR = "triangular"
    CROSS_DEX = "cross_dex"
    FLASH_LOAN = "flash_loan"
    SANDWICH = "sandwich"
    BACKRUN = "backrun"
    FRONTRUN = "frontrun"
    LIQUIDATION = "liquidation"
    STATISTICAL = "statistical"

class TransactionStatus(str, Enum):
    """Transaction execution status"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    REVERTED = "reverted"
    DROPPED = "dropped"

class MEVType(str, Enum):
    """MEV extraction types"""
    SANDWICH = "sandwich"
    FRONTRUN = "frontrun"
    BACKRUN = "backrun"
    LIQUIDATION = "liquidation"
    ARBITRAGE = "arbitrage"
    JIT_LIQUIDITY = "jit_liquidity"

@dataclass
class TransactionModel:
    """
    Comprehensive transaction model with 70+ fields
    Optimized for both real-time processing and ML training
    """
    # Primary identifiers
    transaction_hash: str
    block_number: int
    block_timestamp: datetime
    
    # Transaction details
    from_address: str
    to_address: str
    value: Decimal
    gas_price: int
    gas_used: int
    max_fee_per_gas: Optional[int] = None
    max_priority_fee_per_gas: Optional[int] = None
    transaction_index: int = 0
    nonce: int = 0
    
    # DEX and protocol information
    dex_name: str
    protocol_version: str
    pool_address: str
    token_in: str
    token_out: str
    amount_in: Decimal
    amount_out: Decimal
    
    # Arbitrage specific fields
    is_arbitrage: bool = False
    arbitrage_type: Optional[ArbitrageType] = None
    profit_usd: Decimal = Decimal(0)
    profit_percentage: Decimal = Decimal(0)
    path: List[str] = field(default_factory=list)
    path_length: int = 0
    
    # MEV and competition metrics
    is_mev: bool = False
    mev_type: Optional[MEVType] = None
    bundle_index: Optional[int] = None
    searcher_address: Optional[str] = None
    builder_address: Optional[str] = None
    validator_index: Optional[int] = None
    
    # Risk and performance metrics
    slippage_percentage: Decimal = Decimal(0)
    price_impact: Decimal = Decimal(0)
    gas_efficiency_score: Decimal = Decimal(0)
    execution_time_ms: int = 0
    revert_probability: Decimal = Decimal(0)
    
    # Market conditions
    market_volatility: Decimal = Decimal(0)
    liquidity_depth: Decimal = Decimal(0)
    volume_24h: Decimal = Decimal(0)
    
    # Additional metadata
    input_data: str = ""
    logs: List[str] = field(default_factory=list)
    receipt_status: int = 0
    cumulative_gas_used: int = 0
    effective_gas_price: int = 0
    
    # ML features
    feature_vector: List[float] = field(default_factory=list)
    anomaly_score: float = 0.0
    confidence_score: float = 0.0
    
    # System metadata
    inserted_at: datetime = field(default_factory=datetime.now)
    processing_time_ms: int = 0
    data_version: int = 1
    
    def to_clickhouse_dict(self) -> Dict[str, Any]:
        """Convert to ClickHouse compatible dictionary"""
        return {
            'transaction_hash': self.transaction_hash,
            'block_number': self.block_number,
            'block_timestamp': self.block_timestamp.timestamp(),
            'from_address': self.from_address,
            'to_address': self.to_address,
            'value': str(self.value),
            'gas_price': self.gas_price,
            'gas_used': self.gas_used,
            'max_fee_per_gas': self.max_fee_per_gas,
            'max_priority_fee_per_gas': self.max_priority_fee_per_gas,
            'transaction_index': self.transaction_index,
            'nonce': self.nonce,
            'dex_name': self.dex_name,
            'protocol_version': self.protocol_version,
            'pool_address': self.pool_address,
            'token_in': self.token_in,
            'token_out': self.token_out,
            'amount_in': str(self.amount_in),
            'amount_out': str(self.amount_out),
            'is_arbitrage': self.is_arbitrage,
            'arbitrage_type': self.arbitrage_type.value if self.arbitrage_type else None,
            'profit_usd': float(self.profit_usd),
            'profit_percentage': float(self.profit_percentage),
            'path': self.path,
            'path_length': self.path_length,
            'is_mev': self.is_mev,
            'mev_type': self.mev_type.value if self.mev_type else None,
            'bundle_index': self.bundle_index,
            'searcher_address': self.searcher_address,
            'builder_address': self.builder_address,
            'validator_index': self.validator_index,
            'slippage_percentage': float(self.slippage_percentage),
            'price_impact': float(self.price_impact),
            'gas_efficiency_score': float(self.gas_efficiency_score),
            'execution_time_ms': self.execution_time_ms,
            'revert_probability': float(self.revert_probability),
            'market_volatility': float(self.market_volatility),
            'liquidity_depth': float(self.liquidity_depth),
            'volume_24h': float(self.volume_24h),
            'input_data': self.input_data,
            'logs': self.logs,
            'receipt_status': self.receipt_status,
            'cumulative_gas_used': self.cumulative_gas_used,
            'effective_gas_price': self.effective_gas_price,
            'feature_vector': self.feature_vector,
            'anomaly_score': self.anomaly_score,
            'confidence_score': self.confidence_score,
            'processing_time_ms': self.processing_time_ms,
            'data_version': self.data_version
        }
    
    def to_msgpack(self) -> bytes:
        """Serialize to MessagePack for high-performance transmission"""
        return msgpack.packb(self.to_clickhouse_dict())
    
    @classmethod
    def from_msgpack(cls, data: bytes) -> 'TransactionModel':
        """Deserialize from MessagePack"""
        dict_data = msgpack.unpackb(data, raw=False)
        return cls(**dict_data)

class ArbitrageOpportunity(BaseModel):
    """
    Model for detected arbitrage opportunities
    """
    opportunity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    detected_at: datetime = Field(default_factory=datetime.now)
    block_number: int
    
    # Opportunity details
    type: ArbitrageType
    status: str = "pending"
    
    # Path information
    path_tokens: List[str]
    path_pools: List[str]
    path_dexes: List[str]
    
    # Financial metrics
    expected_profit_usd: Decimal
    actual_profit_usd: Optional[Decimal] = None
    required_capital: Decimal
    roi_percentage: Decimal
    
    # Execution details
    gas_estimate: int
    gas_price_gwei: Decimal
    total_gas_cost_usd: Decimal
    net_profit_usd: Decimal
    
    # Competition analysis
    competitors_detected: int = 0
    win_probability: Decimal = Field(ge=0, le=1)
    optimal_gas_multiplier: Decimal = Decimal("1.0")
    
    # Risk metrics
    impermanent_loss_risk: Decimal = Field(ge=0, le=1)
    slippage_risk: Decimal = Field(ge=0, le=1)
    frontrun_risk: Decimal = Field(ge=0, le=1)
    
    # Execution result
    executed: bool = False
    execution_tx_hash: Optional[str] = None
    execution_timestamp: Optional[datetime] = None
    execution_success: Optional[bool] = None
    failure_reason: Optional[str] = None
    
    @validator('net_profit_usd')
    def validate_net_profit(cls, v, values):
        """Ensure net profit calculation is correct"""
        if 'expected_profit_usd' in values and 'total_gas_cost_usd' in values:
            calculated = values['expected_profit_usd'] - values['total_gas_cost_usd']
            if abs(v - calculated) > Decimal("0.01"):
                raise ValueError(f"Net profit mismatch: {v} != {calculated}")
        return v
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }

@dataclass
class RiskMetrics:
    """
    Comprehensive risk metrics for real-time monitoring
    """
    timestamp: datetime
    metric_type: str = "general"
    
    # Market risk indicators
    volatility_1h: Decimal
    volatility_24h: Decimal
    correlation_matrix: List[List[float]]
    
    # Liquidity metrics
    total_liquidity_usd: Decimal
    liquidity_concentration: Decimal  # Herfindahl index
    depth_imbalance: Decimal
    
    # MEV competition metrics
    avg_gas_price_gwei: Decimal
    mev_competition_score: Decimal
    bundle_success_rate: Decimal
    
    # System performance
    latency_p50_ms: int
    latency_p95_ms: int
    latency_p99_ms: int
    
    # Profit metrics
    total_profit_1h: Decimal
    total_profit_24h: Decimal
    profit_volatility: Decimal
    sharpe_ratio: Decimal
    
    def calculate_risk_score(self) -> Decimal:
        """Calculate overall risk score (0-100)"""
        weights = {
            'volatility': 0.25,
            'liquidity': 0.25,
            'competition': 0.25,
            'performance': 0.25
        }
        
        # Normalize each component to 0-100
        volatility_score = min(self.volatility_24h * 100, 100)
        liquidity_score = max(0, 100 - self.liquidity_concentration * 100)
        competition_score = self.mev_competition_score * 100
        performance_score = max(0, 100 - (self.latency_p99_ms / 10))
        
        risk_score = (
            weights['volatility'] * volatility_score +
            weights['liquidity'] * liquidity_score +
            weights['competition'] * competition_score +
            weights['performance'] * performance_score
        )
        
        return Decimal(str(risk_score))

@dataclass
class MarketSnapshot:
    """
    Point-in-time market state for ML training
    """
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    block_number: int = 0
    
    # Price data
    prices: Dict[str, Decimal] = field(default_factory=dict)
    
    # Volume data
    volumes_1h: Dict[str, Decimal] = field(default_factory=dict)
    volumes_24h: Dict[str, Decimal] = field(default_factory=dict)
    
    # Liquidity data
    liquidity_by_pool: Dict[str, Decimal] = field(default_factory=dict)
    liquidity_by_token: Dict[str, Decimal] = field(default_factory=dict)
    
    # Order book data (aggregated)
    bid_depth: Dict[str, List[Tuple[Decimal, Decimal]]] = field(default_factory=dict)
    ask_depth: Dict[str, List[Tuple[Decimal, Decimal]]] = field(default_factory=dict)
    
    # Network state
    gas_price: int = 0
    base_fee: int = 0
    pending_tx_count: int = 0
    mempool_size_mb: float = 0.0
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert snapshot to ML feature vector"""
        features = []
        
        # Price features
        if self.prices:
            price_values = list(self.prices.values())
            features.extend([
                float(min(price_values)),
                float(max(price_values)),
                float(sum(price_values) / len(price_values)),
                float(np.std([float(p) for p in price_values]))
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Volume features
        if self.volumes_24h:
            volume_values = list(self.volumes_24h.values())
            features.extend([
                float(sum(volume_values)),
                float(max(volume_values)),
                float(sum(volume_values) / len(volume_values))
            ])
        else:
            features.extend([0, 0, 0])
        
        # Liquidity features
        if self.liquidity_by_pool:
            liquidity_values = list(self.liquidity_by_pool.values())
            features.extend([
                float(sum(liquidity_values)),
                float(max(liquidity_values)),
                float(min(liquidity_values))
            ])
        else:
            features.extend([0, 0, 0])
        
        # Network features
        features.extend([
            float(self.gas_price / 1e9),  # Convert to Gwei
            float(self.base_fee / 1e9),
            float(self.pending_tx_count),
            self.mempool_size_mb
        ])
        
        return np.array(features, dtype=np.float32)

@dataclass
class PerformanceMetrics:
    """
    Aggregated performance metrics for monitoring
    """
    period_start: datetime
    period_end: datetime
    metric_type: str = "general"
    
    # Transaction metrics
    total_transactions: int = 0
    successful_transactions: int = 0
    failed_transactions: int = 0
    reverted_transactions: int = 0
    
    # Profit metrics
    gross_profit_usd: Decimal = Decimal(0)
    gas_costs_usd: Decimal = Decimal(0)
    net_profit_usd: Decimal = Decimal(0)
    roi_percentage: Decimal = Decimal(0)
    
    # Efficiency metrics
    avg_gas_used: int = 0
    avg_execution_time_ms: int = 0
    success_rate: Decimal = Decimal(0)
    
    # Competition metrics
    frontrun_count: int = 0
    backrun_count: int = 0
    sandwich_count: int = 0
    
    # Statistical metrics
    profit_mean: Decimal = Decimal(0)
    profit_std: Decimal = Decimal(0)
    profit_min: Decimal = Decimal(0)
    profit_max: Decimal = Decimal(0)
    profit_percentiles: List[Decimal] = field(default_factory=list)
    
    def calculate_efficiency_score(self) -> Decimal:
        """Calculate overall efficiency score"""
        if self.total_transactions == 0:
            return Decimal(0)
        
        success_weight = Decimal("0.4")
        profit_weight = Decimal("0.3")
        gas_weight = Decimal("0.3")
        
        success_score = self.success_rate
        profit_score = min(self.roi_percentage / 100, Decimal(1))
        gas_score = max(0, Decimal(1) - (Decimal(self.avg_gas_used) / Decimal(1000000)))
        
        return (
            success_weight * success_score +
            profit_weight * profit_score +
            gas_weight * gas_score
        ) * 100

# ML Feature Engineering Models
@dataclass
class MLFeatures:
    """
    Pre-engineered features for ML training
    """
    # Temporal features
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    
    # Price features
    price_momentum_1h: float
    price_momentum_24h: float
    price_volatility: float
    
    # Volume features
    volume_ratio_1h_24h: float
    volume_spike_indicator: bool
    
    # Liquidity features
    liquidity_ratio: float
    liquidity_concentration: float
    
    # Network features
    gas_price_percentile: float
    mempool_congestion: float
    
    # Competition features
    mev_activity_score: float
    searcher_competition_index: float
    
    # Historical features
    profit_ma_1h: float
    profit_ma_24h: float
    success_rate_1h: float
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        return np.array([
            self.hour_of_day,
            self.day_of_week,
            int(self.is_weekend),
            self.price_momentum_1h,
            self.price_momentum_24h,
            self.price_volatility,
            self.volume_ratio_1h_24h,
            int(self.volume_spike_indicator),
            self.liquidity_ratio,
            self.liquidity_concentration,
            self.gas_price_percentile,
            self.mempool_congestion,
            self.mev_activity_score,
            self.searcher_competition_index,
            self.profit_ma_1h,
            self.profit_ma_24h,
            self.success_rate_1h
        ], dtype=np.float32)