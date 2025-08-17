"""
Elite Data Models for Arbitrage System
Optimized for high-frequency trading with comprehensive MEV metrics
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field, validator
import numpy as np

class MEVType(str, Enum):
    ARBITRAGE = "arbitrage"
    LIQUIDATION = "liquidation"
    SANDWICH = "sandwich"
    JIT = "jit"
    CEX_DEX = "cex_dex"

class TransactionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"

class OpportunityType(str, Enum):
    SPOT = "spot"
    TRIANGLE = "triangle"
    MULTI_HOP = "multi_hop"
    CROSS_CHAIN = "cross_chain"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class Transaction(BaseModel):
    """Comprehensive transaction model with 70+ fields for ML training"""
    
    # Primary identifiers
    signature: str = Field(..., description="Transaction signature hash")
    block_height: int = Field(..., ge=0)
    block_timestamp: datetime
    slot: int = Field(..., ge=0)
    
    # Transaction metadata
    fee: int = Field(..., ge=0, description="Transaction fee in lamports")
    compute_units_used: int = Field(..., ge=0)
    priority_fee: int = Field(..., ge=0)
    lamports_per_signature: int = Field(..., ge=0)
    
    # MEV specific fields
    is_mev_transaction: bool = False
    mev_type: Optional[MEVType] = None
    bundle_id: Optional[str] = None
    searcher_address: str
    
    # Arbitrage details
    profit_amount: int = Field(..., description="Profit in smallest unit")
    profit_token: str
    gas_cost: int = Field(..., ge=0)
    net_profit: int
    roi_percentage: float = Field(..., description="Return on investment %")
    
    # Path information
    dex_count: int = Field(..., ge=1, le=10)
    hop_count: int = Field(..., ge=1, le=20)
    path_hash: str = Field(..., description="Hash of the arbitrage path")
    dexes: List[str] = Field(..., min_items=1)
    tokens: List[str] = Field(..., min_items=2)
    amounts: List[int] = Field(..., min_items=2)
    
    # Risk metrics
    slippage_percentage: float = Field(..., ge=0, le=100)
    impermanent_loss: float = Field(default=0.0)
    max_drawdown: float = Field(default=0.0)
    sharpe_ratio: float = Field(default=0.0)
    volatility_score: float = Field(..., ge=0, le=100)
    
    # Market conditions
    market_volatility: float = Field(..., ge=0)
    liquidity_depth: int = Field(..., ge=0)
    spread_basis_points: int = Field(..., ge=0)
    volume_24h: int = Field(..., ge=0)
    
    # Performance metrics
    execution_time_ms: int = Field(..., ge=0)
    simulation_time_ms: int = Field(..., ge=0)
    mempool_time_ms: int = Field(..., ge=0)
    confirmation_time_ms: int = Field(..., ge=0)
    
    # ML features (pre-computed)
    price_momentum: float = Field(default=0.0)
    volume_ratio: float = Field(default=0.0)
    liquidity_score: float = Field(..., ge=0, le=100)
    market_impact: float = Field(..., ge=0, le=100)
    cross_dex_correlation: float = Field(..., ge=-1, le=1)
    
    # Additional metadata
    program_ids: List[str] = Field(default_factory=list)
    instruction_count: int = Field(..., ge=1)
    cross_program_invocations: int = Field(default=0)
    error_code: Optional[str] = None
    status: TransactionStatus = TransactionStatus.SUCCESS
    
    # Computed fields
    indexed_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('net_profit')
    def calculate_net_profit(cls, v, values):
        if 'profit_amount' in values and 'gas_cost' in values:
            return values['profit_amount'] - values['gas_cost']
        return v
    
    @validator('roi_percentage')
    def calculate_roi(cls, v, values):
        if 'net_profit' in values and 'gas_cost' in values and values['gas_cost'] > 0:
            return (values['net_profit'] / values['gas_cost']) * 100
        return v
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }

class ArbitrageOpportunity(BaseModel):
    """Model for detected arbitrage opportunities"""
    
    opportunity_id: str = Field(..., description="Unique opportunity identifier")
    detected_at: datetime
    block_height: int
    
    # Opportunity details
    opportunity_type: OpportunityType
    input_token: str
    output_token: str
    input_amount: int
    expected_output: int
    minimum_profit: int
    
    # Path details
    path_json: str = Field(..., description="JSON encoded path details")
    dex_sequence: List[str]
    pool_addresses: List[str]
    
    # Market data snapshot
    pool_reserves: List[Tuple[int, int]]
    pool_fees: List[int]
    price_impacts: List[float]
    
    # Execution details
    executed: bool = False
    execution_tx: Optional[str] = None
    actual_profit: Optional[int] = None
    execution_latency_ms: Optional[int] = None
    
    # Competition metrics
    competing_txs: int = Field(default=0)
    frontrun_attempts: int = Field(default=0)
    backrun_success: bool = False
    
    # Risk assessment
    confidence_score: float = Field(..., ge=0, le=100)
    risk_score: float = Field(..., ge=0, le=100)
    profitability_score: float = Field(..., ge=0, le=100)
    
    def calculate_expected_roi(self) -> float:
        """Calculate expected ROI for the opportunity"""
        if self.input_amount > 0:
            return ((self.expected_output - self.input_amount) / self.input_amount) * 100
        return 0.0
    
    def to_ml_features(self) -> Dict[str, Any]:
        """Convert to ML-ready feature vector"""
        return {
            'opportunity_type': self.opportunity_type.value,
            'path_length': len(self.dex_sequence),
            'expected_profit': self.expected_output - self.input_amount,
            'confidence_score': self.confidence_score,
            'risk_score': self.risk_score,
            'pool_count': len(self.pool_addresses),
            'avg_price_impact': np.mean(self.price_impacts) if self.price_impacts else 0,
            'max_price_impact': max(self.price_impacts) if self.price_impacts else 0,
            'total_fees': sum(self.pool_fees),
            'competition_level': self.competing_txs
        }

class RiskMetrics(BaseModel):
    """Comprehensive risk assessment model"""
    
    timestamp: datetime
    transaction_signature: str
    
    # Market risk
    market_risk_score: float = Field(..., ge=0, le=100)
    volatility_24h: float
    liquidity_risk: float = Field(..., ge=0, le=100)
    
    # Execution risk
    slippage_risk: float = Field(..., ge=0, le=100)
    frontrun_probability: float = Field(..., ge=0, le=1)
    revert_probability: float = Field(..., ge=0, le=1)
    
    # Smart contract risk
    contract_risk_score: float = Field(..., ge=0, le=100)
    audit_status: bool = False
    known_vulnerabilities: List[str] = Field(default_factory=list)
    
    # Capital risk
    position_size_risk: float = Field(..., ge=0, le=100)
    leverage_ratio: float = Field(default=1.0)
    max_loss_amount: int
    
    # Composite scores
    overall_risk_level: RiskLevel
    risk_adjusted_return: float
    var_95: float = Field(..., description="Value at Risk 95%")
    cvar_95: float = Field(..., description="Conditional VaR 95%")
    
    # Historical metrics
    historical_success_rate: float = Field(..., ge=0, le=1)
    avg_historical_slippage: float
    max_historical_drawdown: float
    
    def calculate_risk_score(self) -> float:
        """Calculate composite risk score"""
        weights = {
            'market': 0.25,
            'execution': 0.35,
            'contract': 0.20,
            'capital': 0.20
        }
        
        score = (
            weights['market'] * self.market_risk_score +
            weights['execution'] * (self.slippage_risk + self.frontrun_probability * 100) / 2 +
            weights['contract'] * self.contract_risk_score +
            weights['capital'] * self.position_size_risk
        )
        
        return min(100, score)

class MarketSnapshot(BaseModel):
    """Real-time market state snapshot"""
    
    snapshot_time: datetime
    dex: str
    pool_address: str
    
    # Liquidity metrics
    reserve0: int
    reserve1: int
    total_liquidity: int
    
    # Price data
    price: float
    price_change_1m: float
    price_change_5m: float
    price_change_1h: float
    
    # Volume metrics
    volume_1m: int
    volume_5m: int
    volume_1h: int
    trade_count_1m: int
    
    # Market depth
    bid_liquidity: List[Tuple[float, int]]
    ask_liquidity: List[Tuple[float, int]]
    spread_bps: int
    
    def calculate_market_impact(self, trade_size: int) -> float:
        """Calculate expected market impact for a given trade size"""
        if self.total_liquidity > 0:
            impact_percentage = (trade_size / self.total_liquidity) * 100
            # Apply square root for more realistic impact modeling
            return min(100, np.sqrt(impact_percentage) * 2)
        return 100.0
    
    def get_effective_price(self, is_buy: bool, amount: int) -> float:
        """Calculate effective price including slippage"""
        liquidity_book = self.ask_liquidity if is_buy else self.bid_liquidity
        
        remaining_amount = amount
        total_cost = 0.0
        
        for price, available in liquidity_book:
            if remaining_amount <= 0:
                break
            
            filled = min(remaining_amount, available)
            total_cost += filled * price
            remaining_amount -= filled
        
        if amount > 0:
            return total_cost / amount
        return self.price

class PerformanceMetrics(BaseModel):
    """System and strategy performance metrics"""
    
    metric_date: datetime
    metric_window: str = Field(..., description="1h, 24h, 7d, 30d")
    
    # Transaction metrics
    total_transactions: int
    successful_arbitrages: int
    failed_arbitrages: int
    success_rate: float
    
    # Profit metrics
    total_profit_usd: float
    average_profit_usd: float
    median_profit_usd: float
    max_profit_usd: float
    profit_variance: float
    
    # Gas metrics
    total_gas_used: int
    average_gas_price: float
    gas_efficiency_ratio: float
    
    # Latency metrics
    avg_execution_time_ms: float
    p50_execution_time_ms: int
    p95_execution_time_ms: int
    p99_execution_time_ms: int
    
    # Competition metrics
    frontrun_rate: float
    bundle_inclusion_rate: float
    avg_priority_fee: float
    
    # Strategy performance
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    def calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score"""
        return (
            self.success_rate * 0.3 +
            (1 - self.frontrun_rate) * 0.2 +
            self.gas_efficiency_ratio * 0.2 +
            min(1.0, self.profit_factor / 2) * 0.3
        ) * 100

class MLFeatureSet(BaseModel):
    """ML-ready feature set for model training"""
    
    # Temporal features
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: bool
    
    # Market features
    volatility_percentile: float = Field(..., ge=0, le=100)
    volume_percentile: float = Field(..., ge=0, le=100)
    liquidity_percentile: float = Field(..., ge=0, le=100)
    
    # Path features
    path_complexity: float
    unique_dex_count: int
    unique_token_count: int
    max_hop_distance: int
    
    # Competition features
    mempool_congestion: float = Field(..., ge=0, le=100)
    competitor_activity: float = Field(..., ge=0, le=100)
    gas_price_percentile: float = Field(..., ge=0, le=100)
    
    # Technical indicators
    rsi: float = Field(..., ge=0, le=100)
    macd_signal: float
    bollinger_position: float = Field(..., ge=-1, le=1)
    
    # Cross-market features
    cex_dex_spread: float
    cross_chain_opportunity: bool
    
    # Risk features
    risk_score: float = Field(..., ge=0, le=100)
    expected_slippage: float
    revert_probability: float
    
    # Target variable
    is_profitable: bool
    profit_amount: Optional[float] = None
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        return np.array([
            self.hour_of_day,
            self.day_of_week,
            int(self.is_weekend),
            self.volatility_percentile,
            self.volume_percentile,
            self.liquidity_percentile,
            self.path_complexity,
            self.unique_dex_count,
            self.unique_token_count,
            self.max_hop_distance,
            self.mempool_congestion,
            self.competitor_activity,
            self.gas_price_percentile,
            self.rsi,
            self.macd_signal,
            self.bollinger_position,
            self.cex_dex_spread,
            int(self.cross_chain_opportunity),
            self.risk_score,
            self.expected_slippage,
            self.revert_probability
        ])

class SystemHealth(BaseModel):
    """System health and monitoring metrics"""
    
    timestamp: datetime
    component: str
    
    # Resource metrics
    cpu_usage: float = Field(..., ge=0, le=100)
    memory_usage: float = Field(..., ge=0, le=100)
    disk_usage: float = Field(..., ge=0, le=100)
    network_io: float
    
    # Component health
    is_healthy: bool
    error_rate: float = Field(..., ge=0, le=100)
    latency_ms: float
    throughput: float
    
    # Queue metrics
    queue_depth: int
    processing_rate: float
    backlog_size: int
    
    # Database metrics
    db_connections: int
    query_latency_ms: float
    write_latency_ms: float
    
    # Alerts
    active_alerts: List[str] = Field(default_factory=list)
    warning_count: int = Field(default=0)
    error_count: int = Field(default=0)