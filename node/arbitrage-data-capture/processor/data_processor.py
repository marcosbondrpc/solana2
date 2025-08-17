"""
Elite Real-Time Data Processing Engine
Sub-50ms arbitrage detection with ML feature extraction
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from collections import deque
import msgpack
import pickle
import json
from dataclasses import dataclass
import logging

import aioredis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import ta  # Technical analysis library
import numba
from numba import jit, vectorize, cuda
import cupy as cp  # GPU acceleration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 300  # 5 minutes
    window_size: int = 1000  # For rolling calculations
    batch_size: int = 100
    use_gpu: bool = True
    feature_cache_size: int = 10000

class ArbitrageDetector:
    """Ultra-fast arbitrage detection engine"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.redis = None
        self.price_cache = {}
        self.opportunity_buffer = deque(maxlen=1000)
        self.feature_scaler = StandardScaler()
        
        # Pre-compile JIT functions
        self._compile_jit_functions()
    
    async def initialize(self):
        """Initialize connections and caches"""
        self.redis = await aioredis.create_redis_pool(
            self.config.redis_url,
            encoding='utf-8'
        )
        logger.info("Arbitrage detector initialized")
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_arbitrage_profit(
        amounts: np.ndarray,
        fees: np.ndarray,
        prices: np.ndarray
    ) -> float:
        """JIT-compiled profit calculation for speed"""
        total_fees = np.sum(fees)
        
        # Calculate output amount through path
        current_amount = amounts[0]
        for i in range(len(prices)):
            fee_multiplier = 1.0 - fees[i]
            current_amount = current_amount * prices[i] * fee_multiplier
        
        profit = current_amount - amounts[0]
        net_profit = profit - total_fees
        
        return net_profit
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _detect_triangular_arbitrage(
        price_matrix: np.ndarray,
        fee_matrix: np.ndarray,
        min_profit_threshold: float = 0.001
    ) -> List[Tuple[int, int, int, float]]:
        """Detect triangular arbitrage opportunities using JIT compilation"""
        n = price_matrix.shape[0]
        opportunities = []
        
        for i in numba.prange(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    # Calculate triangular path profit
                    rate1 = price_matrix[i, j] * (1 - fee_matrix[i, j])
                    rate2 = price_matrix[j, k] * (1 - fee_matrix[j, k])
                    rate3 = price_matrix[k, i] * (1 - fee_matrix[k, i])
                    
                    final_amount = 1.0 * rate1 * rate2 * rate3
                    profit = final_amount - 1.0
                    
                    if profit > min_profit_threshold:
                        opportunities.append((i, j, k, profit))
        
        return opportunities
    
    async def detect_arbitrage(self, transaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect arbitrage opportunity in under 50ms"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract key data
            dexes = transaction.get('dexes', [])
            tokens = transaction.get('tokens', [])
            amounts = np.array(transaction.get('amounts', []), dtype=np.float64)
            
            if len(dexes) < 2 or len(tokens) < 2:
                return None
            
            # Get current prices from cache or fetch
            prices = await self._get_prices_batch(tokens, dexes)
            fees = await self._get_fees_batch(dexes)
            
            # Calculate profit using JIT-compiled function
            profit = self._calculate_arbitrage_profit(amounts, fees, prices)
            
            # Check if profitable
            if profit > 0:
                roi = (profit / amounts[0]) * 100
                
                opportunity = {
                    'opportunity_id': f"arb_{transaction['signature'][:16]}",
                    'detected_at': datetime.utcnow().isoformat(),
                    'block_height': transaction['block_height'],
                    'opportunity_type': self._classify_opportunity(dexes, tokens),
                    'input_token': tokens[0],
                    'output_token': tokens[-1],
                    'input_amount': int(amounts[0]),
                    'expected_output': int(amounts[0] + profit),
                    'minimum_profit': int(profit),
                    'path_json': json.dumps({
                        'dexes': dexes,
                        'tokens': tokens,
                        'amounts': amounts.tolist()
                    }),
                    'dex_sequence': dexes,
                    'pool_addresses': await self._get_pool_addresses(dexes, tokens),
                    'confidence_score': self._calculate_confidence(profit, roi, transaction),
                    'risk_score': await self._calculate_risk_score(transaction, profit),
                    'profitability_score': min(100, roi * 10),
                    'detection_latency_ms': int((asyncio.get_event_loop().time() - start_time) * 1000)
                }
                
                # Cache opportunity
                await self._cache_opportunity(opportunity)
                
                return opportunity
                
        except Exception as e:
            logger.error(f"Arbitrage detection error: {e}")
            return None
        
        finally:
            # Log performance
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            if latency > 50:
                logger.warning(f"Detection latency exceeded 50ms: {latency:.2f}ms")
    
    async def _get_prices_batch(self, tokens: List[str], dexes: List[str]) -> np.ndarray:
        """Get prices with caching for speed"""
        prices = []
        
        for i in range(len(tokens) - 1):
            cache_key = f"price:{dexes[i] if i < len(dexes) else 'default'}:{tokens[i]}:{tokens[i+1]}"
            
            # Check cache first
            cached = await self.redis.get(cache_key)
            if cached:
                prices.append(float(cached))
            else:
                # Simulate price fetch (replace with actual API call)
                price = np.random.uniform(0.9, 1.1)  # Placeholder
                prices.append(price)
                
                # Cache for TTL
                await self.redis.setex(cache_key, self.config.cache_ttl, str(price))
        
        return np.array(prices, dtype=np.float64)
    
    async def _get_fees_batch(self, dexes: List[str]) -> np.ndarray:
        """Get DEX fees with caching"""
        fee_map = {
            'uniswap': 0.003,
            'sushiswap': 0.003,
            'curve': 0.0004,
            'balancer': 0.002,
            'pancakeswap': 0.0025,
            'default': 0.003
        }
        
        fees = [fee_map.get(dex.lower(), fee_map['default']) for dex in dexes]
        return np.array(fees, dtype=np.float64)
    
    def _classify_opportunity(self, dexes: List[str], tokens: List[str]) -> str:
        """Classify the type of arbitrage opportunity"""
        if len(set(dexes)) == 1:
            return 'spot'
        elif len(tokens) == 3 and tokens[0] == tokens[-1]:
            return 'triangle'
        elif len(dexes) > 3:
            return 'multi_hop'
        else:
            return 'spot'
    
    def _calculate_confidence(self, profit: float, roi: float, transaction: Dict) -> float:
        """Calculate confidence score for the opportunity"""
        base_confidence = 50.0
        
        # Add confidence based on profit
        if profit > 1000000:  # > $1 profit
            base_confidence += 20
        elif profit > 100000:  # > $0.10 profit
            base_confidence += 10
        
        # Add confidence based on ROI
        if roi > 5:
            base_confidence += 15
        elif roi > 1:
            base_confidence += 5
        
        # Add confidence based on path complexity
        if len(transaction.get('dexes', [])) <= 3:
            base_confidence += 10
        
        # Historical success rate (placeholder)
        base_confidence += np.random.uniform(0, 15)
        
        return min(100, base_confidence)
    
    async def _calculate_risk_score(self, transaction: Dict, profit: float) -> float:
        """Calculate risk score for the opportunity"""
        risk = 0.0
        
        # Slippage risk
        risk += transaction.get('slippage_percentage', 0) * 2
        
        # Gas cost risk
        gas_ratio = transaction.get('gas_cost', 0) / max(profit, 1)
        risk += min(30, gas_ratio * 100)
        
        # Market volatility risk
        risk += transaction.get('market_volatility', 0) * 0.5
        
        # Competition risk
        if transaction.get('mempool_time_ms', 0) > 100:
            risk += 20
        
        return min(100, risk)
    
    async def _get_pool_addresses(self, dexes: List[str], tokens: List[str]) -> List[str]:
        """Get pool addresses for the path"""
        pools = []
        for i in range(len(tokens) - 1):
            # Generate deterministic pool address (placeholder)
            pool = f"{dexes[i] if i < len(dexes) else 'default'}_{tokens[i][:8]}_{tokens[i+1][:8]}"
            pools.append(pool)
        return pools
    
    async def _cache_opportunity(self, opportunity: Dict):
        """Cache opportunity for quick retrieval"""
        key = f"opportunity:{opportunity['opportunity_id']}"
        await self.redis.setex(
            key,
            self.config.cache_ttl,
            msgpack.packb(opportunity)
        )
        
        # Add to sorted set for ranking
        await self.redis.zadd(
            'opportunities:by_profit',
            opportunity['minimum_profit'],
            opportunity['opportunity_id']
        )
    
    def _compile_jit_functions(self):
        """Pre-compile JIT functions for speed"""
        # Warm up JIT compilation
        dummy_amounts = np.array([1000.0], dtype=np.float64)
        dummy_fees = np.array([0.003], dtype=np.float64)
        dummy_prices = np.array([1.0], dtype=np.float64)
        
        self._calculate_arbitrage_profit(dummy_amounts, dummy_fees, dummy_prices)
        logger.info("JIT functions compiled")

class FeatureExtractor:
    """ML feature extraction with GPU acceleration"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.use_gpu = config.use_gpu and cuda.is_available()
        self.feature_cache = deque(maxlen=config.feature_cache_size)
        
        if self.use_gpu:
            logger.info("GPU acceleration enabled for feature extraction")
    
    @staticmethod
    @vectorize(['float32(float32, float32)'], target='cuda' if cuda.is_available() else 'cpu')
    def _calculate_momentum(price_current, price_previous):
        """GPU-accelerated momentum calculation"""
        return (price_current - price_previous) / price_previous * 100
    
    async def extract_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ML features from transaction"""
        
        features = {
            # Temporal features
            'hour_of_day': datetime.fromisoformat(transaction['block_timestamp']).hour,
            'day_of_week': datetime.fromisoformat(transaction['block_timestamp']).weekday(),
            'is_weekend': datetime.fromisoformat(transaction['block_timestamp']).weekday() >= 5,
            
            # Market features
            'volatility_percentile': self._calculate_percentile(
                transaction.get('market_volatility', 0),
                'volatility'
            ),
            'volume_percentile': self._calculate_percentile(
                transaction.get('volume_24h', 0),
                'volume'
            ),
            'liquidity_percentile': self._calculate_percentile(
                transaction.get('liquidity_depth', 0),
                'liquidity'
            ),
            
            # Path features
            'path_complexity': self._calculate_path_complexity(transaction),
            'unique_dex_count': len(set(transaction.get('dexes', []))),
            'unique_token_count': len(set(transaction.get('tokens', []))),
            'max_hop_distance': len(transaction.get('dexes', [])),
            
            # Competition features
            'mempool_congestion': min(100, transaction.get('mempool_time_ms', 0) / 10),
            'gas_price_percentile': self._calculate_percentile(
                transaction.get('priority_fee', 0),
                'gas_price'
            ),
            
            # Technical indicators
            'rsi': await self._calculate_rsi(transaction),
            'macd_signal': await self._calculate_macd(transaction),
            'bollinger_position': await self._calculate_bollinger_position(transaction),
            
            # Risk features
            'risk_score': transaction.get('volatility_score', 50),
            'expected_slippage': transaction.get('slippage_percentage', 0),
            'revert_probability': self._estimate_revert_probability(transaction),
            
            # Target variable
            'is_profitable': transaction.get('net_profit', 0) > 0,
            'profit_amount': float(transaction.get('net_profit', 0))
        }
        
        # Apply GPU acceleration if available
        if self.use_gpu:
            features = self._gpu_accelerate_features(features)
        
        return features
    
    def _calculate_percentile(self, value: float, metric_type: str) -> float:
        """Calculate percentile rank for a metric"""
        # In production, this would use historical data
        # For now, using a placeholder calculation
        if metric_type == 'volatility':
            return min(100, value * 2)
        elif metric_type == 'volume':
            return min(100, np.log10(max(1, value)) * 10)
        elif metric_type == 'liquidity':
            return min(100, np.log10(max(1, value)) * 8)
        elif metric_type == 'gas_price':
            return min(100, value / 1000)
        return 50.0
    
    def _calculate_path_complexity(self, transaction: Dict) -> float:
        """Calculate complexity score for arbitrage path"""
        base_complexity = len(transaction.get('dexes', [])) * 10
        
        # Add complexity for cross-DEX
        unique_dexes = len(set(transaction.get('dexes', [])))
        base_complexity += unique_dexes * 5
        
        # Add complexity for token hops
        base_complexity += len(transaction.get('tokens', [])) * 3
        
        return min(100, base_complexity)
    
    async def _calculate_rsi(self, transaction: Dict) -> float:
        """Calculate RSI indicator"""
        # Placeholder - would use historical price data
        return np.random.uniform(30, 70)
    
    async def _calculate_macd(self, transaction: Dict) -> float:
        """Calculate MACD signal"""
        # Placeholder - would use historical price data
        return np.random.uniform(-1, 1)
    
    async def _calculate_bollinger_position(self, transaction: Dict) -> float:
        """Calculate position within Bollinger Bands"""
        # Placeholder - would use historical price data
        return np.random.uniform(-1, 1)
    
    def _estimate_revert_probability(self, transaction: Dict) -> float:
        """Estimate probability of transaction revert"""
        prob = 0.0
        
        # High slippage increases revert probability
        if transaction.get('slippage_percentage', 0) > 5:
            prob += 0.3
        
        # Complex paths more likely to revert
        if len(transaction.get('dexes', [])) > 4:
            prob += 0.2
        
        # Low liquidity increases revert probability
        if transaction.get('liquidity_depth', float('inf')) < 100000:
            prob += 0.2
        
        return min(1.0, prob)
    
    def _gpu_accelerate_features(self, features: Dict) -> Dict:
        """Apply GPU acceleration to feature calculations"""
        # Convert to CuPy arrays for GPU processing
        numeric_features = {k: v for k, v in features.items() 
                          if isinstance(v, (int, float))}
        
        gpu_array = cp.array(list(numeric_features.values()), dtype=cp.float32)
        
        # Apply transformations on GPU
        gpu_array = cp.log1p(cp.abs(gpu_array))  # Log transform
        gpu_array = (gpu_array - cp.mean(gpu_array)) / cp.std(gpu_array)  # Normalize
        
        # Convert back to CPU
        processed = cp.asnumpy(gpu_array)
        
        # Update features
        for i, key in enumerate(numeric_features.keys()):
            features[f"{key}_processed"] = float(processed[i])
        
        return features

class DataAggregator:
    """Aggregate data for batch processing and ML training"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.aggregation_buffer = []
        self.window_data = deque(maxlen=config.window_size)
        self.redis = None
    
    async def initialize(self):
        """Initialize aggregator"""
        self.redis = await aioredis.create_redis_pool(
            self.config.redis_url,
            encoding='utf-8'
        )
    
    async def aggregate_batch(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Aggregate batch of transactions"""
        
        if not transactions:
            return {}
        
        df = pd.DataFrame(transactions)
        
        aggregated = {
            'timestamp': datetime.utcnow().isoformat(),
            'batch_size': len(transactions),
            
            # Profit metrics
            'total_profit': df['net_profit'].sum(),
            'avg_profit': df['net_profit'].mean(),
            'median_profit': df['net_profit'].median(),
            'profit_std': df['net_profit'].std(),
            'max_profit': df['net_profit'].max(),
            'min_profit': df['net_profit'].min(),
            
            # Performance metrics
            'avg_roi': df['roi_percentage'].mean(),
            'success_rate': (df['net_profit'] > 0).mean(),
            'avg_execution_time': df['execution_time_ms'].mean(),
            'p95_execution_time': df['execution_time_ms'].quantile(0.95),
            
            # Path metrics
            'avg_path_length': df['hop_count'].mean(),
            'unique_searchers': df['searcher_address'].nunique(),
            'unique_paths': df['path_hash'].nunique(),
            
            # Risk metrics
            'avg_slippage': df['slippage_percentage'].mean(),
            'avg_volatility': df['volatility_score'].mean(),
            
            # Market metrics
            'total_volume': df['amounts'].apply(lambda x: x[0] if x else 0).sum(),
            'avg_liquidity': df['liquidity_depth'].mean(),
        }
        
        # Cache aggregated data
        await self._cache_aggregation(aggregated)
        
        return aggregated
    
    async def _cache_aggregation(self, aggregated: Dict):
        """Cache aggregated data"""
        key = f"aggregation:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        await self.redis.setex(
            key,
            3600,  # 1 hour TTL
            json.dumps(aggregated)
        )

# Main processor orchestrator
class DataProcessor:
    """Main data processing orchestrator"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.detector = ArbitrageDetector(config)
        self.extractor = FeatureExtractor(config)
        self.aggregator = DataAggregator(config)
        self.processing_queue = asyncio.Queue(maxsize=10000)
        self.stats = {
            'processed': 0,
            'opportunities_found': 0,
            'features_extracted': 0,
            'batches_aggregated': 0
        }
    
    async def initialize(self):
        """Initialize all components"""
        await self.detector.initialize()
        await self.aggregator.initialize()
        logger.info("Data processor initialized")
    
    async def process_transaction(self, transaction: Dict) -> Dict[str, Any]:
        """Process single transaction through pipeline"""
        result = {}
        
        # Detect arbitrage opportunity
        opportunity = await self.detector.detect_arbitrage(transaction)
        if opportunity:
            result['opportunity'] = opportunity
            self.stats['opportunities_found'] += 1
        
        # Extract ML features
        features = await self.extractor.extract_features(transaction)
        result['features'] = features
        self.stats['features_extracted'] += 1
        
        # Update stats
        self.stats['processed'] += 1
        
        return result
    
    async def process_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Process batch of transactions in parallel"""
        tasks = [self.process_transaction(tx) for tx in transactions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        # Aggregate batch
        aggregated = await self.aggregator.aggregate_batch(transactions)
        self.stats['batches_aggregated'] += 1
        # Attach current fee percentile snapshot for EV-aware consumers (best-effort)
        try:
            import aioredis  # type: ignore
            redis = await aioredis.create_redis_pool(self.config.redis_url, encoding='utf-8')
            p90 = await redis.get('jito-probe:priority-fees:current:p90')
            if p90 is not None:
                aggregated['lamports_per_cu_p90'] = float(p90)
            redis.close()
            await redis.wait_closed()
        except Exception:
            pass
        
        return valid_results
    
    async def run_processing_loop(self):
        """Main processing loop"""
        batch = []
        
        while True:
            try:
                # Get transaction from queue
                transaction = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )
                
                batch.append(transaction)
                
                # Process batch when full
                if len(batch) >= self.config.batch_size:
                    await self.process_batch(batch)
                    batch = []
                
            except asyncio.TimeoutError:
                # Process partial batch on timeout
                if batch:
                    await self.process_batch(batch)
                    batch = []
            
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            **self.stats,
            'queue_size': self.processing_queue.qsize(),
            'detector_cache_size': len(self.detector.price_cache),
            'feature_cache_size': len(self.extractor.feature_cache)
        }