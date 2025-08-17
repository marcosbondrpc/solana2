"""
Elite Entity Behavioral Analysis System
Profiles MEV actors with sophisticated behavioral metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
from collections import defaultdict, Counter
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd


@dataclass
class EntityProfile:
    """Comprehensive behavioral profile for MEV entities"""
    address: str
    profile_date: datetime
    
    # Activity metrics
    total_transactions: int = 0
    mev_transactions: int = 0
    success_rate: float = 0.0
    
    # Financial metrics
    total_volume: float = 0.0
    total_profit: float = 0.0
    average_profit: float = 0.0
    max_single_profit: float = 0.0
    
    # Behavioral patterns
    attack_style: str = "unknown"  # surgical, shotgun, mixed
    victim_selection_pattern: str = "unknown"  # opportunistic, targeted, random
    risk_appetite: float = 0.0  # 0-1 scale
    fee_posture: str = "unknown"  # aggressive, moderate, conservative
    
    # Temporal patterns
    active_hours: List[int] = field(default_factory=list)
    uptime_percentage: float = 0.0
    avg_txns_per_hour: float = 0.0
    burst_pattern: str = "unknown"  # continuous, burst, irregular
    
    # Pool preferences
    preferred_pools: List[str] = field(default_factory=list)
    preferred_tokens: List[str] = field(default_factory=list)
    
    # Advanced metrics
    slippage_imposed_avg: float = 0.0
    sandwich_success_rate: float = 0.0
    arbitrage_path_complexity: float = 0.0
    
    # Clustering
    behavioral_embedding: np.ndarray = field(default_factory=lambda: np.zeros(64))
    cluster_id: Optional[int] = None
    
    # Decision DNA
    profile_dna: str = ""


class BehavioralAnalyzer:
    """Advanced behavioral analysis for MEV entities"""
    
    # Target addresses for special monitoring
    PRIORITY_ADDRESSES = {
        'B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi': 'high_volume_arbitrageur',
        '6gAnjderE13TGGFeqdPVQ438jp2FPVeyXAszxKu9y338': 'sophisticated_sandwicher',
        'CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C': 'raydium_pool',
        'E6YoRP3adE5XYneSseLee15wJshDxCsmyD2WtLvAmfLi': 'flash_loan_expert',
        'pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA': 'pumpswap_specialist'
    }
    
    def __init__(self):
        self.entity_cache = {}
        self.transaction_history = defaultdict(list)
        self.clustering_model = DBSCAN(eps=0.3, min_samples=5)
        self.scaler = StandardScaler()
        
    def analyze_entity(self, address: str, transactions: List[Dict]) -> EntityProfile:
        """Generate comprehensive behavioral profile for an entity"""
        
        profile = EntityProfile(
            address=address,
            profile_date=datetime.utcnow()
        )
        
        if not transactions:
            return profile
        
        # Sort transactions by timestamp
        transactions = sorted(transactions, key=lambda x: x.get('block_time', 0))
        
        # Basic metrics
        profile.total_transactions = len(transactions)
        profile.mev_transactions = sum(1 for tx in transactions if tx.get('is_mev', False))
        profile.success_rate = sum(1 for tx in transactions if tx.get('success', False)) / len(transactions)
        
        # Financial metrics
        profile.total_volume = sum(tx.get('amount', 0) for tx in transactions)
        profits = [tx.get('profit', 0) for tx in transactions if tx.get('profit', 0) > 0]
        if profits:
            profile.total_profit = sum(profits)
            profile.average_profit = np.mean(profits)
            profile.max_single_profit = max(profits)
        
        # Attack style classification
        profile.attack_style = self._classify_attack_style(transactions)
        
        # Victim selection pattern
        profile.victim_selection_pattern = self._analyze_victim_selection(transactions)
        
        # Risk appetite
        profile.risk_appetite = self._calculate_risk_appetite(transactions)
        
        # Fee posture
        profile.fee_posture = self._classify_fee_posture(transactions)
        
        # Temporal patterns
        profile.active_hours = self._extract_active_hours(transactions)
        profile.uptime_percentage = self._calculate_uptime(transactions)
        profile.avg_txns_per_hour = self._calculate_transaction_rate(transactions)
        profile.burst_pattern = self._classify_burst_pattern(transactions)
        
        # Pool and token preferences
        profile.preferred_pools = self._extract_preferred_pools(transactions)
        profile.preferred_tokens = self._extract_preferred_tokens(transactions)
        
        # Advanced metrics
        profile.slippage_imposed_avg = self._calculate_avg_slippage(transactions)
        profile.sandwich_success_rate = self._calculate_sandwich_success(transactions)
        profile.arbitrage_path_complexity = self._calculate_path_complexity(transactions)
        
        # Generate behavioral embedding
        profile.behavioral_embedding = self._generate_embedding(profile)
        
        # Generate profile DNA
        profile.profile_dna = self._generate_profile_dna(profile)
        
        # Cache the profile
        self.entity_cache[address] = profile
        
        return profile
    
    def _classify_attack_style(self, transactions: List[Dict]) -> str:
        """Classify entity's attack style based on transaction patterns"""
        
        if not transactions:
            return "unknown"
        
        # Analyze transaction patterns
        mev_txs = [tx for tx in transactions if tx.get('is_mev', False)]
        
        if not mev_txs:
            return "passive"
        
        # Calculate metrics
        time_diffs = []
        for i in range(1, len(mev_txs)):
            diff = mev_txs[i].get('block_time', 0) - mev_txs[i-1].get('block_time', 0)
            time_diffs.append(diff)
        
        if not time_diffs:
            return "single_shot"
        
        avg_interval = np.mean(time_diffs)
        std_interval = np.std(time_diffs)
        
        # Number of unique pools targeted
        unique_pools = len(set(tx.get('pool', '') for tx in mev_txs))
        
        # Classification logic
        if std_interval < avg_interval * 0.3 and unique_pools < 3:
            return "surgical"  # Focused, consistent attacks on specific pools
        elif unique_pools > 10 and len(mev_txs) > 100:
            return "shotgun"  # Wide-ranging, high-volume attacks
        else:
            return "mixed"  # Combination of strategies
    
    def _analyze_victim_selection(self, transactions: List[Dict]) -> str:
        """Analyze how entity selects victims"""
        
        sandwich_txs = [tx for tx in transactions if tx.get('mev_type') == 'sandwich']
        
        if not sandwich_txs:
            return "none"
        
        # Analyze victim characteristics
        victim_amounts = [tx.get('victim_amount', 0) for tx in sandwich_txs]
        
        if not victim_amounts:
            return "unknown"
        
        avg_victim_amount = np.mean(victim_amounts)
        std_victim_amount = np.std(victim_amounts)
        
        # Check for patterns
        if std_victim_amount < avg_victim_amount * 0.2:
            return "targeted"  # Consistent victim profile
        elif avg_victim_amount > 10000:
            return "whale_hunter"  # Targets large transactions
        else:
            return "opportunistic"  # Takes any opportunity
    
    def _calculate_risk_appetite(self, transactions: List[Dict]) -> float:
        """Calculate risk appetite score (0-1)"""
        
        if not transactions:
            return 0.0
        
        # Factors for risk appetite
        failed_txs = sum(1 for tx in transactions if not tx.get('success', False))
        failure_rate = failed_txs / len(transactions)
        
        # Average priority fee as percentage of transaction
        priority_fees = [tx.get('priority_fee', 0) / max(tx.get('amount', 1), 1) 
                        for tx in transactions]
        avg_priority_ratio = np.mean(priority_fees) if priority_fees else 0
        
        # Complexity of strategies
        mev_types = set(tx.get('mev_type', 'normal') for tx in transactions)
        strategy_diversity = len(mev_types) / 5.0  # Normalize by max types
        
        # Calculate composite risk score
        risk_score = (
            failure_rate * 0.3 +  # Willingness to fail
            min(avg_priority_ratio * 10, 1.0) * 0.4 +  # Fee aggression
            strategy_diversity * 0.3  # Strategy complexity
        )
        
        return min(risk_score, 1.0)
    
    def _classify_fee_posture(self, transactions: List[Dict]) -> str:
        """Classify fee payment behavior"""
        
        priority_fees = [tx.get('priority_fee', 0) for tx in transactions]
        
        if not priority_fees:
            return "unknown"
        
        avg_fee = np.mean(priority_fees)
        percentile_95 = np.percentile(priority_fees, 95)
        
        # Classification thresholds (in lamports)
        if percentile_95 > 1000000:  # > 0.001 SOL
            return "aggressive"
        elif avg_fee > 100000:  # > 0.0001 SOL
            return "moderate"
        else:
            return "conservative"
    
    def _extract_active_hours(self, transactions: List[Dict]) -> List[int]:
        """Extract hours of day when entity is most active"""
        
        hours = []
        for tx in transactions:
            timestamp = tx.get('block_time', 0)
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
                hours.append(dt.hour)
        
        # Return top 8 most active hours
        hour_counts = Counter(hours)
        return [hour for hour, _ in hour_counts.most_common(8)]
    
    def _calculate_uptime(self, transactions: List[Dict]) -> float:
        """Calculate percentage of time entity is active"""
        
        if len(transactions) < 2:
            return 0.0
        
        timestamps = sorted([tx.get('block_time', 0) for tx in transactions if tx.get('block_time')])
        
        if len(timestamps) < 2:
            return 0.0
        
        total_span = timestamps[-1] - timestamps[0]
        
        if total_span == 0:
            return 100.0
        
        # Calculate active periods (within 1 hour windows)
        active_hours = set()
        for ts in timestamps:
            hour_bucket = ts // 3600
            active_hours.add(hour_bucket)
        
        total_hours = total_span / 3600
        uptime_pct = (len(active_hours) / max(total_hours, 1)) * 100
        
        return min(uptime_pct, 100.0)
    
    def _calculate_transaction_rate(self, transactions: List[Dict]) -> float:
        """Calculate average transactions per hour"""
        
        if len(transactions) < 2:
            return 0.0
        
        timestamps = sorted([tx.get('block_time', 0) for tx in transactions if tx.get('block_time')])
        
        if len(timestamps) < 2:
            return 0.0
        
        total_hours = (timestamps[-1] - timestamps[0]) / 3600
        
        if total_hours == 0:
            return len(transactions)
        
        return len(transactions) / total_hours
    
    def _classify_burst_pattern(self, transactions: List[Dict]) -> str:
        """Classify temporal burst patterns"""
        
        if len(transactions) < 10:
            return "insufficient_data"
        
        timestamps = sorted([tx.get('block_time', 0) for tx in transactions if tx.get('block_time')])
        
        if len(timestamps) < 10:
            return "insufficient_data"
        
        # Calculate inter-transaction times
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv = std_interval / avg_interval if avg_interval > 0 else 0
        
        # Classify based on coefficient of variation
        if cv < 0.5:
            return "continuous"  # Regular, steady activity
        elif cv > 1.5:
            return "burst"  # High variability, burst activity
        else:
            return "irregular"  # Mixed pattern
    
    def _extract_preferred_pools(self, transactions: List[Dict]) -> List[str]:
        """Extract most frequently used pools"""
        
        pools = [tx.get('pool', '') for tx in transactions if tx.get('pool')]
        pool_counts = Counter(pools)
        
        # Return top 5 pools
        return [pool for pool, _ in pool_counts.most_common(5)]
    
    def _extract_preferred_tokens(self, transactions: List[Dict]) -> List[str]:
        """Extract most frequently traded tokens"""
        
        tokens = []
        for tx in transactions:
            tokens.extend(tx.get('tokens', []))
        
        token_counts = Counter(tokens)
        
        # Return top 10 tokens
        return [token for token, _ in token_counts.most_common(10)]
    
    def _calculate_avg_slippage(self, transactions: List[Dict]) -> float:
        """Calculate average slippage imposed on victims"""
        
        slippages = [tx.get('slippage', 0) for tx in transactions 
                    if tx.get('mev_type') == 'sandwich' and tx.get('slippage')]
        
        if not slippages:
            return 0.0
        
        return np.mean(slippages)
    
    def _calculate_sandwich_success(self, transactions: List[Dict]) -> float:
        """Calculate sandwich attack success rate"""
        
        sandwich_txs = [tx for tx in transactions if tx.get('mev_type') == 'sandwich']
        
        if not sandwich_txs:
            return 0.0
        
        successful = sum(1 for tx in sandwich_txs if tx.get('profit', 0) > 0)
        
        return successful / len(sandwich_txs)
    
    def _calculate_path_complexity(self, transactions: List[Dict]) -> float:
        """Calculate average arbitrage path complexity"""
        
        arb_txs = [tx for tx in transactions if tx.get('mev_type') == 'arbitrage']
        
        if not arb_txs:
            return 0.0
        
        path_lengths = [len(tx.get('path', [])) for tx in arb_txs if tx.get('path')]
        
        if not path_lengths:
            return 0.0
        
        return np.mean(path_lengths)
    
    def _generate_embedding(self, profile: EntityProfile) -> np.ndarray:
        """Generate behavioral embedding vector for clustering"""
        
        # Create feature vector
        features = [
            profile.total_transactions,
            profile.mev_transactions,
            profile.success_rate,
            profile.total_volume,
            profile.total_profit,
            profile.average_profit,
            profile.risk_appetite,
            len(profile.active_hours),
            profile.uptime_percentage,
            profile.avg_txns_per_hour,
            len(profile.preferred_pools),
            len(profile.preferred_tokens),
            profile.slippage_imposed_avg,
            profile.sandwich_success_rate,
            profile.arbitrage_path_complexity,
            
            # Categorical features as one-hot
            1 if profile.attack_style == 'surgical' else 0,
            1 if profile.attack_style == 'shotgun' else 0,
            1 if profile.victim_selection_pattern == 'targeted' else 0,
            1 if profile.fee_posture == 'aggressive' else 0,
            1 if profile.burst_pattern == 'burst' else 0,
        ]
        
        # Pad or truncate to 64 dimensions
        embedding = np.zeros(64)
        embedding[:len(features)] = features
        
        return embedding
    
    def _generate_profile_dna(self, profile: EntityProfile) -> str:
        """Generate cryptographic DNA for profile"""
        
        profile_dict = {
            'address': profile.address,
            'date': profile.profile_date.isoformat(),
            'transactions': profile.total_transactions,
            'style': profile.attack_style,
            'risk': profile.risk_appetite,
            'embedding': profile.behavioral_embedding.tolist()
        }
        
        profile_str = json.dumps(profile_dict, sort_keys=True)
        return hashlib.sha256(profile_str.encode()).hexdigest()
    
    def cluster_entities(self, profiles: List[EntityProfile]) -> Dict[int, List[str]]:
        """Cluster entities based on behavioral similarity"""
        
        if len(profiles) < 2:
            return {0: [p.address for p in profiles]}
        
        # Extract embeddings
        embeddings = np.array([p.behavioral_embedding for p in profiles])
        
        # Normalize
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Cluster
        labels = self.clustering_model.fit_predict(embeddings_scaled)
        
        # Group by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            profiles[i].cluster_id = label
            clusters[label].append(profiles[i].address)
        
        return dict(clusters)
    
    def identify_coordinated_actors(self, profiles: List[EntityProfile]) -> List[Tuple[str, str, float]]:
        """Identify potentially coordinated actors based on behavioral similarity"""
        
        coordinated_pairs = []
        
        for i, profile1 in enumerate(profiles):
            for profile2 in profiles[i+1:]:
                # Calculate similarity
                similarity = self._calculate_similarity(profile1, profile2)
                
                # High similarity threshold
                if similarity > 0.85:
                    coordinated_pairs.append((
                        profile1.address,
                        profile2.address,
                        similarity
                    ))
        
        return coordinated_pairs
    
    def _calculate_similarity(self, profile1: EntityProfile, profile2: EntityProfile) -> float:
        """Calculate behavioral similarity between two profiles"""
        
        # Cosine similarity of embeddings
        embedding1 = profile1.behavioral_embedding
        embedding2 = profile2.behavioral_embedding
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Additional similarity factors
        pool_overlap = len(set(profile1.preferred_pools) & set(profile2.preferred_pools))
        hour_overlap = len(set(profile1.active_hours) & set(profile2.active_hours))
        
        # Weighted similarity
        similarity = (
            cosine_sim * 0.6 +
            min(pool_overlap / 5, 1.0) * 0.2 +
            min(hour_overlap / 8, 1.0) * 0.2
        )
        
        return min(similarity, 1.0)


class BehavioralSpectrumAnalyzer:
    """Advanced spectrum analysis for entity behaviors"""
    
    def __init__(self):
        self.analyzer = BehavioralAnalyzer()
        
    def generate_spectrum_report(self, address: str, transactions: List[Dict]) -> Dict:
        """Generate comprehensive behavioral spectrum report"""
        
        profile = self.analyzer.analyze_entity(address, transactions)
        
        # Generate spectrum metrics
        spectrum = {
            'entity': address,
            'classification': self._classify_entity_type(profile),
            'risk_level': self._calculate_risk_level(profile),
            'sophistication_score': self._calculate_sophistication(profile),
            'market_impact': self._estimate_market_impact(profile),
            'detection_confidence': self._calculate_detection_confidence(profile),
            
            'behavioral_metrics': {
                'attack_style': profile.attack_style,
                'victim_selection': profile.victim_selection_pattern,
                'risk_appetite': profile.risk_appetite,
                'fee_posture': profile.fee_posture,
                'temporal_pattern': profile.burst_pattern,
            },
            
            'financial_metrics': {
                'total_volume': profile.total_volume,
                'total_profit': profile.total_profit,
                'average_profit': profile.average_profit,
                'success_rate': profile.success_rate,
            },
            
            'operational_metrics': {
                'uptime': profile.uptime_percentage,
                'transaction_rate': profile.avg_txns_per_hour,
                'active_hours': profile.active_hours,
                'preferred_pools': profile.preferred_pools[:3],
            },
            
            'advanced_metrics': {
                'slippage_impact': profile.slippage_imposed_avg,
                'sandwich_success': profile.sandwich_success_rate,
                'path_complexity': profile.arbitrage_path_complexity,
            },
            
            'profile_dna': profile.profile_dna,
            'cluster_id': profile.cluster_id,
        }
        
        return spectrum
    
    def _classify_entity_type(self, profile: EntityProfile) -> str:
        """Classify entity into behavioral categories"""
        
        if profile.total_transactions < 10:
            return "novice"
        
        if profile.sandwich_success_rate > 0.7 and profile.attack_style == "surgical":
            return "elite_sandwicher"
        elif profile.arbitrage_path_complexity > 3:
            return "complex_arbitrageur"
        elif profile.risk_appetite > 0.8:
            return "high_risk_trader"
        elif profile.success_rate > 0.8 and profile.total_profit > 100000:
            return "professional_mev"
        else:
            return "opportunistic_trader"
    
    def _calculate_risk_level(self, profile: EntityProfile) -> str:
        """Calculate entity risk level"""
        
        risk_score = (
            profile.risk_appetite * 0.3 +
            min(profile.total_volume / 1000000, 1.0) * 0.3 +
            min(profile.avg_txns_per_hour / 100, 1.0) * 0.2 +
            (1 if profile.fee_posture == "aggressive" else 0.5) * 0.2
        )
        
        if risk_score > 0.7:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_sophistication(self, profile: EntityProfile) -> float:
        """Calculate sophistication score (0-100)"""
        
        factors = [
            profile.sandwich_success_rate * 20,
            min(profile.arbitrage_path_complexity / 5, 1.0) * 20,
            (1 if profile.attack_style == "surgical" else 0.5) * 20,
            profile.success_rate * 20,
            min(len(profile.preferred_pools) / 10, 1.0) * 20,
        ]
        
        return sum(factors)
    
    def _estimate_market_impact(self, profile: EntityProfile) -> str:
        """Estimate entity's market impact"""
        
        if profile.total_volume > 10000000:  # > 10M
            return "major"
        elif profile.total_volume > 1000000:  # > 1M
            return "significant"
        elif profile.total_volume > 100000:  # > 100K
            return "moderate"
        else:
            return "minimal"
    
    def _calculate_detection_confidence(self, profile: EntityProfile) -> float:
        """Calculate confidence in behavioral detection"""
        
        # More data = higher confidence
        data_confidence = min(profile.total_transactions / 1000, 1.0)
        
        # Clear patterns = higher confidence
        pattern_confidence = 0.5
        if profile.attack_style != "unknown":
            pattern_confidence += 0.25
        if profile.victim_selection_pattern != "unknown":
            pattern_confidence += 0.25
        
        return (data_confidence * 0.6 + pattern_confidence * 0.4)