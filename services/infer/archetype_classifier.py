#!/usr/bin/env python3
"""
Archetype Classification System for MEV Entities
DEFENSIVE-ONLY: Pure analysis and classification

Classifies MEV entities into behavioral archetypes:
- Empire: High-volume, 24/7 operations, sophisticated infrastructure
- Warlord: Specialized, program-specific, tactical presence
- Guerrilla: Opportunistic, niche pools, wallet rotation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
import clickhouse_connect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EntityProfile:
    """Behavioral profile for an MEV entity"""
    address: str
    total_txs: int
    sandwich_count: int
    victim_count: int
    unique_pools: int
    total_extraction_sol: float
    avg_response_ms: float
    p50_response_ms: float
    p99_response_ms: float
    landing_rate: float
    active_hours_bitmap: int
    linked_wallets: List[str]
    cluster_id: Optional[int]
    
    # Archetype indices
    empire_index: float = 0.0
    warlord_index: float = 0.0
    guerrilla_index: float = 0.0
    
    # Behavioral features
    volume_consistency: float = 0.0
    timing_precision: float = 0.0
    pool_specialization: float = 0.0
    wallet_rotation_rate: float = 0.0
    bundle_sophistication: float = 0.0


class ArchetypeClassifier:
    """
    Multi-dimensional behavioral classifier for MEV entities
    Uses measurable indices to identify operational patterns
    """
    
    def __init__(self, clickhouse_client):
        self.client = clickhouse_client
        self.scaler = StandardScaler()
        
        # Thresholds for archetype classification
        self.thresholds = {
            'empire': {
                'min_daily_txs': 1000,
                'min_landing_rate': 0.65,
                'max_p99_latency': 20.0,
                'min_unique_pools': 50,
                'min_uptime_hours': 20,
            },
            'warlord': {
                'min_specialization': 0.7,
                'min_landing_rate': 0.55,
                'min_tactical_precision': 0.6,
            },
            'guerrilla': {
                'max_consistency': 0.3,
                'min_rotation_rate': 0.5,
                'burst_threshold': 0.7,
            }
        }
    
    def classify_entity(self, address: str, lookback_days: int = 30) -> EntityProfile:
        """
        Classify a single entity based on historical behavior
        """
        # Fetch entity data from ClickHouse
        profile_data = self._fetch_entity_data(address, lookback_days)
        
        # Calculate behavioral features
        features = self._extract_features(profile_data)
        
        # Calculate archetype indices
        empire_index = self._calculate_empire_index(features)
        warlord_index = self._calculate_warlord_index(features)
        guerrilla_index = self._calculate_guerrilla_index(features)
        
        # Detect linked wallets and clusters
        linked_wallets = self._detect_linked_wallets(address, profile_data)
        cluster_id = self._assign_cluster(address, features)
        
        # Build entity profile
        profile = EntityProfile(
            address=address,
            total_txs=profile_data['total_txs'],
            sandwich_count=profile_data['sandwich_count'],
            victim_count=profile_data['victim_count'],
            unique_pools=profile_data['unique_pools'],
            total_extraction_sol=profile_data['total_extraction_sol'],
            avg_response_ms=profile_data['avg_response_ms'],
            p50_response_ms=profile_data['p50_response_ms'],
            p99_response_ms=profile_data['p99_response_ms'],
            landing_rate=profile_data['landing_rate'],
            active_hours_bitmap=profile_data['active_hours_bitmap'],
            linked_wallets=linked_wallets,
            cluster_id=cluster_id,
            empire_index=empire_index,
            warlord_index=warlord_index,
            guerrilla_index=guerrilla_index,
            volume_consistency=features['volume_consistency'],
            timing_precision=features['timing_precision'],
            pool_specialization=features['pool_specialization'],
            wallet_rotation_rate=features['wallet_rotation_rate'],
            bundle_sophistication=features['bundle_sophistication'],
        )
        
        logger.info(f"Classified {address[:8]}: Empire={empire_index:.2f}, "
                   f"Warlord={warlord_index:.2f}, Guerrilla={guerrilla_index:.2f}")
        
        return profile
    
    def _fetch_entity_data(self, address: str, lookback_days: int) -> Dict:
        """
        Fetch entity behavioral data from ClickHouse
        """
        query = f"""
        SELECT
            count() as total_txs,
            countIf(evidence != 'weak') as sandwich_count,
            uniqExact(victim_addr) as victim_count,
            uniqExact(pool) as unique_pools,
            sum(attacker_profit_sol) as total_extraction_sol,
            avg(d_ms) as avg_response_ms,
            quantile(0.5)(d_ms) as p50_response_ms,
            quantile(0.99)(d_ms) as p99_response_ms,
            countIf(landing_status = 'landed') / count() as landing_rate,
            groupBitOr(toHour(detection_ts)) as active_hours_bitmap
        FROM ch.candidates
        WHERE attacker_addr = '{address}'
        AND detection_ts >= now() - INTERVAL {lookback_days} DAY
        """
        
        result = self.client.query(query)
        if result.rows:
            row = result.rows[0]
            return {
                'total_txs': row[0],
                'sandwich_count': row[1],
                'victim_count': row[2],
                'unique_pools': row[3],
                'total_extraction_sol': row[4],
                'avg_response_ms': row[5],
                'p50_response_ms': row[6],
                'p99_response_ms': row[7],
                'landing_rate': row[8],
                'active_hours_bitmap': row[9],
            }
        
        return {}
    
    def _extract_features(self, profile_data: Dict) -> Dict:
        """
        Extract behavioral features for archetype classification
        """
        features = {}
        
        # Volume consistency (coefficient of variation)
        daily_volumes = self._get_daily_volumes(profile_data)
        if len(daily_volumes) > 1:
            features['volume_consistency'] = 1.0 - (np.std(daily_volumes) / (np.mean(daily_volumes) + 1e-9))
        else:
            features['volume_consistency'] = 0.0
        
        # Timing precision (inverse of latency variance)
        features['timing_precision'] = 1.0 / (1.0 + profile_data.get('p99_response_ms', 100) / 10.0)
        
        # Pool specialization (inverse of diversity)
        total_attacks = profile_data.get('sandwich_count', 1)
        unique_pools = profile_data.get('unique_pools', 1)
        features['pool_specialization'] = 1.0 - (unique_pools / (total_attacks + 1))
        
        # Wallet rotation rate
        features['wallet_rotation_rate'] = self._calculate_rotation_rate(profile_data)
        
        # Bundle sophistication
        features['bundle_sophistication'] = self._calculate_bundle_sophistication(profile_data)
        
        return features
    
    def _calculate_empire_index(self, features: Dict) -> float:
        """
        Calculate Empire archetype index
        High volume, consistent presence, sophisticated infrastructure
        """
        index = 0.0
        
        # Volume component (40%)
        if features.get('volume_consistency', 0) > 0.7:
            index += 0.4
        
        # Infrastructure sophistication (30%)
        if features.get('bundle_sophistication', 0) > 0.6:
            index += 0.3
        
        # Timing precision (20%)
        if features.get('timing_precision', 0) > 0.8:
            index += 0.2
        
        # Market coverage (10%)
        if features.get('pool_specialization', 1) < 0.3:  # Low specialization = broad coverage
            index += 0.1
        
        return min(index, 1.0)
    
    def _calculate_warlord_index(self, features: Dict) -> float:
        """
        Calculate Warlord archetype index
        Specialized presence, tactical operations, program-specific
        """
        index = 0.0
        
        # Specialization component (40%)
        if features.get('pool_specialization', 0) > 0.7:
            index += 0.4
        
        # Tactical precision (30%)
        if features.get('timing_precision', 0) > 0.6:
            index += 0.3
        
        # Moderate consistency (20%)
        consistency = features.get('volume_consistency', 0)
        if 0.4 < consistency < 0.7:
            index += 0.2
        
        # Limited rotation (10%)
        if features.get('wallet_rotation_rate', 0) < 0.3:
            index += 0.1
        
        return min(index, 1.0)
    
    def _calculate_guerrilla_index(self, features: Dict) -> float:
        """
        Calculate Guerrilla archetype index
        Opportunistic, high rotation, burst activity
        """
        index = 0.0
        
        # High rotation (40%)
        if features.get('wallet_rotation_rate', 0) > 0.5:
            index += 0.4
        
        # Low consistency (30%)
        if features.get('volume_consistency', 0) < 0.3:
            index += 0.3
        
        # Opportunistic targeting (20%)
        if features.get('pool_specialization', 0) < 0.5:
            index += 0.2
        
        # Lower sophistication (10%)
        if features.get('bundle_sophistication', 0) < 0.4:
            index += 0.1
        
        return min(index, 1.0)
    
    def _get_daily_volumes(self, profile_data: Dict) -> List[float]:
        """
        Fetch daily transaction volumes for consistency analysis
        """
        # Simplified - would query ClickHouse for actual daily volumes
        return [profile_data.get('total_txs', 0) / 30] * 30
    
    def _calculate_rotation_rate(self, profile_data: Dict) -> float:
        """
        Calculate wallet rotation rate based on linked wallet patterns
        """
        # Simplified - would analyze wallet creation/abandonment patterns
        return 0.2  # Placeholder
    
    def _calculate_bundle_sophistication(self, profile_data: Dict) -> float:
        """
        Calculate bundle sophistication score
        """
        landing_rate = profile_data.get('landing_rate', 0)
        avg_response = profile_data.get('avg_response_ms', 100)
        
        # High landing rate + low latency = sophisticated
        sophistication = (landing_rate * 0.6) + ((100 - min(avg_response, 100)) / 100 * 0.4)
        return sophistication
    
    def _detect_linked_wallets(self, address: str, profile_data: Dict) -> List[str]:
        """
        Detect wallets linked to the same operator
        """
        query = f"""
        WITH similar_patterns AS (
            SELECT 
                attacker_addr,
                groupArray(pool) as pools,
                avg(d_ms) as avg_timing,
                avg(priority_fee) as avg_fee
            FROM ch.candidates
            WHERE detection_ts >= now() - INTERVAL 7 DAY
            GROUP BY attacker_addr
        )
        SELECT 
            s1.attacker_addr
        FROM similar_patterns s1
        CROSS JOIN similar_patterns s2
        WHERE s2.attacker_addr = '{address}'
        AND s1.attacker_addr != s2.attacker_addr
        AND abs(s1.avg_timing - s2.avg_timing) < 10
        AND abs(s1.avg_fee - s2.avg_fee) / s2.avg_fee < 0.2
        AND length(arrayIntersect(s1.pools, s2.pools)) > length(s1.pools) * 0.5
        LIMIT 10
        """
        
        result = self.client.query(query)
        return [row[0] for row in result.rows]
    
    def _assign_cluster(self, address: str, features: Dict) -> Optional[int]:
        """
        Assign entity to a behavioral cluster
        """
        # Simplified clustering - would use DBSCAN or similar
        if features.get('empire_index', 0) > 0.7:
            return 1  # Empire cluster
        elif features.get('warlord_index', 0) > 0.7:
            return 2  # Warlord cluster
        elif features.get('guerrilla_index', 0) > 0.7:
            return 3  # Guerrilla cluster
        return None
    
    def batch_classify(self, addresses: List[str], lookback_days: int = 30) -> pd.DataFrame:
        """
        Classify multiple entities and return results as DataFrame
        """
        profiles = []
        for address in addresses:
            try:
                profile = self.classify_entity(address, lookback_days)
                profiles.append(profile)
            except Exception as e:
                logger.error(f"Failed to classify {address}: {e}")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'address': p.address,
                'archetype': self._get_primary_archetype(p),
                'empire_index': p.empire_index,
                'warlord_index': p.warlord_index,
                'guerrilla_index': p.guerrilla_index,
                'total_extraction_sol': p.total_extraction_sol,
                'landing_rate': p.landing_rate,
                'cluster_id': p.cluster_id,
            }
            for p in profiles
        ])
        
        return df
    
    def _get_primary_archetype(self, profile: EntityProfile) -> str:
        """
        Determine primary archetype based on indices
        """
        indices = {
            'Empire': profile.empire_index,
            'Warlord': profile.warlord_index,
            'Guerrilla': profile.guerrilla_index,
        }
        
        max_archetype = max(indices, key=indices.get)
        if indices[max_archetype] < 0.3:
            return 'Unknown'
        return max_archetype
    
    def generate_archetype_report(self, lookback_days: int = 30) -> Dict:
        """
        Generate comprehensive archetype analysis report
        """
        # Fetch top MEV entities
        top_entities_query = """
        SELECT 
            attacker_addr,
            sum(attacker_profit_sol) as total_profit
        FROM ch.candidates
        WHERE detection_ts >= now() - INTERVAL 30 DAY
        GROUP BY attacker_addr
        ORDER BY total_profit DESC
        LIMIT 100
        """
        
        result = self.client.query(top_entities_query)
        addresses = [row[0] for row in result.rows]
        
        # Classify all entities
        df = self.batch_classify(addresses, lookback_days)
        
        # Generate report
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': lookback_days,
            'total_entities': len(df),
            'archetype_distribution': df['archetype'].value_counts().to_dict(),
            'total_extraction_sol': df['total_extraction_sol'].sum(),
            'avg_landing_rate': df['landing_rate'].mean(),
            'top_empires': df[df['archetype'] == 'Empire'].nlargest(5, 'total_extraction_sol').to_dict('records'),
            'top_warlords': df[df['archetype'] == 'Warlord'].nlargest(5, 'total_extraction_sol').to_dict('records'),
            'top_guerrillas': df[df['archetype'] == 'Guerrilla'].nlargest(5, 'total_extraction_sol').to_dict('records'),
            'cluster_sizes': df.groupby('cluster_id').size().to_dict(),
        }
        
        logger.info(f"Generated archetype report: {report['archetype_distribution']}")
        
        return report


class FleetDetector:
    """
    Detect coordinated wallet fleets using behavioral clustering
    """
    
    def __init__(self, clickhouse_client):
        self.client = clickhouse_client
        self.dbscan = DBSCAN(eps=0.3, min_samples=2)
    
    def detect_fleets(self, lookback_days: int = 7) -> List[Dict]:
        """
        Identify coordinated wallet fleets
        """
        # Fetch behavioral features for all active wallets
        features_query = f"""
        SELECT
            attacker_addr,
            avg(d_ms) as avg_timing,
            avg(priority_fee) as avg_fee,
            avg(slippage_victim) as avg_slippage,
            groupArray(pool) as pools,
            count() as tx_count,
            sum(attacker_profit_sol) as total_profit
        FROM ch.candidates
        WHERE detection_ts >= now() - INTERVAL {lookback_days} DAY
        GROUP BY attacker_addr
        HAVING tx_count > 10
        """
        
        result = self.client.query(features_query)
        
        if not result.rows:
            return []
        
        # Build feature matrix
        addresses = []
        features = []
        
        for row in result.rows:
            addresses.append(row[0])
            features.append([
                row[1],  # avg_timing
                row[2],  # avg_fee
                row[3],  # avg_slippage
                len(row[4]),  # num_pools
                row[5],  # tx_count
            ])
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Cluster wallets
        clusters = self.dbscan.fit_predict(features_scaled)
        
        # Build fleet results
        fleets = {}
        for addr, cluster_id in zip(addresses, clusters):
            if cluster_id != -1:  # -1 means noise/unclustered
                if cluster_id not in fleets:
                    fleets[cluster_id] = []
                fleets[cluster_id].append(addr)
        
        # Format results
        fleet_list = []
        for cluster_id, members in fleets.items():
            if len(members) >= 2:  # Only report fleets with 2+ members
                fleet_list.append({
                    'fleet_id': cluster_id,
                    'member_count': len(members),
                    'members': members,
                    'detected_at': datetime.utcnow().isoformat(),
                })
        
        logger.info(f"Detected {len(fleet_list)} wallet fleets")
        
        return fleet_list