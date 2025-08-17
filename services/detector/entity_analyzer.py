#!/usr/bin/env python3
"""
Entity Behavioral Analyzer
DETECTION-ONLY: Pure behavioral profiling and pattern recognition
Tracks attack styles, victim selection, risk profiles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import clickhouse_driver
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
import logging
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityMetrics:
    """Behavioral metrics for an entity"""
    entity_addr: str
    
    # Attack patterns
    total_attacks: int = 0
    successful_attacks: int = 0
    failed_attacks: int = 0
    
    # Victim selection
    retail_victims: int = 0  # < 100 SOL wallets
    whale_victims: int = 0   # > 10000 SOL wallets
    bot_victims: int = 0     # Known bots/arbitrageurs
    
    # Attack style
    surgical_score: float = 0.0  # Precision targeting
    shotgun_score: float = 0.0   # Volume spray
    adaptive_score: float = 0.0  # Pattern changes
    
    # Risk profile
    max_position_sol: float = 0.0
    avg_position_sol: float = 0.0
    loss_tolerance: float = 0.0
    
    # Fee behavior
    avg_priority_fee: float = 0.0
    max_priority_fee: float = 0.0
    fee_escalation_rate: float = 0.0
    
    # Timing patterns
    avg_response_ms: float = 0.0
    p50_response_ms: float = 0.0
    p99_response_ms: float = 0.0
    
    # Economic impact
    total_extraction_sol: float = 0.0
    total_fees_paid_sol: float = 0.0
    profit_margin: float = 0.0
    
    # Network analysis
    unique_pools: int = 0
    unique_venues: int = 0
    unique_victims: int = 0
    
    # Uptime patterns
    active_hours: List[int] = None
    active_days: List[int] = None
    uptime_ratio: float = 0.0

class BehavioralAnalyzer:
    """Analyze entity behavioral patterns"""
    
    def __init__(self, ch_host='localhost', ch_port=9000):
        self.ch = clickhouse_driver.Client(
            host=ch_host,
            port=ch_port,
            settings={'use_numpy': True}
        )
        self.entity_cache = {}
        
    def analyze_entity(self, entity_addr: str, lookback_days: int = 30) -> EntityMetrics:
        """Comprehensive behavioral analysis of an entity"""
        
        metrics = EntityMetrics(entity_addr=entity_addr)
        
        # Get attack history
        attacks = self._get_attack_history(entity_addr, lookback_days)
        if not attacks:
            return metrics
        
        # Basic counts
        metrics.total_attacks = len(attacks)
        metrics.successful_attacks = sum(1 for a in attacks if a['landed'])
        metrics.failed_attacks = metrics.total_attacks - metrics.successful_attacks
        
        # Victim analysis
        victim_profiles = self._analyze_victims(attacks)
        metrics.retail_victims = victim_profiles['retail']
        metrics.whale_victims = victim_profiles['whale']
        metrics.bot_victims = victim_profiles['bot']
        
        # Attack style scoring
        style_scores = self._calculate_attack_style(attacks)
        metrics.surgical_score = style_scores['surgical']
        metrics.shotgun_score = style_scores['shotgun']
        metrics.adaptive_score = style_scores['adaptive']
        
        # Risk analysis
        risk_profile = self._analyze_risk_profile(attacks)
        metrics.max_position_sol = risk_profile['max_position']
        metrics.avg_position_sol = risk_profile['avg_position']
        metrics.loss_tolerance = risk_profile['loss_tolerance']
        
        # Fee behavior
        fee_analysis = self._analyze_fee_behavior(attacks)
        metrics.avg_priority_fee = fee_analysis['avg_priority']
        metrics.max_priority_fee = fee_analysis['max_priority']
        metrics.fee_escalation_rate = fee_analysis['escalation_rate']
        
        # Timing patterns
        timing = self._analyze_timing_patterns(attacks)
        metrics.avg_response_ms = timing['avg_ms']
        metrics.p50_response_ms = timing['p50_ms']
        metrics.p99_response_ms = timing['p99_ms']
        
        # Economic impact
        economics = self._calculate_economics(attacks)
        metrics.total_extraction_sol = economics['total_extraction']
        metrics.total_fees_paid_sol = economics['total_fees']
        metrics.profit_margin = economics['profit_margin']
        
        # Network diversity
        metrics.unique_pools = len(set(a['pool'] for a in attacks if a['pool']))
        metrics.unique_venues = len(set(a['venue'] for a in attacks if a['venue']))
        metrics.unique_victims = len(set(a['victim'] for a in attacks if a['victim']))
        
        # Uptime patterns
        uptime = self._analyze_uptime_patterns(attacks)
        metrics.active_hours = uptime['active_hours']
        metrics.active_days = uptime['active_days']
        metrics.uptime_ratio = uptime['ratio']
        
        return metrics
    
    def _get_attack_history(self, entity_addr: str, lookback_days: int) -> List[Dict]:
        """Fetch attack history from ClickHouse"""
        
        query = f"""
        SELECT 
            c.slot,
            c.victim_sig,
            c.victim_addr,
            c.pool,
            c.d_ms,
            c.slippage_victim,
            c.attacker_profit_sol,
            c.fee_burn_sol,
            c.detection_ts,
            r.venue,
            r.priority_fee,
            r.landing_status = 'landed' as landed,
            r.amount_in,
            r.amount_out
        FROM ch.candidates c
        LEFT JOIN ch.raw_tx r ON c.attacker_a_sig = r.sig
        WHERE c.attacker_addr = '{entity_addr}'
          AND c.detection_ts >= now() - INTERVAL {lookback_days} DAY
        ORDER BY c.slot
        """
        
        results = self.ch.execute(query)
        
        attacks = []
        for row in results:
            attacks.append({
                'slot': row[0],
                'victim_sig': row[1],
                'victim': row[2],
                'pool': row[3],
                'response_ms': row[4],
                'slippage': row[5],
                'profit': row[6],
                'fees': row[7],
                'timestamp': row[8],
                'venue': row[9],
                'priority_fee': row[10],
                'landed': row[11],
                'amount_in': row[12],
                'amount_out': row[13]
            })
        
        return attacks
    
    def _analyze_victims(self, attacks: List[Dict]) -> Dict[str, int]:
        """Classify victim types"""
        
        victims = {'retail': 0, 'whale': 0, 'bot': 0}
        
        for attack in attacks:
            if not attack['victim']:
                continue
            
            # Get victim balance (would need real balance lookup)
            balance = self._estimate_victim_balance(attack['victim'])
            
            if balance < 100:
                victims['retail'] += 1
            elif balance > 10000:
                victims['whale'] += 1
            else:
                # Check if bot (simplified heuristic)
                if self._is_likely_bot(attack['victim']):
                    victims['bot'] += 1
                else:
                    victims['retail'] += 1
        
        return victims
    
    def _calculate_attack_style(self, attacks: List[Dict]) -> Dict[str, float]:
        """Calculate attack style scores"""
        
        if not attacks:
            return {'surgical': 0, 'shotgun': 0, 'adaptive': 0}
        
        # Surgical: High success rate, targeted pools, consistent timing
        success_rate = sum(1 for a in attacks if a['landed']) / len(attacks)
        pool_concentration = 1 - (len(set(a['pool'] for a in attacks)) / len(attacks))
        timing_consistency = 1 - (np.std([a['response_ms'] for a in attacks if a['response_ms']]) / 1000)
        
        surgical = (success_rate * 0.4 + pool_concentration * 0.3 + 
                   max(0, timing_consistency) * 0.3)
        
        # Shotgun: High volume, diverse targets, lower success
        volume_score = min(1, len(attacks) / 100)  # Normalize to 100 attacks
        target_diversity = len(set(a['victim'] for a in attacks)) / len(attacks)
        
        shotgun = volume_score * 0.5 + target_diversity * 0.5 - success_rate * 0.2
        
        # Adaptive: Pattern changes over time
        if len(attacks) > 10:
            # Split into time windows
            mid = len(attacks) // 2
            early_pools = set(a['pool'] for a in attacks[:mid])
            late_pools = set(a['pool'] for a in attacks[mid:])
            
            pool_shift = len(early_pools ^ late_pools) / len(early_pools | late_pools)
            
            early_fees = np.mean([a['priority_fee'] for a in attacks[:mid] if a['priority_fee']])
            late_fees = np.mean([a['priority_fee'] for a in attacks[mid:] if a['priority_fee']])
            fee_adaptation = abs(late_fees - early_fees) / max(early_fees, 1)
            
            adaptive = pool_shift * 0.5 + min(1, fee_adaptation) * 0.5
        else:
            adaptive = 0
        
        # Normalize scores
        total = surgical + shotgun + adaptive
        if total > 0:
            surgical /= total
            shotgun /= total
            adaptive /= total
        
        return {
            'surgical': surgical,
            'shotgun': shotgun,
            'adaptive': adaptive
        }
    
    def _analyze_risk_profile(self, attacks: List[Dict]) -> Dict[str, float]:
        """Analyze risk-taking behavior"""
        
        if not attacks:
            return {'max_position': 0, 'avg_position': 0, 'loss_tolerance': 0}
        
        positions = [a['amount_in'] / 1e9 if a['amount_in'] else 0 for a in attacks]
        profits = [a['profit'] if a['profit'] else 0 for a in attacks]
        
        # Calculate loss tolerance (ratio of losing trades accepted)
        losses = sum(1 for p in profits if p < 0)
        loss_tolerance = losses / len(profits) if profits else 0
        
        return {
            'max_position': max(positions) if positions else 0,
            'avg_position': np.mean(positions) if positions else 0,
            'loss_tolerance': loss_tolerance
        }
    
    def _analyze_fee_behavior(self, attacks: List[Dict]) -> Dict[str, float]:
        """Analyze priority fee patterns"""
        
        fees = [a['priority_fee'] for a in attacks if a['priority_fee']]
        
        if not fees:
            return {'avg_priority': 0, 'max_priority': 0, 'escalation_rate': 0}
        
        # Calculate escalation rate (how fees change over time)
        if len(fees) > 1:
            fee_changes = [fees[i+1] - fees[i] for i in range(len(fees)-1)]
            escalation = np.mean([c for c in fee_changes if c > 0])
        else:
            escalation = 0
        
        return {
            'avg_priority': np.mean(fees),
            'max_priority': max(fees),
            'escalation_rate': escalation
        }
    
    def _analyze_timing_patterns(self, attacks: List[Dict]) -> Dict[str, float]:
        """Analyze response time patterns"""
        
        times = [a['response_ms'] for a in attacks if a['response_ms'] and a['response_ms'] > 0]
        
        if not times:
            return {'avg_ms': 0, 'p50_ms': 0, 'p99_ms': 0}
        
        return {
            'avg_ms': np.mean(times),
            'p50_ms': np.percentile(times, 50),
            'p99_ms': np.percentile(times, 99)
        }
    
    def _calculate_economics(self, attacks: List[Dict]) -> Dict[str, float]:
        """Calculate economic impact"""
        
        total_extraction = sum(a['profit'] for a in attacks if a['profit'] and a['profit'] > 0)
        total_fees = sum(a['fees'] for a in attacks if a['fees'])
        
        profit_margin = (total_extraction - total_fees) / total_extraction if total_extraction > 0 else 0
        
        return {
            'total_extraction': total_extraction,
            'total_fees': total_fees,
            'profit_margin': profit_margin
        }
    
    def _analyze_uptime_patterns(self, attacks: List[Dict]) -> Dict:
        """Analyze activity patterns"""
        
        if not attacks:
            return {'active_hours': [], 'active_days': [], 'ratio': 0}
        
        timestamps = [a['timestamp'] for a in attacks if a['timestamp']]
        
        if not timestamps:
            return {'active_hours': [], 'active_days': [], 'ratio': 0}
        
        # Extract hours and days
        hours = [ts.hour for ts in timestamps]
        days = [ts.weekday() for ts in timestamps]
        
        # Calculate uptime ratio (active hours / total hours in period)
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        total_hours = (last_ts - first_ts).total_seconds() / 3600
        active_hours = len(set((ts.date(), ts.hour) for ts in timestamps))
        
        uptime_ratio = active_hours / total_hours if total_hours > 0 else 0
        
        return {
            'active_hours': list(set(hours)),
            'active_days': list(set(days)),
            'ratio': min(1, uptime_ratio)
        }
    
    def _estimate_victim_balance(self, victim_addr: str) -> float:
        """Estimate victim wallet balance (placeholder)"""
        # In production, would query actual balance
        return np.random.lognormal(3, 2)  # Log-normal distribution
    
    def _is_likely_bot(self, addr: str) -> bool:
        """Check if address is likely a bot (placeholder)"""
        # Simple heuristic: check transaction frequency
        query = f"""
        SELECT count() as tx_count
        FROM ch.raw_tx
        WHERE payer = '{addr}'
          AND ts >= now() - INTERVAL 1 DAY
        """
        
        result = self.ch.execute(query)
        if result and result[0][0] > 100:  # More than 100 tx/day
            return True
        return False

class WalletClusterer:
    """Identify linked wallets and entity clusters"""
    
    def __init__(self, ch_client):
        self.ch = ch_client
        self.graph = nx.Graph()
        
    def find_linked_wallets(self, seed_addr: str, max_depth: int = 2) -> List[str]:
        """Find wallets linked to seed address"""
        
        linked = set([seed_addr])
        to_process = [seed_addr]
        depth = 0
        
        while to_process and depth < max_depth:
            current_batch = to_process
            to_process = []
            
            for addr in current_batch:
                # Find wallets that interact frequently
                connections = self._find_connections(addr)
                
                for connected_addr, strength in connections:
                    if connected_addr not in linked and strength > 0.7:
                        linked.add(connected_addr)
                        to_process.append(connected_addr)
            
            depth += 1
        
        return list(linked)
    
    def _find_connections(self, addr: str) -> List[Tuple[str, float]]:
        """Find strongly connected addresses"""
        
        query = f"""
        SELECT 
            arrayJoin(accounts) as connected_addr,
            count() as interactions,
            sum(amount_in + amount_out) as total_volume
        FROM ch.raw_tx
        WHERE payer = '{addr}'
          AND ts >= now() - INTERVAL 7 DAY
        GROUP BY connected_addr
        HAVING interactions > 10
        ORDER BY interactions DESC
        LIMIT 20
        """
        
        results = self.ch.execute(query)
        
        connections = []
        for connected, interactions, volume in results:
            if connected != addr:
                # Calculate connection strength
                strength = min(1, interactions / 100) * 0.5 + min(1, volume / 1e12) * 0.5
                connections.append((connected, strength))
        
        return connections
    
    def cluster_entities(self, addresses: List[str]) -> Dict[int, List[str]]:
        """Cluster addresses into entity groups"""
        
        # Build feature matrix
        features = []
        valid_addrs = []
        
        for addr in addresses:
            feat = self._extract_features(addr)
            if feat is not None:
                features.append(feat)
                valid_addrs.append(addr)
        
        if not features:
            return {}
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=2)
        labels = clustering.fit_predict(features_scaled)
        
        # Group by cluster
        clusters = {}
        for addr, label in zip(valid_addrs, labels):
            if label >= 0:  # Ignore noise points (-1)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(addr)
        
        return clusters
    
    def _extract_features(self, addr: str) -> Optional[np.ndarray]:
        """Extract clustering features for an address"""
        
        query = f"""
        SELECT 
            count() as tx_count,
            avg(fee) as avg_fee,
            avg(priority_fee) as avg_priority,
            uniqExact(arrayJoin(pool_keys)) as unique_pools,
            sum(amount_in) / 1e9 as total_in,
            sum(amount_out) / 1e9 as total_out
        FROM ch.raw_tx
        WHERE payer = '{addr}'
          AND ts >= now() - INTERVAL 7 DAY
        """
        
        result = self.ch.execute(query)
        
        if result and result[0][0] > 0:
            return np.array(result[0])
        return None

def generate_behavior_report(entity_addr: str, output_file: str = None):
    """Generate comprehensive behavioral report for an entity"""
    
    analyzer = BehavioralAnalyzer(ch_host='clickhouse')
    metrics = analyzer.analyze_entity(entity_addr, lookback_days=30)
    
    report = {
        'entity': entity_addr,
        'generated': datetime.utcnow().isoformat(),
        'metrics': {
            'attacks': {
                'total': metrics.total_attacks,
                'successful': metrics.successful_attacks,
                'failed': metrics.failed_attacks,
                'success_rate': metrics.successful_attacks / max(metrics.total_attacks, 1)
            },
            'style': {
                'primary': 'surgical' if metrics.surgical_score > 0.5 else 
                          'shotgun' if metrics.shotgun_score > 0.5 else 'adaptive',
                'scores': {
                    'surgical': metrics.surgical_score,
                    'shotgun': metrics.shotgun_score,
                    'adaptive': metrics.adaptive_score
                }
            },
            'victims': {
                'retail': metrics.retail_victims,
                'whale': metrics.whale_victims,
                'bot': metrics.bot_victims,
                'unique': metrics.unique_victims
            },
            'risk': {
                'max_position_sol': metrics.max_position_sol,
                'avg_position_sol': metrics.avg_position_sol,
                'loss_tolerance': metrics.loss_tolerance
            },
            'economics': {
                'total_extraction_sol': metrics.total_extraction_sol,
                'total_fees_sol': metrics.total_fees_paid_sol,
                'profit_margin': metrics.profit_margin
            },
            'behavior': {
                'avg_response_ms': metrics.avg_response_ms,
                'p50_response_ms': metrics.p50_response_ms,
                'p99_response_ms': metrics.p99_response_ms,
                'avg_priority_fee': metrics.avg_priority_fee,
                'fee_escalation': metrics.fee_escalation_rate
            },
            'activity': {
                'uptime_ratio': metrics.uptime_ratio,
                'active_hours': metrics.active_hours,
                'active_days': metrics.active_days
            },
            'network': {
                'unique_pools': metrics.unique_pools,
                'unique_venues': metrics.unique_venues
            }
        }
    }
    
    # Find linked wallets
    clusterer = WalletClusterer(analyzer.ch)
    linked = clusterer.find_linked_wallets(entity_addr)
    report['linked_wallets'] = linked
    
    # Generate DNA fingerprint
    dna = hashlib.blake2b(json.dumps(report, sort_keys=True).encode(), digest_size=32).hexdigest()
    report['dna_fingerprint'] = dna
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_file}")
    
    return report

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        entity = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else None
        report = generate_behavior_report(entity, output)
        print(json.dumps(report, indent=2))
    else:
        print("Usage: python entity_analyzer.py <entity_address> [output_file]")