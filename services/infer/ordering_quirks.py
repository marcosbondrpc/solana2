#!/usr/bin/env python3
"""
Transaction Ordering Quirks Detection Service
Detects statistically significant ordering patterns using bootstrap null distributions
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Set
import clickhouse_driver
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OrderingPattern:
    """Represents a detected ordering pattern"""
    attacker_entity: str
    pool_address: str
    pattern_type: str  # SANDWICH, BACKRUN, FRONTRUN, ATOMIC_ARB
    observed_frequency: float
    expected_frequency: float
    adjacency_count: int
    total_observations: int
    bootstrap_p_value: float
    chi_square_statistic: float
    effect_size: float
    temporal_clustering: float
    profit_correlation: float
    decision_dna: str


@dataclass
class QuirkProfile:
    """Profile of ordering quirks for an entity"""
    entity_id: str
    observation_period: timedelta
    total_transactions: int
    sandwich_patterns: List[OrderingPattern]
    backrun_patterns: List[OrderingPattern]
    atomic_patterns: List[OrderingPattern]
    quirk_score: float
    most_targeted_pools: List[Tuple[str, int]]
    timing_precision_std: float


class OrderingQuirksAnalyzer:
    """
    Detects statistically significant transaction ordering patterns
    """
    
    def __init__(self, clickhouse_url: str = "http://localhost:8123"):
        self.client = clickhouse_driver.Client(host=clickhouse_url.replace("http://", "").split(":")[0])
        self.bootstrap_iterations = 1000
        self.significance_level = 0.01  # More stringent for ordering detection
        self.min_observations = 20
        
    def get_transaction_sequences(self, 
                                 lookback_hours: int = 72) -> pd.DataFrame:
        """
        Fetch transaction sequences from ClickHouse
        """
        query = f"""
        WITH ordered_txs AS (
            SELECT 
                block_height,
                slot,
                transaction_index,
                transaction_hash,
                signer as wallet_address,
                entity_id,
                pool_address,
                instruction_type,
                token_in,
                token_out,
                amount_in,
                amount_out,
                timestamp,
                -- Next transaction in same block
                lead(transaction_hash, 1) OVER w as next_tx_hash,
                lead(entity_id, 1) OVER w as next_entity,
                lead(pool_address, 1) OVER w as next_pool,
                lead(instruction_type, 1) OVER w as next_instruction,
                lead(transaction_index, 1) OVER w as next_tx_index,
                -- Previous transaction
                lag(transaction_hash, 1) OVER w as prev_tx_hash,
                lag(entity_id, 1) OVER w as prev_entity,
                lag(pool_address, 1) OVER w as prev_pool,
                lag(instruction_type, 1) OVER w as prev_instruction,
                lag(transaction_index, 1) OVER w as prev_tx_index,
                -- Sandwich detection
                lead(entity_id, 2) OVER w as next2_entity,
                lead(pool_address, 2) OVER w as next2_pool
            FROM mev_transactions
            WHERE timestamp >= now() - INTERVAL {lookback_hours} HOUR
            WINDOW w AS (PARTITION BY block_height ORDER BY transaction_index)
        )
        SELECT 
            entity_id,
            pool_address,
            block_height,
            transaction_index,
            -- Adjacency counts
            countIf(next_entity = entity_id AND next_pool = pool_address) as self_adjacent,
            countIf(prev_entity != entity_id AND next_entity = entity_id 
                   AND prev_pool = pool_address AND next_pool = pool_address) as sandwich_pattern,
            countIf(prev_entity != entity_id AND prev_pool = pool_address) as backrun_pattern,
            countIf(next_entity != entity_id AND next_pool = pool_address) as frontrun_pattern,
            -- Timing metrics
            avg(next_tx_index - transaction_index) as avg_index_gap,
            stddevPop(next_tx_index - transaction_index) as index_gap_std,
            -- Profit metrics
            sum(CASE WHEN next_entity = entity_id THEN amount_out - amount_in ELSE 0 END) as adjacent_profit,
            count() as transaction_count,
            groupArray(transaction_hash) as tx_hashes,
            groupArray(block_height) as blocks,
            groupArray(transaction_index) as indices
        FROM ordered_txs
        WHERE entity_id IS NOT NULL 
            AND pool_address IS NOT NULL
        GROUP BY entity_id, pool_address
        HAVING transaction_count >= {self.min_observations}
        """
        
        result = self.client.execute(query)
        
        columns = ['entity_id', 'pool_address', 'block_height', 'transaction_index',
                  'self_adjacent', 'sandwich_pattern', 'backrun_pattern', 
                  'frontrun_pattern', 'avg_index_gap', 'index_gap_std',
                  'adjacent_profit', 'transaction_count', 'tx_hashes', 
                  'blocks', 'indices']
        
        return pd.DataFrame(result, columns=columns)
    
    def bootstrap_null_distribution(self, 
                                   observations: List[int],
                                   n_iterations: int = 1000) -> np.ndarray:
        """
        Generate null distribution via bootstrap resampling
        """
        n = len(observations)
        null_stats = []
        
        for _ in range(n_iterations):
            # Randomly shuffle to break any ordering patterns
            shuffled = np.random.permutation(observations)
            
            # Calculate statistic under null hypothesis
            # (e.g., number of adjacencies in random ordering)
            adjacencies = 0
            for i in range(len(shuffled) - 1):
                if shuffled[i+1] == shuffled[i] + 1:  # Adjacent indices
                    adjacencies += 1
            
            null_stats.append(adjacencies)
        
        return np.array(null_stats)
    
    def calculate_temporal_clustering(self, 
                                     blocks: List[int],
                                     indices: List[int]) -> float:
        """
        Measure how clustered transactions are in time
        """
        if len(blocks) < 2:
            return 0.0
        
        # Group by block
        block_groups = defaultdict(list)
        for block, idx in zip(blocks, indices):
            block_groups[block].append(idx)
        
        # Calculate clustering metric
        clustering_scores = []
        
        for block, block_indices in block_groups.items():
            if len(block_indices) < 2:
                continue
            
            # Sort indices
            sorted_indices = sorted(block_indices)
            
            # Calculate gaps
            gaps = [sorted_indices[i+1] - sorted_indices[i] 
                   for i in range(len(sorted_indices)-1)]
            
            if gaps:
                # Low mean gap = high clustering
                mean_gap = np.mean(gaps)
                expected_gap = max(sorted_indices) / len(sorted_indices)
                
                if expected_gap > 0:
                    clustering = 1 - (mean_gap / expected_gap)
                    clustering_scores.append(max(0, clustering))
        
        return np.mean(clustering_scores) if clustering_scores else 0.0
    
    def detect_sandwich_patterns(self, 
                                df: pd.DataFrame) -> List[OrderingPattern]:
        """
        Detect sandwich attack patterns
        """
        patterns = []
        
        for idx, row in df.iterrows():
            if row['sandwich_pattern'] < 2:  # Need at least 2 sandwiches
                continue
            
            # Calculate expected frequency under random ordering
            n_txs = row['transaction_count']
            pool_popularity = n_txs / df['transaction_count'].sum()
            
            # Probability of sandwich by chance
            # (simplified - assumes uniform distribution)
            expected_sandwiches = n_txs * pool_popularity * 0.01  # 1% base rate
            
            # Bootstrap test
            null_dist = self.bootstrap_null_distribution(
                row['indices'],
                self.bootstrap_iterations
            )
            
            # Calculate p-value
            observed = row['sandwich_pattern']
            p_value = np.mean(null_dist >= observed)
            
            # Chi-square test
            if expected_sandwiches > 0:
                chi_square = ((observed - expected_sandwiches) ** 2) / expected_sandwiches
            else:
                chi_square = 0
            
            # Effect size (Cohen's h for proportions)
            p_observed = observed / n_txs
            p_expected = expected_sandwiches / n_txs
            effect_size = 2 * (np.arcsin(np.sqrt(p_observed)) - 
                              np.arcsin(np.sqrt(max(p_expected, 0.001))))
            
            # Temporal clustering
            temporal_clustering = self.calculate_temporal_clustering(
                row['blocks'],
                row['indices']
            )
            
            # Profit correlation
            profit_correlation = row['adjacent_profit'] / max(observed, 1)
            
            # Generate decision DNA
            dna_input = f"sandwich:{row['entity_id']}:{row['pool_address']}:{observed}:{p_value:.6f}"
            decision_dna = hashlib.sha256(dna_input.encode()).hexdigest()[:16]
            
            pattern = OrderingPattern(
                attacker_entity=row['entity_id'],
                pool_address=row['pool_address'],
                pattern_type="SANDWICH",
                observed_frequency=p_observed,
                expected_frequency=p_expected,
                adjacency_count=observed,
                total_observations=n_txs,
                bootstrap_p_value=p_value,
                chi_square_statistic=chi_square,
                effect_size=effect_size,
                temporal_clustering=temporal_clustering,
                profit_correlation=profit_correlation,
                decision_dna=decision_dna
            )
            
            if p_value < self.significance_level:
                patterns.append(pattern)
        
        return patterns
    
    def detect_backrun_patterns(self, 
                               df: pd.DataFrame) -> List[OrderingPattern]:
        """
        Detect backrunning patterns
        """
        patterns = []
        
        for idx, row in df.iterrows():
            if row['backrun_pattern'] < self.min_observations // 2:
                continue
            
            n_txs = row['transaction_count']
            
            # Expected backruns under random ordering
            expected_backruns = n_txs * 0.05  # 5% base rate assumption
            
            # Bootstrap test
            null_dist = self.bootstrap_null_distribution(
                row['indices'],
                self.bootstrap_iterations
            )
            
            observed = row['backrun_pattern']
            p_value = np.mean(null_dist >= observed)
            
            # Statistical tests
            if expected_backruns > 0:
                chi_square = ((observed - expected_backruns) ** 2) / expected_backruns
            else:
                chi_square = 0
            
            p_observed = observed / n_txs
            p_expected = expected_backruns / n_txs
            effect_size = 2 * (np.arcsin(np.sqrt(p_observed)) - 
                              np.arcsin(np.sqrt(max(p_expected, 0.001))))
            
            temporal_clustering = self.calculate_temporal_clustering(
                row['blocks'],
                row['indices']
            )
            
            # Generate decision DNA
            dna_input = f"backrun:{row['entity_id']}:{row['pool_address']}:{observed}:{p_value:.6f}"
            decision_dna = hashlib.sha256(dna_input.encode()).hexdigest()[:16]
            
            pattern = OrderingPattern(
                attacker_entity=row['entity_id'],
                pool_address=row['pool_address'],
                pattern_type="BACKRUN",
                observed_frequency=p_observed,
                expected_frequency=p_expected,
                adjacency_count=observed,
                total_observations=n_txs,
                bootstrap_p_value=p_value,
                chi_square_statistic=chi_square,
                effect_size=effect_size,
                temporal_clustering=temporal_clustering,
                profit_correlation=row['adjacent_profit'] / max(observed, 1),
                decision_dna=decision_dna
            )
            
            if p_value < self.significance_level:
                patterns.append(pattern)
        
        return patterns
    
    def detect_atomic_arbitrage(self, 
                               df: pd.DataFrame) -> List[OrderingPattern]:
        """
        Detect atomic arbitrage patterns (multiple pools in same tx)
        """
        patterns = []
        
        # Group by entity and look for multi-pool patterns
        entity_groups = df.groupby('entity_id')
        
        for entity_id, group in entity_groups:
            if len(group) < 2:  # Need multiple pools
                continue
            
            # Find pools frequently accessed together
            pools = group['pool_address'].tolist()
            pool_pairs = []
            
            for i in range(len(pools)):
                for j in range(i+1, len(pools)):
                    pool_pairs.append((pools[i], pools[j]))
            
            # Count co-occurrences
            pair_counts = Counter(pool_pairs)
            
            for (pool1, pool2), count in pair_counts.most_common(10):
                if count < 5:  # Minimum threshold
                    continue
                
                # Calculate expected co-occurrence
                p1 = len(df[df['pool_address'] == pool1]) / len(df)
                p2 = len(df[df['pool_address'] == pool2]) / len(df)
                expected = len(df) * p1 * p2
                
                # Statistical test
                if expected > 0:
                    chi_square = ((count - expected) ** 2) / expected
                    # Simplified p-value calculation
                    p_value = 1 - stats.chi2.cdf(chi_square, df=1)
                else:
                    chi_square = 0
                    p_value = 1.0
                
                if p_value < self.significance_level:
                    # Generate decision DNA
                    dna_input = f"atomic:{entity_id}:{pool1}:{pool2}:{count}"
                    decision_dna = hashlib.sha256(dna_input.encode()).hexdigest()[:16]
                    
                    pattern = OrderingPattern(
                        attacker_entity=entity_id,
                        pool_address=f"{pool1[:8]}...{pool2[:8]}",  # Combined identifier
                        pattern_type="ATOMIC_ARB",
                        observed_frequency=count / len(group),
                        expected_frequency=expected / len(df),
                        adjacency_count=count,
                        total_observations=len(group),
                        bootstrap_p_value=p_value,
                        chi_square_statistic=chi_square,
                        effect_size=np.sqrt(chi_square),
                        temporal_clustering=0,  # Not applicable for atomic
                        profit_correlation=0,  # Would need profit data
                        decision_dna=decision_dna
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def analyze_entity_quirks(self, 
                             lookback_hours: int = 72) -> List[QuirkProfile]:
        """
        Main analysis pipeline for ordering quirks
        """
        df = self.get_transaction_sequences(lookback_hours)
        
        if df.empty:
            return []
        
        # Detect different pattern types
        sandwich_patterns = self.detect_sandwich_patterns(df)
        backrun_patterns = self.detect_backrun_patterns(df)
        atomic_patterns = self.detect_atomic_arbitrage(df)
        
        # Group patterns by entity
        entity_patterns = defaultdict(lambda: {
            'sandwich': [],
            'backrun': [],
            'atomic': []
        })
        
        for pattern in sandwich_patterns:
            entity_patterns[pattern.attacker_entity]['sandwich'].append(pattern)
        
        for pattern in backrun_patterns:
            entity_patterns[pattern.attacker_entity]['backrun'].append(pattern)
        
        for pattern in atomic_patterns:
            entity_patterns[pattern.attacker_entity]['atomic'].append(pattern)
        
        # Create profiles
        profiles = []
        
        for entity_id, patterns_dict in entity_patterns.items():
            # Get entity statistics
            entity_df = df[df['entity_id'] == entity_id]
            
            if entity_df.empty:
                continue
            
            total_txs = entity_df['transaction_count'].sum()
            
            # Most targeted pools
            pool_counts = entity_df.groupby('pool_address')['transaction_count'].sum()
            most_targeted = list(pool_counts.nlargest(5).items())
            
            # Timing precision
            timing_std = entity_df['index_gap_std'].mean()
            
            # Calculate quirk score (weighted sum of pattern significances)
            quirk_score = 0
            pattern_count = 0
            
            for pattern_list in patterns_dict.values():
                for pattern in pattern_list:
                    # Weight by effect size and inverse p-value
                    weight = abs(pattern.effect_size) * (1 - pattern.bootstrap_p_value)
                    quirk_score += weight
                    pattern_count += 1
            
            if pattern_count > 0:
                quirk_score /= pattern_count
            
            profile = QuirkProfile(
                entity_id=entity_id,
                observation_period=timedelta(hours=lookback_hours),
                total_transactions=total_txs,
                sandwich_patterns=patterns_dict['sandwich'],
                backrun_patterns=patterns_dict['backrun'],
                atomic_patterns=patterns_dict['atomic'],
                quirk_score=quirk_score,
                most_targeted_pools=most_targeted,
                timing_precision_std=timing_std
            )
            
            profiles.append(profile)
        
        # Sort by quirk score
        profiles.sort(key=lambda x: x.quirk_score, reverse=True)
        
        return profiles
    
    def generate_report(self, profiles: List[QuirkProfile]) -> str:
        """
        Generate ordering quirks analysis report
        """
        report = []
        report.append("=" * 80)
        report.append("TRANSACTION ORDERING QUIRKS ANALYSIS")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_entities = len(profiles)
        entities_with_quirks = len([p for p in profiles if p.quirk_score > 0])
        
        all_patterns = []
        for profile in profiles:
            all_patterns.extend(profile.sandwich_patterns)
            all_patterns.extend(profile.backrun_patterns)
            all_patterns.extend(profile.atomic_patterns)
        
        report.append(f"Total entities analyzed: {total_entities}")
        report.append(f"Entities with significant quirks: {entities_with_quirks}")
        report.append(f"Total patterns detected: {len(all_patterns)}")
        report.append(f"Significance level: {self.significance_level}")
        report.append("")
        
        # Top entities by quirk score
        if profiles:
            report.append("TOP ENTITIES BY QUIRK SCORE:")
            report.append("-" * 40)
            
            for profile in profiles[:10]:
                if profile.quirk_score == 0:
                    continue
                
                report.append(f"\nEntity: {profile.entity_id}")
                report.append(f"  Quirk Score: {profile.quirk_score:.4f}")
                report.append(f"  Total Transactions: {profile.total_transactions:,}")
                report.append(f"  Timing Precision (std): {profile.timing_precision_std:.2f}")
                
                # Pattern breakdown
                n_sandwich = len(profile.sandwich_patterns)
                n_backrun = len(profile.backrun_patterns)
                n_atomic = len(profile.atomic_patterns)
                
                report.append(f"  Patterns: {n_sandwich} sandwich, {n_backrun} backrun, {n_atomic} atomic")
                
                # Most significant patterns
                all_entity_patterns = (
                    profile.sandwich_patterns + 
                    profile.backrun_patterns + 
                    profile.atomic_patterns
                )
                
                if all_entity_patterns:
                    # Sort by p-value
                    all_entity_patterns.sort(key=lambda x: x.bootstrap_p_value)
                    
                    report.append("  Most Significant Patterns:")
                    for pattern in all_entity_patterns[:3]:
                        report.append(f"    • {pattern.pattern_type} on {pattern.pool_address[:16]}...")
                        report.append(f"      Observed: {pattern.adjacency_count} "
                                    f"({pattern.observed_frequency:.2%})")
                        report.append(f"      Expected: {pattern.expected_frequency:.2%}")
                        report.append(f"      P-value: {pattern.bootstrap_p_value:.6f}")
                        report.append(f"      Effect Size: {pattern.effect_size:.3f}")
                        report.append(f"      DNA: {pattern.decision_dna}")
                
                # Most targeted pools
                if profile.most_targeted_pools:
                    report.append("  Most Targeted Pools:")
                    for pool, count in profile.most_targeted_pools[:3]:
                        report.append(f"    • {pool[:16]}...: {count} txs")
        
        # Pattern type distribution
        report.append("\n" + "=" * 40)
        report.append("PATTERN TYPE DISTRIBUTION:")
        report.append("-" * 40)
        
        pattern_counts = {'SANDWICH': 0, 'BACKRUN': 0, 'ATOMIC_ARB': 0}
        for pattern in all_patterns:
            pattern_counts[pattern.pattern_type] += 1
        
        for ptype, count in pattern_counts.items():
            report.append(f"{ptype}: {count} instances")
        
        # Statistical significance summary
        report.append("\n" + "=" * 40)
        report.append("STATISTICAL SIGNIFICANCE:")
        report.append("-" * 40)
        
        significant_patterns = [p for p in all_patterns if p.bootstrap_p_value < 0.001]
        report.append(f"Patterns with p < 0.001: {len(significant_patterns)}")
        
        highly_significant = [p for p in all_patterns if p.bootstrap_p_value < 0.0001]
        report.append(f"Patterns with p < 0.0001: {len(highly_significant)}")
        
        return "\n".join(report)
    
    def export_to_clickhouse(self, profiles: List[QuirkProfile]):
        """
        Export ordering quirks analysis to ClickHouse
        """
        if not profiles:
            return
        
        # Profile data
        profile_data = []
        for profile in profiles:
            profile_data.append({
                'timestamp': datetime.now(),
                'entity_id': profile.entity_id,
                'observation_hours': profile.observation_period.total_seconds() / 3600,
                'total_transactions': profile.total_transactions,
                'sandwich_count': len(profile.sandwich_patterns),
                'backrun_count': len(profile.backrun_patterns),
                'atomic_count': len(profile.atomic_patterns),
                'quirk_score': profile.quirk_score,
                'timing_precision_std': profile.timing_precision_std
            })
        
        # Pattern details
        pattern_data = []
        for profile in profiles:
            all_patterns = (
                profile.sandwich_patterns +
                profile.backrun_patterns +
                profile.atomic_patterns
            )
            
            for pattern in all_patterns:
                pattern_data.append({
                    'timestamp': datetime.now(),
                    'entity_id': pattern.attacker_entity,
                    'pool_address': pattern.pool_address,
                    'pattern_type': pattern.pattern_type,
                    'observed_frequency': pattern.observed_frequency,
                    'expected_frequency': pattern.expected_frequency,
                    'adjacency_count': pattern.adjacency_count,
                    'total_observations': pattern.total_observations,
                    'bootstrap_p_value': pattern.bootstrap_p_value,
                    'chi_square_statistic': pattern.chi_square_statistic,
                    'effect_size': pattern.effect_size,
                    'temporal_clustering': pattern.temporal_clustering,
                    'profit_correlation': pattern.profit_correlation,
                    'decision_dna': pattern.decision_dna
                })
        
        # Create tables
        create_profile_table = """
        CREATE TABLE IF NOT EXISTS ordering_quirk_profiles (
            timestamp DateTime,
            entity_id String,
            observation_hours Float32,
            total_transactions UInt32,
            sandwich_count UInt32,
            backrun_count UInt32,
            atomic_count UInt32,
            quirk_score Float32,
            timing_precision_std Float32
        ) ENGINE = MergeTree()
        ORDER BY (timestamp, entity_id)
        TTL timestamp + INTERVAL 90 DAY
        """
        
        create_pattern_table = """
        CREATE TABLE IF NOT EXISTS ordering_patterns (
            timestamp DateTime,
            entity_id String,
            pool_address String,
            pattern_type String,
            observed_frequency Float32,
            expected_frequency Float32,
            adjacency_count UInt32,
            total_observations UInt32,
            bootstrap_p_value Float64,
            chi_square_statistic Float32,
            effect_size Float32,
            temporal_clustering Float32,
            profit_correlation Float32,
            decision_dna String
        ) ENGINE = MergeTree()
        ORDER BY (timestamp, entity_id, pattern_type)
        TTL timestamp + INTERVAL 90 DAY
        """
        
        self.client.execute(create_profile_table)
        self.client.execute(create_pattern_table)
        
        # Insert data
        if profile_data:
            self.client.execute("INSERT INTO ordering_quirk_profiles VALUES", profile_data)
        if pattern_data:
            self.client.execute("INSERT INTO ordering_patterns VALUES", pattern_data)


def main():
    """
    Main execution
    """
    analyzer = OrderingQuirksAnalyzer()
    
    print("Analyzing transaction ordering quirks...")
    profiles = analyzer.analyze_entity_quirks(lookback_hours=72)
    
    # Generate report
    report = analyzer.generate_report(profiles)
    print(report)
    
    # Export to ClickHouse
    analyzer.export_to_clickhouse(profiles)
    print(f"\nResults exported to ClickHouse")


if __name__ == "__main__":
    main()