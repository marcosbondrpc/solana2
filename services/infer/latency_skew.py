#!/usr/bin/env python3
"""
Latency Distribution Analysis Service
Detects ultra-optimized MEV operators through latency variance analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import clickhouse_driver
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import warnings

warnings.filterwarnings('ignore')


@dataclass
class LatencyProfile:
    """Latency distribution metrics for an entity"""
    entity_id: str
    observation_window: timedelta
    sample_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    std_latency_ms: float
    skewness: float
    kurtosis: float
    p95_p50_spread: float
    levene_statistic: float
    levene_p_value: float
    is_ultra_optimized: bool
    percentile_ratios: Dict[str, float]
    decision_dna: str


class LatencySkewAnalyzer:
    """
    Analyzes latency distributions to identify ultra-optimized MEV operators
    """
    
    def __init__(self, clickhouse_url: str = "http://localhost:8123"):
        self.client = clickhouse_driver.Client(host=clickhouse_url.replace("http://", "").split(":")[0])
        self.ultra_tight_threshold = 50  # ms - P95-P50 spread indicating optimization
        self.min_samples = 100  # Minimum samples for reliable statistics
        
    def get_latency_distributions(self, 
                                 lookback_hours: int = 24) -> pd.DataFrame:
        """
        Fetch latency data from ClickHouse
        """
        query = f"""
        SELECT 
            entity_id,
            groupArray(decision_latency_ms) as latencies,
            count() as sample_count,
            quantile(0.50)(decision_latency_ms) as p50,
            quantile(0.75)(decision_latency_ms) as p75,
            quantile(0.90)(decision_latency_ms) as p90,
            quantile(0.95)(decision_latency_ms) as p95,
            quantile(0.99)(decision_latency_ms) as p99,
            quantile(0.999)(decision_latency_ms) as p999,
            avg(decision_latency_ms) as mean_latency,
            stddevPop(decision_latency_ms) as std_latency,
            min(decision_latency_ms) as min_latency,
            max(decision_latency_ms) as max_latency,
            -- Network path metrics
            avg(rpc_latency_ms) as avg_rpc_latency,
            avg(bundle_send_latency_ms) as avg_bundle_latency,
            -- Timing precision
            stddevPop(block_production_offset_ms) as timing_precision,
            -- Success correlation
            corr(decision_latency_ms, toFloat32(landed)) as latency_success_corr
        FROM mev_decisions
        WHERE timestamp >= now() - INTERVAL {lookback_hours} HOUR
            AND decision_latency_ms > 0
            AND decision_latency_ms < 10000  -- Filter outliers
        GROUP BY entity_id
        HAVING sample_count >= {self.min_samples}
        ORDER BY p95 - p50 ASC  -- Tightest distributions first
        """
        
        result = self.client.execute(query)
        
        columns = ['entity_id', 'latencies', 'sample_count', 'p50', 'p75', 'p90', 
                  'p95', 'p99', 'p999', 'mean_latency', 'std_latency',
                  'min_latency', 'max_latency', 'avg_rpc_latency', 
                  'avg_bundle_latency', 'timing_precision', 'latency_success_corr']
        
        return pd.DataFrame(result, columns=columns)
    
    def calculate_distribution_metrics(self, latencies: List[float]) -> Dict:
        """
        Calculate advanced distribution metrics
        """
        arr = np.array(latencies)
        
        # Basic statistics
        metrics = {
            'mean': np.mean(arr),
            'std': np.std(arr, ddof=1),
            'skewness': stats.skew(arr),
            'kurtosis': stats.kurtosis(arr),
            'iqr': np.percentile(arr, 75) - np.percentile(arr, 25),
            'mad': np.median(np.abs(arr - np.median(arr))),  # Median absolute deviation
        }
        
        # Percentile ratios (indicators of distribution shape)
        p50 = np.percentile(arr, 50)
        p90 = np.percentile(arr, 90)
        p95 = np.percentile(arr, 95)
        p99 = np.percentile(arr, 99)
        
        metrics['percentile_ratios'] = {
            'p90_p50': p90 / p50 if p50 > 0 else float('inf'),
            'p95_p50': p95 / p50 if p50 > 0 else float('inf'),
            'p99_p50': p99 / p50 if p50 > 0 else float('inf'),
            'p99_p95': p99 / p95 if p95 > 0 else float('inf'),
        }
        
        # Tail behavior
        metrics['tail_weight'] = len([x for x in arr if x > p95]) / len(arr)
        
        # Bimodality test (Hartigan's dip test approximation)
        sorted_arr = np.sort(arr)
        n = len(sorted_arr)
        if n > 10:
            # Simple bimodality indicator based on gap analysis
            gaps = np.diff(sorted_arr)
            max_gap_idx = np.argmax(gaps)
            if max_gap_idx > 0 and max_gap_idx < n-2:
                gap_ratio = gaps[max_gap_idx] / np.median(gaps)
                metrics['bimodality_indicator'] = gap_ratio
            else:
                metrics['bimodality_indicator'] = 1.0
        else:
            metrics['bimodality_indicator'] = 1.0
        
        return metrics
    
    def levene_test_vs_population(self, 
                                  entity_latencies: List[float],
                                  population_latencies: List[List[float]]) -> Tuple[float, float]:
        """
        Levene's test for variance homogeneity
        Tests if entity has significantly different variance from population
        """
        # Combine all population samples
        all_population = []
        for latencies in population_latencies:
            all_population.extend(latencies[:100])  # Sample to avoid memory issues
        
        # Perform Levene's test
        if len(all_population) > 0 and len(entity_latencies) > 0:
            statistic, p_value = stats.levene(entity_latencies, all_population)
            return statistic, p_value
        else:
            return 0.0, 1.0
    
    def detect_ultra_optimized(self, 
                              lookback_hours: int = 24) -> List[LatencyProfile]:
        """
        Main detection pipeline for ultra-optimized operators
        """
        # Fetch data
        df = self.get_latency_distributions(lookback_hours)
        
        if df.empty:
            return []
        
        results = []
        
        # Collect all latencies for population comparison
        population_latencies = df['latencies'].tolist()
        
        for idx, row in df.iterrows():
            entity_id = row['entity_id']
            latencies = row['latencies']
            
            # Calculate distribution metrics
            dist_metrics = self.calculate_distribution_metrics(latencies)
            
            # Calculate P95-P50 spread (key indicator)
            p95_p50_spread = row['p95'] - row['p50']
            
            # Levene's test against population
            other_latencies = [l for i, l in enumerate(population_latencies) if i != idx]
            levene_stat, levene_p = self.levene_test_vs_population(latencies, other_latencies)
            
            # Determine if ultra-optimized
            is_ultra = (
                p95_p50_spread < self.ultra_tight_threshold and  # Tight distribution
                row['p99'] < 100 and  # Sub-100ms P99
                row['std_latency'] < 20 and  # Low variance
                dist_metrics['skewness'] < 1.0 and  # Not heavily right-skewed
                levene_p < 0.05  # Significantly different variance
            )
            
            # Special markers for extreme optimization
            if row['p99'] < 20:  # Sub-20ms P99 is elite
                is_ultra = True
            
            # Generate decision DNA
            dna_input = f"{entity_id}:{p95_p50_spread:.2f}:{row['p99']:.2f}:{levene_stat:.4f}"
            decision_dna = hashlib.sha256(dna_input.encode()).hexdigest()[:16]
            
            # Create profile
            profile = LatencyProfile(
                entity_id=entity_id,
                observation_window=timedelta(hours=lookback_hours),
                sample_count=row['sample_count'],
                p50_latency_ms=row['p50'],
                p95_latency_ms=row['p95'],
                p99_latency_ms=row['p99'],
                mean_latency_ms=row['mean_latency'],
                std_latency_ms=row['std_latency'],
                skewness=dist_metrics['skewness'],
                kurtosis=dist_metrics['kurtosis'],
                p95_p50_spread=p95_p50_spread,
                levene_statistic=levene_stat,
                levene_p_value=levene_p,
                is_ultra_optimized=is_ultra,
                percentile_ratios=dist_metrics['percentile_ratios'],
                decision_dna=decision_dna
            )
            
            results.append(profile)
        
        # Sort by optimization level (tightest distributions first)
        results.sort(key=lambda x: x.p95_p50_spread)
        
        return results
    
    def analyze_optimization_techniques(self, profile: LatencyProfile) -> List[str]:
        """
        Infer likely optimization techniques based on latency profile
        """
        techniques = []
        
        # Sub-10ms P50 suggests co-location
        if profile.p50_latency_ms < 10:
            techniques.append("CO-LOCATION: Likely running in same datacenter as validators")
        
        # Ultra-tight distribution suggests custom networking
        if profile.p95_p50_spread < 20:
            techniques.append("CUSTOM_NETWORKING: DPDK/kernel-bypass likely in use")
        
        # Very low P99 suggests dedicated infrastructure
        if profile.p99_latency_ms < 50:
            techniques.append("DEDICATED_INFRA: Private nodes and dedicated RPC")
        
        # Consistent sub-slot timing
        if profile.std_latency_ms < 10:
            techniques.append("SLOT_TIMING: Precise slot boundary prediction")
        
        # Bimodal distribution might indicate hedging
        if profile.kurtosis < -0.5:
            techniques.append("HEDGED_SENDING: Multiple path execution detected")
        
        # Heavy right tail suggests retry logic
        if profile.skewness > 2.0:
            techniques.append("RETRY_LOGIC: Aggressive retry on failure")
        
        return techniques
    
    def generate_report(self, results: List[LatencyProfile]) -> str:
        """
        Generate comprehensive latency analysis report
        """
        report = []
        report.append("=" * 80)
        report.append("LATENCY DISTRIBUTION ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        ultra_optimized = [r for r in results if r.is_ultra_optimized]
        report.append(f"Total entities analyzed: {len(results)}")
        report.append(f"Ultra-optimized operators detected: {len(ultra_optimized)}")
        report.append("")
        
        # Ultra-optimized operators
        if ultra_optimized:
            report.append("ULTRA-OPTIMIZED OPERATORS:")
            report.append("-" * 40)
            
            for profile in ultra_optimized:
                report.append(f"\nEntity: {profile.entity_id}")
                report.append(f"  P50: {profile.p50_latency_ms:.1f}ms")
                report.append(f"  P95: {profile.p95_latency_ms:.1f}ms")
                report.append(f"  P99: {profile.p99_latency_ms:.1f}ms")
                report.append(f"  P95-P50 Spread: {profile.p95_p50_spread:.1f}ms")
                report.append(f"  Std Dev: {profile.std_latency_ms:.1f}ms")
                report.append(f"  Levene's Test: p={profile.levene_p_value:.6f}")
                report.append(f"  Decision DNA: {profile.decision_dna}")
                
                # Inferred techniques
                techniques = self.analyze_optimization_techniques(profile)
                if techniques:
                    report.append("  Likely Optimizations:")
                    for tech in techniques:
                        report.append(f"    • {tech}")
        
        # Latency leaderboard
        report.append("\n" + "=" * 40)
        report.append("LATENCY LEADERBOARD (P99):")
        report.append("-" * 40)
        
        sorted_by_p99 = sorted(results, key=lambda x: x.p99_latency_ms)[:10]
        for i, profile in enumerate(sorted_by_p99, 1):
            marker = "🏆" if profile.is_ultra_optimized else "  "
            report.append(f"{i:2}. {marker} {profile.entity_id}: "
                        f"P99={profile.p99_latency_ms:.1f}ms, "
                        f"P50={profile.p50_latency_ms:.1f}ms, "
                        f"Spread={profile.p95_p50_spread:.1f}ms")
        
        # Distribution tightness ranking
        report.append("\n" + "=" * 40)
        report.append("TIGHTEST DISTRIBUTIONS (P95-P50 Spread):")
        report.append("-" * 40)
        
        for i, profile in enumerate(results[:10], 1):
            report.append(f"{i:2}. {profile.entity_id}: "
                        f"Spread={profile.p95_p50_spread:.1f}ms "
                        f"(σ={profile.std_latency_ms:.1f}ms)")
        
        return "\n".join(report)
    
    def export_to_clickhouse(self, results: List[LatencyProfile]):
        """
        Export analysis results to ClickHouse
        """
        if not results:
            return
        
        # Prepare data
        data = []
        for profile in results:
            data.append({
                'timestamp': datetime.now(),
                'entity_id': profile.entity_id,
                'observation_window_hours': profile.observation_window.total_seconds() / 3600,
                'sample_count': profile.sample_count,
                'p50_latency_ms': profile.p50_latency_ms,
                'p95_latency_ms': profile.p95_latency_ms,
                'p99_latency_ms': profile.p99_latency_ms,
                'mean_latency_ms': profile.mean_latency_ms,
                'std_latency_ms': profile.std_latency_ms,
                'skewness': profile.skewness,
                'kurtosis': profile.kurtosis,
                'p95_p50_spread': profile.p95_p50_spread,
                'levene_statistic': profile.levene_statistic,
                'levene_p_value': profile.levene_p_value,
                'is_ultra_optimized': 1 if profile.is_ultra_optimized else 0,
                'decision_dna': profile.decision_dna
            })
        
        # Create table
        create_table = """
        CREATE TABLE IF NOT EXISTS latency_analysis (
            timestamp DateTime,
            entity_id String,
            observation_window_hours Float32,
            sample_count UInt32,
            p50_latency_ms Float32,
            p95_latency_ms Float32,
            p99_latency_ms Float32,
            mean_latency_ms Float32,
            std_latency_ms Float32,
            skewness Float32,
            kurtosis Float32,
            p95_p50_spread Float32,
            levene_statistic Float32,
            levene_p_value Float64,
            is_ultra_optimized UInt8,
            decision_dna String
        ) ENGINE = MergeTree()
        ORDER BY (timestamp, entity_id)
        TTL timestamp + INTERVAL 90 DAY
        """
        
        self.client.execute(create_table)
        
        # Insert data
        self.client.execute(
            "INSERT INTO latency_analysis VALUES",
            data
        )


def main():
    """
    Main execution
    """
    analyzer = LatencySkewAnalyzer()
    
    print("Analyzing latency distributions...")
    results = analyzer.detect_ultra_optimized(lookback_hours=24)
    
    # Generate report
    report = analyzer.generate_report(results)
    print(report)
    
    # Export to ClickHouse
    analyzer.export_to_clickhouse(results)
    print(f"\nResults exported to ClickHouse")


if __name__ == "__main__":
    main()