#!/usr/bin/env python3
"""
Bundle Landing Rate Anomaly Detection Service
Statistical testing for MEV bundle success rates with Benjamini-Hochberg correction
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import clickhouse_driver
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json

warnings.filterwarnings('ignore')


@dataclass
class LandingRateMetrics:
    """Metrics for bundle landing rate analysis"""
    entity_id: str
    observation_window: timedelta
    total_bundles: int
    landed_bundles: int
    landing_rate: float
    peer_mean_rate: float
    peer_std_rate: float
    t_statistic: float
    p_value: float
    adjusted_p_value: float
    is_anomalous: bool
    confidence_interval: Tuple[float, float]
    effect_size: float
    decision_dna: str


class LandingRatioAnalyzer:
    """
    Detects anomalous bundle landing rates using statistical hypothesis testing
    """
    
    def __init__(self, clickhouse_url: str = "http://localhost:8123"):
        self.client = clickhouse_driver.Client(host=clickhouse_url.replace("http://", "").split(":")[0])
        self.alpha = 0.05  # Significance level
        self.min_sample_size = 30  # Minimum bundles for reliable statistics
        
    def get_landing_rates(self, 
                          lookback_hours: int = 24,
                          entity_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch bundle landing rates from ClickHouse
        """
        query = f"""
        SELECT 
            entity_id,
            COUNT(*) as total_bundles,
            SUM(CASE WHEN landed = 1 THEN 1 ELSE 0 END) as landed_bundles,
            landed_bundles / total_bundles as landing_rate,
            groupArray(bundle_hash) as bundle_hashes,
            groupArray(block_height) as block_heights,
            groupArray(tip_lamports) as tips,
            avg(tip_lamports) as avg_tip,
            stddevPop(tip_lamports) as std_tip,
            min(timestamp) as first_seen,
            max(timestamp) as last_seen
        FROM mev_bundles
        WHERE timestamp >= now() - INTERVAL {lookback_hours} HOUR
            {'AND entity_id = %(entity_id)s' if entity_filter else ''}
        GROUP BY entity_id
        HAVING total_bundles >= {self.min_sample_size}
        ORDER BY landing_rate DESC
        """
        
        params = {'entity_id': entity_filter} if entity_filter else {}
        result = self.client.execute(query, params)
        
        columns = ['entity_id', 'total_bundles', 'landed_bundles', 'landing_rate',
                  'bundle_hashes', 'block_heights', 'tips', 'avg_tip', 'std_tip',
                  'first_seen', 'last_seen']
        
        return pd.DataFrame(result, columns=columns)
    
    def perform_t_test(self, 
                       entity_rate: float,
                       entity_n: int,
                       peer_rates: np.ndarray) -> Dict:
        """
        Perform one-sample t-test comparing entity to peer group
        """
        # Calculate peer statistics
        peer_mean = np.mean(peer_rates)
        peer_std = np.std(peer_rates, ddof=1)
        
        # Calculate standard error
        se = peer_std / np.sqrt(entity_n)
        
        # T-statistic
        t_stat = (entity_rate - peer_mean) / se if se > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=entity_n-1))
        
        # Effect size (Cohen's d)
        effect_size = (entity_rate - peer_mean) / peer_std if peer_std > 0 else 0
        
        # Confidence interval
        margin = stats.t.ppf(1 - self.alpha/2, entity_n-1) * se
        ci_lower = entity_rate - margin
        ci_upper = entity_rate + margin
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper),
            'peer_mean': peer_mean,
            'peer_std': peer_std
        }
    
    def benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """
        Apply Benjamini-Hochberg correction for multiple hypothesis testing
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        # Calculate adjusted p-values
        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[i] = min(1.0, sorted_p_values[i] * n / (i + 1))
        
        # Enforce monotonicity
        for i in range(n-2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i+1])
        
        # Restore original order
        result = np.zeros(n)
        for i, idx in enumerate(sorted_indices):
            result[idx] = adjusted[i]
        
        return result.tolist()
    
    def detect_anomalies(self, 
                        lookback_hours: int = 24,
                        target_entity: Optional[str] = 'B91') -> List[LandingRateMetrics]:
        """
        Main detection pipeline with statistical testing
        """
        # Fetch data
        df = self.get_landing_rates(lookback_hours)
        
        if df.empty:
            return []
        
        results = []
        p_values = []
        
        # Analyze each entity
        for idx, row in df.iterrows():
            entity_id = row['entity_id']
            
            # Get peer group (exclude current entity)
            peer_rates = df[df['entity_id'] != entity_id]['landing_rate'].values
            
            if len(peer_rates) < 3:  # Need at least 3 peers for comparison
                continue
            
            # Perform statistical test
            test_results = self.perform_t_test(
                row['landing_rate'],
                row['total_bundles'],
                peer_rates
            )
            
            # Generate decision DNA
            dna_input = f"{entity_id}:{row['landing_rate']:.4f}:{test_results['t_statistic']:.4f}"
            decision_dna = hashlib.sha256(dna_input.encode()).hexdigest()[:16]
            
            # Create metrics object
            metrics = LandingRateMetrics(
                entity_id=entity_id,
                observation_window=timedelta(hours=lookback_hours),
                total_bundles=row['total_bundles'],
                landed_bundles=row['landed_bundles'],
                landing_rate=row['landing_rate'],
                peer_mean_rate=test_results['peer_mean'],
                peer_std_rate=test_results['peer_std'],
                t_statistic=test_results['t_statistic'],
                p_value=test_results['p_value'],
                adjusted_p_value=0,  # Will be filled after correction
                is_anomalous=False,  # Will be determined after correction
                confidence_interval=test_results['confidence_interval'],
                effect_size=test_results['effect_size'],
                decision_dna=decision_dna
            )
            
            results.append(metrics)
            p_values.append(test_results['p_value'])
        
        # Apply multiple testing correction
        if p_values:
            adjusted_p_values = self.benjamini_hochberg_correction(p_values)
            
            for i, metrics in enumerate(results):
                metrics.adjusted_p_value = adjusted_p_values[i]
                metrics.is_anomalous = adjusted_p_values[i] < self.alpha
        
        # Sort by effect size (most anomalous first)
        results.sort(key=lambda x: abs(x.effect_size), reverse=True)
        
        return results
    
    def generate_report(self, results: List[LandingRateMetrics]) -> str:
        """
        Generate human-readable report of findings
        """
        report = []
        report.append("=" * 80)
        report.append("BUNDLE LANDING RATE ANOMALY DETECTION REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        anomalous = [r for r in results if r.is_anomalous]
        report.append(f"Total entities analyzed: {len(results)}")
        report.append(f"Anomalous entities detected: {len(anomalous)}")
        report.append(f"Significance level (alpha): {self.alpha}")
        report.append("")
        
        # Detailed findings
        if anomalous:
            report.append("ANOMALOUS ENTITIES:")
            report.append("-" * 40)
            
            for metrics in anomalous:
                report.append(f"\nEntity: {metrics.entity_id}")
                report.append(f"  Landing Rate: {metrics.landing_rate:.2%} "
                            f"(Peer Mean: {metrics.peer_mean_rate:.2%})")
                report.append(f"  Sample Size: {metrics.total_bundles} bundles")
                report.append(f"  T-Statistic: {metrics.t_statistic:.4f}")
                report.append(f"  P-Value: {metrics.p_value:.6f} "
                            f"(Adjusted: {metrics.adjusted_p_value:.6f})")
                report.append(f"  Effect Size (Cohen's d): {metrics.effect_size:.4f}")
                report.append(f"  95% CI: [{metrics.confidence_interval[0]:.2%}, "
                            f"{metrics.confidence_interval[1]:.2%}]")
                report.append(f"  Decision DNA: {metrics.decision_dna}")
                
                # Interpretation
                if metrics.effect_size > 0.8:
                    report.append("  → LARGE effect size - highly significant difference")
                elif metrics.effect_size > 0.5:
                    report.append("  → MEDIUM effect size - moderate difference")
                elif metrics.effect_size > 0.2:
                    report.append("  → SMALL effect size - minor difference")
        
        # Top performers (even if not statistically anomalous)
        report.append("\n" + "=" * 40)
        report.append("TOP PERFORMERS BY LANDING RATE:")
        report.append("-" * 40)
        
        for metrics in results[:5]:
            status = "⚠️ ANOMALOUS" if metrics.is_anomalous else "✓ Normal"
            report.append(f"{metrics.entity_id}: {metrics.landing_rate:.2%} "
                        f"({metrics.total_bundles} bundles) {status}")
        
        return "\n".join(report)
    
    def export_to_clickhouse(self, results: List[LandingRateMetrics]):
        """
        Export analysis results back to ClickHouse for dashboarding
        """
        if not results:
            return
        
        # Prepare data for insertion
        data = []
        for metrics in results:
            data.append({
                'timestamp': datetime.now(),
                'entity_id': metrics.entity_id,
                'observation_window_hours': metrics.observation_window.total_seconds() / 3600,
                'total_bundles': metrics.total_bundles,
                'landed_bundles': metrics.landed_bundles,
                'landing_rate': metrics.landing_rate,
                'peer_mean_rate': metrics.peer_mean_rate,
                'peer_std_rate': metrics.peer_std_rate,
                't_statistic': metrics.t_statistic,
                'p_value': metrics.p_value,
                'adjusted_p_value': metrics.adjusted_p_value,
                'is_anomalous': 1 if metrics.is_anomalous else 0,
                'effect_size': metrics.effect_size,
                'ci_lower': metrics.confidence_interval[0],
                'ci_upper': metrics.confidence_interval[1],
                'decision_dna': metrics.decision_dna
            })
        
        # Create table if not exists
        create_table = """
        CREATE TABLE IF NOT EXISTS landing_rate_analysis (
            timestamp DateTime,
            entity_id String,
            observation_window_hours Float32,
            total_bundles UInt32,
            landed_bundles UInt32,
            landing_rate Float32,
            peer_mean_rate Float32,
            peer_std_rate Float32,
            t_statistic Float32,
            p_value Float64,
            adjusted_p_value Float64,
            is_anomalous UInt8,
            effect_size Float32,
            ci_lower Float32,
            ci_upper Float32,
            decision_dna String
        ) ENGINE = MergeTree()
        ORDER BY (timestamp, entity_id)
        TTL timestamp + INTERVAL 90 DAY
        """
        
        self.client.execute(create_table)
        
        # Insert data
        self.client.execute(
            "INSERT INTO landing_rate_analysis VALUES",
            data
        )


def main():
    """
    Main execution for standalone testing
    """
    analyzer = LandingRatioAnalyzer()
    
    # Run detection
    print("Running bundle landing rate anomaly detection...")
    results = analyzer.detect_anomalies(lookback_hours=24)
    
    # Generate and print report
    report = analyzer.generate_report(results)
    print(report)
    
    # Export to ClickHouse
    analyzer.export_to_clickhouse(results)
    print(f"\nResults exported to ClickHouse")
    
    # Special focus on B91 if present
    b91_results = [r for r in results if r.entity_id == 'B91']
    if b91_results:
        print("\n" + "=" * 40)
        print("B91 SPECIFIC ANALYSIS:")
        print("-" * 40)
        b91 = b91_results[0]
        print(f"Landing Rate: {b91.landing_rate:.2%}")
        print(f"Statistical Significance: {'YES' if b91.is_anomalous else 'NO'}")
        print(f"Performance vs Peers: {'+' if b91.effect_size > 0 else ''}{b91.effect_size:.2f} sigma")


if __name__ == "__main__":
    main()