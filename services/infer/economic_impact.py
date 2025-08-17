#!/usr/bin/env python3
"""
Economic Impact Measurement Service
Calculates SOL extraction, victim slippage, and congestion externalities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import clickhouse_driver
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EconomicMetrics:
    """Economic impact metrics for an entity"""
    entity_id: str
    measurement_period: timedelta
    # Extraction metrics
    sol_extracted_7d: float
    sol_extracted_30d: float
    sol_extracted_90d: float
    gross_profit_sol: float
    net_profit_sol: float
    # Victim metrics
    total_victims: int
    unique_victims: int
    avg_victim_loss_sol: float
    median_victim_loss_sol: float
    max_victim_loss_sol: float
    victim_loss_distribution: Dict[str, int]
    # Cost metrics
    total_tips_paid: float
    total_fees_paid: float
    avg_tip_per_bundle: float
    tip_efficiency_ratio: float
    # Congestion metrics
    blocks_congested: int
    congestion_ratio: float
    avg_block_utilization: float
    compute_units_consumed: int
    # Efficiency metrics
    profit_per_sandwich: float
    success_rate: float
    roi_percentage: float
    # Market impact
    market_share: float
    dominance_score: float
    decision_dna: str


@dataclass
class VictimProfile:
    """Profile of MEV victims"""
    victim_address: str
    total_losses_sol: float
    sandwich_count: int
    avg_loss_per_sandwich: float
    max_single_loss: float
    first_victimized: datetime
    last_victimized: datetime
    attacker_entities: List[str]
    pools_affected: List[str]


class EconomicImpactAnalyzer:
    """
    Measures economic impact of MEV activities
    """
    
    def __init__(self, clickhouse_url: str = "http://localhost:8123"):
        self.client = clickhouse_driver.Client(host=clickhouse_url.replace("http://", "").split(":")[0])
        # Reference: B91 ~7,800 SOL/month gross
        self.b91_monthly_reference = 7800
        
    def get_extraction_metrics(self, 
                              entity_id: Optional[str] = None,
                              lookback_days: int = 90) -> pd.DataFrame:
        """
        Calculate SOL extraction metrics
        """
        entity_filter = f"AND entity_id = '{entity_id}'" if entity_id else ""
        
        query = f"""
        WITH extraction_data AS (
            SELECT 
                entity_id,
                toDate(timestamp) as date,
                -- Gross extraction
                sum(profit_sol) as daily_gross_sol,
                -- Costs
                sum(tip_lamports) / 1e9 as daily_tips_sol,
                sum(priority_fee_lamports) / 1e9 as daily_fees_sol,
                -- Transaction counts
                count() as daily_transactions,
                countIf(is_sandwich = 1) as daily_sandwiches,
                countIf(is_backrun = 1) as daily_backruns,
                countIf(is_atomic_arb = 1) as daily_atomic_arbs,
                -- Success metrics
                countIf(landed = 1) as daily_landed,
                -- Victim metrics
                countIf(victim_address IS NOT NULL) as daily_victim_txs,
                count(DISTINCT victim_address) as daily_unique_victims,
                sum(victim_loss_sol) as daily_victim_losses,
                -- Bundle metrics
                count(DISTINCT bundle_hash) as daily_bundles,
                avg(bundle_size) as avg_bundle_size
            FROM mev_transactions
            WHERE timestamp >= now() - INTERVAL {lookback_days} DAY
                {entity_filter}
            GROUP BY entity_id, date
        )
        SELECT 
            entity_id,
            -- 7-day metrics
            sum(daily_gross_sol) OVER (
                PARTITION BY entity_id 
                ORDER BY date 
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as sol_7d,
            -- 30-day metrics
            sum(daily_gross_sol) OVER (
                PARTITION BY entity_id 
                ORDER BY date 
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            ) as sol_30d,
            -- 90-day total
            sum(daily_gross_sol) as sol_90d,
            -- Costs
            sum(daily_tips_sol) as tips_90d,
            sum(daily_fees_sol) as fees_90d,
            -- Counts
            sum(daily_transactions) as txs_90d,
            sum(daily_sandwiches) as sandwiches_90d,
            sum(daily_landed) as landed_90d,
            sum(daily_bundles) as bundles_90d,
            -- Victims
            sum(daily_victim_txs) as victim_txs_90d,
            max(daily_unique_victims) as max_daily_victims,
            sum(daily_victim_losses) as victim_losses_90d,
            -- Latest metrics
            last_value(daily_gross_sol) as latest_daily_sol,
            last_value(date) as latest_date
        FROM extraction_data
        GROUP BY entity_id
        ORDER BY sol_90d DESC
        """
        
        result = self.client.execute(query)
        
        columns = ['entity_id', 'sol_7d', 'sol_30d', 'sol_90d', 'tips_90d', 
                  'fees_90d', 'txs_90d', 'sandwiches_90d', 'landed_90d',
                  'bundles_90d', 'victim_txs_90d', 'max_daily_victims',
                  'victim_losses_90d', 'latest_daily_sol', 'latest_date']
        
        return pd.DataFrame(result, columns=columns)
    
    def get_victim_analysis(self, 
                           lookback_days: int = 30) -> pd.DataFrame:
        """
        Analyze victim impact
        """
        query = f"""
        SELECT 
            victim_address,
            count() as sandwich_count,
            sum(victim_loss_sol) as total_loss_sol,
            avg(victim_loss_sol) as avg_loss_sol,
            max(victim_loss_sol) as max_loss_sol,
            min(victim_loss_sol) as min_loss_sol,
            stddevPop(victim_loss_sol) as loss_std,
            -- Timing
            min(timestamp) as first_victimized,
            max(timestamp) as last_victimized,
            -- Attackers
            groupArray(DISTINCT entity_id) as attacker_entities,
            count(DISTINCT entity_id) as unique_attackers,
            -- Pools
            groupArray(DISTINCT pool_address) as pools_affected,
            count(DISTINCT pool_address) as unique_pools,
            -- Slippage analysis
            avg(slippage_bps) as avg_slippage_bps,
            max(slippage_bps) as max_slippage_bps
        FROM mev_transactions
        WHERE victim_address IS NOT NULL
            AND timestamp >= now() - INTERVAL {lookback_days} DAY
        GROUP BY victim_address
        HAVING sandwich_count >= 2  -- Multiple victimizations
        ORDER BY total_loss_sol DESC
        LIMIT 10000
        """
        
        result = self.client.execute(query)
        
        columns = ['victim_address', 'sandwich_count', 'total_loss_sol',
                  'avg_loss_sol', 'max_loss_sol', 'min_loss_sol', 'loss_std',
                  'first_victimized', 'last_victimized', 'attacker_entities',
                  'unique_attackers', 'pools_affected', 'unique_pools',
                  'avg_slippage_bps', 'max_slippage_bps']
        
        return pd.DataFrame(result, columns=columns)
    
    def calculate_congestion_impact(self, 
                                   entity_id: str,
                                   lookback_hours: int = 168) -> Dict:
        """
        Calculate network congestion impact
        """
        query = f"""
        SELECT 
            -- Block congestion
            count(DISTINCT block_height) as blocks_touched,
            countIf(block_utilization > 0.9) as congested_blocks,
            avg(block_utilization) as avg_block_util,
            max(block_utilization) as max_block_util,
            -- Compute usage
            sum(compute_units_consumed) as total_compute,
            avg(compute_units_consumed) as avg_compute,
            -- Timing impact
            avg(block_production_time_ms) as avg_block_time,
            stddevPop(block_production_time_ms) as block_time_std,
            -- Priority fee escalation
            avg(priority_fee_lamports) as avg_priority_fee,
            max(priority_fee_lamports) as max_priority_fee,
            corr(toUnixTimestamp(timestamp), priority_fee_lamports) as fee_escalation_corr
        FROM mev_transactions t
        JOIN block_metrics b ON t.block_height = b.block_height
        WHERE t.entity_id = '{entity_id}'
            AND t.timestamp >= now() - INTERVAL {lookback_hours} HOUR
        """
        
        result = self.client.execute(query)
        
        if result:
            row = result[0]
            return {
                'blocks_touched': row[0],
                'congested_blocks': row[1],
                'congestion_ratio': row[1] / max(row[0], 1),
                'avg_block_utilization': row[2],
                'max_block_utilization': row[3],
                'total_compute_units': row[4],
                'avg_compute_units': row[5],
                'avg_block_time_ms': row[6],
                'block_time_std': row[7],
                'avg_priority_fee': row[8],
                'max_priority_fee': row[9],
                'fee_escalation_correlation': row[10]
            }
        
        return {}
    
    def calculate_market_dominance(self, 
                                  entity_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market share and dominance metrics
        """
        if entity_metrics.empty:
            return entity_metrics
        
        # Total market size
        total_extraction = entity_metrics['sol_90d'].sum()
        total_sandwiches = entity_metrics['sandwiches_90d'].sum()
        
        # Add market share columns
        entity_metrics['market_share_sol'] = (
            entity_metrics['sol_90d'] / total_extraction * 100
        )
        entity_metrics['market_share_sandwiches'] = (
            entity_metrics['sandwiches_90d'] / total_sandwiches * 100
        )
        
        # Dominance score (Herfindahl-Hirschman Index contribution)
        entity_metrics['hhi_contribution'] = entity_metrics['market_share_sol'] ** 2
        
        # Efficiency metrics
        entity_metrics['sol_per_sandwich'] = (
            entity_metrics['sol_90d'] / entity_metrics['sandwiches_90d'].replace(0, 1)
        )
        entity_metrics['success_rate'] = (
            entity_metrics['landed_90d'] / entity_metrics['txs_90d'].replace(0, 1) * 100
        )
        entity_metrics['tip_efficiency'] = (
            entity_metrics['sol_90d'] / entity_metrics['tips_90d'].replace(0, 1)
        )
        
        return entity_metrics
    
    def build_economic_profiles(self, 
                               lookback_days: int = 90) -> List[EconomicMetrics]:
        """
        Build comprehensive economic impact profiles
        """
        # Get extraction metrics
        extraction_df = self.get_extraction_metrics(lookback_days=lookback_days)
        
        if extraction_df.empty:
            return []
        
        # Add market dominance metrics
        extraction_df = self.calculate_market_dominance(extraction_df)
        
        # Get victim data
        victim_df = self.get_victim_analysis(lookback_days=min(lookback_days, 30))
        
        profiles = []
        
        for idx, row in extraction_df.iterrows():
            entity_id = row['entity_id']
            
            # Get congestion impact
            congestion = self.calculate_congestion_impact(
                entity_id, 
                lookback_hours=lookback_days * 24
            )
            
            # Calculate victim metrics for this entity
            entity_victims = victim_df[
                victim_df['attacker_entities'].apply(lambda x: entity_id in x)
            ]
            
            if not entity_victims.empty:
                total_victims = len(entity_victims)
                avg_victim_loss = entity_victims['total_loss_sol'].mean()
                median_victim_loss = entity_victims['total_loss_sol'].median()
                max_victim_loss = entity_victims['total_loss_sol'].max()
                
                # Victim loss distribution
                loss_bins = [0, 0.1, 0.5, 1, 5, 10, float('inf')]
                loss_labels = ['<0.1', '0.1-0.5', '0.5-1', '1-5', '5-10', '>10']
                victim_losses = entity_victims['total_loss_sol'].values
                loss_dist = pd.cut(victim_losses, bins=loss_bins, labels=loss_labels)
                victim_loss_distribution = dict(loss_dist.value_counts())
            else:
                total_victims = 0
                avg_victim_loss = 0
                median_victim_loss = 0
                max_victim_loss = 0
                victim_loss_distribution = {}
            
            # Calculate derived metrics
            gross_profit = row['sol_90d']
            costs = row['tips_90d'] + row['fees_90d']
            net_profit = gross_profit - costs
            
            roi = (net_profit / costs * 100) if costs > 0 else float('inf')
            
            profit_per_sandwich = (
                gross_profit / row['sandwiches_90d'] 
                if row['sandwiches_90d'] > 0 else 0
            )
            
            avg_tip = row['tips_90d'] / row['bundles_90d'] if row['bundles_90d'] > 0 else 0
            
            # Generate decision DNA
            dna_input = f"{entity_id}:{gross_profit:.2f}:{net_profit:.2f}:{total_victims}"
            decision_dna = hashlib.sha256(dna_input.encode()).hexdigest()[:16]
            
            metrics = EconomicMetrics(
                entity_id=entity_id,
                measurement_period=timedelta(days=lookback_days),
                # Extraction
                sol_extracted_7d=row.get('sol_7d', 0),
                sol_extracted_30d=row.get('sol_30d', 0),
                sol_extracted_90d=gross_profit,
                gross_profit_sol=gross_profit,
                net_profit_sol=net_profit,
                # Victims
                total_victims=total_victims,
                unique_victims=row.get('max_daily_victims', 0),
                avg_victim_loss_sol=avg_victim_loss,
                median_victim_loss_sol=median_victim_loss,
                max_victim_loss_sol=max_victim_loss,
                victim_loss_distribution=victim_loss_distribution,
                # Costs
                total_tips_paid=row['tips_90d'],
                total_fees_paid=row['fees_90d'],
                avg_tip_per_bundle=avg_tip,
                tip_efficiency_ratio=row.get('tip_efficiency', 0),
                # Congestion
                blocks_congested=congestion.get('congested_blocks', 0),
                congestion_ratio=congestion.get('congestion_ratio', 0),
                avg_block_utilization=congestion.get('avg_block_utilization', 0),
                compute_units_consumed=congestion.get('total_compute_units', 0),
                # Efficiency
                profit_per_sandwich=profit_per_sandwich,
                success_rate=row.get('success_rate', 0),
                roi_percentage=roi,
                # Market
                market_share=row.get('market_share_sol', 0),
                dominance_score=row.get('hhi_contribution', 0),
                decision_dna=decision_dna
            )
            
            profiles.append(metrics)
        
        # Sort by gross profit
        profiles.sort(key=lambda x: x.gross_profit_sol, reverse=True)
        
        return profiles
    
    def generate_report(self, profiles: List[EconomicMetrics]) -> str:
        """
        Generate comprehensive economic impact report
        """
        report = []
        report.append("=" * 80)
        report.append("ECONOMIC IMPACT ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        report.append("")
        
        if not profiles:
            report.append("No data available for analysis")
            return "\n".join(report)
        
        # Summary statistics
        total_extraction = sum(p.gross_profit_sol for p in profiles)
        total_victims = sum(p.total_victims for p in profiles)
        total_tips = sum(p.total_tips_paid for p in profiles)
        
        report.append("MARKET OVERVIEW:")
        report.append("-" * 40)
        report.append(f"Total SOL Extracted (90d): {total_extraction:,.2f} SOL")
        report.append(f"Total Unique Victims: {total_victims:,}")
        report.append(f"Total Tips Paid: {total_tips:,.2f} SOL")
        report.append(f"Number of Active Entities: {len(profiles)}")
        report.append("")
        
        # Top extractors
        report.append("TOP EXTRACTORS BY GROSS PROFIT (90 DAYS):")
        report.append("-" * 40)
        
        for i, profile in enumerate(profiles[:10], 1):
            monthly_rate = profile.sol_extracted_30d
            b91_comparison = (monthly_rate / self.b91_monthly_reference * 100) if self.b91_monthly_reference > 0 else 0
            
            report.append(f"\n{i}. Entity: {profile.entity_id}")
            report.append(f"   90-Day Extraction: {profile.gross_profit_sol:,.2f} SOL")
            report.append(f"   30-Day Extraction: {profile.sol_extracted_30d:,.2f} SOL")
            report.append(f"   7-Day Extraction: {profile.sol_extracted_7d:,.2f} SOL")
            report.append(f"   vs B91 Reference: {b91_comparison:.1f}%")
            report.append(f"   Net Profit: {profile.net_profit_sol:,.2f} SOL")
            report.append(f"   ROI: {profile.roi_percentage:.1f}%")
            report.append(f"   Market Share: {profile.market_share:.2f}%")
            report.append(f"   Success Rate: {profile.success_rate:.1f}%")
            report.append(f"   Profit/Sandwich: {profile.profit_per_sandwich:.4f} SOL")
            report.append(f"   Total Victims: {profile.total_victims:,}")
            report.append(f"   Avg Victim Loss: {profile.avg_victim_loss_sol:.4f} SOL")
            report.append(f"   Decision DNA: {profile.decision_dna}")
        
        # Victim impact analysis
        report.append("\n" + "=" * 40)
        report.append("VICTIM IMPACT ANALYSIS:")
        report.append("-" * 40)
        
        all_victim_losses = []
        for profile in profiles:
            if profile.victim_loss_distribution:
                for range_label, count in profile.victim_loss_distribution.items():
                    all_victim_losses.extend([range_label] * count)
        
        if all_victim_losses:
            loss_counter = pd.Series(all_victim_losses).value_counts()
            report.append("Loss Distribution (SOL):")
            for loss_range, count in loss_counter.items():
                percentage = count / len(all_victim_losses) * 100
                report.append(f"  {loss_range} SOL: {count:,} victims ({percentage:.1f}%)")
        
        # Congestion impact
        report.append("\n" + "=" * 40)
        report.append("NETWORK CONGESTION IMPACT:")
        report.append("-" * 40)
        
        high_congestion = [p for p in profiles if p.congestion_ratio > 0.1]
        report.append(f"Entities causing high congestion (>10%): {len(high_congestion)}")
        
        if high_congestion:
            total_congested_blocks = sum(p.blocks_congested for p in high_congestion)
            total_compute = sum(p.compute_units_consumed for p in high_congestion)
            
            report.append(f"Total congested blocks: {total_congested_blocks:,}")
            report.append(f"Total compute units consumed: {total_compute:,}")
            
            report.append("\nTop Congestion Contributors:")
            for profile in sorted(high_congestion, key=lambda x: x.congestion_ratio, reverse=True)[:5]:
                report.append(f"  {profile.entity_id}: {profile.congestion_ratio:.1%} blocks congested")
        
        # Efficiency rankings
        report.append("\n" + "=" * 40)
        report.append("EFFICIENCY RANKINGS:")
        report.append("-" * 40)
        
        # Sort by tip efficiency
        by_efficiency = sorted(profiles, key=lambda x: x.tip_efficiency_ratio, reverse=True)[:5]
        report.append("\nMost Tip-Efficient (SOL earned per SOL tipped):")
        for profile in by_efficiency:
            report.append(f"  {profile.entity_id}: {profile.tip_efficiency_ratio:.2f}x")
        
        # Sort by profit per sandwich
        by_profit_per = sorted(profiles, key=lambda x: x.profit_per_sandwich, reverse=True)[:5]
        report.append("\nHighest Profit per Sandwich:")
        for profile in by_profit_per:
            report.append(f"  {profile.entity_id}: {profile.profit_per_sandwich:.4f} SOL")
        
        # Market concentration (HHI)
        report.append("\n" + "=" * 40)
        report.append("MARKET CONCENTRATION:")
        report.append("-" * 40)
        
        hhi = sum(p.dominance_score for p in profiles)
        report.append(f"Herfindahl-Hirschman Index: {hhi:.0f}")
        
        if hhi < 1500:
            report.append("→ Market is COMPETITIVE")
        elif hhi < 2500:
            report.append("→ Market is MODERATELY CONCENTRATED")
        else:
            report.append("→ Market is HIGHLY CONCENTRATED")
        
        # Top 3 market share
        top3_share = sum(p.market_share for p in profiles[:3])
        report.append(f"Top 3 entities control: {top3_share:.1f}% of extraction")
        
        return "\n".join(report)
    
    def export_to_clickhouse(self, profiles: List[EconomicMetrics]):
        """
        Export economic metrics to ClickHouse
        """
        if not profiles:
            return
        
        data = []
        for profile in profiles:
            data.append({
                'timestamp': datetime.now(),
                'entity_id': profile.entity_id,
                'measurement_days': profile.measurement_period.days,
                'sol_extracted_7d': profile.sol_extracted_7d,
                'sol_extracted_30d': profile.sol_extracted_30d,
                'sol_extracted_90d': profile.sol_extracted_90d,
                'gross_profit_sol': profile.gross_profit_sol,
                'net_profit_sol': profile.net_profit_sol,
                'total_victims': profile.total_victims,
                'unique_victims': profile.unique_victims,
                'avg_victim_loss_sol': profile.avg_victim_loss_sol,
                'median_victim_loss_sol': profile.median_victim_loss_sol,
                'max_victim_loss_sol': profile.max_victim_loss_sol,
                'total_tips_paid': profile.total_tips_paid,
                'total_fees_paid': profile.total_fees_paid,
                'avg_tip_per_bundle': profile.avg_tip_per_bundle,
                'tip_efficiency_ratio': profile.tip_efficiency_ratio,
                'blocks_congested': profile.blocks_congested,
                'congestion_ratio': profile.congestion_ratio,
                'avg_block_utilization': profile.avg_block_utilization,
                'compute_units_consumed': profile.compute_units_consumed,
                'profit_per_sandwich': profile.profit_per_sandwich,
                'success_rate': profile.success_rate,
                'roi_percentage': profile.roi_percentage,
                'market_share': profile.market_share,
                'dominance_score': profile.dominance_score,
                'decision_dna': profile.decision_dna
            })
        
        # Create table
        create_table = """
        CREATE TABLE IF NOT EXISTS economic_impact_analysis (
            timestamp DateTime,
            entity_id String,
            measurement_days UInt16,
            sol_extracted_7d Float32,
            sol_extracted_30d Float32,
            sol_extracted_90d Float32,
            gross_profit_sol Float32,
            net_profit_sol Float32,
            total_victims UInt32,
            unique_victims UInt32,
            avg_victim_loss_sol Float32,
            median_victim_loss_sol Float32,
            max_victim_loss_sol Float32,
            total_tips_paid Float32,
            total_fees_paid Float32,
            avg_tip_per_bundle Float32,
            tip_efficiency_ratio Float32,
            blocks_congested UInt32,
            congestion_ratio Float32,
            avg_block_utilization Float32,
            compute_units_consumed UInt64,
            profit_per_sandwich Float32,
            success_rate Float32,
            roi_percentage Float32,
            market_share Float32,
            dominance_score Float32,
            decision_dna String
        ) ENGINE = MergeTree()
        ORDER BY (timestamp, entity_id)
        TTL timestamp + INTERVAL 180 DAY
        """
        
        self.client.execute(create_table)
        
        # Insert data
        self.client.execute(
            "INSERT INTO economic_impact_analysis VALUES",
            data
        )


def main():
    """
    Main execution
    """
    analyzer = EconomicImpactAnalyzer()
    
    print("Analyzing economic impact of MEV activities...")
    profiles = analyzer.build_economic_profiles(lookback_days=90)
    
    # Generate report
    report = analyzer.generate_report(profiles)
    print(report)
    
    # Export to ClickHouse
    analyzer.export_to_clickhouse(profiles)
    print(f"\nResults exported to ClickHouse")
    
    # Special analysis for B91 if present
    b91_profiles = [p for p in profiles if p.entity_id == 'B91']
    if b91_profiles:
        b91 = b91_profiles[0]
        print("\n" + "=" * 40)
        print("B91 ECONOMIC PROFILE:")
        print("-" * 40)
        print(f"Monthly Extraction: {b91.sol_extracted_30d:,.2f} SOL")
        print(f"Reference: ~{analyzer.b91_monthly_reference:,} SOL/month")
        print(f"Market Share: {b91.market_share:.2f}%")
        print(f"ROI: {b91.roi_percentage:.1f}%")
        print(f"Victims Affected: {b91.total_victims:,}")


if __name__ == "__main__":
    main()