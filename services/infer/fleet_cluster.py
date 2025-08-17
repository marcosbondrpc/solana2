#!/usr/bin/env python3
"""
Wallet Fleet Clustering Service
Detects coordinated wallet fleets using HDBSCAN and behavioral similarity
"""

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Set, Tuple, Optional
import clickhouse_driver
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import networkx as nx
import warnings

warnings.filterwarnings('ignore')


@dataclass
class WalletCluster:
    """Represents a detected wallet fleet/cluster"""
    cluster_id: str
    wallet_addresses: List[str]
    cluster_size: int
    behavioral_similarity: float
    temporal_correlation: float
    pool_overlap_ratio: float
    anti_correlation_score: float
    rotation_pattern: str
    shared_pools: Set[str]
    cluster_confidence: float
    detection_method: str
    decision_dna: str


@dataclass 
class FleetProfile:
    """Profile of a coordinated wallet fleet"""
    fleet_id: str
    primary_entity: str
    wallet_count: int
    active_wallets: int
    rotation_frequency_hours: float
    avg_lifetime_hours: float
    pool_diversity: float
    timing_precision_ms: float
    coordination_score: float
    clusters: List[WalletCluster] = field(default_factory=list)


class FleetClusterAnalyzer:
    """
    Detects coordinated wallet fleets using unsupervised clustering
    """
    
    def __init__(self, clickhouse_url: str = "http://localhost:8123"):
        self.client = clickhouse_driver.Client(host=clickhouse_url.replace("http://", "").split(":")[0])
        self.min_cluster_size = 3  # Minimum wallets to form a fleet
        self.similarity_threshold = 0.7  # Cosine similarity threshold
        
    def get_wallet_behaviors(self, 
                            lookback_hours: int = 168) -> pd.DataFrame:  # 7 days
        """
        Extract wallet behavioral features from ClickHouse
        """
        query = f"""
        SELECT 
            wallet_address,
            entity_id,
            -- Activity patterns
            count() as transaction_count,
            count(DISTINCT toStartOfHour(timestamp)) as active_hours,
            count(DISTINCT toDate(timestamp)) as active_days,
            min(timestamp) as first_seen,
            max(timestamp) as last_seen,
            -- Pool interactions
            groupArray(DISTINCT pool_address) as pools_touched,
            count(DISTINCT pool_address) as unique_pools,
            -- Timing patterns
            avg(toHour(timestamp)) as avg_hour_of_day,
            stddevPop(toHour(timestamp)) as hour_std,
            -- MEV patterns
            sum(CASE WHEN is_sandwich = 1 THEN 1 ELSE 0 END) as sandwich_count,
            sum(profit_sol) as total_profit,
            avg(profit_sol) as avg_profit,
            -- Burst detection
            max(transactions_per_minute) as max_burst_rate,
            -- Pool preference vector (top 20 pools)
            topK(20)(pool_address) as top_pools,
            -- Temporal features
            groupArray(toUnixTimestamp(timestamp)) as timestamps
        FROM (
            SELECT 
                *,
                countIf(timestamp BETWEEN timestamp - INTERVAL 1 MINUTE 
                    AND timestamp) OVER (PARTITION BY wallet_address 
                    ORDER BY timestamp) as transactions_per_minute
            FROM mev_transactions
            WHERE timestamp >= now() - INTERVAL {lookback_hours} HOUR
        )
        GROUP BY wallet_address, entity_id
        HAVING transaction_count >= 10  -- Active wallets only
        """
        
        result = self.client.execute(query)
        
        columns = ['wallet_address', 'entity_id', 'transaction_count', 'active_hours',
                  'active_days', 'first_seen', 'last_seen', 'pools_touched',
                  'unique_pools', 'avg_hour_of_day', 'hour_std', 'sandwich_count',
                  'total_profit', 'avg_profit', 'max_burst_rate', 'top_pools',
                  'timestamps']
        
        return pd.DataFrame(result, columns=columns)
    
    def create_behavioral_vectors(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create feature vectors for clustering
        """
        features = []
        
        for idx, row in df.iterrows():
            # Temporal features
            lifetime_hours = (row['last_seen'] - row['first_seen']).total_seconds() / 3600
            activity_density = row['active_hours'] / max(lifetime_hours, 1)
            
            # Pool diversity
            pool_diversity = row['unique_pools'] / max(row['transaction_count'], 1)
            
            # MEV intensity
            mev_ratio = row['sandwich_count'] / max(row['transaction_count'], 1)
            
            # Timing features
            hour_concentration = 1 / max(row['hour_std'], 1)  # Higher = more concentrated
            
            # Profit features
            profit_per_tx = row['avg_profit'] if row['avg_profit'] else 0
            
            # Activity pattern
            burst_intensity = row['max_burst_rate'] / max(row['transaction_count'], 1)
            
            feature_vector = [
                activity_density,
                pool_diversity,
                mev_ratio,
                hour_concentration,
                profit_per_tx,
                burst_intensity,
                row['unique_pools'],
                row['transaction_count'],
                lifetime_hours
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def create_pool_vectors(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create pool interaction vectors for cosine similarity
        """
        # Get all unique pools
        all_pools = set()
        for pools in df['pools_touched']:
            all_pools.update(pools)
        
        pool_list = sorted(list(all_pools))
        pool_to_idx = {pool: idx for idx, pool in enumerate(pool_list)}
        
        # Create binary vectors
        vectors = []
        for pools in df['pools_touched']:
            vector = np.zeros(len(pool_list))
            for pool in pools:
                if pool in pool_to_idx:
                    vector[pool_to_idx[pool]] = 1
            vectors.append(vector)
        
        return np.array(vectors)
    
    def detect_anti_correlation(self, 
                               timestamps_list: List[List[int]]) -> float:
        """
        Detect anti-correlated activity (wallet rotation pattern)
        """
        if len(timestamps_list) < 2:
            return 0.0
        
        # Convert to hourly activity matrices
        all_timestamps = []
        for timestamps in timestamps_list:
            all_timestamps.extend(timestamps)
        
        if not all_timestamps:
            return 0.0
        
        min_ts = min(all_timestamps)
        max_ts = max(all_timestamps)
        n_hours = int((max_ts - min_ts) / 3600) + 1
        
        # Create activity matrix
        activity_matrix = np.zeros((len(timestamps_list), n_hours))
        
        for i, timestamps in enumerate(timestamps_list):
            for ts in timestamps:
                hour_idx = int((ts - min_ts) / 3600)
                if hour_idx < n_hours:
                    activity_matrix[i, hour_idx] = 1
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(timestamps_list)):
            for j in range(i+1, len(timestamps_list)):
                # Check for anti-correlation (one active when other is not)
                overlap = np.sum(activity_matrix[i] * activity_matrix[j])
                total = np.sum(activity_matrix[i]) + np.sum(activity_matrix[j])
                
                if total > 0:
                    anti_corr = 1 - (2 * overlap / total)
                    correlations.append(anti_corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def cluster_wallets(self, 
                       df: pd.DataFrame) -> List[WalletCluster]:
        """
        Perform HDBSCAN clustering on wallet behaviors
        """
        if len(df) < self.min_cluster_size:
            return []
        
        # Create feature vectors
        behavioral_features = self.create_behavioral_vectors(df)
        pool_vectors = self.create_pool_vectors(df)
        
        # Standardize features
        scaler = StandardScaler()
        behavioral_features_scaled = scaler.fit_transform(behavioral_features)
        
        # Perform HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_epsilon=0.5
        )
        
        cluster_labels = clusterer.fit_predict(behavioral_features_scaled)
        
        # Analyze each cluster
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            # Get cluster members
            cluster_mask = cluster_labels == label
            cluster_wallets = df[cluster_mask]['wallet_address'].tolist()
            
            if len(cluster_wallets) < self.min_cluster_size:
                continue
            
            # Calculate cluster metrics
            cluster_pool_vectors = pool_vectors[cluster_mask]
            
            # Behavioral similarity (average pairwise cosine similarity)
            if len(cluster_pool_vectors) > 1:
                similarities = cosine_similarity(cluster_pool_vectors)
                behavioral_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            else:
                behavioral_similarity = 1.0
            
            # Pool overlap
            cluster_pools = df[cluster_mask]['pools_touched'].tolist()
            shared_pools = set(cluster_pools[0])
            for pools in cluster_pools[1:]:
                shared_pools = shared_pools.intersection(set(pools))
            
            all_pools = set()
            for pools in cluster_pools:
                all_pools.update(pools)
            
            pool_overlap_ratio = len(shared_pools) / len(all_pools) if all_pools else 0
            
            # Temporal correlation
            timestamps_list = df[cluster_mask]['timestamps'].tolist()
            anti_correlation = self.detect_anti_correlation(timestamps_list)
            
            # Determine rotation pattern
            if anti_correlation > 0.7:
                rotation_pattern = "STRICT_ROTATION"
            elif anti_correlation > 0.4:
                rotation_pattern = "LOOSE_ROTATION"
            else:
                rotation_pattern = "CONCURRENT"
            
            # Cluster confidence
            confidence = min(
                behavioral_similarity * 0.4 +
                pool_overlap_ratio * 0.3 +
                (anti_correlation if rotation_pattern != "CONCURRENT" else 0) * 0.3,
                1.0
            )
            
            # Generate decision DNA
            dna_input = f"cluster:{label}:{len(cluster_wallets)}:{behavioral_similarity:.3f}"
            decision_dna = hashlib.sha256(dna_input.encode()).hexdigest()[:16]
            
            cluster = WalletCluster(
                cluster_id=f"C{label}_{datetime.now().strftime('%Y%m%d%H')}",
                wallet_addresses=cluster_wallets,
                cluster_size=len(cluster_wallets),
                behavioral_similarity=behavioral_similarity,
                temporal_correlation=1 - anti_correlation,  # Convert to correlation
                pool_overlap_ratio=pool_overlap_ratio,
                anti_correlation_score=anti_correlation,
                rotation_pattern=rotation_pattern,
                shared_pools=shared_pools,
                cluster_confidence=confidence,
                detection_method="HDBSCAN",
                decision_dna=decision_dna
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def analyze_entity_fleets(self, 
                             lookback_hours: int = 168) -> List[FleetProfile]:
        """
        Analyze wallet fleets per entity
        """
        df = self.get_wallet_behaviors(lookback_hours)
        
        if df.empty:
            return []
        
        profiles = []
        
        # Group by entity
        for entity_id in df['entity_id'].unique():
            entity_df = df[df['entity_id'] == entity_id]
            
            if len(entity_df) < self.min_cluster_size:
                continue
            
            # Detect clusters within entity
            clusters = self.cluster_wallets(entity_df)
            
            # Calculate fleet-level metrics
            all_wallets = entity_df['wallet_address'].tolist()
            active_wallets = len(entity_df[
                entity_df['last_seen'] >= datetime.now() - timedelta(hours=24)
            ])
            
            # Rotation frequency
            lifetimes = []
            for idx, row in entity_df.iterrows():
                lifetime = (row['last_seen'] - row['first_seen']).total_seconds() / 3600
                lifetimes.append(lifetime)
            
            avg_lifetime = np.mean(lifetimes) if lifetimes else 0
            
            # Pool diversity
            all_entity_pools = set()
            for pools in entity_df['pools_touched']:
                all_entity_pools.update(pools)
            
            pool_diversity = len(all_entity_pools) / max(len(all_wallets), 1)
            
            # Coordination score
            coordination_score = 0
            if clusters:
                # Average cluster confidence weighted by size
                total_weight = sum(c.cluster_size for c in clusters)
                coordination_score = sum(
                    c.cluster_confidence * c.cluster_size for c in clusters
                ) / total_weight if total_weight > 0 else 0
            
            profile = FleetProfile(
                fleet_id=f"{entity_id}_fleet",
                primary_entity=entity_id,
                wallet_count=len(all_wallets),
                active_wallets=active_wallets,
                rotation_frequency_hours=168 / len(all_wallets) if all_wallets else 0,
                avg_lifetime_hours=avg_lifetime,
                pool_diversity=pool_diversity,
                timing_precision_ms=0,  # Would need more granular data
                coordination_score=coordination_score,
                clusters=clusters
            )
            
            profiles.append(profile)
        
        # Sort by coordination score
        profiles.sort(key=lambda x: x.coordination_score, reverse=True)
        
        return profiles
    
    def build_wallet_network(self, clusters: List[WalletCluster]) -> nx.Graph:
        """
        Build network graph of wallet relationships
        """
        G = nx.Graph()
        
        # Add nodes for all wallets
        all_wallets = set()
        for cluster in clusters:
            all_wallets.update(cluster.wallet_addresses)
        
        for wallet in all_wallets:
            G.add_node(wallet)
        
        # Add edges within clusters
        for cluster in clusters:
            wallets = cluster.wallet_addresses
            for i in range(len(wallets)):
                for j in range(i+1, len(wallets)):
                    G.add_edge(
                        wallets[i], 
                        wallets[j],
                        weight=cluster.behavioral_similarity,
                        cluster_id=cluster.cluster_id
                    )
        
        return G
    
    def generate_report(self, profiles: List[FleetProfile]) -> str:
        """
        Generate comprehensive fleet detection report
        """
        report = []
        report.append("=" * 80)
        report.append("WALLET FLEET CLUSTERING ANALYSIS")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        total_fleets = len(profiles)
        total_clusters = sum(len(p.clusters) for p in profiles)
        coordinated_fleets = [p for p in profiles if p.coordination_score > 0.5]
        
        report.append(f"Total entities with fleets: {total_fleets}")
        report.append(f"Total clusters detected: {total_clusters}")
        report.append(f"Highly coordinated fleets: {len(coordinated_fleets)}")
        report.append("")
        
        # Detailed fleet analysis
        if profiles:
            report.append("TOP COORDINATED FLEETS:")
            report.append("-" * 40)
            
            for profile in profiles[:10]:
                report.append(f"\nEntity: {profile.primary_entity}")
                report.append(f"  Fleet Size: {profile.wallet_count} wallets")
                report.append(f"  Active Wallets: {profile.active_wallets}")
                report.append(f"  Coordination Score: {profile.coordination_score:.2%}")
                report.append(f"  Avg Wallet Lifetime: {profile.avg_lifetime_hours:.1f} hours")
                report.append(f"  Pool Diversity: {profile.pool_diversity:.2f}")
                
                if profile.clusters:
                    report.append(f"  Detected Clusters: {len(profile.clusters)}")
                    
                    for cluster in profile.clusters:
                        report.append(f"\n    Cluster {cluster.cluster_id}:")
                        report.append(f"      Size: {cluster.cluster_size} wallets")
                        report.append(f"      Similarity: {cluster.behavioral_similarity:.2%}")
                        report.append(f"      Rotation: {cluster.rotation_pattern}")
                        report.append(f"      Pool Overlap: {cluster.pool_overlap_ratio:.2%}")
                        report.append(f"      Confidence: {cluster.cluster_confidence:.2%}")
                        report.append(f"      DNA: {cluster.decision_dna}")
        
        # Rotation patterns summary
        report.append("\n" + "=" * 40)
        report.append("ROTATION PATTERNS DETECTED:")
        report.append("-" * 40)
        
        pattern_counts = {}
        for profile in profiles:
            for cluster in profile.clusters:
                pattern = cluster.rotation_pattern
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"{pattern}: {count} clusters")
        
        return "\n".join(report)
    
    def export_to_clickhouse(self, profiles: List[FleetProfile]):
        """
        Export fleet analysis to ClickHouse
        """
        if not profiles:
            return
        
        # Fleet profiles table
        fleet_data = []
        for profile in profiles:
            fleet_data.append({
                'timestamp': datetime.now(),
                'fleet_id': profile.fleet_id,
                'entity_id': profile.primary_entity,
                'wallet_count': profile.wallet_count,
                'active_wallets': profile.active_wallets,
                'rotation_frequency_hours': profile.rotation_frequency_hours,
                'avg_lifetime_hours': profile.avg_lifetime_hours,
                'pool_diversity': profile.pool_diversity,
                'coordination_score': profile.coordination_score,
                'cluster_count': len(profile.clusters)
            })
        
        # Cluster details table
        cluster_data = []
        for profile in profiles:
            for cluster in profile.clusters:
                cluster_data.append({
                    'timestamp': datetime.now(),
                    'fleet_id': profile.fleet_id,
                    'cluster_id': cluster.cluster_id,
                    'cluster_size': cluster.cluster_size,
                    'behavioral_similarity': cluster.behavioral_similarity,
                    'temporal_correlation': cluster.temporal_correlation,
                    'pool_overlap_ratio': cluster.pool_overlap_ratio,
                    'anti_correlation_score': cluster.anti_correlation_score,
                    'rotation_pattern': cluster.rotation_pattern,
                    'cluster_confidence': cluster.cluster_confidence,
                    'decision_dna': cluster.decision_dna
                })
        
        # Create tables
        create_fleet_table = """
        CREATE TABLE IF NOT EXISTS wallet_fleet_analysis (
            timestamp DateTime,
            fleet_id String,
            entity_id String,
            wallet_count UInt32,
            active_wallets UInt32,
            rotation_frequency_hours Float32,
            avg_lifetime_hours Float32,
            pool_diversity Float32,
            coordination_score Float32,
            cluster_count UInt32
        ) ENGINE = MergeTree()
        ORDER BY (timestamp, entity_id)
        TTL timestamp + INTERVAL 90 DAY
        """
        
        create_cluster_table = """
        CREATE TABLE IF NOT EXISTS wallet_cluster_details (
            timestamp DateTime,
            fleet_id String,
            cluster_id String,
            cluster_size UInt32,
            behavioral_similarity Float32,
            temporal_correlation Float32,
            pool_overlap_ratio Float32,
            anti_correlation_score Float32,
            rotation_pattern String,
            cluster_confidence Float32,
            decision_dna String
        ) ENGINE = MergeTree()
        ORDER BY (timestamp, fleet_id, cluster_id)
        TTL timestamp + INTERVAL 90 DAY
        """
        
        self.client.execute(create_fleet_table)
        self.client.execute(create_cluster_table)
        
        # Insert data
        if fleet_data:
            self.client.execute("INSERT INTO wallet_fleet_analysis VALUES", fleet_data)
        if cluster_data:
            self.client.execute("INSERT INTO wallet_cluster_details VALUES", cluster_data)


def main():
    """
    Main execution
    """
    analyzer = FleetClusterAnalyzer()
    
    print("Analyzing wallet fleets and clusters...")
    profiles = analyzer.analyze_entity_fleets(lookback_hours=168)
    
    # Generate report
    report = analyzer.generate_report(profiles)
    print(report)
    
    # Export to ClickHouse
    analyzer.export_to_clickhouse(profiles)
    print(f"\nResults exported to ClickHouse")
    
    # Build network graph for visualization
    all_clusters = []
    for profile in profiles:
        all_clusters.extend(profile.clusters)
    
    if all_clusters:
        G = analyzer.build_wallet_network(all_clusters)
        print(f"\nWallet network graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


if __name__ == "__main__":
    main()