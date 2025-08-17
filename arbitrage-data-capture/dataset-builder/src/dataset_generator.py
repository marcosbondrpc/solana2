import asyncio
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils import resample
import pyarrow as pa
import pyarrow.parquet as pq
from clickhouse_driver import Client as ClickHouseClient
import structlog

logger = structlog.get_logger()

@dataclass
class DatasetConfig:
    name: str
    version: str
    description: str
    source_table: str
    target_variable: str
    features: List[str]
    time_range: Tuple[datetime, datetime]
    filters: Dict[str, Any]
    split_strategy: str  # random, temporal, stratified
    test_size: float
    validation_size: float
    balance_strategy: Optional[str]  # oversample, undersample, smote
    export_format: List[str]  # json, parquet, csv

class DatasetBuilder:
    def __init__(self, clickhouse_config: Dict[str, Any]):
        self.clickhouse = ClickHouseClient(
            host=clickhouse_config['host'],
            port=clickhouse_config['port'],
            database=clickhouse_config['database']
        )
        self.datasets = {}
        self.metadata = {}
        
    async def build_dataset(self, config: DatasetConfig) -> Dict[str, Any]:
        """Build ML-ready dataset according to configuration"""
        logger.info(f"Building dataset: {config.name} v{config.version}")
        
        # Fetch raw data
        raw_data = await self.fetch_data(config)
        
        if raw_data.empty:
            logger.warning("No data found for dataset configuration")
            return {}
            
        # Feature engineering
        engineered_data = await self.engineer_features(raw_data, config)
        
        # Clean and validate data
        clean_data = await self.clean_data(engineered_data, config)
        
        # Balance dataset if needed
        balanced_data = await self.balance_dataset(clean_data, config)
        
        # Split dataset
        splits = await self.split_dataset(balanced_data, config)
        
        # Generate statistics
        stats = await self.generate_statistics(splits)
        
        # Export dataset
        export_paths = await self.export_dataset(splits, config, stats)
        
        # Store metadata
        metadata = self.create_metadata(config, stats, export_paths)
        self.metadata[config.name] = metadata
        
        return {
            'dataset': splits,
            'metadata': metadata,
            'statistics': stats,
            'export_paths': export_paths
        }
        
    async def fetch_data(self, config: DatasetConfig) -> pd.DataFrame:
        """Fetch data from ClickHouse"""
        where_clauses = []
        
        # Time range filter
        where_clauses.append(
            f"timestamp BETWEEN '{config.time_range[0]}' AND '{config.time_range[1]}'"
        )
        
        # Additional filters
        for field, value in config.filters.items():
            if isinstance(value, str):
                where_clauses.append(f"{field} = '{value}'")
            elif isinstance(value, (list, tuple)):
                values_str = "', '".join(map(str, value))
                where_clauses.append(f"{field} IN ('{values_str}')")
            else:
                where_clauses.append(f"{field} = {value}")
                
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        SELECT *
        FROM {config.source_table}
        WHERE {where_clause}
        ORDER BY timestamp
        """
        
        logger.info(f"Executing query: {query[:200]}...")
        
        result = self.clickhouse.execute(query)
        
        if not result:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(result)
        
        logger.info(f"Fetched {len(df)} records")
        
        return df
        
    async def engineer_features(
        self, 
        data: pd.DataFrame, 
        config: DatasetConfig
    ) -> pd.DataFrame:
        """Engineer features for ML"""
        logger.info("Engineering features")
        
        df = data.copy()
        
        # Price-based features
        if 'price' in df.columns:
            df['price_log'] = np.log1p(df['price'])
            df['price_change'] = df['price'].pct_change()
            df['price_volatility'] = df['price'].rolling(window=20).std()
            df['price_ma_7'] = df['price'].rolling(window=7).mean()
            df['price_ma_30'] = df['price'].rolling(window=30).mean()
            df['price_ma_ratio'] = df['price_ma_7'] / df['price_ma_30']
            
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['price'].rolling(window=20).mean()
            bb_std = df['price'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
        # Volume-based features
        if 'volume' in df.columns:
            df['volume_log'] = np.log1p(df['volume'])
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
            df['volume_spike'] = df['volume'] / df['volume_ma_7']
            
            # VWAP
            if 'price' in df.columns:
                df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
                df['price_vwap_ratio'] = df['price'] / df['vwap']
                
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
        # Network features
        if 'gas_price' in df.columns:
            df['gas_price_log'] = np.log1p(df['gas_price'])
            df['gas_price_ma'] = df['gas_price'].rolling(window=10).mean()
            df['gas_spike'] = df['gas_price'] / df['gas_price_ma']
            
        # DEX-specific features
        if 'num_dex' in df.columns:
            df['is_multi_dex'] = (df['num_dex'] > 1).astype(int)
            df['is_triangular'] = (df['num_dex'] == 3).astype(int)
            
        # Profit features
        if 'profit_usd' in df.columns:
            df['profit_log'] = np.log1p(df['profit_usd'].clip(lower=0))
            df['is_profitable'] = (df['profit_usd'] > 0).astype(int)
            df['profit_category'] = pd.cut(
                df['profit_usd'],
                bins=[-np.inf, 0, 10, 100, 1000, np.inf],
                labels=['loss', 'small', 'medium', 'large', 'whale']
            )
            
        # Lag features
        lag_columns = ['price', 'volume', 'gas_price', 'profit_usd']
        for col in lag_columns:
            if col in df.columns:
                for lag in [1, 3, 6, 12, 24]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
        # Rolling statistics
        window_sizes = [5, 10, 30]
        stat_columns = ['price', 'volume', 'profit_usd']
        
        for col in stat_columns:
            if col in df.columns:
                for window in window_sizes:
                    df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window).max()
                    
        # Interaction features
        if 'price' in df.columns and 'volume' in df.columns:
            df['price_volume_interaction'] = df['price'] * df['volume']
            
        if 'gas_price' in df.columns and 'profit_usd' in df.columns:
            df['gas_profit_ratio'] = df['gas_price'] / (df['profit_usd'] + 1)
            
        # Remove NaN rows created by rolling/lag features
        df = df.dropna()
        
        logger.info(f"Engineered {len(df.columns)} features")
        
        return df
        
    async def clean_data(
        self, 
        data: pd.DataFrame, 
        config: DatasetConfig
    ) -> pd.DataFrame:
        """Clean and validate data"""
        logger.info("Cleaning data")
        
        df = data.copy()
        initial_size = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('unknown')
        
        # Remove outliers using IQR method
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Validate data types
        for col in config.features:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric if possible
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
                        
        final_size = len(df)
        logger.info(f"Cleaned data: {initial_size} -> {final_size} records")
        
        return df
        
    async def balance_dataset(
        self, 
        data: pd.DataFrame, 
        config: DatasetConfig
    ) -> pd.DataFrame:
        """Balance dataset if needed"""
        if not config.balance_strategy or config.target_variable not in data.columns:
            return data
            
        logger.info(f"Balancing dataset using {config.balance_strategy}")
        
        df = data.copy()
        target = df[config.target_variable]
        
        if config.balance_strategy == 'oversample':
            # Oversample minority class
            value_counts = target.value_counts()
            max_count = value_counts.max()
            
            balanced_dfs = []
            for value in value_counts.index:
                class_df = df[target == value]
                if len(class_df) < max_count:
                    class_df = resample(
                        class_df,
                        replace=True,
                        n_samples=max_count,
                        random_state=42
                    )
                balanced_dfs.append(class_df)
                
            df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif config.balance_strategy == 'undersample':
            # Undersample majority class
            value_counts = target.value_counts()
            min_count = value_counts.min()
            
            balanced_dfs = []
            for value in value_counts.index:
                class_df = df[target == value]
                if len(class_df) > min_count:
                    class_df = class_df.sample(n=min_count, random_state=42)
                balanced_dfs.append(class_df)
                
            df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif config.balance_strategy == 'smote':
            # SMOTE oversampling
            from imblearn.over_sampling import SMOTE
            
            features = [col for col in df.columns if col != config.target_variable]
            X = df[features]
            y = df[config.target_variable]
            
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            df = pd.DataFrame(X_balanced, columns=features)
            df[config.target_variable] = y_balanced
            
        logger.info(f"Balanced dataset size: {len(df)}")
        
        return df
        
    async def split_dataset(
        self, 
        data: pd.DataFrame, 
        config: DatasetConfig
    ) -> Dict[str, pd.DataFrame]:
        """Split dataset into train/validation/test sets"""
        logger.info(f"Splitting dataset using {config.split_strategy} strategy")
        
        if config.target_variable not in data.columns:
            target = None
        else:
            target = data[config.target_variable]
            
        if config.split_strategy == 'random':
            # Random split
            train_val, test = train_test_split(
                data,
                test_size=config.test_size,
                random_state=42,
                stratify=target if target is not None else None
            )
            
            val_size_adjusted = config.validation_size / (1 - config.test_size)
            train, val = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=42,
                stratify=train_val[config.target_variable] if config.target_variable in train_val.columns else None
            )
            
        elif config.split_strategy == 'temporal':
            # Temporal split (time series)
            data_sorted = data.sort_values('timestamp')
            n = len(data_sorted)
            
            train_end = int(n * (1 - config.test_size - config.validation_size))
            val_end = int(n * (1 - config.test_size))
            
            train = data_sorted.iloc[:train_end]
            val = data_sorted.iloc[train_end:val_end]
            test = data_sorted.iloc[val_end:]
            
        elif config.split_strategy == 'stratified':
            # Stratified K-fold for better distribution
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Use first fold for test, second for validation, rest for train
            splits = list(skf.split(data, target))
            
            train_idx = np.concatenate([splits[2][0], splits[3][0], splits[4][0]])
            val_idx = splits[1][0]
            test_idx = splits[0][0]
            
            train = data.iloc[train_idx]
            val = data.iloc[val_idx]
            test = data.iloc[test_idx]
            
        else:
            raise ValueError(f"Unknown split strategy: {config.split_strategy}")
            
        splits = {
            'train': train,
            'validation': val,
            'test': test
        }
        
        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return splits
        
    async def generate_statistics(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive statistics for the dataset"""
        logger.info("Generating dataset statistics")
        
        stats = {}
        
        for split_name, df in splits.items():
            split_stats = {
                'n_samples': len(df),
                'n_features': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns),
                'missing_values': df.isnull().sum().to_dict(),
                'feature_statistics': {}
            }
            
            # Statistics for numeric features
            numeric_df = df.select_dtypes(include=[np.number])
            for col in numeric_df.columns:
                split_stats['feature_statistics'][col] = {
                    'mean': float(numeric_df[col].mean()),
                    'std': float(numeric_df[col].std()),
                    'min': float(numeric_df[col].min()),
                    'max': float(numeric_df[col].max()),
                    'median': float(numeric_df[col].median()),
                    'q25': float(numeric_df[col].quantile(0.25)),
                    'q75': float(numeric_df[col].quantile(0.75)),
                    'skewness': float(numeric_df[col].skew()),
                    'kurtosis': float(numeric_df[col].kurtosis()),
                    'n_unique': int(numeric_df[col].nunique()),
                    'n_missing': int(numeric_df[col].isnull().sum())
                }
                
            # Statistics for categorical features
            categorical_df = df.select_dtypes(include=['object'])
            for col in categorical_df.columns:
                value_counts = categorical_df[col].value_counts()
                split_stats['feature_statistics'][col] = {
                    'n_unique': int(categorical_df[col].nunique()),
                    'n_missing': int(categorical_df[col].isnull().sum()),
                    'mode': str(categorical_df[col].mode()[0]) if not categorical_df[col].mode().empty else None,
                    'top_values': value_counts.head(10).to_dict()
                }
                
            stats[split_name] = split_stats
            
        # Calculate class balance if target variable exists
        if 'train' in splits:
            for col in splits['train'].columns:
                if 'target' in col.lower() or 'label' in col.lower():
                    class_distribution = {}
                    for split_name, df in splits.items():
                        if col in df.columns:
                            class_distribution[split_name] = df[col].value_counts().to_dict()
                    stats['class_distribution'] = class_distribution
                    break
                    
        return stats
        
    async def export_dataset(
        self, 
        splits: Dict[str, pd.DataFrame], 
        config: DatasetConfig,
        stats: Dict[str, Any]
    ) -> Dict[str, str]:
        """Export dataset in multiple formats"""
        logger.info(f"Exporting dataset in formats: {config.export_format}")
        
        export_dir = f"/data/datasets/{config.name}/v{config.version}"
        os.makedirs(export_dir, exist_ok=True)
        
        export_paths = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for format_type in config.export_format:
            if format_type == 'json':
                # Export as JSON with SOTA-1.0 schema
                for split_name, df in splits.items():
                    json_path = f"{export_dir}/{split_name}_{timestamp}.json"
                    
                    # Convert to SOTA-1.0 format
                    sota_data = self.convert_to_sota_format(df, config, split_name)
                    
                    with open(json_path, 'w') as f:
                        json.dump(sota_data, f, indent=2, default=str)
                        
                    export_paths[f'{split_name}_json'] = json_path
                    
            elif format_type == 'parquet':
                # Export as Parquet for efficient storage
                for split_name, df in splits.items():
                    parquet_path = f"{export_dir}/{split_name}_{timestamp}.parquet"
                    df.to_parquet(
                        parquet_path,
                        compression='snappy',
                        index=False
                    )
                    export_paths[f'{split_name}_parquet'] = parquet_path
                    
            elif format_type == 'csv':
                # Export as CSV for compatibility
                for split_name, df in splits.items():
                    csv_path = f"{export_dir}/{split_name}_{timestamp}.csv"
                    df.to_csv(csv_path, index=False)
                    export_paths[f'{split_name}_csv'] = csv_path
                    
        # Export metadata
        metadata_path = f"{export_dir}/metadata_{timestamp}.json"
        metadata = {
            'config': asdict(config) if hasattr(config, '__dict__') else config,
            'statistics': stats,
            'export_paths': export_paths,
            'created_at': datetime.now().isoformat(),
            'dataset_hash': self.calculate_dataset_hash(splits)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        export_paths['metadata'] = metadata_path
        
        logger.info(f"Dataset exported to {export_dir}")
        
        return export_paths
        
    def convert_to_sota_format(
        self, 
        df: pd.DataFrame, 
        config: DatasetConfig,
        split_name: str
    ) -> Dict[str, Any]:
        """Convert DataFrame to SOTA-1.0 format"""
        
        sota_data = {
            'version': 'SOTA-1.0',
            'dataset_name': config.name,
            'dataset_version': config.version,
            'split': split_name,
            'created_at': datetime.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(df.columns),
            'features': df.columns.tolist(),
            'target_variable': config.target_variable,
            'data': []
        }
        
        # Convert each row to SOTA format
        for _, row in df.iterrows():
            sample = {
                'id': hashlib.md5(str(row).encode()).hexdigest(),
                'features': row.to_dict(),
                'metadata': {
                    'timestamp': row.get('timestamp', ''),
                    'source': config.source_table
                }
            }
            
            if config.target_variable in row:
                sample['label'] = row[config.target_variable]
                
            sota_data['data'].append(sample)
            
        return sota_data
        
    def calculate_dataset_hash(self, splits: Dict[str, pd.DataFrame]) -> str:
        """Calculate hash of dataset for versioning"""
        hasher = hashlib.sha256()
        
        for split_name, df in splits.items():
            # Hash shape and column names
            hasher.update(f"{split_name}:{df.shape}:{df.columns.tolist()}".encode())
            
            # Hash sample of data
            sample = df.sample(min(100, len(df)), random_state=42)
            hasher.update(sample.to_string().encode())
            
        return hasher.hexdigest()
        
    def create_metadata(
        self, 
        config: DatasetConfig, 
        stats: Dict[str, Any],
        export_paths: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for dataset"""
        
        metadata = {
            'dataset_id': hashlib.md5(f"{config.name}:{config.version}".encode()).hexdigest(),
            'name': config.name,
            'version': config.version,
            'description': config.description,
            'created_at': datetime.now().isoformat(),
            'configuration': asdict(config) if hasattr(config, '__dict__') else config,
            'statistics': stats,
            'export_paths': export_paths,
            'schema': {
                'features': config.features,
                'target': config.target_variable,
                'data_types': {}
            },
            'quality_metrics': {
                'completeness': 1.0,  # Would calculate actual completeness
                'consistency': 1.0,   # Would calculate actual consistency
                'accuracy': None,     # Would need ground truth
                'timeliness': (datetime.now() - config.time_range[1]).days
            }
        }
        
        return metadata

async def main():
    # Example usage
    clickhouse_config = {
        'host': 'localhost',
        'port': 9000,
        'database': 'solana_arbitrage'
    }
    
    builder = DatasetBuilder(clickhouse_config)
    
    # Define dataset configuration
    config = DatasetConfig(
        name='solana_arbitrage_v1',
        version='1.0.0',
        description='Solana arbitrage detection dataset with advanced features',
        source_table='arbitrage_transactions',
        target_variable='is_arbitrage',
        features=[
            'price_spread_percentage', 'volume_24h', 'liquidity_depth_usd',
            'gas_price_gwei', 'num_dex_involved', 'sandwich_risk_score'
        ],
        time_range=(
            datetime.now() - timedelta(days=30),
            datetime.now()
        ),
        filters={'profit_usd': ('>', 0)},
        split_strategy='temporal',
        test_size=0.2,
        validation_size=0.1,
        balance_strategy='oversample',
        export_format=['json', 'parquet']
    )
    
    # Build dataset
    result = await builder.build_dataset(config)
    
    print(f"Dataset built successfully: {result['metadata']}")

if __name__ == "__main__":
    asyncio.run(main())