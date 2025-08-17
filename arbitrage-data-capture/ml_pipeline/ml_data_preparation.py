"""
Elite ML Pipeline for Arbitrage Data Preparation
Feature engineering, normalization, and export for model training
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce
import featuretools as ft
from imbalanced_learn.over_sampling import SMOTE
from imbalanced_learn.under_sampling import RandomUnderSampler
import pickle
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for arbitrage data"""
    
    def __init__(self):
        self.feature_names = []
        self.categorical_encoders = {}
        self.scalers = {}
        self.feature_importance = {}
    
    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from timestamps"""
        
        if 'block_timestamp' in df.columns:
            df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
            
            # Time-based features
            df['hour'] = df['block_timestamp'].dt.hour
            df['day_of_week'] = df['block_timestamp'].dt.dayofweek
            df['day_of_month'] = df['block_timestamp'].dt.day
            df['week_of_year'] = df['block_timestamp'].dt.isocalendar().week
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Cyclical encoding for temporal features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Trading session features
            df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            
            # Time since epoch (for trend)
            df['timestamp_numeric'] = df['block_timestamp'].astype(np.int64) // 10**9
            
        return df
    
    def engineer_path_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from arbitrage paths"""
        
        if 'dexes' in df.columns:
            # Path complexity
            df['unique_dex_count'] = df['dexes'].apply(lambda x: len(set(x)) if x else 0)
            df['path_length'] = df['dexes'].apply(len)
            df['path_complexity'] = df['unique_dex_count'] * df['path_length']
            
            # DEX diversity score
            df['dex_diversity'] = df['dexes'].apply(
                lambda x: len(set(x)) / len(x) if x and len(x) > 0 else 0
            )
            
            # Popular DEX features
            popular_dexes = ['uniswap', 'sushiswap', 'curve', 'balancer']
            for dex in popular_dexes:
                df[f'uses_{dex}'] = df['dexes'].apply(
                    lambda x: int(dex in [d.lower() for d in x]) if x else 0
                )
        
        if 'tokens' in df.columns:
            # Token features
            df['unique_token_count'] = df['tokens'].apply(lambda x: len(set(x)) if x else 0)
            df['token_hop_ratio'] = df['unique_token_count'] / (df['path_length'] + 1)
            
            # Stablecoin involvement
            stablecoins = ['usdc', 'usdt', 'dai', 'busd', 'tusd']
            df['uses_stablecoin'] = df['tokens'].apply(
                lambda x: int(any(stable in str(x).lower() for stable in stablecoins)) if x else 0
            )
        
        if 'amounts' in df.columns:
            # Amount features
            df['input_amount_log'] = df['amounts'].apply(
                lambda x: np.log1p(x[0]) if x and len(x) > 0 else 0
            )
            df['amount_variance'] = df['amounts'].apply(
                lambda x: np.var(x) if x and len(x) > 1 else 0
            )
        
        return df
    
    def engineer_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer market condition features"""
        
        # Volatility features
        if 'market_volatility' in df.columns:
            df['volatility_squared'] = df['market_volatility'] ** 2
            df['volatility_log'] = np.log1p(df['market_volatility'])
            
            # Volatility buckets
            df['volatility_bucket'] = pd.cut(
                df['market_volatility'],
                bins=[0, 10, 25, 50, 100, float('inf')],
                labels=['very_low', 'low', 'medium', 'high', 'extreme']
            )
        
        # Liquidity features
        if 'liquidity_depth' in df.columns:
            df['liquidity_log'] = np.log1p(df['liquidity_depth'])
            df['liquidity_inverse'] = 1 / (df['liquidity_depth'] + 1)
            
            # Liquidity categories
            df['liquidity_category'] = pd.cut(
                df['liquidity_depth'],
                bins=[0, 100000, 1000000, 10000000, float('inf')],
                labels=['low', 'medium', 'high', 'very_high']
            )
        
        # Volume features
        if 'volume_24h' in df.columns:
            df['volume_log'] = np.log1p(df['volume_24h'])
            df['volume_sqrt'] = np.sqrt(df['volume_24h'])
        
        # Spread features
        if 'spread_basis_points' in df.columns:
            df['spread_category'] = pd.cut(
                df['spread_basis_points'],
                bins=[0, 5, 10, 25, 50, float('inf')],
                labels=['tight', 'normal', 'wide', 'very_wide', 'extreme']
            )
        
        return df
    
    def engineer_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer risk-related features"""
        
        # Slippage features
        if 'slippage_percentage' in df.columns:
            df['slippage_squared'] = df['slippage_percentage'] ** 2
            df['high_slippage'] = (df['slippage_percentage'] > 2).astype(int)
            df['slippage_risk_score'] = df['slippage_percentage'] * df.get('market_volatility', 1)
        
        # Gas features
        if 'gas_cost' in df.columns and 'net_profit' in df.columns:
            df['gas_to_profit_ratio'] = df['gas_cost'] / (df['net_profit'].abs() + 1)
            df['gas_efficiency'] = df['net_profit'] / (df['gas_cost'] + 1)
            df['profitable_after_gas'] = (df['net_profit'] > 0).astype(int)
        
        # Risk scores
        if 'volatility_score' in df.columns:
            df['risk_category'] = pd.cut(
                df['volatility_score'],
                bins=[0, 20, 40, 60, 80, 100],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        # Execution risk
        if 'execution_time_ms' in df.columns:
            df['execution_risk'] = df['execution_time_ms'] / 1000  # Convert to seconds
            df['fast_execution'] = (df['execution_time_ms'] < 100).astype(int)
        
        return df
    
    def engineer_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer performance-related features"""
        
        # ROI features
        if 'roi_percentage' in df.columns:
            df['roi_log'] = np.sign(df['roi_percentage']) * np.log1p(np.abs(df['roi_percentage']))
            df['roi_squared'] = df['roi_percentage'] ** 2
            df['high_roi'] = (df['roi_percentage'] > 5).astype(int)
            
            # ROI buckets
            df['roi_bucket'] = pd.cut(
                df['roi_percentage'],
                bins=[-float('inf'), 0, 1, 5, 10, 20, float('inf')],
                labels=['negative', 'low', 'medium', 'good', 'high', 'exceptional']
            )
        
        # Profit features
        if 'net_profit' in df.columns:
            df['profit_log'] = np.sign(df['net_profit']) * np.log1p(np.abs(df['net_profit']))
            df['is_profitable'] = (df['net_profit'] > 0).astype(int)
            
            # Profit categories
            df['profit_category'] = pd.cut(
                df['net_profit'],
                bins=[-float('inf'), 0, 1000, 10000, 100000, float('inf')],
                labels=['loss', 'small', 'medium', 'large', 'whale']
            )
        
        # Sharpe ratio approximation
        if 'net_profit' in df.columns and 'volatility_score' in df.columns:
            df['pseudo_sharpe'] = df['net_profit'] / (df['volatility_score'] + 1)
        
        return df
    
    def engineer_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        
        interactions = []
        
        # Volatility * Liquidity interaction
        if 'market_volatility' in df.columns and 'liquidity_depth' in df.columns:
            df['volatility_liquidity_interaction'] = (
                df['market_volatility'] * np.log1p(df['liquidity_depth'])
            )
        
        # Path complexity * Gas cost
        if 'path_length' in df.columns and 'gas_cost' in df.columns:
            df['complexity_cost_interaction'] = df['path_length'] * df['gas_cost']
        
        # ROI * Risk interaction
        if 'roi_percentage' in df.columns and 'volatility_score' in df.columns:
            df['roi_risk_interaction'] = df['roi_percentage'] * df['volatility_score']
        
        # Time * Volatility interaction (captures time-varying volatility)
        if 'hour' in df.columns and 'market_volatility' in df.columns:
            df['time_volatility_interaction'] = df['hour'] * df['market_volatility']
        
        # Slippage * Volume interaction
        if 'slippage_percentage' in df.columns and 'volume_24h' in df.columns:
            df['slippage_volume_interaction'] = (
                df['slippage_percentage'] * np.log1p(df['volume_24h'])
            )
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str], 
                           lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Create lagged features for time series"""
        
        df = df.sort_values('block_timestamp')
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
                # Rolling statistics
                df[f'{col}_roll_mean_10'] = df[col].rolling(10, min_periods=1).mean()
                df[f'{col}_roll_std_10'] = df[col].rolling(10, min_periods=1).std()
                df[f'{col}_roll_max_10'] = df[col].rolling(10, min_periods=1).max()
                df[f'{col}_roll_min_10'] = df[col].rolling(10, min_periods=1).min()
        
        return df
    
    def apply_all_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        
        logger.info(f"Starting feature engineering on {len(df)} records")
        
        # Apply all engineering steps
        df = self.engineer_temporal_features(df)
        df = self.engineer_path_features(df)
        df = self.engineer_market_features(df)
        df = self.engineer_risk_features(df)
        df = self.engineer_performance_features(df)
        df = self.engineer_interaction_features(df)
        
        # Create lag features for important columns
        lag_columns = ['net_profit', 'roi_percentage', 'market_volatility', 'gas_cost']
        df = self.create_lag_features(df, lag_columns)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        logger.info(f"Feature engineering complete. Created {len(self.feature_names)} features")
        
        return df

class DataNormalizer:
    """Normalize and scale data for ML"""
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scalers = {}
        self.encoders = {}
    
    def fit_transform(self, df: pd.DataFrame, 
                     numeric_cols: List[str],
                     categorical_cols: List[str]) -> pd.DataFrame:
        """Fit and transform data"""
        
        # Handle numeric columns
        if numeric_cols:
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'robust':
                scaler = RobustScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.method}")
            
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['numeric'] = scaler
        
        # Handle categorical columns
        if categorical_cols:
            for col in categorical_cols:
                if col in df.columns:
                    # Use target encoding for high cardinality
                    if df[col].nunique() > 10:
                        encoder = ce.TargetEncoder()
                        if 'net_profit' in df.columns:
                            df[col] = encoder.fit_transform(df[col], df['net_profit'])
                        else:
                            # Fall back to ordinal encoding
                            encoder = ce.OrdinalEncoder()
                            df[col] = encoder.fit_transform(df[col])
                    else:
                        # Use one-hot for low cardinality
                        encoder = ce.OneHotEncoder(use_cat_names=True)
                        encoded = encoder.fit_transform(df[col])
                        df = pd.concat([df, encoded], axis=1)
                        df = df.drop(col, axis=1)
                    
                    self.encoders[col] = encoder
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scalers"""
        
        # Apply numeric scaling
        if 'numeric' in self.scalers:
            numeric_cols = [col for col in self.scalers['numeric'].feature_names_in_ 
                           if col in df.columns]
            df[numeric_cols] = self.scalers['numeric'].transform(df[numeric_cols])
        
        # Apply categorical encoding
        for col, encoder in self.encoders.items():
            if col in df.columns:
                if isinstance(encoder, ce.OneHotEncoder):
                    encoded = encoder.transform(df[col])
                    df = pd.concat([df, encoded], axis=1)
                    df = df.drop(col, axis=1)
                else:
                    df[col] = encoder.transform(df[col])
        
        return df
    
    def save(self, path: str):
        """Save fitted scalers and encoders"""
        joblib.dump({
            'scalers': self.scalers,
            'encoders': self.encoders,
            'method': self.method
        }, path)
    
    def load(self, path: str):
        """Load fitted scalers and encoders"""
        data = joblib.load(path)
        self.scalers = data['scalers']
        self.encoders = data['encoders']
        self.method = data['method']

class DataSplitter:
    """Advanced data splitting for time series"""
    
    def __init__(self, method: str = 'time_series'):
        self.method = method
    
    def split(self, df: pd.DataFrame, 
             target_col: str,
             test_size: float = 0.2,
             val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        
        if self.method == 'time_series':
            # Time-based split
            df = df.sort_values('block_timestamp')
            
            n = len(df)
            train_end = int(n * (1 - test_size - val_size))
            val_end = int(n * (1 - test_size))
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
        elif self.method == 'random':
            # Random split
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, random_state=42
            )
            
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_size/(1-test_size), random_state=42
            )
            
        elif self.method == 'stratified':
            # Stratified split based on profit categories
            if 'profit_category' in df.columns:
                stratify_col = 'profit_category'
            else:
                # Create profit bins for stratification
                df['profit_bin'] = pd.qcut(df[target_col], q=5, labels=False)
                stratify_col = 'profit_bin'
            
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, stratify=df[stratify_col], random_state=42
            )
            
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_size/(1-test_size), 
                stratify=train_val_df[stratify_col], random_state=42
            )
        
        else:
            raise ValueError(f"Unknown split method: {self.method}")
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_time_series_cv(self, df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time series cross-validation splits"""
        
        df = df.sort_values('block_timestamp')
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for train_idx, test_idx in tscv.split(df):
            splits.append((train_idx, test_idx))
        
        return splits

class ImbalanceHandler:
    """Handle imbalanced data for classification"""
    
    def __init__(self, method: str = 'smote'):
        self.method = method
        self.sampler = None
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Resample data to handle imbalance"""
        
        if self.method == 'smote':
            self.sampler = SMOTE(random_state=42)
        elif self.method == 'undersample':
            self.sampler = RandomUnderSampler(random_state=42)
        elif self.method == 'combined':
            # Combine over and under sampling
            from imblearn.combine import SMOTETomek
            self.sampler = SMOTETomek(random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        logger.info(f"Resampled data: {len(X)} -> {len(X_resampled)}")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

class FeatureSelector:
    """Select most important features"""
    
    def __init__(self, method: str = 'mutual_info', n_features: int = 50):
        self.method = method
        self.n_features = n_features
        self.selector = None
        self.selected_features = []
    
    def fit_select(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selector and transform data"""
        
        if self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_regression, k=self.n_features)
        elif self.method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.selector = RFE(estimator, n_features_to_select=self.n_features)
        elif self.method == 'pca':
            self.selector = PCA(n_components=self.n_features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        X_selected = self.selector.fit_transform(X, y)
        
        # Get selected feature names
        if self.method != 'pca':
            mask = self.selector.get_support()
            self.selected_features = X.columns[mask].tolist()
        else:
            self.selected_features = [f'pca_{i}' for i in range(self.n_features)]
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return pd.DataFrame(X_selected, columns=self.selected_features)

class MLDataExporter:
    """Export data in ML-ready formats"""
    
    def __init__(self):
        self.export_stats = {}
    
    async def export_to_parquet(self, df: pd.DataFrame, path: str, compression: str = 'snappy'):
        """Export to Parquet format"""
        
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, compression=compression)
        
        self.export_stats['parquet'] = {
            'path': path,
            'rows': len(df),
            'columns': len(df.columns),
            'size_mb': os.path.getsize(path) / 1024 / 1024
        }
        
        logger.info(f"Exported {len(df)} rows to Parquet: {path}")
    
    async def export_to_hdf5(self, train_df: pd.DataFrame, 
                            val_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            path: str):
        """Export to HDF5 format for deep learning"""
        
        with h5py.File(path, 'w') as f:
            # Create datasets
            train_group = f.create_group('train')
            val_group = f.create_group('validation')
            test_group = f.create_group('test')
            
            # Save data
            for col in train_df.columns:
                train_group.create_dataset(col, data=train_df[col].values)
                val_group.create_dataset(col, data=val_df[col].values)
                test_group.create_dataset(col, data=test_df[col].values)
            
            # Save metadata
            f.attrs['train_size'] = len(train_df)
            f.attrs['val_size'] = len(val_df)
            f.attrs['test_size'] = len(test_df)
            f.attrs['features'] = list(train_df.columns)
        
        logger.info(f"Exported data to HDF5: {path}")
    
    async def export_to_tfrecord(self, df: pd.DataFrame, path: str):
        """Export to TFRecord format for TensorFlow"""
        
        import tensorflow as tf
        
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        
        with tf.io.TFRecordWriter(path) as writer:
            for _, row in df.iterrows():
                feature_dict = {}
                
                for col in df.columns:
                    if df[col].dtype == 'object':
                        feature_dict[col] = _bytes_feature(str(row[col]).encode())
                    else:
                        feature_dict[col] = _float_feature(float(row[col]))
                
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature_dict)
                )
                
                writer.write(example.SerializeToString())
        
        logger.info(f"Exported {len(df)} records to TFRecord: {path}")

# Main ML pipeline
class MLPipeline:
    """Complete ML data preparation pipeline"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.normalizer = DataNormalizer()
        self.splitter = DataSplitter()
        self.selector = FeatureSelector()
        self.exporter = MLDataExporter()
    
    async def prepare_data(self, 
                          df: pd.DataFrame,
                          target_col: str = 'net_profit',
                          task_type: str = 'regression') -> Dict[str, Any]:
        """Complete data preparation pipeline"""
        
        logger.info(f"Starting ML pipeline for {len(df)} records")
        
        # 1. Feature engineering
        df = self.feature_engineer.apply_all_engineering(df)
        
        # 2. Handle missing values
        df = df.fillna(0)  # Simple fill, could be more sophisticated
        
        # 3. Identify column types
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from features
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # 4. Normalize data
        df = self.normalizer.fit_transform(df, numeric_cols, categorical_cols)
        
        # 5. Split data
        train_df, val_df, test_df = self.splitter.split(df, target_col)
        
        # 6. Separate features and target
        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        
        X_val = val_df.drop(target_col, axis=1)
        y_val = val_df[target_col]
        
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]
        
        # 7. Feature selection
        X_train = self.selector.fit_select(X_train, y_train)
        X_val = X_val[self.selector.selected_features]
        X_test = X_test[self.selector.selected_features]
        
        # 8. Handle imbalance for classification
        if task_type == 'classification':
            handler = ImbalanceHandler()
            X_train, y_train = handler.fit_resample(X_train, y_train)
        
        # 9. Export data
        await self.exporter.export_to_parquet(
            pd.concat([X_train, y_train], axis=1),
            'data/train.parquet'
        )
        
        await self.exporter.export_to_parquet(
            pd.concat([X_val, y_val], axis=1),
            'data/validation.parquet'
        )
        
        await self.exporter.export_to_parquet(
            pd.concat([X_test, y_test], axis=1),
            'data/test.parquet'
        )
        
        return {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test),
            'features': self.selector.selected_features,
            'stats': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'n_features': len(self.selector.selected_features)
            }
        }

import os