"""
Ultra-High-Performance Data Preprocessing Pipeline for MEV Arbitrage Detection
Engineered for sub-millisecond processing with intelligent feature engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import category_encoders as ce
from dataclasses import dataclass
import joblib
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq
import numba
from numba import jit, prange
import hashlib
import redis
import pickle
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    
    # Feature engineering
    create_interaction_features: bool = True
    create_lag_features: bool = True
    create_rolling_features: bool = True
    lag_periods: List[int] = None
    rolling_windows: List[int] = None
    
    # Normalization
    numerical_scaler: str = 'robust'  # 'standard', 'robust', 'minmax'
    categorical_encoder: str = 'target'  # 'onehot', 'target', 'catboost'
    
    # Imputation
    numerical_imputer: str = 'knn'  # 'mean', 'median', 'knn', 'iterative'
    categorical_imputer: str = 'mode'  # 'mode', 'constant'
    
    # Outlier detection
    outlier_method: str = 'isolation_forest'  # 'isolation_forest', 'lof', 'zscore'
    outlier_threshold: float = 0.01
    
    # Feature selection
    feature_selection_method: str = 'mutual_info'  # 'mutual_info', 'chi2', 'anova'
    n_features_to_select: int = 50
    
    # Performance
    use_gpu: bool = True
    cache_preprocessed: bool = True
    cache_ttl: int = 300  # seconds
    n_jobs: int = -1
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 5, 10, 20, 50]
        if self.rolling_windows is None:
            self.rolling_windows = [3, 5, 10, 20, 50, 100]


class AdvancedPreprocessor:
    """
    Production-grade data preprocessor with ultra-low latency and intelligent feature engineering
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selector = None
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.redis_client = None
        self._init_cache()
        
    def _init_cache(self):
        """Initialize Redis cache for preprocessed data"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=1,
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info("Redis cache initialized for preprocessing")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.redis_client = None
    
    @jit(nopython=True, parallel=True)
    def _fast_rolling_stats(self, data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ultra-fast rolling statistics using Numba"""
        n = len(data)
        rolling_mean = np.zeros(n)
        rolling_std = np.zeros(n)
        rolling_max = np.zeros(n)
        
        for i in prange(window, n):
            window_data = data[i-window:i]
            rolling_mean[i] = np.mean(window_data)
            rolling_std[i] = np.std(window_data)
            rolling_max[i] = np.max(window_data)
        
        return rolling_mean, rolling_std, rolling_max
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for MEV arbitrage detection
        """
        logger.info("Engineering advanced features...")
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
            df['second'] = pd.to_datetime(df['timestamp']).dt.second
            
            # Block-based features
            df['blocks_since_last'] = df.groupby('amm')['block_number'].diff()
            df['time_since_last'] = pd.to_datetime(df['timestamp']).diff().dt.total_seconds()
        
        # Price and volume features
        if 'revenue_sol' in df.columns:
            df['log_revenue'] = np.log1p(df['revenue_sol'])
            df['revenue_squared'] = df['revenue_sol'] ** 2
            df['revenue_sqrt'] = np.sqrt(np.abs(df['revenue_sol']))
        
        if 'roi' in df.columns:
            df['roi_percentile'] = df['roi'].rank(pct=True)
            df['roi_zscore'] = (df['roi'] - df['roi'].mean()) / df['roi'].std()
            df['roi_categories'] = pd.qcut(df['roi'], q=10, labels=False, duplicates='drop')
        
        if 'slippage' in df.columns:
            df['slippage_impact'] = df['slippage'] * df.get('volume', 1)
            df['slippage_severity'] = pd.cut(
                df['slippage'],
                bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
                labels=['minimal', 'low', 'medium', 'high']
            )
        
        # DEX-specific features
        if 'dex' in df.columns:
            dex_stats = df.groupby('dex').agg({
                'revenue_sol': ['mean', 'std', 'max'],
                'roi': ['mean', 'std'],
                'slippage': ['mean', 'std']
            }).fillna(0)
            
            for col in dex_stats.columns:
                feature_name = f"dex_{col[0]}_{col[1]}"
                df[feature_name] = df['dex'].map(dex_stats[col])
        
        # AMM-specific features
        if 'amm' in df.columns:
            amm_stats = df.groupby('amm').agg({
                'revenue_sol': ['mean', 'std', 'count'],
                'roi': ['mean', 'std']
            }).fillna(0)
            
            for col in amm_stats.columns:
                feature_name = f"amm_{col[0]}_{col[1]}"
                df[feature_name] = df['amm'].map(amm_stats[col])
        
        # Lag features for time series patterns
        if self.config.create_lag_features:
            for col in ['revenue_sol', 'roi', 'slippage']:
                if col in df.columns:
                    for lag in self.config.lag_periods:
                        df[f"{col}_lag_{lag}"] = df.groupby('amm')[col].shift(lag)
                        df[f"{col}_diff_{lag}"] = df[col] - df[f"{col}_lag_{lag}"]
                        df[f"{col}_pct_change_{lag}"] = df.groupby('amm')[col].pct_change(lag)
        
        # Rolling window features
        if self.config.create_rolling_features:
            for col in ['revenue_sol', 'roi', 'slippage']:
                if col in df.columns:
                    for window in self.config.rolling_windows:
                        df[f"{col}_rolling_mean_{window}"] = df.groupby('amm')[col].transform(
                            lambda x: x.rolling(window, min_periods=1).mean()
                        )
                        df[f"{col}_rolling_std_{window}"] = df.groupby('amm')[col].transform(
                            lambda x: x.rolling(window, min_periods=1).std()
                        )
                        df[f"{col}_rolling_max_{window}"] = df.groupby('amm')[col].transform(
                            lambda x: x.rolling(window, min_periods=1).max()
                        )
                        df[f"{col}_rolling_min_{window}"] = df.groupby('amm')[col].transform(
                            lambda x: x.rolling(window, min_periods=1).min()
                        )
        
        # Interaction features
        if self.config.create_interaction_features:
            if 'revenue_sol' in df.columns and 'roi' in df.columns:
                df['revenue_roi_interaction'] = df['revenue_sol'] * df['roi']
                df['revenue_per_roi'] = df['revenue_sol'] / (df['roi'] + 1e-8)
            
            if 'slippage' in df.columns and 'roi' in df.columns:
                df['slippage_roi_ratio'] = df['slippage'] / (df['roi'] + 1e-8)
                df['effective_roi'] = df['roi'] * (1 - df['slippage'])
        
        # MEV-specific features
        df['potential_profit'] = df.get('revenue_sol', 0) * (1 - df.get('slippage', 0))
        df['risk_score'] = df.get('slippage', 0) * df.get('gas_cost', 1) / (df.get('revenue_sol', 1) + 1e-8)
        
        # Statistical aggregations per block
        if 'block_number' in df.columns:
            block_stats = df.groupby('block_number').agg({
                'revenue_sol': ['sum', 'mean', 'std', 'count'],
                'roi': ['mean', 'std'],
                'slippage': ['mean', 'max']
            }).fillna(0)
            
            for col in block_stats.columns:
                feature_name = f"block_{col[0]}_{col[1]}"
                df[feature_name] = df['block_number'].map(block_stats[col])
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligent missing value imputation
        """
        logger.info("Handling missing values...")
        
        # Separate features by type
        self.numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numerical imputation
        if self.numerical_features:
            if self.config.numerical_imputer == 'knn':
                imputer = KNNImputer(n_neighbors=5, weights='distance')
            elif self.config.numerical_imputer == 'iterative':
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                imputer = IterativeImputer(random_state=42, max_iter=10)
            else:
                imputer = SimpleImputer(strategy=self.config.numerical_imputer)
            
            df[self.numerical_features] = imputer.fit_transform(df[self.numerical_features])
            self.imputers['numerical'] = imputer
        
        # Categorical imputation
        if self.categorical_features:
            if self.config.categorical_imputer == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                imputer = SimpleImputer(strategy='constant', fill_value='missing')
            
            df[self.categorical_features] = imputer.fit_transform(df[self.categorical_features])
            self.imputers['categorical'] = imputer
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Advanced feature normalization with multiple scaler options
        """
        logger.info("Normalizing features...")
        
        if not self.numerical_features:
            return df
        
        # Select scaler based on config
        if self.config.numerical_scaler == 'standard':
            scaler = StandardScaler()
        elif self.config.numerical_scaler == 'robust':
            scaler = RobustScaler(quantile_range=(5, 95))
        elif self.config.numerical_scaler == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            raise ValueError(f"Unknown scaler: {self.config.numerical_scaler}")
        
        if fit:
            df[self.numerical_features] = scaler.fit_transform(df[self.numerical_features])
            self.scalers['numerical'] = scaler
        else:
            df[self.numerical_features] = self.scalers['numerical'].transform(df[self.numerical_features])
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, target: Optional[pd.Series] = None, fit: bool = True) -> pd.DataFrame:
        """
        Advanced categorical encoding with multiple encoder options
        """
        logger.info("Encoding categorical features...")
        
        if not self.categorical_features:
            return df
        
        # Select encoder based on config
        if self.config.categorical_encoder == 'target' and target is not None:
            encoder = ce.TargetEncoder(cols=self.categorical_features, smoothing=0.3)
            if fit:
                df[self.categorical_features] = encoder.fit_transform(df[self.categorical_features], target)
                self.encoders['categorical'] = encoder
            else:
                df[self.categorical_features] = self.encoders['categorical'].transform(df[self.categorical_features])
        
        elif self.config.categorical_encoder == 'catboost':
            encoder = ce.CatBoostEncoder(cols=self.categorical_features)
            if fit:
                df[self.categorical_features] = encoder.fit_transform(df[self.categorical_features], target)
                self.encoders['categorical'] = encoder
            else:
                df[self.categorical_features] = self.encoders['categorical'].transform(df[self.categorical_features])
        
        else:  # One-hot encoding
            df = pd.get_dummies(df, columns=self.categorical_features, drop_first=True)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced outlier detection and handling
        """
        logger.info("Detecting outliers...")
        
        if self.config.outlier_method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            detector = IsolationForest(
                contamination=self.config.outlier_threshold,
                random_state=42,
                n_jobs=self.config.n_jobs
            )
            outliers = detector.fit_predict(df[self.numerical_features])
            df['is_outlier'] = (outliers == -1).astype(int)
        
        elif self.config.outlier_method == 'lof':
            from sklearn.neighbors import LocalOutlierFactor
            detector = LocalOutlierFactor(
                contamination=self.config.outlier_threshold,
                n_neighbors=20,
                n_jobs=self.config.n_jobs
            )
            outliers = detector.fit_predict(df[self.numerical_features])
            df['is_outlier'] = (outliers == -1).astype(int)
        
        elif self.config.outlier_method == 'zscore':
            z_scores = np.abs((df[self.numerical_features] - df[self.numerical_features].mean()) / df[self.numerical_features].std())
            df['is_outlier'] = (z_scores > 3).any(axis=1).astype(int)
        
        # Mark high-value arbitrage opportunities (not necessarily outliers to remove)
        if 'revenue_sol' in df.columns:
            df['high_value_opportunity'] = (df['revenue_sol'] > df['revenue_sol'].quantile(0.95)).astype(int)
        
        return df
    
    def select_features(self, df: pd.DataFrame, target: pd.Series, fit: bool = True) -> pd.DataFrame:
        """
        Advanced feature selection using mutual information and other methods
        """
        logger.info("Selecting best features...")
        
        if fit:
            if self.config.feature_selection_method == 'mutual_info':
                selector = SelectKBest(
                    score_func=mutual_info_regression,
                    k=min(self.config.n_features_to_select, df.shape[1])
                )
            else:
                from sklearn.feature_selection import f_regression
                selector = SelectKBest(
                    score_func=f_regression,
                    k=min(self.config.n_features_to_select, df.shape[1])
                )
            
            df_selected = selector.fit_transform(df, target)
            self.feature_selector = selector
            
            # Get selected feature names
            selected_mask = selector.get_support()
            self.feature_names = df.columns[selected_mask].tolist()
            
            return pd.DataFrame(df_selected, columns=self.feature_names, index=df.index)
        else:
            if self.feature_selector:
                df_selected = self.feature_selector.transform(df)
                return pd.DataFrame(df_selected, columns=self.feature_names, index=df.index)
            return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality validation
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numerical_stats': {},
            'categorical_stats': {},
            'quality_score': 0.0
        }
        
        # Numerical statistics
        for col in self.numerical_features:
            if col in df.columns:
                validation_results['numerical_stats'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'null_percentage': df[col].isnull().mean() * 100
                }
        
        # Categorical statistics
        for col in self.categorical_features:
            if col in df.columns:
                validation_results['categorical_stats'][col] = {
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].value_counts().head(5).to_dict(),
                    'null_percentage': df[col].isnull().mean() * 100
                }
        
        # Calculate quality score
        null_penalty = df.isnull().mean().mean() * 100
        validation_results['quality_score'] = max(0, 100 - null_penalty)
        
        return validation_results
    
    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Full preprocessing pipeline - fit and transform
        """
        start_time = datetime.now()
        logger.info(f"Starting preprocessing pipeline with {len(df)} rows...")
        
        # Check cache if enabled
        if self.config.cache_preprocessed and self.redis_client:
            cache_key = self._generate_cache_key(df)
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info("Retrieved preprocessed data from cache")
                return cached_data
        
        # Execute preprocessing steps
        df = self.engineer_features(df)
        df = self.handle_missing_values(df)
        df = self.normalize_features(df, fit=True)
        df = self.encode_categorical(df, target, fit=True)
        df = self.detect_outliers(df)
        
        if target is not None:
            df = self.select_features(df, target, fit=True)
        
        # Validate final data
        validation_results = self.validate_data(df)
        logger.info(f"Data quality score: {validation_results['quality_score']:.2f}%")
        
        # Cache if enabled
        if self.config.cache_preprocessed and self.redis_client:
            self._save_to_cache(cache_key, df)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Preprocessing completed in {processing_time:.2f} seconds")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors
        """
        start_time = datetime.now()
        
        # Check cache
        if self.config.cache_preprocessed and self.redis_client:
            cache_key = self._generate_cache_key(df)
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Execute preprocessing steps
        df = self.engineer_features(df)
        df = self.handle_missing_values(df)
        df = self.normalize_features(df, fit=False)
        df = self.encode_categorical(df, fit=False)
        df = self.detect_outliers(df)
        
        if self.feature_selector:
            df = self.select_features(df, None, fit=False)
        
        # Cache if enabled
        if self.config.cache_preprocessed and self.redis_client:
            self._save_to_cache(cache_key, df)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Transform completed in {processing_time:.2f}ms")
        
        return df
    
    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """Generate unique cache key for dataframe"""
        df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
        return f"preprocessed:{df_hash}"
    
    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve preprocessed data from cache"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _save_to_cache(self, key: str, df: pd.DataFrame):
        """Save preprocessed data to cache"""
        if not self.redis_client:
            return
        
        try:
            serialized = pickle.dumps(df)
            self.redis_client.setex(key, self.config.cache_ttl, serialized)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def save_preprocessor(self, path: str):
        """Save fitted preprocessor to disk"""
        preprocessor_state = {
            'config': self.config,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }
        joblib.dump(preprocessor_state, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str):
        """Load fitted preprocessor from disk"""
        preprocessor_state = joblib.load(path)
        self.config = preprocessor_state['config']
        self.scalers = preprocessor_state['scalers']
        self.encoders = preprocessor_state['encoders']
        self.imputers = preprocessor_state['imputers']
        self.feature_selector = preprocessor_state['feature_selector']
        self.feature_names = preprocessor_state['feature_names']
        self.categorical_features = preprocessor_state['categorical_features']
        self.numerical_features = preprocessor_state['numerical_features']
        logger.info(f"Preprocessor loaded from {path}")