#!/usr/bin/env python3
"""
XGBoost to Treelite ML Pipeline for Ultra-Fast Sandwich Detection
Target: <100μs inference latency
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import treelite
import treelite_runtime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
import clickhouse_driver
import joblib
import json
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SandwichMLPipeline:
    def __init__(self, clickhouse_host: str = "localhost"):
        self.ch_client = clickhouse_driver.Client(host=clickhouse_host)
        self.model = None
        self.predictor = None
        
    def extract_training_data(self, days: int = 7) -> pd.DataFrame:
        """Extract labeled training data from ClickHouse"""
        query = """
        SELECT 
            s.timestamp,
            s.confidence,
            s.expected_profit,
            s.gas_cost,
            s.tip_amount,
            JSONExtractFloat(s.features, 'input') as input_amount,
            JSONExtractFloat(s.features, 'output') as output_amount,
            JSONExtractFloat(s.features, 'reserves', 1) as reserve_a,
            JSONExtractFloat(s.features, 'reserves', 2) as reserve_b,
            JSONExtractFloat(s.features, 'gas_price') as gas_price,
            JSONExtractFloat(s.features, 'slippage') as slippage,
            JSONExtractFloat(s.features, 'depth') as mempool_depth,
            o.landed as label,
            o.actual_profit
        FROM mev_sandwich_db.mev_sandwich s
        JOIN mev_sandwich_db.bundle_outcomes o ON s.target_tx = o.tx_hash
        WHERE s.timestamp >= now() - INTERVAL {days} DAY
        AND o.landed IS NOT NULL
        """.format(days=days)
        
        data = self.ch_client.execute(query, with_column_types=True)
        df = pd.DataFrame(data[0], columns=[col[0] for col in data[1]])
        
        logger.info(f"Extracted {len(df)} training samples")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for sandwich detection"""
        
        # Price impact features
        df['price_impact'] = df['input_amount'] / (df['reserve_a'] + 1e-9)
        df['price_impact_sqrt'] = np.sqrt(df['price_impact'])
        df['price_impact_log'] = np.log1p(df['price_impact'])
        
        # Slippage features
        df['actual_slippage'] = 1 - (df['output_amount'] / (df['input_amount'] + 1e-9))
        df['slippage_ratio'] = df['slippage'] / (df['actual_slippage'] + 1e-9)
        
        # Pool features
        df['pool_ratio'] = df['reserve_a'] / (df['reserve_b'] + 1e-9)
        df['pool_ratio_log'] = np.log1p(df['pool_ratio'])
        df['pool_liquidity'] = df['reserve_a'] + df['reserve_b']
        df['pool_liquidity_log'] = np.log1p(df['pool_liquidity'])
        
        # Profitability features
        df['profit_ratio'] = df['expected_profit'] / (df['gas_cost'] + 1e-9)
        df['profit_per_gas'] = df['expected_profit'] / (df['gas_price'] + 1e-9)
        df['tip_ratio'] = df['tip_amount'] / (df['expected_profit'] + 1e-9)
        
        # Market microstructure
        df['mempool_congestion'] = df['mempool_depth'] / 1000
        df['gas_premium'] = df['gas_price'] / 50000  # Normalized by baseline
        
        # Interaction features
        df['impact_x_congestion'] = df['price_impact'] * df['mempool_congestion']
        df['profit_x_slippage'] = df['profit_ratio'] * df['actual_slippage']
        
        # Time features (if timestamp available)
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> xgb.Booster:
        """Train XGBoost model optimized for speed"""
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,           # Shallow for speed
            'min_child_weight': 10,   # Prevent overfitting
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eta': 0.1,
            'gamma': 0.1,
            'alpha': 0.1,             # L1 regularization
            'lambda': 1.0,            # L2 regularization
            'tree_method': 'hist',    # Fast histogram method
            'grow_policy': 'lossguide',
            'max_leaves': 31,         # Limit complexity
            'seed': 42
        }
        
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=evallist,
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        return model
    
    def compile_to_treelite(self, xgb_model: xgb.Booster, output_path: str):
        """Compile XGBoost to Treelite for <100μs inference"""
        
        # Convert to Treelite model
        tl_model = treelite.Model.from_xgboost(xgb_model)
        
        # Compile with aggressive optimizations
        tl_model.export_lib(
            toolchain='gcc',
            libpath=output_path,
            params={
                'parallel_comp': 8,          # Parallel compilation
                'optimize': 3,                # O3 optimization
                'native': True,               # Native CPU instructions
                'simd': True,                 # Enable SIMD
                'avx512': True,               # AVX512 if available
                'fast_math': True,            # Fast math operations
                'unroll': True,               # Loop unrolling
                'inline': True,               # Aggressive inlining
            },
            verbose=True
        )
        
        logger.info(f"Compiled model to {output_path}")
        
        # Test inference speed
        predictor = treelite_runtime.Predictor(output_path)
        
        # Warmup
        test_data = np.random.randn(1000, xgb_model.num_features()).astype(np.float32)
        for _ in range(100):
            _ = predictor.predict(test_data)
        
        # Benchmark
        import time
        start = time.perf_counter()
        for _ in range(1000):
            _ = predictor.predict(test_data[:1])  # Single sample
        elapsed = (time.perf_counter() - start) / 1000
        
        logger.info(f"Single inference latency: {elapsed*1e6:.2f}μs")
        
        if elapsed > 100e-6:
            logger.warning(f"Inference slower than 100μs target!")
        
        return predictor
    
    def train_pipeline(self):
        """Complete training pipeline"""
        
        # Extract data
        logger.info("Extracting training data...")
        df = self.extract_training_data(days=7)
        
        # Engineer features
        logger.info("Engineering features...")
        df = self.engineer_features(df)
        
        # Select features
        feature_cols = [
            'price_impact', 'price_impact_sqrt', 'price_impact_log',
            'actual_slippage', 'slippage_ratio',
            'pool_ratio', 'pool_ratio_log', 'pool_liquidity_log',
            'profit_ratio', 'profit_per_gas', 'tip_ratio',
            'mempool_congestion', 'gas_premium',
            'impact_x_congestion', 'profit_x_slippage',
            'hour_sin', 'hour_cos'
        ]
        
        X = df[feature_cols].values.astype(np.float32)
        y = df['label'].values.astype(np.float32)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        logger.info(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        # Train model
        logger.info("Training XGBoost model...")
        model = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Evaluate
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        auc = roc_auc_score(y_test, y_pred)
        logger.info(f"Test AUC: {auc:.4f}")
        
        # Save XGBoost model
        model.save_model('/home/kidgordones/0solana/node/mev-sandwich-detector/models/sandwich_xgb.json')
        
        # Compile to Treelite
        logger.info("Compiling to Treelite for ultra-fast inference...")
        predictor = self.compile_to_treelite(
            model, 
            '/home/kidgordones/0solana/node/mev-sandwich-detector/models/sandwich_treelite.so'
        )
        
        # Save feature names and stats
        feature_info = {
            'features': feature_cols,
            'auc': float(auc),
            'num_trees': model.num_boosted_rounds(),
            'num_features': len(feature_cols)
        }
        
        with open('/home/kidgordones/0solana/node/mev-sandwich-detector/models/feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info("Training pipeline complete!")
        
        return model, predictor
    
    def continuous_retraining(self, interval_hours: int = 6):
        """Continuous model retraining from ClickHouse data"""
        import schedule
        import time
        
        def retrain_job():
            try:
                logger.info("Starting scheduled retraining...")
                self.train_pipeline()
                logger.info("Retraining complete")
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
        
        # Schedule retraining
        schedule.every(interval_hours).hours.do(retrain_job)
        
        logger.info(f"Starting continuous retraining every {interval_hours} hours")
        
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    pipeline = SandwichMLPipeline()
    
    # Initial training
    pipeline.train_pipeline()
    
    # Start continuous retraining
    # pipeline.continuous_retraining(interval_hours=6)