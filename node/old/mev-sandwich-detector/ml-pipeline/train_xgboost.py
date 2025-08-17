#!/usr/bin/env python3
"""
MEV Sandwich Detector - XGBoost Training Pipeline
Trains model from ClickHouse data and compiles to Treelite for <100μs inference
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import treelite
import treelite_runtime
from clickhouse_driver import Client
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
import logging
from datetime import datetime, timedelta
import optuna
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SandwichModelTrainer:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.client = Client(
            host=self.config['clickhouse']['host'],
            port=self.config['clickhouse']['port'],
            database='mev_sandwich'
        )
        
        self.feature_columns = self.config['features']['columns']
        self.target_column = self.config['features']['target']
        
    def fetch_training_data(self, days_back: int = 7) -> pd.DataFrame:
        """Fetch training data from ClickHouse"""
        logger.info(f"Fetching {days_back} days of training data from ClickHouse")
        
        query = f"""
        SELECT
            timestamp,
            decision_time_us,
            
            -- Core features
            JSONExtractFloat(features, 'input') AS input_amount,
            JSONExtractFloat(features, 'output') AS output_amount,
            JSONExtractFloat(features, 'gas_price') AS gas_price,
            JSONExtractFloat(features, 'slippage') AS slippage,
            JSONExtractUInt(features, 'depth') AS mempool_depth,
            
            -- Pool features
            JSONExtractFloat(features, 'reserves.0') AS reserve_0,
            JSONExtractFloat(features, 'reserves.1') AS reserve_1,
            
            -- Time features
            toHour(timestamp) AS hour_of_day,
            toDayOfWeek(timestamp) AS day_of_week,
            
            -- Derived features
            input_amount / NULLIF(output_amount, 0) AS price_ratio,
            input_amount / NULLIF(reserve_0, 0) AS input_to_reserve_ratio,
            gas_cost / NULLIF(expected_profit, 0) AS gas_to_profit_ratio,
            
            -- Context features
            confidence,
            priority,
            tip_amount,
            tip_amount / NULLIF(expected_profit, 0) AS tip_ratio,
            
            -- Target
            landed AS label,
            net_profit
            
        FROM mev_sandwich
        WHERE 
            date >= today() - {days_back}
            AND landed IS NOT NULL
            AND expected_profit > 0
            AND gas_cost > 0
        ORDER BY timestamp
        """
        
        df = pd.DataFrame(self.client.execute(query))
        df.columns = [
            'timestamp', 'decision_time_us', 'input_amount', 'output_amount',
            'gas_price', 'slippage', 'mempool_depth', 'reserve_0', 'reserve_1',
            'hour_of_day', 'day_of_week', 'price_ratio', 'input_to_reserve_ratio',
            'gas_to_profit_ratio', 'confidence', 'priority', 'tip_amount',
            'tip_ratio', 'label', 'net_profit'
        ]
        
        logger.info(f"Fetched {len(df)} samples, {df['label'].sum()} positive")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        logger.info("Engineering features")
        
        # Rolling statistics
        df['input_amount_rolling_mean'] = df['input_amount'].rolling(100, min_periods=1).mean()
        df['gas_price_rolling_std'] = df['gas_price'].rolling(100, min_periods=1).std()
        
        # Lag features
        for col in ['input_amount', 'gas_price', 'mempool_depth']:
            for lag in [1, 5, 10]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Interaction features
        df['input_gas_interaction'] = df['input_amount'] * df['gas_price']
        df['reserve_imbalance'] = abs(df['reserve_0'] - df['reserve_1']) / (df['reserve_0'] + df['reserve_1'])
        
        # Time-based features
        df['is_peak_hour'] = df['hour_of_day'].isin([8, 9, 10, 14, 15, 16]).astype(int)
        df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)
        
        # Market microstructure
        df['price_impact'] = df['input_amount'] / (df['reserve_0'] + df['input_amount'])
        df['effective_slippage'] = df['slippage'] * (1 + df['price_impact'])
        
        # Fill NaN values
        df = df.fillna(0)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val) -> Dict:
        """Optimize XGBoost hyperparameters using Optuna"""
        logger.info("Starting hyperparameter optimization")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
                'tree_method': 'hist',
                'device': 'cuda',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, n_jobs=4)
        
        logger.info(f"Best AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def train_model(self, df: pd.DataFrame) -> xgb.XGBClassifier:
        """Train XGBoost model with optimized parameters"""
        logger.info("Training XGBoost model")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'label', 'net_profit']]
        X = df[feature_cols]
        y = df['label']
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        best_model = None
        best_score = 0
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Optimize hyperparameters
            best_params = self.optimize_hyperparameters(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Train final model
            model = xgb.XGBClassifier(**best_params)
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=50,
                verbose=True
            )
            
            # Evaluate
            y_pred = model.predict_proba(X_val_scaled)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            
            if auc > best_score:
                best_score = auc
                best_model = model
                self.scaler = scaler
                self.feature_names = feature_cols
        
        logger.info(f"Best cross-validation AUC: {best_score:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 important features:")
        print(importance.head(10))
        
        return best_model
    
    def compile_to_treelite(self, model: xgb.XGBClassifier, output_path: str):
        """Compile XGBoost model to Treelite for ultra-fast inference"""
        logger.info("Compiling model to Treelite")
        
        # Convert to Treelite
        tl_model = treelite.Model.from_xgboost(model.get_booster())
        
        # Compile with aggressive optimizations
        tl_model.export_lib(
            toolchain='gcc',
            libpath=output_path,
            params={
                'parallel_comp': 32,  # Parallel compilation
                'optimize': 3,         # -O3 optimization
                'native': True,        # CPU-specific optimizations
                'fast_math': True,     # Fast math operations
                'prefetch': True,      # Enable prefetching
            },
            verbose=True
        )
        
        # Test inference speed
        predictor = treelite_runtime.Predictor(output_path)
        
        # Generate test data
        test_data = np.random.randn(1000, len(self.feature_names)).astype(np.float32)
        
        # Benchmark
        import time
        start = time.perf_counter()
        for _ in range(1000):
            _ = predictor.predict(treelite_runtime.DMatrix(test_data))
        elapsed = (time.perf_counter() - start) / 1000
        
        logger.info(f"Treelite inference time: {elapsed*1e6:.2f} microseconds per batch")
        
        if elapsed * 1e6 > 100:
            logger.warning("Inference time exceeds 100μs target!")
    
    def export_model(self, model: xgb.XGBClassifier):
        """Export model and artifacts"""
        logger.info("Exporting model artifacts")
        
        # Save XGBoost model
        model.save_model('../models/sandwich_xgboost.json')
        
        # Save scaler
        joblib.dump(self.scaler, '../models/scaler.pkl')
        
        # Save feature names
        with open('../models/features.yaml', 'w') as f:
            yaml.dump({
                'features': self.feature_names,
                'n_features': len(self.feature_names),
                'model_version': datetime.now().strftime('%Y%m%d_%H%M%S')
            }, f)
        
        # Compile to Treelite
        self.compile_to_treelite(model, '../models/sandwich_treelite.so')
        
        # Convert to ONNX for GPU inference
        try:
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            initial_types = [('input', FloatTensorType([None, len(self.feature_names)]))]
            onnx_model = convert_xgboost(model, initial_types=initial_types)
            
            with open('../models/sandwich_model.onnx', 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info("Model exported to ONNX format")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
    
    def validate_model(self, model: xgb.XGBClassifier, df: pd.DataFrame):
        """Validate model performance"""
        logger.info("Validating model performance")
        
        # Use last 20% for final validation
        split_idx = int(len(df) * 0.8)
        df_val = df.iloc[split_idx:]
        
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'label', 'net_profit']]
        X_val = self.scaler.transform(df_val[feature_cols])
        y_val = df_val['label']
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.75).astype(int)  # High threshold for precision
        
        # Metrics
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"\nROC AUC: {auc:.4f}")
        
        # Profit analysis
        df_val['predicted'] = y_pred
        df_val['pred_proba'] = y_pred_proba
        
        # Only trade when confident
        confident_trades = df_val[df_val['pred_proba'] > 0.75]
        
        if len(confident_trades) > 0:
            total_profit = confident_trades[confident_trades['predicted'] == 1]['net_profit'].sum()
            n_trades = len(confident_trades[confident_trades['predicted'] == 1])
            
            print(f"\nProfit Analysis (confidence > 0.75):")
            print(f"Total profit: {total_profit / 1e9:.4f} SOL")
            print(f"Number of trades: {n_trades}")
            print(f"Average profit per trade: {total_profit / n_trades / 1e9:.6f} SOL")
    
    def run_pipeline(self):
        """Run complete training pipeline"""
        logger.info("Starting MEV Sandwich model training pipeline")
        
        # Fetch data
        df = self.fetch_training_data(days_back=14)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Train model
        model = self.train_model(df)
        
        # Validate
        self.validate_model(model, df)
        
        # Export
        self.export_model(model)
        
        logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    # Configuration
    config = {
        'clickhouse': {
            'host': 'localhost',
            'port': 9000
        },
        'features': {
            'columns': [
                'input_amount', 'output_amount', 'gas_price', 'slippage',
                'mempool_depth', 'reserve_0', 'reserve_1', 'confidence',
                'priority', 'tip_amount'
            ],
            'target': 'landed'
        }
    }
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Run training
    trainer = SandwichModelTrainer()
    trainer.run_pipeline()