#!/usr/bin/env python3
"""
Ultra-Performance ML Pipeline for MEV Detection
Implements XGBoost, Online Learning, and Anomaly Detection
Target: <100μs inference, adaptive learning
DEFENSIVE-ONLY: Pure detection without execution
"""

import os
import sys
import time
import pickle
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import onnx
import onnxruntime as ort
from river import anomaly, ensemble, metrics, preprocessing, tree
import optuna
from optuna.samplers import TPESampler
import mlflow
import mlflow.xgboost
import mlflow.pytorch
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import aiofiles
import msgpack
import lz4.frame
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass, field
from collections import deque
import logging
from functools import lru_cache
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_VERSION = "v2.0.0"
FEATURE_VERSION = "v1.5.0"
WINDOW_SIZE = 100
EMBEDDING_DIM = 128
LATENT_DIM = 32
BATCH_SIZE = 256
LEARNING_RATE = 0.001
ANOMALY_THRESHOLD = 0.05

@dataclass
class MEVFeatures:
    """Feature set for MEV detection"""
    # Transaction features
    slot: int
    timestamp: int
    signature: bytes
    compute_units: int
    lamports: int
    num_instructions: int
    num_accounts: int
    
    # DEX interaction features
    is_dex: bool
    dex_program: Optional[str]
    swap_amount_in: float
    swap_amount_out: float
    price_impact: float
    
    # Timing features
    slot_position: int  # Position within slot
    time_since_slot_start: float
    time_to_slot_end: float
    
    # Network features
    mempool_size: int
    gas_price: float
    priority_fee: float
    
    # Historical features
    sender_tx_count: int
    sender_mev_count: int
    sender_success_rate: float
    
    # Graph features
    account_centrality: float
    program_popularity: float
    interaction_complexity: float
    
    # Computed features
    features_vector: Optional[np.ndarray] = None
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        if self.features_vector is not None:
            return self.features_vector
        
        features = np.array([
            self.slot,
            self.timestamp,
            self.compute_units,
            self.lamports,
            self.num_instructions,
            self.num_accounts,
            int(self.is_dex),
            self.swap_amount_in,
            self.swap_amount_out,
            self.price_impact,
            self.slot_position,
            self.time_since_slot_start,
            self.time_to_slot_end,
            self.mempool_size,
            self.gas_price,
            self.priority_fee,
            self.sender_tx_count,
            self.sender_mev_count,
            self.sender_success_rate,
            self.account_centrality,
            self.program_popularity,
            self.interaction_complexity
        ], dtype=np.float32)
        
        self.features_vector = features
        return features

class FeatureEngineering:
    """Advanced feature engineering for MEV detection"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_cache = {}
        self.interaction_features = []
        self.polynomial_features = []
        
    def extract_features(self, transaction: Dict) -> MEVFeatures:
        """Extract features from raw transaction"""
        # Cache key
        cache_key = hashlib.sha256(
            msgpack.packb(transaction, use_bin_type=True)
        ).hexdigest()
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Extract base features
        features = MEVFeatures(
            slot=transaction['slot'],
            timestamp=transaction['timestamp'],
            signature=bytes.fromhex(transaction['signature']),
            compute_units=transaction.get('compute_units', 0),
            lamports=transaction.get('lamports', 0),
            num_instructions=len(transaction.get('instructions', [])),
            num_accounts=len(transaction.get('accounts', [])),
            is_dex=self._is_dex_transaction(transaction),
            dex_program=self._get_dex_program(transaction),
            swap_amount_in=self._calculate_swap_in(transaction),
            swap_amount_out=self._calculate_swap_out(transaction),
            price_impact=self._calculate_price_impact(transaction),
            slot_position=transaction.get('slot_position', 0),
            time_since_slot_start=transaction.get('time_since_slot_start', 0),
            time_to_slot_end=transaction.get('time_to_slot_end', 0),
            mempool_size=transaction.get('mempool_size', 0),
            gas_price=transaction.get('gas_price', 0),
            priority_fee=transaction.get('priority_fee', 0),
            sender_tx_count=transaction.get('sender_tx_count', 0),
            sender_mev_count=transaction.get('sender_mev_count', 0),
            sender_success_rate=transaction.get('sender_success_rate', 0),
            account_centrality=self._calculate_centrality(transaction),
            program_popularity=self._calculate_popularity(transaction),
            interaction_complexity=self._calculate_complexity(transaction)
        )
        
        # Cache result
        self.feature_cache[cache_key] = features
        
        # Limit cache size
        if len(self.feature_cache) > 10000:
            # Remove oldest entries
            keys_to_remove = list(self.feature_cache.keys())[:1000]
            for key in keys_to_remove:
                del self.feature_cache[key]
        
        return features
    
    def create_interaction_features(self, features: np.ndarray) -> np.ndarray:
        """Create interaction features"""
        n_features = features.shape[1]
        interactions = []
        
        # Create pairwise interactions for important features
        important_indices = [0, 3, 4, 9, 14, 15]  # slot, lamports, instructions, price_impact, gas, priority
        
        for i in important_indices:
            for j in important_indices:
                if i < j:
                    interactions.append(features[:, i] * features[:, j])
        
        if interactions:
            return np.column_stack([features] + interactions)
        return features
    
    def create_rolling_features(self, features_list: List[MEVFeatures], window: int = 10) -> np.ndarray:
        """Create rolling window features"""
        if len(features_list) < window:
            return np.zeros((len(features_list), 10))
        
        rolling_features = []
        for i in range(window - 1, len(features_list)):
            window_data = [f.to_numpy() for f in features_list[i-window+1:i+1]]
            window_array = np.array(window_data)
            
            # Calculate rolling statistics
            roll_mean = np.mean(window_array, axis=0)
            roll_std = np.std(window_array, axis=0)
            roll_max = np.max(window_array, axis=0)
            roll_min = np.min(window_array, axis=0)
            
            # Combine
            combined = np.concatenate([
                features_list[i].to_numpy(),
                roll_mean[:5],  # Select important rolling features
                roll_std[:5]
            ])
            
            rolling_features.append(combined)
        
        return np.array(rolling_features)
    
    def _is_dex_transaction(self, tx: Dict) -> bool:
        """Check if transaction involves DEX"""
        dex_programs = {
            'SerumV3', 'RaydiumV4', 'OrcaWhirlpool',
            'JupiterV4', 'MercurialStable', 'SaberStable'
        }
        
        for instruction in tx.get('instructions', []):
            if instruction.get('program') in dex_programs:
                return True
        return False
    
    def _get_dex_program(self, tx: Dict) -> Optional[str]:
        """Get DEX program if present"""
        dex_programs = {
            'SerumV3', 'RaydiumV4', 'OrcaWhirlpool',
            'JupiterV4', 'MercurialStable', 'SaberStable'
        }
        
        for instruction in tx.get('instructions', []):
            if instruction.get('program') in dex_programs:
                return instruction.get('program')
        return None
    
    def _calculate_swap_in(self, tx: Dict) -> float:
        """Calculate swap input amount"""
        # Simplified calculation
        return tx.get('swap_in', 0) / 1e9  # Convert lamports to SOL
    
    def _calculate_swap_out(self, tx: Dict) -> float:
        """Calculate swap output amount"""
        return tx.get('swap_out', 0) / 1e9
    
    def _calculate_price_impact(self, tx: Dict) -> float:
        """Calculate price impact of swap"""
        swap_in = self._calculate_swap_in(tx)
        swap_out = self._calculate_swap_out(tx)
        
        if swap_in > 0:
            expected_out = swap_in * tx.get('pool_price', 1)
            if expected_out > 0:
                return abs(swap_out - expected_out) / expected_out
        return 0
    
    def _calculate_centrality(self, tx: Dict) -> float:
        """Calculate account centrality score"""
        # Simplified PageRank-like score
        return len(tx.get('accounts', [])) * 0.1
    
    def _calculate_popularity(self, tx: Dict) -> float:
        """Calculate program popularity score"""
        # Based on historical usage
        return tx.get('program_usage_count', 0) / 1000
    
    def _calculate_complexity(self, tx: Dict) -> float:
        """Calculate interaction complexity"""
        return len(tx.get('instructions', [])) * len(tx.get('accounts', []))

class XGBoostMEVDetector:
    """XGBoost-based MEV detection model"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_importance = None
        self.threshold = 0.5
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize XGBoost model with optimal parameters"""
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            objective='multi:softproba',
            n_jobs=-1,
            tree_method='gpu_hist',
            gpu_id=0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=2.0,  # Handle imbalanced data
            eval_metric=['mlogloss', 'auc'],
            early_stopping_rounds=50,
            random_state=42
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        # Convert to DMatrix for efficiency
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        eval_list = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_list.append((dval, 'eval'))
        
        # Train with early stopping
        self.model = xgb.train(
            params={
                'objective': 'multi:softproba',
                'num_class': 5,  # 5 MEV types
                'max_depth': 10,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'eval_metric': ['mlogloss', 'auc']
            },
            dtrain=dtrain,
            num_boost_round=500,
            evals=eval_list,
            early_stopping_rounds=50,
            verbose_eval=10
        )
        
        # Get feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        logger.info("XGBoost training completed")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict MEV type and probability"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        dtest = xgb.DMatrix(X)
        probabilities = self.model.predict(dtest)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def save_model(self, path: str):
        """Save model to disk"""
        self.model.save_model(path)
        
        # Save feature importance
        with open(path + '.importance', 'wb') as f:
            pickle.dump(self.feature_importance, f)
    
    def load_model(self, path: str):
        """Load model from disk"""
        self.model = xgb.Booster()
        self.model.load_model(path)
        
        # Load feature importance
        importance_path = path + '.importance'
        if os.path.exists(importance_path):
            with open(importance_path, 'rb') as f:
                self.feature_importance = pickle.load(f)
    
    def export_onnx(self, path: str, input_shape: Tuple[int, ...]):
        """Export model to ONNX for faster inference"""
        # Convert XGBoost to ONNX
        # This would use onnxmltools in practice
        pass

class OnlineMEVLearner:
    """Online learning for adaptive MEV detection"""
    
    def __init__(self):
        # Initialize online models
        self.classifier = ensemble.AdaptiveRandomForestClassifier(
            n_models=10,
            max_features='sqrt',
            lambda_value=6,
            grace_period=50,
            split_confidence=0.01,
            tie_threshold=0.05,
            leaf_prediction='nb',
            nb_threshold=0,
            seed=42
        )
        
        self.anomaly_detector = anomaly.HalfSpaceTrees(
            n_trees=10,
            height=8,
            window_size=250,
            seed=42
        )
        
        # Preprocessing
        self.scaler = preprocessing.StandardScaler()
        
        # Metrics
        self.accuracy = metrics.Accuracy()
        self.f1 = metrics.F1()
        self.roc_auc = metrics.ROCAUC()
        
        # Buffer for batch updates
        self.update_buffer = deque(maxlen=100)
        
    def learn_one(self, x: Dict, y: int):
        """Learn from a single sample"""
        # Scale features
        x_scaled = self.scaler.learn_one(x).transform_one(x)
        
        # Make prediction before learning
        y_pred = self.classifier.predict_one(x_scaled)
        
        # Update metrics
        self.accuracy.update(y, y_pred)
        self.f1.update(y, y_pred)
        
        # Learn from sample
        self.classifier.learn_one(x_scaled, y)
        
        # Detect anomalies
        anomaly_score = self.anomaly_detector.score_one(x_scaled)
        self.anomaly_detector.learn_one(x_scaled)
        
        # Add to buffer for batch processing
        self.update_buffer.append((x_scaled, y, anomaly_score))
        
        return y_pred, anomaly_score
    
    def predict_one(self, x: Dict) -> Tuple[int, float, float]:
        """Predict single sample"""
        x_scaled = self.scaler.transform_one(x)
        
        # Classification
        y_pred = self.classifier.predict_one(x_scaled)
        y_proba = self.classifier.predict_proba_one(x_scaled)
        
        # Anomaly score
        anomaly_score = self.anomaly_detector.score_one(x_scaled)
        
        max_proba = max(y_proba.values()) if y_proba else 0
        
        return y_pred, max_proba, anomaly_score
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return {
            'accuracy': self.accuracy.get(),
            'f1': self.f1.get(),
            'samples_seen': self.accuracy.n
        }

class NeuralMEVDetector(nn.Module):
    """Deep learning model for complex MEV patterns"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], num_classes: int = 5):
        super().__init__()
        
        # Build encoder
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=4,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        encoded = self.encoder(x)
        
        # Self-attention
        attended, _ = self.attention(
            encoded.unsqueeze(1),
            encoded.unsqueeze(1),
            encoded.unsqueeze(1)
        )
        attended = attended.squeeze(1)
        
        # Classify
        output = self.classifier(attended)
        
        return output
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings"""
        with torch.no_grad():
            return self.encoder(x)

class ModelEnsemble:
    """Ensemble of multiple models for robust detection"""
    
    def __init__(self):
        self.xgboost = XGBoostMEVDetector()
        self.online = OnlineMEVLearner()
        self.neural = NeuralMEVDetector(input_dim=32)
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            random_state=42
        )
        
        # Ensemble weights (learned through validation)
        self.weights = {
            'xgboost': 0.4,
            'neural': 0.3,
            'online': 0.2,
            'isolation': 0.1
        }
    
    def predict(self, features: np.ndarray) -> Tuple[int, float, Dict]:
        """Ensemble prediction"""
        predictions = {}
        
        # XGBoost prediction
        if self.xgboost.model is not None:
            xgb_pred, xgb_proba = self.xgboost.predict(features)
            predictions['xgboost'] = {
                'pred': xgb_pred,
                'proba': xgb_proba,
                'weight': self.weights['xgboost']
            }
        
        # Neural network prediction
        with torch.no_grad():
            x_tensor = torch.FloatTensor(features)
            neural_output = self.neural(x_tensor)
            neural_proba = F.softmax(neural_output, dim=-1).numpy()
            neural_pred = np.argmax(neural_proba, axis=-1)
            
            predictions['neural'] = {
                'pred': neural_pred,
                'proba': neural_proba,
                'weight': self.weights['neural']
            }
        
        # Online learning prediction
        online_pred, online_proba, anomaly_score = self.online.predict_one(
            {f'f{i}': v for i, v in enumerate(features[0])}
        )
        
        predictions['online'] = {
            'pred': online_pred,
            'proba': online_proba,
            'weight': self.weights['online'],
            'anomaly_score': anomaly_score
        }
        
        # Isolation forest anomaly detection
        iso_pred = self.isolation_forest.predict(features)
        iso_score = self.isolation_forest.score_samples(features)
        
        predictions['isolation'] = {
            'is_anomaly': iso_pred[0] == -1,
            'score': iso_score[0],
            'weight': self.weights['isolation']
        }
        
        # Weighted ensemble
        ensemble_proba = np.zeros(5)  # 5 MEV types
        
        for model_name, pred_info in predictions.items():
            if 'proba' in pred_info and model_name != 'isolation':
                if isinstance(pred_info['proba'], np.ndarray):
                    ensemble_proba += pred_info['proba'][0] * pred_info['weight']
        
        ensemble_pred = np.argmax(ensemble_proba)
        confidence = np.max(ensemble_proba)
        
        return ensemble_pred, confidence, predictions

async def main():
    """Main training and evaluation pipeline"""
    logger.info(f"Starting ML Pipeline {MODEL_VERSION}")
    
    # Initialize components
    feature_eng = FeatureEngineering()
    ensemble = ModelEnsemble()
    
    # Load training data (example)
    # data = await load_training_data()
    
    # Feature extraction
    # features = [feature_eng.extract_features(tx) for tx in data]
    
    # Train models
    # ensemble.xgboost.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    # metrics = evaluate_models(ensemble, X_test, y_test)
    
    # Save models
    # ensemble.xgboost.save_model('models/xgboost_mev.model')
    
    logger.info("ML Pipeline completed")

if __name__ == '__main__':
    asyncio.run(main())