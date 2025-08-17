"""
Elite Model Training Pipeline for MEV Arbitrage Detection
Multi-model ensemble with hyperparameter optimization and GPU acceleration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import joblib
import pickle
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training"""
    
    # Model selection
    models_to_train: List[str] = None
    enable_ensemble: bool = True
    ensemble_method: str = 'voting'  # 'voting', 'stacking', 'blending'
    
    # Hyperparameter tuning
    enable_hyperopt: bool = True
    n_trials: int = 100
    hyperopt_timeout: int = 3600  # seconds
    
    # Training parameters
    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = 42
    early_stopping_rounds: int = 50
    
    # Performance targets
    target_accuracy: float = 0.95
    target_precision: float = 0.90
    target_recall: float = 0.85
    max_inference_time_ms: float = 10.0
    
    # GPU settings
    use_gpu: bool = True
    gpu_device: int = 0
    
    # Model persistence
    save_models: bool = True
    model_save_path: str = '/home/kidgordones/0solana/node/arbitrage-data-capture/ml-pipeline/models/'
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = [
                'random_forest', 'xgboost', 'lightgbm', 'catboost',
                'lstm', 'transformer', 'ensemble'
            ]


class LSTMModel(nn.Module):
    """LSTM model for time-series arbitrage detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, output_dim: int = 1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output
        out = attn_out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.fc_layers(out)
        
        return torch.sigmoid(out)


class TransformerModel(nn.Module):
    """Transformer model for advanced pattern recognition"""
    
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1, output_dim: int = 1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = x.transpose(0, 1)  # (seq_len, batch, features)
        transformer_out = self.transformer(x)
        
        # Global average pooling
        out = transformer_out.mean(dim=0)
        
        # Final classification
        out = self.fc_layers(out)
        
        return torch.sigmoid(out)


class AdvancedModelTrainer:
    """
    Production-grade model trainer with ensemble methods and hyperparameter optimization
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.best_params = {}
        self.model_scores = {}
        self.ensemble_model = None
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        """Setup CUDA device if available"""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu_device}')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(self.config.gpu_device)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for training")
        return device
    
    def optimize_hyperparameters(self, model_name: str, X_train: np.ndarray, 
                                 y_train: np.ndarray) -> Dict[str, Any]:
        """
        Bayesian hyperparameter optimization using Optuna
        """
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'n_jobs': -1,
                    'random_state': self.config.random_state
                }
                model = RandomForestClassifier(**params)
            
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                    'use_label_encoder': False,
                    'eval_metric': 'logloss',
                    'random_state': self.config.random_state
                }
                if self.config.use_gpu:
                    params['tree_method'] = 'gpu_hist'
                    params['gpu_id'] = self.config.gpu_device
                model = xgb.XGBClassifier(**params)
            
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 2),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 2),
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'random_state': self.config.random_state,
                    'verbosity': -1
                }
                if self.config.use_gpu:
                    params['device'] = 'gpu'
                    params['gpu_device_id'] = self.config.gpu_device
                model = lgb.LGBMClassifier(**params)
            
            elif model_name == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                    'random_strength': trial.suggest_float('random_strength', 0, 1),
                    'loss_function': 'Logloss',
                    'eval_metric': 'AUC',
                    'random_state': self.config.random_state,
                    'verbose': False
                }
                if self.config.use_gpu:
                    params['task_type'] = 'GPU'
                    params['devices'] = str(self.config.gpu_device)
                model = cb.CatBoostClassifier(**params)
            
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='roc_auc', n_jobs=-1)
            
            return scores.mean()
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.random_state)
        )
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.hyperopt_timeout,
            n_jobs=1  # Use 1 job since models already use parallelization
        )
        
        logger.info(f"Best parameters for {model_name}: {study.best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        
        if self.config.enable_hyperopt:
            params = self.optimize_hyperparameters('random_forest', X_train, y_train)
            self.best_params['random_forest'] = params
        else:
            params = {
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'n_jobs': -1,
                'random_state': self.config.random_state
            }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        logger.info("Training XGBoost...")
        
        if self.config.enable_hyperopt:
            params = self.optimize_hyperparameters('xgboost', X_train, y_train)
            self.best_params['xgboost'] = params
        else:
            params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'random_state': self.config.random_state
            }
            if self.config.use_gpu:
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = self.config.gpu_device
        
        model = xgb.XGBClassifier(**params)
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None) -> lgb.LGBMClassifier:
        """Train LightGBM model"""
        logger.info("Training LightGBM...")
        
        if self.config.enable_hyperopt:
            params = self.optimize_hyperparameters('lightgbm', X_train, y_train)
            self.best_params['lightgbm'] = params
        else:
            params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'random_state': self.config.random_state,
                'verbosity': -1
            }
            if self.config.use_gpu:
                params['device'] = 'gpu'
                params['gpu_device_id'] = self.config.gpu_device
        
        model = lgb.LGBMClassifier(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(self.config.early_stopping_rounds)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def train_catboost(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None) -> cb.CatBoostClassifier:
        """Train CatBoost model"""
        logger.info("Training CatBoost...")
        
        if self.config.enable_hyperopt:
            params = self.optimize_hyperparameters('catboost', X_train, y_train)
            self.best_params['catboost'] = params
        else:
            params = {
                'iterations': 500,
                'depth': 8,
                'learning_rate': 0.1,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_state': self.config.random_state,
                'verbose': False
            }
            if self.config.use_gpu:
                params['task_type'] = 'GPU'
                params['devices'] = str(self.config.gpu_device)
        
        model = cb.CatBoostClassifier(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose=False
            )
        else:
            model.fit(X_train, y_train, verbose=False)
        
        return model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None) -> LSTMModel:
        """Train LSTM model for time-series patterns"""
        logger.info("Training LSTM model...")
        
        # Prepare data for LSTM
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = LSTMModel(input_dim=input_dim).to(self.device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(100):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if X_val is not None and y_val is not None:
                model.eval()
                X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
                
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_rounds:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}")
        
        return model
    
    def train_transformer(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray = None, y_val: np.ndarray = None) -> TransformerModel:
        """Train Transformer model for advanced pattern recognition"""
        logger.info("Training Transformer model...")
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = TransformerModel(input_dim=input_dim).to(self.device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(100):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            
            # Validation
            if X_val is not None and y_val is not None:
                model.eval()
                X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
                
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}")
        
        return model
    
    def create_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> Union[VotingClassifier, StackingClassifier]:
        """Create ensemble model"""
        logger.info(f"Creating {self.config.ensemble_method} ensemble...")
        
        base_models = []
        
        if 'random_forest' in self.models:
            base_models.append(('rf', self.models['random_forest']))
        if 'xgboost' in self.models:
            base_models.append(('xgb', self.models['xgboost']))
        if 'lightgbm' in self.models:
            base_models.append(('lgb', self.models['lightgbm']))
        if 'catboost' in self.models:
            base_models.append(('cb', self.models['catboost']))
        
        if self.config.ensemble_method == 'voting':
            ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft',
                n_jobs=-1
            )
        elif self.config.ensemble_method == 'stacking':
            # Use XGBoost as meta-learner
            meta_learner = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.config.random_state
            )
            ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.ensemble_method}")
        
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        if isinstance(model, (LSTMModel, TransformerModel)):
            model.eval()
            X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(self.device)
            with torch.no_grad():
                y_pred_proba = model(X_test_tensor).cpu().numpy().squeeze()
                y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        # Measure inference time
        start_time = datetime.now()
        for _ in range(100):
            if isinstance(model, (LSTMModel, TransformerModel)):
                with torch.no_grad():
                    _ = model(X_test_tensor[:1])
            else:
                _ = model.predict(X_test[:1])
        
        inference_time = (datetime.now() - start_time).total_seconds() * 10  # ms per prediction
        metrics['inference_time_ms'] = inference_time
        
        # Log results
        logger.info(f"{model_name} Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Inference Time: {metrics['inference_time_ms']:.2f}ms")
        
        # Check against targets
        if metrics['accuracy'] >= self.config.target_accuracy:
            logger.info(f"  ✓ Meets accuracy target ({self.config.target_accuracy})")
        if metrics['inference_time_ms'] <= self.config.max_inference_time_ms:
            logger.info(f"  ✓ Meets latency target ({self.config.max_inference_time_ms}ms)")
        
        return metrics
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train all configured models"""
        logger.info("Starting model training pipeline...")
        
        for model_name in self.config.models_to_train:
            if model_name == 'random_forest':
                self.models['random_forest'] = self.train_random_forest(X_train, y_train)
            elif model_name == 'xgboost':
                self.models['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
            elif model_name == 'lightgbm':
                self.models['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
            elif model_name == 'catboost':
                self.models['catboost'] = self.train_catboost(X_train, y_train, X_val, y_val)
            elif model_name == 'lstm':
                self.models['lstm'] = self.train_lstm(X_train, y_train, X_val, y_val)
            elif model_name == 'transformer':
                self.models['transformer'] = self.train_transformer(X_train, y_train, X_val, y_val)
        
        # Create ensemble if enabled
        if self.config.enable_ensemble and len(self.models) > 1:
            self.ensemble_model = self.create_ensemble(X_train, y_train)
            self.models['ensemble'] = self.ensemble_model
        
        logger.info(f"Training completed. Trained {len(self.models)} models.")
        return self.models
    
    def save_models(self):
        """Save all trained models"""
        import os
        os.makedirs(self.config.model_save_path, exist_ok=True)
        
        for model_name, model in self.models.items():
            save_path = os.path.join(self.config.model_save_path, f"{model_name}_model.pkl")
            
            if isinstance(model, (LSTMModel, TransformerModel)):
                torch.save(model.state_dict(), save_path)
            else:
                joblib.dump(model, save_path)
            
            logger.info(f"Saved {model_name} to {save_path}")
        
        # Save best parameters
        params_path = os.path.join(self.config.model_save_path, "best_params.json")
        import json
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save model scores
        scores_path = os.path.join(self.config.model_save_path, "model_scores.json")
        with open(scores_path, 'w') as f:
            json.dump(self.model_scores, f, indent=2)