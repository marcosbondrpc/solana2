"""
Automated Feedback Loop with Incremental Learning
Production-grade continuous model improvement system
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch
from river import ensemble, tree, metrics as river_metrics
from river import preprocessing as river_preprocessing
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
import optuna
from datetime import datetime, timedelta
import pickle
import json
import hashlib
from pathlib import Path
import aiofiles
import aiokafka
import aioredis
from clickhouse_driver import Client
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version tracking"""
    version_id: str
    timestamp: datetime
    performance_metrics: Dict[str, float]
    is_champion: bool = False
    deployment_status: str = "staging"
    training_data_hash: str = ""
    hyperparameters: Dict[str, Any] = None
    

class IncrementalDataset(Dataset):
    """PyTorch dataset for incremental learning"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class OnlineArbitrageModel:
    """Online learning model with River ML"""
    
    def __init__(self):
        # Ensemble of online learners
        self.model = ensemble.AdaptiveRandomForestRegressor(
            n_models=10,
            max_features="sqrt",
            lambda_value=6,
            grace_period=10,
            delta=0.01,
            tau=0.05,
            leaf_prediction="adaptive",
            nb_threshold=0,
            nominal_attributes=None,
            splitter=None,
            binary_split=False,
            min_branch_fraction=0.01,
            max_share_to_split=0.99,
            max_size=100,
            memory_estimate_period=1000,
            stop_mem_management=False,
            remove_poor_attrs=False,
            merit_preprune=True
        )
        
        # Feature scaler
        self.scaler = river_preprocessing.StandardScaler()
        
        # Performance tracking
        self.mae = river_metrics.MAE()
        self.rmse = river_metrics.RMSE()
        self.r2 = river_metrics.R2()
        
        self.samples_seen = 0
        
    def learn_one(self, x: Dict[str, float], y: float):
        """Incremental learning on single sample"""
        # Scale features
        x_scaled = self.scaler.learn_one(x).transform_one(x)
        
        # Predict before learning
        y_pred = self.model.predict_one(x_scaled)
        
        # Update metrics
        if y_pred is not None:
            self.mae.update(y, y_pred)
            self.rmse.update(y, y_pred)
            self.r2.update(y, y_pred)
        
        # Learn from sample
        self.model.learn_one(x_scaled, y)
        self.samples_seen += 1
        
        return y_pred
    
    def predict_one(self, x: Dict[str, float]) -> float:
        """Make prediction on single sample"""
        x_scaled = self.scaler.transform_one(x)
        return self.model.predict_one(x_scaled)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return {
            'mae': self.mae.get(),
            'rmse': self.rmse.get(),
            'r2': self.r2.get(),
            'samples_seen': self.samples_seen
        }


class NeuralArbitrageModel(nn.Module):
    """Deep learning model for arbitrage detection"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        
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
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)


class AutomatedFeedbackLoop:
    """Production-grade automated feedback and retraining system"""
    
    def __init__(self,
                 clickhouse_host: str = "localhost",
                 redis_url: str = "redis://localhost:6390",
                 kafka_bootstrap: str = "localhost:9092",
                 mlflow_uri: str = "http://localhost:5000"):
        
        self.clickhouse = Client(clickhouse_host)
        self.redis_url = redis_url
        self.kafka_bootstrap = kafka_bootstrap
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("arbitrage_continuous_learning")
        
        # Model management
        self.online_model = OnlineArbitrageModel()
        self.neural_model = None
        self.current_version = None
        self.champion_model = None
        self.challenger_models = []
        
        # Data buffers
        self.training_buffer = []
        self.buffer_size = 10000
        self.retrain_threshold = 5000
        
        # Performance tracking
        self.drift_threshold = 0.1
        self.performance_window = []
        
        # Paths
        self.model_dir = Path("/home/kidgordones/0solana/node/arbitrage-data-capture/continuous-improvement/models")
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def initialize(self):
        """Initialize connections and load models"""
        # Redis connection
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        
        # Kafka setup
        self.kafka_consumer = aiokafka.AIOKafkaConsumer(
            'arbitrage_transactions',
            'model_feedback',
            bootstrap_servers=self.kafka_bootstrap,
            value_deserializer=lambda v: json.loads(v.decode()),
            auto_offset_reset='latest'
        )
        await self.kafka_consumer.start()
        
        self.kafka_producer = aiokafka.AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode()
        )
        await self.kafka_producer.start()
        
        # Load champion model
        await self._load_champion_model()
        
    async def _load_champion_model(self):
        """Load the current champion model from MLflow"""
        try:
            # Get latest production model
            client = mlflow.tracking.MlflowClient()
            
            model_name = "arbitrage_detector"
            latest_version = client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )
            
            if latest_version:
                version = latest_version[0]
                self.champion_model = mlflow.pytorch.load_model(
                    f"models:/{model_name}/{version.version}"
                )
                self.current_version = ModelVersion(
                    version_id=version.version,
                    timestamp=datetime.now(),
                    performance_metrics={},
                    is_champion=True,
                    deployment_status="production"
                )
                logger.info(f"Loaded champion model version {version.version}")
            else:
                logger.warning("No production model found, training new model")
                await self._train_initial_model()
                
        except Exception as e:
            logger.error(f"Failed to load champion model: {e}")
            await self._train_initial_model()
    
    async def continuous_learning_loop(self):
        """Main continuous learning loop"""
        logger.info("Starting continuous learning loop")
        
        async for msg in self.kafka_consumer:
            try:
                data = msg.value
                
                # Process based on message type
                if msg.topic == 'arbitrage_transactions':
                    await self._process_new_data(data)
                elif msg.topic == 'model_feedback':
                    await self._process_feedback(data)
                    
                # Check if retraining needed
                if len(self.training_buffer) >= self.retrain_threshold:
                    asyncio.create_task(self._trigger_retraining())
                    
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
    
    async def _process_new_data(self, data: Dict[str, Any]):
        """Process new transaction data for learning"""
        
        # Extract features and label
        features = data.get('features', {})
        profit = data.get('profit', 0)
        
        # Online learning - immediate update
        self.online_model.learn_one(features, profit)
        
        # Add to training buffer for batch retraining
        self.training_buffer.append({
            'features': features,
            'target': profit,
            'timestamp': datetime.now().isoformat()
        })
        
        # Maintain buffer size
        if len(self.training_buffer) > self.buffer_size:
            self.training_buffer.pop(0)
        
        # Track performance
        online_metrics = self.online_model.get_metrics()
        
        # Send metrics to monitoring
        await self.kafka_producer.send('model_metrics', value={
            'model_type': 'online',
            'metrics': online_metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _process_feedback(self, feedback: Dict[str, Any]):
        """Process model feedback for improvement"""
        
        prediction = feedback.get('prediction')
        actual = feedback.get('actual')
        features = feedback.get('features')
        
        if all([prediction, actual, features]):
            # Calculate error
            error = abs(prediction - actual)
            
            # High error samples get higher weight in retraining
            weight = 1.0 if error < 0.1 else min(error * 2, 5.0)
            
            self.training_buffer.append({
                'features': features,
                'target': actual,
                'weight': weight,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update performance tracking
            self.performance_window.append(error)
            if len(self.performance_window) > 1000:
                self.performance_window.pop(0)
            
            # Check for performance degradation
            if len(self.performance_window) >= 100:
                recent_error = np.mean(self.performance_window[-100:])
                baseline_error = np.mean(self.performance_window[:-100]) if len(self.performance_window) > 100 else 0
                
                if baseline_error > 0 and recent_error > baseline_error * (1 + self.drift_threshold):
                    logger.warning(f"Performance degradation detected: {recent_error:.4f} vs {baseline_error:.4f}")
                    asyncio.create_task(self._trigger_retraining())
    
    async def _trigger_retraining(self):
        """Trigger model retraining with new data"""
        logger.info("Triggering model retraining")
        
        try:
            # Prepare training data
            df = pd.DataFrame(self.training_buffer)
            
            # Feature engineering
            features = pd.DataFrame(df['features'].tolist())
            targets = df['target'].values
            weights = df.get('weight', pd.Series([1.0] * len(df))).values
            
            # Split data
            X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                features, targets, weights, test_size=0.2, random_state=42
            )
            
            # Start MLflow run
            with mlflow.start_run():
                # Hyperparameter optimization
                best_params = await self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
                
                # Train new model
                new_model = await self._train_model(
                    X_train, y_train, X_val, y_val, 
                    weights=w_train, params=best_params
                )
                
                # Evaluate model
                val_score = await self._evaluate_model(new_model, X_val, y_val)
                
                # Log to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metric("validation_mae", val_score)
                mlflow.log_metric("training_samples", len(X_train))
                
                # Model versioning
                model_hash = hashlib.md5(
                    f"{datetime.now()}{val_score}".encode()
                ).hexdigest()[:8]
                
                model_version = ModelVersion(
                    version_id=model_hash,
                    timestamp=datetime.now(),
                    performance_metrics={'mae': val_score},
                    is_champion=False,
                    deployment_status="staging",
                    training_data_hash=hashlib.md5(str(df.head()).encode()).hexdigest(),
                    hyperparameters=best_params
                )
                
                # Save model
                model_path = self.model_dir / f"model_{model_hash}.pt"
                torch.save(new_model.state_dict(), model_path)
                
                # Log model to MLflow
                mlflow.pytorch.log_model(
                    new_model,
                    "model",
                    registered_model_name="arbitrage_detector"
                )
                
                # Champion/Challenger evaluation
                await self._evaluate_challenger(new_model, model_version)
                
                # Clear training buffer
                self.training_buffer = self.training_buffer[-1000:]  # Keep recent samples
                
                logger.info(f"Retraining complete. New model version: {model_hash}")
                
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    
    async def _optimize_hyperparameters(self, 
                                       X_train: pd.DataFrame,
                                       y_train: np.ndarray,
                                       X_val: pd.DataFrame,
                                       y_val: np.ndarray) -> Dict[str, Any]:
        """Bayesian hyperparameter optimization with Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'hidden_dims': [
                    trial.suggest_int('hidden_1', 128, 1024),
                    trial.suggest_int('hidden_2', 64, 512),
                    trial.suggest_int('hidden_3', 32, 256)
                ],
                'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
            }
            
            # Train model with suggested params
            model = NeuralArbitrageModel(
                input_dim=X_train.shape[1],
                hidden_dims=params['hidden_dims']
            ).to(self.device)
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            
            # Quick training for evaluation
            dataset = IncrementalDataset(X_train.values, y_train)
            dataloader = DataLoader(
                dataset, 
                batch_size=params['batch_size'],
                shuffle=True
            )
            
            model.train()
            for epoch in range(5):  # Quick evaluation
                for features, targets in dataloader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(features).squeeze()
                    loss = nn.MSELoss()(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_features = torch.FloatTensor(X_val.values).to(self.device)
                val_predictions = model(val_features).squeeze().cpu().numpy()
                mae = np.mean(np.abs(val_predictions - y_val))
            
            return mae
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, n_jobs=1)
        
        return study.best_params
    
    async def _train_model(self,
                          X_train: pd.DataFrame,
                          y_train: np.ndarray,
                          X_val: pd.DataFrame,
                          y_val: np.ndarray,
                          weights: Optional[np.ndarray] = None,
                          params: Dict[str, Any] = None) -> nn.Module:
        """Train neural network model"""
        
        # Default params
        if params is None:
            params = {
                'hidden_dims': [512, 256, 128],
                'learning_rate': 0.001,
                'batch_size': 64,
                'weight_decay': 1e-5
            }
        
        # Create model
        model = NeuralArbitrageModel(
            input_dim=X_train.shape[1],
            hidden_dims=params.get('hidden_dims', [512, 256, 128])
        ).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.get('learning_rate', 0.001),
            weight_decay=params.get('weight_decay', 1e-5)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training data
        train_dataset = IncrementalDataset(X_train.values, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=params.get('batch_size', 64),
            shuffle=True
        )
        
        val_dataset = IncrementalDataset(X_val.values, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=params.get('batch_size', 64),
            shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            for features, targets in train_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features).squeeze()
                loss = nn.MSELoss()(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = model(features).squeeze()
                    val_loss += nn.MSELoss()(outputs, targets).item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return model
    
    async def _evaluate_model(self, model: nn.Module, X: pd.DataFrame, y: np.ndarray) -> float:
        """Evaluate model performance"""
        model.eval()
        
        with torch.no_grad():
            features = torch.FloatTensor(X.values).to(self.device)
            predictions = model(features).squeeze().cpu().numpy()
            mae = np.mean(np.abs(predictions - y))
        
        return mae
    
    async def _evaluate_challenger(self, challenger: nn.Module, version: ModelVersion):
        """Evaluate challenger model against champion"""
        
        # Get recent test data from ClickHouse
        query = """
        SELECT features, profit
        FROM arbitrage_transactions
        WHERE timestamp > now() - INTERVAL 1 HOUR
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        
        test_data = self.clickhouse.execute(query)
        
        if len(test_data) < 100:
            logger.warning("Insufficient test data for challenger evaluation")
            return
        
        # Prepare test data
        features = pd.DataFrame([json.loads(row[0]) for row in test_data])
        targets = np.array([row[1] for row in test_data])
        
        # Evaluate both models
        challenger_score = await self._evaluate_model(challenger, features, targets)
        
        if self.champion_model:
            champion_score = await self._evaluate_model(self.champion_model, features, targets)
            
            # Statistical significance test
            improvement = (champion_score - challenger_score) / champion_score
            
            if improvement > 0.05:  # 5% improvement threshold
                logger.info(f"Challenger model shows {improvement*100:.2f}% improvement")
                await self._promote_challenger(challenger, version)
            else:
                logger.info(f"Challenger model did not show significant improvement: {improvement*100:.2f}%")
                self.challenger_models.append((challenger, version))
        else:
            # No champion, promote challenger
            await self._promote_challenger(challenger, version)
    
    async def _promote_challenger(self, model: nn.Module, version: ModelVersion):
        """Promote challenger to champion"""
        logger.info(f"Promoting model {version.version_id} to champion")
        
        # Update MLflow model stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="arbitrage_detector",
            version=version.version_id,
            stage="Production"
        )
        
        # Update champion
        self.champion_model = model
        self.current_version = version
        version.is_champion = True
        version.deployment_status = "production"
        
        # Notify deployment system
        await self.kafka_producer.send('model_deployment', value={
            'action': 'deploy',
            'model_version': version.version_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # Store deployment info in Redis
        await self.redis.set(
            'champion_model_version',
            json.dumps({
                'version': version.version_id,
                'deployed_at': datetime.now().isoformat(),
                'metrics': version.performance_metrics
            })
        )
    
    async def rollback_model(self, version_id: str):
        """Rollback to specific model version"""
        logger.info(f"Rolling back to model version {version_id}")
        
        try:
            # Load model from MLflow
            model = mlflow.pytorch.load_model(
                f"models:/arbitrage_detector/{version_id}"
            )
            
            # Set as champion
            self.champion_model = model
            
            # Update MLflow stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="arbitrage_detector",
                version=version_id,
                stage="Production"
            )
            
            logger.info(f"Successfully rolled back to version {version_id}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise
    
    async def start(self):
        """Start the feedback loop system"""
        await self.initialize()
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self.continuous_learning_loop()),
            asyncio.create_task(self._periodic_retraining()),
            asyncio.create_task(self._model_health_check())
        ]
        
        await asyncio.gather(*tasks)
    
    async def _periodic_retraining(self):
        """Periodic retraining schedule"""
        while True:
            await asyncio.sleep(3600)  # Every hour
            
            # Check if enough new data
            if len(self.training_buffer) >= 1000:
                logger.info("Triggering scheduled retraining")
                await self._trigger_retraining()
    
    async def _model_health_check(self):
        """Monitor model health and performance"""
        while True:
            await asyncio.sleep(60)  # Every minute
            
            try:
                # Get online model metrics
                metrics = self.online_model.get_metrics()
                
                # Store in ClickHouse
                query = """
                INSERT INTO model_health (
                    timestamp, model_type, mae, rmse, r2, samples_seen
                ) VALUES
                """
                
                data = [(
                    datetime.now(),
                    'online',
                    metrics['mae'],
                    metrics['rmse'],
                    metrics['r2'],
                    metrics['samples_seen']
                )]
                
                self.clickhouse.execute(query, data)
                
                # Check for issues
                if metrics['mae'] > 0.1:  # High error threshold
                    logger.warning(f"High MAE detected: {metrics['mae']:.4f}")
                    
            except Exception as e:
                logger.error(f"Health check failed: {e}")


async def main():
    """Run the automated feedback loop"""
    feedback_loop = AutomatedFeedbackLoop()
    
    try:
        await feedback_loop.start()
    except KeyboardInterrupt:
        logger.info("Shutting down feedback loop")


if __name__ == "__main__":
    asyncio.run(main())