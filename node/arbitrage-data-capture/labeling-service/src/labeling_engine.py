import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from clickhouse_driver import Client as ClickHouseClient
import redis.asyncio as redis
from kafka import KafkaConsumer, KafkaProducer
import structlog

logger = structlog.get_logger()

@dataclass
class ArbitrageLabel:
    transaction_id: str
    timestamp: int
    block_slot: int
    is_arbitrage: bool
    confidence_score: float
    arbitrage_type: Optional[str]
    profit_class: str  # profitable, unprofitable, failed
    ground_truth: Optional[bool]
    human_verified: bool
    features: Dict[str, float]
    metadata: Dict[str, Any]

class LabelingEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clickhouse = ClickHouseClient(
            host=config['clickhouse_host'],
            port=config['clickhouse_port'],
            database=config['clickhouse_db']
        )
        self.redis_client = None
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.label_buffer = []
        self.metrics_history = []
        
    async def initialize(self):
        """Initialize connections and load models"""
        logger.info("Initializing Labeling Engine")
        
        # Initialize Redis
        self.redis_client = await redis.from_url(
            self.config['redis_url'],
            encoding='utf-8',
            decode_responses=True
        )
        
        # Load or train models
        await self.load_models()
        
        # Start background tasks
        asyncio.create_task(self.consume_transactions())
        asyncio.create_task(self.periodic_model_update())
        asyncio.create_task(self.export_labels())
        
    async def load_models(self):
        """Load pre-trained models or train new ones"""
        logger.info("Loading ML models for labeling")
        
        # Try to load existing models
        models_loaded = await self.load_saved_models()
        
        if not models_loaded:
            # Train new models with historical data
            await self.train_models()
            
    async def train_models(self):
        """Train ensemble of models for arbitrage detection"""
        logger.info("Training new models on historical data")
        
        # Fetch training data
        df = await self.fetch_training_data()
        
        if df.empty:
            logger.warning("No training data available")
            return
            
        # Feature engineering
        X, y = self.prepare_features(df)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train Random Forest
        logger.info("Training Random Forest")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
        self.models['rf'].fit(X_train_scaled, y_train)
        
        # Train XGBoost
        logger.info("Training XGBoost")
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.01,
            objective='binary:logistic',
            n_jobs=-1,
            random_state=42
        )
        self.models['xgb'].fit(X_train_scaled, y_train)
        
        # Train LightGBM
        logger.info("Training LightGBM")
        self.models['lgb'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.01,
            num_leaves=31,
            n_jobs=-1,
            random_state=42
        )
        self.models['lgb'].fit(X_train_scaled, y_train)
        
        # Train CatBoost
        logger.info("Training CatBoost")
        self.models['catboost'] = CatBoostClassifier(
            iterations=200,
            depth=10,
            learning_rate=0.01,
            loss_function='Logloss',
            random_state=42,
            verbose=False
        )
        self.models['catboost'].fit(X_train_scaled, y_train)
        
        # Evaluate models
        await self.evaluate_models(X_test_scaled, y_test)
        
        # Extract feature importance
        self.extract_feature_importance(X.columns)
        
        # Save models
        await self.save_models()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features for ML models"""
        features = []
        
        # Price features
        features.extend([
            'price_spread_percentage',
            'price_volatility_1h',
            'price_volatility_24h',
            'price_momentum',
            'price_rsi',
            'price_bollinger_position',
        ])
        
        # Volume features
        features.extend([
            'volume_24h',
            'volume_spike_ratio',
            'liquidity_depth_usd',
            'liquidity_imbalance',
            'order_book_depth',
        ])
        
        # Network features
        features.extend([
            'gas_price_gwei',
            'priority_fee_sol',
            'network_congestion_score',
            'slot_timing_ms',
            'transactions_per_slot',
        ])
        
        # DEX features
        features.extend([
            'num_dex_involved',
            'cross_dex_spread',
            'pool_fee_total_bps',
            'slippage_estimate',
        ])
        
        # Risk features
        features.extend([
            'sandwich_risk_score',
            'backrun_probability',
            'token_age_days',
            'ownership_concentration',
            'rugpull_risk_score',
        ])
        
        # Historical features
        features.extend([
            'success_rate_7d',
            'avg_profit_7d',
            'similar_arbs_1h',
            'competition_level',
        ])
        
        # Ensure all features exist in dataframe
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
                
        X = df[features]
        y = df['is_arbitrage'].astype(int)
        
        return X, y
        
    async def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            auc = roc_auc_score(y_test, y_prob)
            
            results[name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            logger.info(f"Model {name} - Precision: {precision:.3f}, "
                       f"Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
            
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
    def extract_feature_importance(self, feature_names: List[str]):
        """Extract and store feature importance from models"""
        
        # Random Forest feature importance
        if 'rf' in self.models:
            self.feature_importance['rf'] = dict(zip(
                feature_names,
                self.models['rf'].feature_importances_
            ))
            
        # XGBoost feature importance
        if 'xgb' in self.models:
            importance = self.models['xgb'].get_booster().get_score(
                importance_type='gain'
            )
            self.feature_importance['xgb'] = {
                feature_names[int(k[1:])]: v 
                for k, v in importance.items()
            }
            
        # LightGBM feature importance
        if 'lgb' in self.models:
            self.feature_importance['lgb'] = dict(zip(
                feature_names,
                self.models['lgb'].feature_importances_
            ))
            
        # Log top features
        for model_name, importance in self.feature_importance.items():
            sorted_features = sorted(
                importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            logger.info(f"Top features for {model_name}: {sorted_features}")
            
    async def label_transaction(self, transaction: Dict[str, Any]) -> ArbitrageLabel:
        """Label a single transaction"""
        
        # Extract features
        features = await self.extract_transaction_features(transaction)
        
        # Prepare features for model
        feature_vector = self.prepare_feature_vector(features)
        
        # Scale features
        if 'standard' in self.scalers:
            feature_vector_scaled = self.scalers['standard'].transform(
                feature_vector.reshape(1, -1)
            )
        else:
            feature_vector_scaled = feature_vector.reshape(1, -1)
            
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(feature_vector_scaled)[0]
            prob = model.predict_proba(feature_vector_scaled)[0, 1]
            predictions[name] = pred
            probabilities[name] = prob
            
        # Ensemble prediction (weighted average)
        weights = {'rf': 0.2, 'xgb': 0.3, 'lgb': 0.3, 'catboost': 0.2}
        ensemble_prob = sum(
            probabilities.get(name, 0) * weight 
            for name, weight in weights.items()
        )
        
        is_arbitrage = ensemble_prob > 0.5
        
        # Determine profit class
        profit_class = self.classify_profit(transaction, is_arbitrage)
        
        # Determine arbitrage type
        arb_type = self.determine_arbitrage_type(transaction) if is_arbitrage else None
        
        label = ArbitrageLabel(
            transaction_id=transaction['signature'],
            timestamp=transaction['timestamp'],
            block_slot=transaction['slot'],
            is_arbitrage=is_arbitrage,
            confidence_score=ensemble_prob,
            arbitrage_type=arb_type,
            profit_class=profit_class,
            ground_truth=None,
            human_verified=False,
            features=features,
            metadata={
                'model_predictions': predictions,
                'model_probabilities': probabilities,
                'ensemble_weight': weights
            }
        )
        
        return label
        
    async def extract_transaction_features(
        self, 
        transaction: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract features from a transaction"""
        features = {}
        
        # Basic transaction features
        features['num_instructions'] = len(transaction.get('instructions', []))
        features['num_accounts'] = len(transaction.get('accounts', []))
        features['compute_units'] = transaction.get('compute_units', 0)
        
        # Analyze programs involved
        programs = [inst['program_id'] for inst in transaction.get('instructions', [])]
        dex_programs = self.identify_dex_programs(programs)
        features['num_dex_involved'] = len(dex_programs)
        
        # Token flow analysis
        token_flows = self.analyze_token_flows(transaction)
        features['num_tokens'] = len(token_flows)
        features['is_cyclic'] = self.detect_cyclic_flow(token_flows)
        
        # Price analysis
        if 'price_data' in transaction:
            price_data = transaction['price_data']
            features['price_spread_percentage'] = self.calculate_price_spread(price_data)
            features['cross_dex_spread'] = self.calculate_cross_dex_spread(price_data)
            
        # Gas and fees
        features['gas_price_gwei'] = transaction.get('gas_price', 0)
        features['priority_fee_sol'] = transaction.get('priority_fee', 0) / 1e9
        
        # Risk features from external data
        risk_data = await self.fetch_risk_metrics(transaction)
        features.update(risk_data)
        
        # Historical context
        historical = await self.fetch_historical_context(transaction)
        features.update(historical)
        
        return features
        
    def identify_dex_programs(self, programs: List[str]) -> List[str]:
        """Identify DEX programs from program IDs"""
        dex_programs = {
            '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8': 'Raydium',
            'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc': 'Orca',
            'PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY': 'Phoenix',
            'LBUZKhRxPF3XgDs2K2ESXfp7cVRxWjpQwo7hjUDPPY': 'Meteora',
            'srmqPvymJeFKQ4zGQed1GFppgkRHL9kaE  Mz9F5YD4J': 'OpenBook',
            '2wT8Yq49kHgDzGu2KLAzyN1FoAGv5V3UDvqWGHJ8VQg': 'Lifinity',
        }
        
        return [
            dex_programs[prog] 
            for prog in programs 
            if prog in dex_programs
        ]
        
    def analyze_token_flows(self, transaction: Dict[str, Any]) -> List[Dict]:
        """Analyze token flows in transaction"""
        flows = []
        
        # Parse pre and post token balances
        pre_balances = transaction.get('pre_token_balances', [])
        post_balances = transaction.get('post_token_balances', [])
        
        for i, (pre, post) in enumerate(zip(pre_balances, post_balances)):
            if pre['mint'] == post['mint']:
                amount_change = post['amount'] - pre['amount']
                if amount_change != 0:
                    flows.append({
                        'mint': pre['mint'],
                        'account': pre['account'],
                        'change': amount_change,
                        'decimals': pre['decimals']
                    })
                    
        return flows
        
    def detect_cyclic_flow(self, token_flows: List[Dict]) -> bool:
        """Detect if token flow forms a cycle"""
        if len(token_flows) < 3:
            return False
            
        # Check if tokens form a cycle (A -> B -> C -> A)
        mints = [flow['mint'] for flow in token_flows]
        return len(set(mints)) < len(mints)
        
    def calculate_price_spread(self, price_data: Dict) -> float:
        """Calculate price spread percentage"""
        if not price_data or 'prices' not in price_data:
            return 0.0
            
        prices = price_data['prices']
        if len(prices) < 2:
            return 0.0
            
        max_price = max(prices.values())
        min_price = min(prices.values())
        
        if min_price == 0:
            return 0.0
            
        return ((max_price - min_price) / min_price) * 100
        
    def calculate_cross_dex_spread(self, price_data: Dict) -> float:
        """Calculate spread across different DEXs"""
        if not price_data or 'dex_prices' not in price_data:
            return 0.0
            
        dex_prices = price_data['dex_prices']
        if len(dex_prices) < 2:
            return 0.0
            
        prices = list(dex_prices.values())
        return np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0.0
        
    async def fetch_risk_metrics(self, transaction: Dict) -> Dict[str, float]:
        """Fetch risk metrics for transaction"""
        # This would query risk analysis service
        return {
            'sandwich_risk_score': 0.0,
            'backrun_probability': 0.0,
            'rugpull_risk_score': 0.0,
            'token_age_days': 30,
            'ownership_concentration': 0.1,
        }
        
    async def fetch_historical_context(self, transaction: Dict) -> Dict[str, float]:
        """Fetch historical context for transaction"""
        # Query historical data from ClickHouse
        query = f"""
        SELECT 
            COUNT(*) as similar_count,
            AVG(profit_usd) as avg_profit,
            AVG(success) as success_rate
        FROM arbitrage_transactions
        WHERE timestamp > now() - INTERVAL 7 DAY
            AND token_a = %(token_a)s
            AND token_b = %(token_b)s
        """
        
        # Execute query (simplified)
        result = {'similar_arbs_1h': 5, 'avg_profit_7d': 100, 'success_rate_7d': 0.8}
        
        return result
        
    def prepare_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for model input"""
        # Define feature order (must match training)
        feature_order = [
            'price_spread_percentage', 'price_volatility_1h', 'price_volatility_24h',
            'price_momentum', 'price_rsi', 'price_bollinger_position',
            'volume_24h', 'volume_spike_ratio', 'liquidity_depth_usd',
            'liquidity_imbalance', 'order_book_depth', 'gas_price_gwei',
            'priority_fee_sol', 'network_congestion_score', 'slot_timing_ms',
            'transactions_per_slot', 'num_dex_involved', 'cross_dex_spread',
            'pool_fee_total_bps', 'slippage_estimate', 'sandwich_risk_score',
            'backrun_probability', 'token_age_days', 'ownership_concentration',
            'rugpull_risk_score', 'success_rate_7d', 'avg_profit_7d',
            'similar_arbs_1h', 'competition_level'
        ]
        
        vector = np.array([features.get(f, 0.0) for f in feature_order])
        return vector
        
    def classify_profit(self, transaction: Dict, is_arbitrage: bool) -> str:
        """Classify profit level of arbitrage"""
        if not is_arbitrage:
            return 'non_arbitrage'
            
        profit = transaction.get('net_profit_usd', 0)
        
        if profit > 100:
            return 'highly_profitable'
        elif profit > 10:
            return 'profitable'
        elif profit > 0:
            return 'marginally_profitable'
        elif profit == 0:
            return 'break_even'
        else:
            return 'unprofitable'
            
    def determine_arbitrage_type(self, transaction: Dict) -> str:
        """Determine the type of arbitrage"""
        num_dex = transaction.get('num_dex_involved', 0)
        is_cyclic = transaction.get('is_cyclic', False)
        has_flashloan = transaction.get('has_flashloan', False)
        
        if has_flashloan:
            return 'flashloan'
        elif is_cyclic:
            return 'cyclic'
        elif num_dex == 2:
            return 'two_leg'
        elif num_dex == 3:
            return 'triangular'
        elif num_dex > 3:
            return f'multi_leg_{num_dex}'
        else:
            return 'unknown'
            
    async def consume_transactions(self):
        """Consume transactions from Kafka for labeling"""
        consumer = KafkaConsumer(
            'solana_transactions',
            bootstrap_servers=self.config['kafka_brokers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        for message in consumer:
            try:
                transaction = message.value
                label = await self.label_transaction(transaction)
                
                # Buffer labels for batch processing
                self.label_buffer.append(label)
                
                if len(self.label_buffer) >= 100:
                    await self.process_label_batch()
                    
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
                
    async def process_label_batch(self):
        """Process batch of labels"""
        if not self.label_buffer:
            return
            
        batch = self.label_buffer.copy()
        self.label_buffer.clear()
        
        # Store in ClickHouse
        await self.store_labels(batch)
        
        # Update Redis cache
        await self.update_cache(batch)
        
        # Send to downstream services
        await self.publish_labels(batch)
        
    async def store_labels(self, labels: List[ArbitrageLabel]):
        """Store labels in ClickHouse"""
        data = [asdict(label) for label in labels]
        
        self.clickhouse.execute(
            """
            INSERT INTO arbitrage_labels 
            (transaction_id, timestamp, is_arbitrage, confidence_score, 
             arbitrage_type, profit_class, features, metadata)
            VALUES
            """,
            data
        )
        
    async def update_cache(self, labels: List[ArbitrageLabel]):
        """Update Redis cache with latest labels"""
        pipe = self.redis_client.pipeline()
        
        for label in labels:
            key = f"label:{label.transaction_id}"
            value = json.dumps(asdict(label))
            pipe.setex(key, 3600, value)  # 1 hour TTL
            
        await pipe.execute()
        
    async def publish_labels(self, labels: List[ArbitrageLabel]):
        """Publish labels to Kafka for downstream consumption"""
        producer = KafkaProducer(
            bootstrap_servers=self.config['kafka_brokers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        for label in labels:
            producer.send('arbitrage_labels', value=asdict(label))
            
        producer.flush()
        
    async def periodic_model_update(self):
        """Periodically retrain models with new data"""
        while True:
            await asyncio.sleep(3600)  # Every hour
            
            try:
                # Check if enough new labels for retraining
                new_labels_count = await self.count_new_labels()
                
                if new_labels_count > 1000:
                    logger.info(f"Retraining models with {new_labels_count} new labels")
                    await self.train_models()
                    
            except Exception as e:
                logger.error(f"Error in periodic model update: {e}")
                
    async def count_new_labels(self) -> int:
        """Count new labels since last training"""
        result = self.clickhouse.execute(
            """
            SELECT COUNT(*) 
            FROM arbitrage_labels
            WHERE timestamp > now() - INTERVAL 1 HOUR
            """
        )
        return result[0][0] if result else 0
        
    async def export_labels(self):
        """Export labels for ML training datasets"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            try:
                # Export recent labels to JSON format
                labels = await self.fetch_recent_labels()
                
                if labels:
                    await self.export_to_json(labels)
                    await self.export_to_parquet(labels)
                    
            except Exception as e:
                logger.error(f"Error exporting labels: {e}")
                
    async def fetch_recent_labels(self) -> List[Dict]:
        """Fetch recent labels from database"""
        result = self.clickhouse.execute(
            """
            SELECT *
            FROM arbitrage_labels
            WHERE timestamp > now() - INTERVAL 5 MINUTE
            ORDER BY timestamp DESC
            """
        )
        return result
        
    async def export_to_json(self, labels: List[Dict]):
        """Export labels to JSON format"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"labels_{timestamp}.json"
        
        with open(f"/data/exports/{filename}", 'w') as f:
            json.dump(labels, f, indent=2, default=str)
            
        logger.info(f"Exported {len(labels)} labels to {filename}")
        
    async def export_to_parquet(self, labels: List[Dict]):
        """Export labels to Parquet format"""
        df = pd.DataFrame(labels)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"labels_{timestamp}.parquet"
        
        df.to_parquet(f"/data/exports/{filename}", compression='snappy')
        
        logger.info(f"Exported {len(labels)} labels to {filename}")
        
    async def save_models(self):
        """Save trained models to disk"""
        import joblib
        
        for name, model in self.models.items():
            joblib.dump(model, f"/models/{name}_model.pkl")
            
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"/models/{name}_scaler.pkl")
            
        logger.info("Models saved successfully")
        
    async def load_saved_models(self) -> bool:
        """Load models from disk if they exist"""
        import joblib
        import os
        
        model_names = ['rf', 'xgb', 'lgb', 'catboost']
        
        try:
            for name in model_names:
                model_path = f"/models/{name}_model.pkl"
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                else:
                    return False
                    
            scaler_path = "/models/standard_scaler.pkl"
            if os.path.exists(scaler_path):
                self.scalers['standard'] = joblib.load(scaler_path)
            else:
                return False
                
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
            
    async def fetch_training_data(self) -> pd.DataFrame:
        """Fetch training data from ClickHouse"""
        query = """
        SELECT *
        FROM arbitrage_transactions
        WHERE timestamp > now() - INTERVAL 30 DAY
        LIMIT 100000
        """
        
        result = self.clickhouse.execute(query)
        
        if not result:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(result)
        
        return df

async def main():
    config = {
        'clickhouse_host': 'localhost',
        'clickhouse_port': 9000,
        'clickhouse_db': 'solana_arbitrage',
        'redis_url': 'redis://localhost:6390',
        'kafka_brokers': ['localhost:9092'],
    }
    
    engine = LabelingEngine(config)
    await engine.initialize()
    
    # Keep running
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())