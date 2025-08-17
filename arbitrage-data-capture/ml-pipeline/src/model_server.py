"""
Ultra-Low-Latency Model Serving API for Real-Time MEV Arbitrage Detection
FastAPI service with GPU acceleration, caching, and streaming predictions
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import joblib
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import uvloop

# FastAPI and async
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# ML and data processing
import torch
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Caching and streaming
import redis
import aioredis
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError

# Monitoring
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Performance optimization
import numba
from numba import jit
import pyarrow as pa
import pyarrow.parquet as pq

# Set up async event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total number of predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
active_connections = Gauge('active_websocket_connections', 'Number of active WebSocket connections')
cache_hits = Counter('cache_hits_total', 'Total number of cache hits')
cache_misses = Counter('cache_misses_total', 'Total number of cache misses')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy', ['model'])


class PredictionRequest(BaseModel):
    """Input schema for prediction requests"""
    
    features: Dict[str, Union[float, int, str]]
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    model_name: Optional[str] = Field("ensemble", description="Model to use for prediction")
    return_probabilities: bool = Field(True, description="Return probability scores")
    enable_explanation: bool = Field(False, description="Return SHAP explanations")
    
    @validator('features')
    def validate_features(cls, v):
        required_features = ['revenue_sol', 'roi', 'slippage', 'amm', 'dex']
        for feature in required_features:
            if feature not in v:
                raise ValueError(f"Missing required feature: {feature}")
        return v


class BatchPredictionRequest(BaseModel):
    """Input schema for batch prediction requests"""
    
    batch: List[Dict[str, Union[float, int, str]]]
    model_name: Optional[str] = Field("ensemble", description="Model to use")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    async_processing: bool = Field(False, description="Process asynchronously")


class PredictionResponse(BaseModel):
    """Output schema for predictions"""
    
    request_id: str
    prediction: int
    probability: Optional[float] = None
    confidence: float
    model_used: str
    latency_ms: float
    timestamp: str
    explanation: Optional[Dict[str, float]] = None
    risk_score: Optional[float] = None
    expected_profit: Optional[float] = None


class ModelServer:
    """
    High-performance model server with caching and GPU acceleration
    """
    
    def __init__(self, model_dir: str = "/home/kidgordones/0solana/node/arbitrage-data-capture/ml-pipeline/models/"):
        self.model_dir = model_dir
        self.models = {}
        self.preprocessor = None
        self.redis_client = None
        self.kafka_producer = None
        self.device = self._setup_device()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._load_models()
        self._init_cache()
        self._init_kafka()
        
    def _setup_device(self) -> torch.device:
        """Setup CUDA device if available"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for inference")
        return device
    
    def _load_models(self):
        """Load all trained models"""
        logger.info("Loading models...")
        
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'lightgbm': 'lightgbm_model.pkl',
            'catboost': 'catboost_model.pkl',
            'ensemble': 'ensemble_model.pkl'
        }
        
        for model_name, file_name in model_files.items():
            model_path = os.path.join(self.model_dir, file_name)
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
        
        # Load preprocessor
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info("Loaded preprocessor")
        
        # Load deep learning models if available
        for model_name in ['lstm', 'transformer']:
            model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                # Load model architecture and weights
                # This would need the model class definitions from model_trainer.py
                pass
        
        if not self.models:
            raise RuntimeError("No models loaded. Please train models first.")
    
    async def _init_cache(self):
        """Initialize Redis cache"""
        try:
            self.redis_client = await aioredis.create_redis_pool(
                'redis://localhost:6379',
                db=2,
                encoding='utf-8'
            )
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
    
    async def _init_kafka(self):
        """Initialize Kafka producer for streaming predictions"""
        try:
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers='localhost:9092',
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='snappy',
                acks='all'
            )
            await self.kafka_producer.start()
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.warning(f"Kafka not available: {e}")
    
    @staticmethod
    @jit(nopython=True)
    def _fast_feature_processing(features: np.ndarray) -> np.ndarray:
        """Ultra-fast feature processing with Numba"""
        # Apply transformations
        processed = np.empty_like(features)
        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                # Log transform for revenue
                if j == 0:  # Assuming first column is revenue
                    processed[i, j] = np.log1p(features[i, j])
                # Clip extreme values
                elif features[i, j] > 1000:
                    processed[i, j] = 1000
                elif features[i, j] < -1000:
                    processed[i, j] = -1000
                else:
                    processed[i, j] = features[i, j]
        return processed
    
    async def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess features for model input"""
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Apply preprocessor if available
        if self.preprocessor:
            df = self.preprocessor.transform(df)
        
        return df.values
    
    async def get_prediction_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction"""
        if not self.redis_client:
            return None
        
        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                cache_hits.inc()
                return json.loads(cached)
            cache_misses.inc()
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def cache_prediction(self, cache_key: str, prediction: Dict[str, Any], ttl: int = 60):
        """Cache prediction result"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(prediction)
            )
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make a single prediction"""
        start_time = datetime.now()
        
        # Generate cache key
        feature_hash = hashlib.md5(
            json.dumps(request.features, sort_keys=True).encode()
        ).hexdigest()
        cache_key = f"prediction:{request.model_name}:{feature_hash}"
        
        # Check cache
        cached_result = await self.get_prediction_from_cache(cache_key)
        if cached_result:
            return PredictionResponse(**cached_result)
        
        # Preprocess features
        X = await self.preprocess_features(request.features)
        
        # Get model
        model = self.models.get(request.model_name, self.models.get('ensemble'))
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
        
        # Make prediction
        try:
            prediction = model.predict(X)[0]
            
            if request.return_probabilities and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
                confidence = max(probabilities)
            else:
                probability = None
                confidence = 1.0
            
            # Calculate risk score and expected profit
            risk_score = self._calculate_risk_score(request.features)
            expected_profit = self._calculate_expected_profit(request.features, probability)
            
            # Get SHAP explanation if requested
            explanation = None
            if request.enable_explanation:
                explanation = await self._get_shap_explanation(model, X, request.features)
            
            # Create response
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            response_data = {
                'request_id': request.request_id or feature_hash[:8],
                'prediction': int(prediction),
                'probability': probability,
                'confidence': float(confidence),
                'model_used': request.model_name,
                'latency_ms': latency_ms,
                'timestamp': datetime.utcnow().isoformat(),
                'explanation': explanation,
                'risk_score': risk_score,
                'expected_profit': expected_profit
            }
            
            # Cache result
            await self.cache_prediction(cache_key, response_data)
            
            # Update metrics
            prediction_counter.inc()
            prediction_latency.observe(latency_ms / 1000)
            
            # Stream to Kafka if available
            if self.kafka_producer:
                await self.kafka_producer.send('predictions', response_data)
            
            return PredictionResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def predict_batch(self, request: BatchPredictionRequest) -> List[PredictionResponse]:
        """Make batch predictions"""
        if request.async_processing:
            # Process asynchronously
            task_id = request.batch_id or hashlib.md5(
                json.dumps(request.batch).encode()
            ).hexdigest()[:8]
            
            # Submit to background task
            asyncio.create_task(self._process_batch_async(request, task_id))
            
            return {"task_id": task_id, "status": "processing"}
        
        # Process synchronously
        predictions = []
        for item in request.batch:
            pred_request = PredictionRequest(
                features=item,
                model_name=request.model_name
            )
            prediction = await self.predict(pred_request)
            predictions.append(prediction)
        
        return predictions
    
    async def _process_batch_async(self, request: BatchPredictionRequest, task_id: str):
        """Process batch asynchronously"""
        results = []
        
        for item in request.batch:
            pred_request = PredictionRequest(
                features=item,
                model_name=request.model_name
            )
            prediction = await self.predict(pred_request)
            results.append(asdict(prediction))
        
        # Store results in Redis
        if self.redis_client:
            await self.redis_client.setex(
                f"batch_result:{task_id}",
                3600,  # 1 hour TTL
                json.dumps(results)
            )
        
        # Send notification via Kafka
        if self.kafka_producer:
            await self.kafka_producer.send('batch_completions', {
                'task_id': task_id,
                'status': 'completed',
                'num_predictions': len(results)
            })
    
    def _calculate_risk_score(self, features: Dict[str, Any]) -> float:
        """Calculate risk score for arbitrage opportunity"""
        slippage = features.get('slippage', 0)
        gas_cost = features.get('gas_cost', 0.001)
        revenue = features.get('revenue_sol', 0)
        
        if revenue > 0:
            risk_score = (slippage * gas_cost) / revenue
        else:
            risk_score = 1.0
        
        return min(max(risk_score, 0), 1)
    
    def _calculate_expected_profit(self, features: Dict[str, Any], probability: Optional[float]) -> float:
        """Calculate expected profit from arbitrage"""
        revenue = features.get('revenue_sol', 0)
        slippage = features.get('slippage', 0)
        gas_cost = features.get('gas_cost', 0.001)
        
        gross_profit = revenue * (1 - slippage)
        net_profit = gross_profit - gas_cost
        
        if probability:
            expected_profit = net_profit * probability
        else:
            expected_profit = net_profit
        
        return expected_profit
    
    async def _get_shap_explanation(self, model: Any, X: np.ndarray, 
                                   features: Dict[str, Any]) -> Dict[str, float]:
        """Generate SHAP explanation for prediction"""
        try:
            import shap
            
            # Create explainer
            if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier, lgb.LGBMClassifier)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                
                # Map SHAP values to feature names
                feature_names = list(features.keys())
                explanation = {
                    feature_names[i]: float(shap_values[0][i])
                    for i in range(min(len(feature_names), len(shap_values[0])))
                }
                
                return explanation
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
        
        return None
    
    async def stream_predictions(self, websocket: WebSocket):
        """Stream real-time predictions via WebSocket"""
        await websocket.accept()
        active_connections.inc()
        
        try:
            # Create Kafka consumer for real-time data
            consumer = AIOKafkaConsumer(
                'arbitrage_opportunities',
                bootstrap_servers='localhost:9092',
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            await consumer.start()
            
            async for msg in consumer:
                # Process incoming arbitrage opportunity
                features = msg.value
                
                # Make prediction
                pred_request = PredictionRequest(
                    features=features,
                    model_name='ensemble'
                )
                prediction = await self.predict(pred_request)
                
                # Send to WebSocket client
                await websocket.send_json(asdict(prediction))
                
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            active_connections.dec()
            await consumer.stop()
    
    async def update_model(self, model_name: str, model_path: str):
        """Hot-reload model without downtime"""
        try:
            # Load new model
            new_model = joblib.load(model_path)
            
            # Validate model
            test_input = np.random.randn(1, 10)  # Adjust based on your feature count
            _ = new_model.predict(test_input)
            
            # Atomic swap
            self.models[model_name] = new_model
            logger.info(f"Successfully updated {model_name}")
            
            return {"status": "success", "model": model_name}
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_model_metrics(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        metrics = {
            'models_loaded': list(self.models.keys()),
            'total_predictions': prediction_counter._value.get(),
            'avg_latency_ms': prediction_latency._sum.get() / max(prediction_latency._count.get(), 1) * 1000,
            'cache_hit_rate': cache_hits._value.get() / max(cache_hits._value.get() + cache_misses._value.get(), 1),
            'active_connections': active_connections._value.get()
        }
        
        return metrics


# Initialize FastAPI app
app = FastAPI(
    title="MEV Arbitrage Detection API",
    description="Ultra-low-latency model serving for real-time arbitrage detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model server
model_server = None


@app.on_event("startup")
async def startup_event():
    """Initialize model server on startup"""
    global model_server
    model_server = ModelServer()
    logger.info("Model server initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if model_server:
        if model_server.redis_client:
            model_server.redis_client.close()
            await model_server.redis_client.wait_closed()
        if model_server.kafka_producer:
            await model_server.kafka_producer.stop()
    logger.info("Model server shutdown")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    return await model_server.predict(request)


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    return await model_server.predict_batch(request)


@app.get("/batch/status/{task_id}")
async def get_batch_status(task_id: str):
    """Get batch processing status"""
    if not model_server or not model_server.redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    result = await model_server.redis_client.get(f"batch_result:{task_id}")
    if result:
        return {"status": "completed", "results": json.loads(result)}
    
    return {"status": "processing", "task_id": task_id}


@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """WebSocket endpoint for streaming predictions"""
    if not model_server:
        await websocket.close()
        return
    
    await model_server.stream_predictions(websocket)


@app.post("/model/update")
async def update_model(model_name: str, model_path: str):
    """Update model endpoint"""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    return await model_server.update_model(model_name, model_path)


@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics"""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    return await model_server.get_model_metrics()


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        loop="uvloop",
        log_level="info",
        access_log=True
    )