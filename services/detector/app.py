"""
Elite MEV Detection FastAPI Service
Real-time inference endpoints with ONNX model serving
Target: P50 ≤1 slot, P95 ≤2 slots, ROC-AUC ≥0.95
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import numpy as np
import torch
import time
import hashlib
import json
from datetime import datetime, timedelta
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
from collections import defaultdict
import clickhouse_driver
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from contextlib import asynccontextmanager

# Import detection models
from models import (
    RuleBasedDetector,
    StatisticalAnomalyDetector,
    GNNDetector,
    TransformerDetector,
    HybridMEVDetector,
    ONNXModelServer,
    DetectionResult
)
from entity_analyzer import BehavioralAnalyzer, BehavioralSpectrumAnalyzer

# Metrics
detection_counter = Counter('mev_detections_total', 'Total MEV detections', ['mev_type'])
latency_histogram = Histogram('detection_latency_ms', 'Detection latency in milliseconds')
model_accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy')
entity_profiles_gauge = Gauge('entity_profiles_total', 'Total entity profiles')

# Pydantic models
class TransactionFeatures(BaseModel):
    """Transaction features for detection"""
    signature: str
    slot: int
    block_time: Optional[float] = None
    program_ids: List[str]
    instruction_count: int
    account_keys: List[str]
    fee: int
    compute_units: Optional[int] = 0
    priority_fee: Optional[int] = 0
    token_transfers: Optional[List[Dict]] = []
    feature_hash: Optional[str] = None

class DetectionRequest(BaseModel):
    """Detection request payload"""
    transaction: Dict[str, Any]
    features: TransactionFeatures
    priority: bool = False

class BatchDetectionRequest(BaseModel):
    """Batch detection request"""
    transactions: List[DetectionRequest]

class DetectionResponse(BaseModel):
    """Detection response with DNA tracking"""
    is_mev: bool
    mev_type: Optional[str]
    confidence: float
    attacker_address: Optional[str]
    victim_address: Optional[str]
    profit_estimate: Optional[float]
    feature_importance: Dict[str, float]
    decision_dna: str
    inference_latency_ms: float
    model_scores: Dict[str, float]

class EntityProfileRequest(BaseModel):
    """Entity profile request"""
    address: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class EntityProfileResponse(BaseModel):
    """Entity behavioral profile response"""
    address: str
    classification: str
    risk_level: str
    sophistication_score: float
    behavioral_metrics: Dict[str, Any]
    financial_metrics: Dict[str, Any]
    operational_metrics: Dict[str, Any]
    advanced_metrics: Dict[str, Any]
    profile_dna: str
    cluster_id: Optional[int]

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    print("🚀 Initializing MEV Detection Service...")
    
    # Initialize models
    app.state.models = await initialize_models()
    
    # Initialize connections
    app.state.clickhouse = await connect_clickhouse()
    app.state.redis = await connect_redis()
    
    # Initialize analyzers
    app.state.behavioral_analyzer = BehavioralAnalyzer()
    app.state.spectrum_analyzer = BehavioralSpectrumAnalyzer()
    
    # Start background tasks
    app.state.executor = ThreadPoolExecutor(max_workers=10)
    app.state.decision_chain = []
    
    print("✅ MEV Detection Service initialized")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down MEV Detection Service...")
    app.state.executor.shutdown(wait=True)
    if app.state.clickhouse:
        app.state.clickhouse.disconnect()
    if app.state.redis:
        await app.state.redis.close()

# Create FastAPI app
app = FastAPI(
    title="Elite MEV Detection Service",
    description="State-of-the-art MEV behavioral analysis with sub-slot detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_models():
    """Initialize all detection models"""
    models = {
        'rule_based': RuleBasedDetector(),
        'statistical': StatisticalAnomalyDetector(),
    }
    
    # Load neural network models if available
    try:
        models['gnn'] = GNNDetector()
        models['gnn'].load_state_dict(torch.load('models/gnn_model.pth', map_location='cpu'))
        models['gnn'].eval()
    except:
        print("⚠️ GNN model not available")
    
    try:
        models['transformer'] = TransformerDetector()
        models['transformer'].load_state_dict(torch.load('models/transformer_model.pth', map_location='cpu'))
        models['transformer'].eval()
    except:
        print("⚠️ Transformer model not available")
    
    try:
        models['hybrid'] = HybridMEVDetector()
        models['hybrid'].load_state_dict(torch.load('models/hybrid_model.pth', map_location='cpu'))
        models['hybrid'].eval()
    except:
        print("⚠️ Hybrid model not available")
    
    # Load ONNX models for production inference
    try:
        models['onnx_ensemble'] = ONNXModelServer('models/ensemble.onnx')
    except:
        print("⚠️ ONNX ensemble not available")
    
    return models

async def connect_clickhouse():
    """Connect to ClickHouse database"""
    try:
        client = clickhouse_driver.Client(
            host='localhost',
            port=9000,
            database='default',
            settings={'use_numpy': True}
        )
        return client
    except Exception as e:
        print(f"⚠️ ClickHouse connection failed: {e}")
        return None

async def connect_redis():
    """Connect to Redis for caching"""
    try:
        import aioredis
        redis = await aioredis.create_redis_pool(
            'redis://localhost',
            minsize=5,
            maxsize=10
        )
        return redis
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        return None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MEV Detection Service",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": list(app.state.models.keys())
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_mev(request: DetectionRequest):
    """Single transaction MEV detection endpoint"""
    start_time = time.perf_counter()
    
    # Run detection through ensemble
    results = {}
    model_scores = {}
    
    # Rule-based detection
    if 'rule_based' in app.state.models:
        rule_result = app.state.models['rule_based'].detect_sandwich([request.transaction])
        results['rule_based'] = rule_result
        model_scores['rule_based'] = rule_result.confidence
    
    # Statistical anomaly detection
    if 'statistical' in app.state.models:
        stat_result = app.state.models['statistical'].detect_anomaly(request.transaction)
        results['statistical'] = stat_result
        model_scores['statistical'] = stat_result.confidence
    
    # ONNX ensemble (fastest for production)
    if 'onnx_ensemble' in app.state.models:
        features = extract_features_vector(request.features)
        onnx_result = app.state.models['onnx_ensemble'].predict(features)
        results['onnx'] = onnx_result
        model_scores['onnx'] = onnx_result.confidence
    
    # Ensemble voting
    ensemble_result = ensemble_vote(results)
    
    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000
    latency_histogram.observe(latency_ms)
    
    # Track detection
    if ensemble_result.is_mev:
        detection_counter.labels(mev_type=ensemble_result.mev_type).inc()
    
    # Store in ClickHouse
    if app.state.clickhouse and ensemble_result.is_mev:
        await store_detection(app.state.clickhouse, ensemble_result, request)
    
    # Add to decision chain
    app.state.decision_chain.append({
        'signature': request.features.signature,
        'decision_dna': ensemble_result.decision_dna,
        'timestamp': datetime.utcnow().isoformat()
    })
    
    return DetectionResponse(
        is_mev=ensemble_result.is_mev,
        mev_type=ensemble_result.mev_type,
        confidence=ensemble_result.confidence,
        attacker_address=ensemble_result.attacker_address,
        victim_address=ensemble_result.victim_address,
        profit_estimate=ensemble_result.profit_estimate,
        feature_importance=ensemble_result.feature_importance,
        decision_dna=ensemble_result.decision_dna,
        inference_latency_ms=latency_ms,
        model_scores=model_scores
    )

@app.post("/detect/batch")
async def detect_batch(request: BatchDetectionRequest):
    """Batch detection endpoint for high throughput"""
    start_time = time.perf_counter()
    
    # Process in parallel
    tasks = []
    for tx_request in request.transactions:
        task = detect_mev(tx_request)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        "results": results,
        "batch_size": len(request.transactions),
        "total_latency_ms": latency_ms,
        "avg_latency_ms": latency_ms / len(request.transactions)
    }

@app.post("/detect/priority")
async def detect_priority(request: DetectionRequest):
    """Priority detection for monitored addresses"""
    # Fast path for priority addresses
    request.priority = True
    result = await detect_mev(request)
    
    # Alert if high confidence MEV
    if result.is_mev and result.confidence > 0.9:
        await send_alert(result, request.features.account_keys[0])
    
    return result

@app.get("/entity/{address}", response_model=EntityProfileResponse)
async def get_entity_profile(address: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
    """Get behavioral profile for an entity"""
    
    # Check cache first
    if app.state.redis:
        cached = await app.state.redis.get(f"profile:{address}")
        if cached:
            return json.loads(cached)
    
    # Fetch transactions from ClickHouse
    if app.state.clickhouse:
        transactions = await fetch_entity_transactions(
            app.state.clickhouse, 
            address, 
            start_date, 
            end_date
        )
    else:
        transactions = []
    
    # Generate profile
    spectrum_report = app.state.spectrum_analyzer.generate_spectrum_report(
        address, 
        transactions
    )
    
    # Cache for 5 minutes
    if app.state.redis:
        await app.state.redis.setex(
            f"profile:{address}",
            300,
            json.dumps(spectrum_report)
        )
    
    # Update gauge
    entity_profiles_gauge.inc()
    
    return EntityProfileResponse(
        address=address,
        classification=spectrum_report['classification'],
        risk_level=spectrum_report['risk_level'],
        sophistication_score=spectrum_report['sophistication_score'],
        behavioral_metrics=spectrum_report['behavioral_metrics'],
        financial_metrics=spectrum_report['financial_metrics'],
        operational_metrics=spectrum_report['operational_metrics'],
        advanced_metrics=spectrum_report['advanced_metrics'],
        profile_dna=spectrum_report['profile_dna'],
        cluster_id=spectrum_report.get('cluster_id')
    )

@app.get("/clusters")
async def get_entity_clusters():
    """Get clustered entities based on behavioral similarity"""
    
    # Fetch all profiles
    if app.state.clickhouse:
        query = """
        SELECT DISTINCT entity_address 
        FROM entity_profiles 
        WHERE profile_date >= today() - 7
        """
        addresses = app.state.clickhouse.execute(query)
        
        profiles = []
        for (addr,) in addresses:
            tx = await fetch_entity_transactions(app.state.clickhouse, addr)
            profile = app.state.behavioral_analyzer.analyze_entity(addr, tx)
            profiles.append(profile)
        
        # Cluster entities
        clusters = app.state.behavioral_analyzer.cluster_entities(profiles)
        
        # Find coordinated actors
        coordinated = app.state.behavioral_analyzer.identify_coordinated_actors(profiles)
        
        return {
            "clusters": clusters,
            "coordinated_pairs": coordinated,
            "total_entities": len(profiles)
        }
    
    return {"error": "ClickHouse not available"}

@app.get("/stats")
async def get_stats():
    """Get detection statistics"""
    stats = {
        "total_detections": sum(detection_counter._metrics.values()),
        "decision_chain_length": len(app.state.decision_chain),
        "models_available": list(app.state.models.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add ClickHouse stats if available
    if app.state.clickhouse:
        ch_stats = app.state.clickhouse.execute("""
        SELECT 
            count(*) as total_transactions,
            countIf(is_mev_candidate = 1) as mev_candidates,
            avg(inference_latency_ms) as avg_latency
        FROM model_scores
        WHERE score_timestamp >= now() - INTERVAL 1 HOUR
        """)
        
        if ch_stats:
            stats['clickhouse'] = {
                'recent_transactions': ch_stats[0][0],
                'mev_candidates': ch_stats[0][1],
                'avg_latency_ms': ch_stats[0][2]
            }
    
    return stats

@app.get("/merkle")
async def get_merkle_root():
    """Get current Merkle root of decision chain"""
    if not app.state.decision_chain:
        return {"merkle_root": None}
    
    # Build Merkle tree
    leaves = [
        hashlib.sha256(json.dumps(decision).encode()).digest()
        for decision in app.state.decision_chain
    ]
    
    level = leaves
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else level[i]
            combined = left + right
            next_level.append(hashlib.sha256(combined).digest())
        level = next_level
    
    merkle_root = level[0].hex()
    
    return {
        "merkle_root": merkle_root,
        "chain_length": len(app.state.decision_chain),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

# Helper functions
def extract_features_vector(features: TransactionFeatures) -> np.ndarray:
    """Extract feature vector from transaction features"""
    vector = np.zeros(128)  # Fixed size feature vector
    
    # Basic features
    vector[0] = features.slot
    vector[1] = features.fee
    vector[2] = features.compute_units
    vector[3] = features.priority_fee
    vector[4] = features.instruction_count
    vector[5] = len(features.account_keys)
    vector[6] = len(features.program_ids)
    vector[7] = len(features.token_transfers)
    
    # Program flags (one-hot encoding for known programs)
    known_programs = [
        '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',  # Raydium V4
        'CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK',  # Raydium CPMM
        '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP',  # Orca
        '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P',  # Pump.fun
    ]
    
    for i, program in enumerate(known_programs):
        if program in features.program_ids:
            vector[10 + i] = 1
    
    # Normalize
    vector = vector / (np.linalg.norm(vector) + 1e-10)
    
    return vector

def ensemble_vote(results: Dict[str, DetectionResult]) -> DetectionResult:
    """Ensemble voting from multiple models"""
    if not results:
        return DetectionResult(
            is_mev=False,
            mev_type=None,
            confidence=0.0,
            attacker_address=None,
            victim_address=None,
            profit_estimate=None,
            feature_importance={},
            decision_dna=hashlib.sha256(b"no_results").hexdigest(),
            inference_latency_ms=0
        )
    
    # Weighted voting
    weights = {
        'rule_based': 0.3,
        'statistical': 0.2,
        'onnx': 0.5
    }
    
    total_confidence = 0
    mev_votes = defaultdict(float)
    
    for model_name, result in results.items():
        weight = weights.get(model_name, 0.25)
        total_confidence += result.confidence * weight
        
        if result.is_mev and result.mev_type:
            mev_votes[result.mev_type] += weight
    
    # Determine final decision
    is_mev = total_confidence > 0.5
    mev_type = max(mev_votes, key=mev_votes.get) if mev_votes else None
    
    # Combine feature importance
    combined_importance = {}
    for result in results.values():
        for feature, importance in result.feature_importance.items():
            if feature not in combined_importance:
                combined_importance[feature] = 0
            combined_importance[feature] += importance
    
    # Generate ensemble DNA
    ensemble_data = {
        'models': list(results.keys()),
        'confidence': total_confidence,
        'mev_type': mev_type
    }
    decision_dna = hashlib.sha256(json.dumps(ensemble_data).encode()).hexdigest()
    
    return DetectionResult(
        is_mev=is_mev,
        mev_type=mev_type,
        confidence=total_confidence,
        attacker_address=None,  # Would need consensus
        victim_address=None,
        profit_estimate=None,
        feature_importance=combined_importance,
        decision_dna=decision_dna,
        inference_latency_ms=0
    )

async def store_detection(client, result: DetectionResult, request: DetectionRequest):
    """Store detection result in ClickHouse"""
    try:
        query = """
        INSERT INTO sandwich_candidates 
        (detection_id, detection_timestamp, confidence_score, victim_signature, 
         victim_slot, attacker_address, victim_address, decision_dna, feature_hash)
        VALUES
        """
        
        data = (
            hashlib.sha256(result.decision_dna.encode()).hexdigest()[:16],
            datetime.utcnow(),
            result.confidence,
            request.features.signature,
            request.features.slot,
            result.attacker_address or '',
            result.victim_address or '',
            result.decision_dna,
            request.features.feature_hash or ''
        )
        
        client.execute(query, [data])
    except Exception as e:
        print(f"Failed to store detection: {e}")

async def fetch_entity_transactions(client, address: str, start_date=None, end_date=None):
    """Fetch entity transactions from ClickHouse"""
    try:
        where_clause = f"account_keys[1] = '{address}'"
        
        if start_date:
            where_clause += f" AND block_time >= '{start_date}'"
        if end_date:
            where_clause += f" AND block_time <= '{end_date}'"
        
        query = f"""
        SELECT 
            signature, slot, block_time, fee, compute_units_consumed,
            program_ids, instruction_count, priority_fee_lamports
        FROM solana_transactions
        WHERE {where_clause}
        ORDER BY block_time DESC
        LIMIT 1000
        """
        
        results = client.execute(query)
        
        transactions = []
        for row in results:
            transactions.append({
                'signature': row[0],
                'slot': row[1],
                'block_time': row[2].timestamp(),
                'fee': row[3],
                'compute_units': row[4],
                'program_ids': row[5],
                'instruction_count': row[6],
                'priority_fee': row[7]
            })
        
        return transactions
    except Exception as e:
        print(f"Failed to fetch transactions: {e}")
        return []

async def send_alert(result: DetectionResponse, address: str):
    """Send alert for high-confidence detections"""
    alert = {
        "type": "HIGH_CONFIDENCE_MEV",
        "address": address,
        "mev_type": result.mev_type,
        "confidence": result.confidence,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    print(f"🚨 ALERT: {alert}")
    
    # In production, send to monitoring system
    # await send_to_pagerduty(alert)
    # await send_to_slack(alert)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )