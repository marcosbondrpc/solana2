#!/usr/bin/env python3
"""
FastAPI Detection Service with ONNX Runtime
DETECTION-ONLY: Pure inference, no execution
Target: P50 <100μs, P99 <500μs inference latency
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import onnxruntime as ort
import numpy as np
import torch
import asyncio
import aioredis
import clickhouse_driver
from datetime import datetime
import hashlib
import json
import time
import logging
from contextlib import asynccontextmanager
import uvloop
import ed25519

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model sessions
gnn_session: Optional[ort.InferenceSession] = None
transformer_session: Optional[ort.InferenceSession] = None
redis_client: Optional[aioredis.Redis] = None
ch_client: Optional[clickhouse_driver.Client] = None

# Ed25519 signing for Decision DNA
signing_key = ed25519.SigningKey(b'0' * 32)  # Replace with secure key
verifying_key = signing_key.verifying_key

class TransactionData(BaseModel):
    """Input transaction data for detection"""
    slot: int
    sig: str
    payer: str
    programs: List[str]
    ix_kinds: List[int]
    accounts: List[str]
    pool_keys: List[str] = []
    amount_in: float = 0
    amount_out: float = 0
    token_in: Optional[str] = None
    token_out: Optional[str] = None
    fee: int = 0
    priority_fee: int = 0
    venue: Optional[str] = None

class DetectionRequest(BaseModel):
    """Batch detection request"""
    transactions: List[TransactionData]
    window_size: int = Field(default=10, ge=1, le=100)
    include_confidence: bool = True
    include_dna: bool = True

class DetectionResult(BaseModel):
    """Detection result with evidence"""
    slot: int
    sig: str
    is_sandwich: bool
    sandwich_score: float = Field(ge=0, le=1)
    pattern_type: str  # 'normal', 'sandwich', 'backrun', 'liquidation', 'arbitrage'
    confidence: float = Field(ge=0, le=1)
    evidence: Dict[str, any]
    dna_fingerprint: Optional[str] = None
    signature: Optional[str] = None
    latency_us: int

class BehaviorProfile(BaseModel):
    """Entity behavioral profile"""
    entity_addr: str
    attack_style: str  # 'surgical', 'shotgun', 'adaptive'
    victim_selection: str  # 'retail', 'whale', 'bot', 'mixed'
    risk_appetite: float = Field(ge=0, le=1)
    fee_aggressiveness: float = Field(ge=0, le=1)
    avg_response_ms: float
    landing_rate: float = Field(ge=0, le=1)
    total_extraction_sol: float
    linked_wallets: List[str] = []
    cluster_id: Optional[int] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global gnn_session, transformer_session, redis_client, ch_client
    
    # Load ONNX models
    logger.info("Loading ONNX models...")
    
    # Configure ONNX Runtime for low latency
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.inter_op_num_threads = 4
    sess_options.intra_op_num_threads = 4
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        gnn_session = ort.InferenceSession(
            "/home/kidgordones/0solana/solana2/models/sandwich_gnn.onnx",
            sess_options=sess_options,
            providers=providers
        )
        logger.info("GNN model loaded successfully")
    except Exception as e:
        logger.warning(f"GNN model not found or failed to load: {e}")
    
    try:
        transformer_session = ort.InferenceSession(
            "/home/kidgordones/0solana/solana2/models/mev_transformer.onnx",
            sess_options=sess_options,
            providers=providers
        )
        logger.info("Transformer model loaded successfully")
    except Exception as e:
        logger.warning(f"Transformer model not found or failed to load: {e}")
    
    # Connect to Redis for caching
    try:
        redis_client = await aioredis.create_redis_pool(
            'redis://redis:6390',
            minsize=5,
            maxsize=10
        )
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
    
    # Connect to ClickHouse
    try:
        ch_client = clickhouse_driver.Client(
            host='clickhouse',
            port=9000,
            settings={'use_numpy': True}
        )
        logger.info("ClickHouse connected")
    except Exception as e:
        logger.warning(f"ClickHouse connection failed: {e}")
    
    yield
    
    # Cleanup
    if redis_client:
        redis_client.close()
        await redis_client.wait_closed()

app = FastAPI(
    title="MEV Detection Service",
    description="DETECTION-ONLY behavioral analysis for Solana MEV",
    version="1.0.0",
    lifespan=lifespan
)

def generate_dna_fingerprint(data: Dict) -> str:
    """Generate Decision DNA fingerprint with Ed25519 signature"""
    # Create deterministic hash
    data_str = json.dumps(data, sort_keys=True)
    dna = hashlib.blake2b(data_str.encode(), digest_size=32).hexdigest()
    return dna

def sign_decision(dna: str, decision: Dict) -> str:
    """Sign decision with Ed25519"""
    message = f"{dna}:{json.dumps(decision, sort_keys=True)}"
    signature = signing_key.sign(message.encode())
    return signature.hex()

async def encode_instruction_sequence(programs: List[str], ix_kinds: List[int]) -> np.ndarray:
    """Encode instruction sequence for transformer"""
    # Simplified encoding - should match training
    vocab = {
        '11111111111111111111111111111111': 2,  # System
        'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA': 4,  # Token
        '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8': 10,  # Raydium
        'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4': 13,  # Jupiter
        '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P': 20,  # Pump
    }
    
    sequence = []
    for prog in programs[:128]:  # Limit sequence length
        sequence.append(vocab.get(prog, 1))  # 1 for unknown
    
    # Pad to 128
    while len(sequence) < 128:
        sequence.append(0)
    
    return np.array(sequence[:128], dtype=np.int64)

async def run_gnn_inference(graph_features: np.ndarray) -> Tuple[float, str]:
    """Run GNN inference"""
    if not gnn_session:
        return 0.0, "gnn_unavailable"
    
    start = time.perf_counter()
    
    # Prepare inputs
    inputs = {
        'x': graph_features,
        'edge_index': np.random.randint(0, 10, (2, 20), dtype=np.int64),  # Placeholder
        'batch': np.zeros(graph_features.shape[0], dtype=np.int64)
    }
    
    # Run inference
    outputs = gnn_session.run(None, inputs)
    scores = outputs[0]
    
    latency_us = int((time.perf_counter() - start) * 1_000_000)
    
    # Softmax to get probabilities
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()
    
    sandwich_prob = probs[0, 1] if len(probs.shape) > 1 else probs[1]
    
    return float(sandwich_prob), f"gnn_{latency_us}us"

async def run_transformer_inference(sequence: np.ndarray) -> Tuple[float, str, float]:
    """Run transformer inference"""
    if not transformer_session:
        return 0.0, "normal", 0.5
    
    start = time.perf_counter()
    
    # Prepare inputs
    inputs = {
        'sequence': sequence.reshape(1, -1),
        'mask': None
    }
    
    # Run inference
    outputs = transformer_session.run(None, inputs)
    pattern_logits = outputs[0]
    confidence = outputs[1]
    
    latency_us = int((time.perf_counter() - start) * 1_000_000)
    
    # Get pattern type
    pattern_types = ['normal', 'sandwich', 'backrun', 'liquidation', 'arbitrage']
    pattern_idx = np.argmax(pattern_logits[0])
    pattern_type = pattern_types[min(pattern_idx, len(pattern_types)-1)]
    
    # Get sandwich probability
    exp_logits = np.exp(pattern_logits[0][:2] - np.max(pattern_logits[0][:2]))
    sandwich_prob = exp_logits[1] / exp_logits.sum()
    
    return float(sandwich_prob), pattern_type, float(confidence[0])

async def detect_sandwich_heuristic(tx: TransactionData, window: List[TransactionData]) -> Dict:
    """Heuristic sandwich detection"""
    evidence = {
        'bracket': False,
        'slip_rebound': False,
        'timing': False,
        'pool_match': False
    }
    
    # Check for bracket pattern
    if len(window) >= 3:
        # Look for same pool in transactions before and after
        for i, w_tx in enumerate(window):
            if w_tx.sig == tx.sig:
                # Check transactions before and after
                if i > 0 and i < len(window) - 1:
                    before = window[i-1]
                    after = window[i+1]
                    
                    # Check if same attacker
                    if before.payer == after.payer and before.payer != tx.payer:
                        evidence['bracket'] = True
                        
                        # Check pool overlap
                        pools_before = set(before.pool_keys)
                        pools_after = set(after.pool_keys)
                        pools_victim = set(tx.pool_keys)
                        
                        if pools_before & pools_victim and pools_after & pools_victim:
                            evidence['pool_match'] = True
                        
                        # Check timing
                        if before.slot == tx.slot and after.slot == tx.slot:
                            evidence['timing'] = True
                        
                        # Check for price reversion
                        if before.amount_out > 0 and after.amount_in > 0:
                            if abs(before.amount_out - after.amount_in) / before.amount_out < 0.1:
                                evidence['slip_rebound'] = True
    
    return evidence

@app.post("/infer", response_model=List[DetectionResult])
async def detect_mev(request: DetectionRequest, background_tasks: BackgroundTasks):
    """
    Main detection endpoint
    Returns detection results with evidence and DNA
    """
    results = []
    
    for tx in request.transactions:
        start_time = time.perf_counter()
        
        # Run detections in parallel
        tasks = []
        
        # Prepare features
        graph_features = np.random.randn(10, 128).astype(np.float32)  # Placeholder
        sequence = await encode_instruction_sequence(tx.programs, tx.ix_kinds)
        
        # Run models
        gnn_task = asyncio.create_task(run_gnn_inference(graph_features))
        transformer_task = asyncio.create_task(run_transformer_inference(sequence))
        heuristic_task = asyncio.create_task(
            detect_sandwich_heuristic(tx, request.transactions)
        )
        
        # Wait for all
        gnn_score, gnn_info = await gnn_task
        transformer_score, pattern_type, confidence = await transformer_task
        heuristic_evidence = await heuristic_task
        
        # Ensemble scoring
        ensemble_score = (gnn_score * 0.4 + transformer_score * 0.4 + 
                         (0.2 if heuristic_evidence['bracket'] else 0))
        
        is_sandwich = ensemble_score > 0.5
        
        # Generate Decision DNA
        dna_data = {
            'slot': tx.slot,
            'sig': tx.sig,
            'models': {
                'gnn': gnn_score,
                'transformer': transformer_score,
                'ensemble': ensemble_score
            },
            'evidence': heuristic_evidence,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        dna_fingerprint = generate_dna_fingerprint(dna_data) if request.include_dna else None
        
        # Sign decision
        decision = {
            'is_sandwich': is_sandwich,
            'score': ensemble_score,
            'pattern': pattern_type
        }
        signature = sign_decision(dna_fingerprint, decision) if request.include_dna else None
        
        # Calculate latency
        latency_us = int((time.perf_counter() - start_time) * 1_000_000)
        
        # Store result
        result = DetectionResult(
            slot=tx.slot,
            sig=tx.sig,
            is_sandwich=is_sandwich,
            sandwich_score=ensemble_score,
            pattern_type=pattern_type,
            confidence=confidence if request.include_confidence else 0.5,
            evidence={
                'heuristic': heuristic_evidence,
                'gnn_score': gnn_score,
                'transformer_score': transformer_score,
                'models_info': {
                    'gnn': gnn_info,
                    'latency': f"{latency_us}us"
                }
            },
            dna_fingerprint=dna_fingerprint,
            signature=signature,
            latency_us=latency_us
        )
        
        results.append(result)
        
        # Store in ClickHouse asynchronously
        if ch_client and is_sandwich:
            background_tasks.add_task(
                store_detection,
                tx,
                result
            )
    
    return results

async def store_detection(tx: TransactionData, result: DetectionResult):
    """Store detection result in ClickHouse"""
    try:
        query = """
        INSERT INTO ch.candidates (
            detection_ts, slot, victim_sig, attacker_a_sig, attacker_b_sig,
            attacker_addr, victim_addr, pool, d_ms, d_slots,
            slippage_victim, price_reversion, evidence, score_rule,
            score_gnn, score_transformer, ensemble_score,
            dna_fingerprint, model_version
        ) VALUES
        """
        
        # Simplified insert - would need full data in production
        ch_client.execute(query, [{
            'detection_ts': datetime.utcnow(),
            'slot': tx.slot,
            'victim_sig': tx.sig,
            'attacker_a_sig': '',
            'attacker_b_sig': '',
            'attacker_addr': '',
            'victim_addr': tx.payer,
            'pool': tx.pool_keys[0] if tx.pool_keys else '',
            'd_ms': 0,
            'd_slots': 0,
            'slippage_victim': 0,
            'price_reversion': 0,
            'evidence': 1,
            'score_rule': 0,
            'score_gnn': result.evidence['gnn_score'],
            'score_transformer': result.evidence['transformer_score'],
            'ensemble_score': result.sandwich_score,
            'dna_fingerprint': result.dna_fingerprint or '',
            'model_version': 'v1.0.0'
        }])
    except Exception as e:
        logger.error(f"Failed to store detection: {e}")

@app.get("/profile/{entity_addr}", response_model=BehaviorProfile)
async def get_entity_profile(entity_addr: str):
    """Get behavioral profile for an entity"""
    
    if not ch_client:
        raise HTTPException(status_code=503, detail="ClickHouse unavailable")
    
    # Query entity profile
    query = f"""
    SELECT 
        attack_style_surgical,
        attack_style_shotgun,
        victim_retail_ratio,
        victim_whale_ratio,
        risk_appetite,
        fee_aggressiveness,
        avg_response_ms,
        landing_rate,
        total_extraction_sol,
        linked_wallets,
        cluster_id
    FROM ch.entity_profiles
    WHERE entity_addr = '{entity_addr}'
    ORDER BY profile_date DESC
    LIMIT 1
    """
    
    result = ch_client.execute(query)
    
    if not result:
        # Generate profile on the fly
        profile = await generate_entity_profile(entity_addr)
    else:
        row = result[0]
        
        # Determine attack style
        if row[0] > 0.7:
            attack_style = 'surgical'
        elif row[1] > 0.7:
            attack_style = 'shotgun'
        else:
            attack_style = 'adaptive'
        
        # Determine victim selection
        if row[2] > 0.7:
            victim_selection = 'retail'
        elif row[3] > 0.7:
            victim_selection = 'whale'
        else:
            victim_selection = 'mixed'
        
        profile = BehaviorProfile(
            entity_addr=entity_addr,
            attack_style=attack_style,
            victim_selection=victim_selection,
            risk_appetite=row[4],
            fee_aggressiveness=row[5],
            avg_response_ms=row[6],
            landing_rate=row[7],
            total_extraction_sol=row[8],
            linked_wallets=row[9] or [],
            cluster_id=row[10]
        )
    
    return profile

async def generate_entity_profile(entity_addr: str) -> BehaviorProfile:
    """Generate behavioral profile from transaction history"""
    
    # Placeholder implementation
    return BehaviorProfile(
        entity_addr=entity_addr,
        attack_style='unknown',
        victim_selection='unknown',
        risk_appetite=0.5,
        fee_aggressiveness=0.5,
        avg_response_ms=100,
        landing_rate=0.5,
        total_extraction_sol=0,
        linked_wallets=[],
        cluster_id=None
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'models': {
            'gnn': gnn_session is not None,
            'transformer': transformer_session is not None
        },
        'services': {
            'redis': redis_client is not None,
            'clickhouse': ch_client is not None
        },
        'timestamp': datetime.utcnow().isoformat()
    }
    return JSONResponse(content=status)

@app.get("/metrics")
async def get_metrics():
    """Get detection metrics"""
    if not ch_client:
        raise HTTPException(status_code=503, detail="ClickHouse unavailable")
    
    query = """
    SELECT 
        model_name,
        avg(roc_auc) as avg_auc,
        avg(precision) as avg_precision,
        avg(recall) as avg_recall,
        avg(false_positive_rate) as avg_fpr,
        avg(inference_p50_us) as p50_latency,
        avg(inference_p99_us) as p99_latency,
        sum(predictions_count) as total_predictions
    FROM ch.model_metrics
    WHERE ts >= now() - INTERVAL 1 DAY
    GROUP BY model_name
    """
    
    results = ch_client.execute(query)
    
    metrics = {}
    for row in results:
        metrics[row[0]] = {
            'auc': row[1],
            'precision': row[2],
            'recall': row[3],
            'fpr': row[4],
            'p50_latency_us': row[5],
            'p99_latency_us': row[6],
            'total_predictions': row[7]
        }
    
    return metrics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800, workers=4)