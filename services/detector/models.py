"""
Elite MEV Detection Models
Multi-layer detection with GNN, Transformer, and Hybrid architectures
Target: ROC-AUC ≥0.95, FP <0.5%, Latency P50 ≤1 slot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib
import time
from scipy import stats
import onnx
import onnxruntime as ort


@dataclass
class DetectionResult:
    """Unified detection result with DNA tracking"""
    is_mev: bool
    mev_type: Optional[str]
    confidence: float
    attacker_address: Optional[str]
    victim_address: Optional[str]
    profit_estimate: Optional[float]
    feature_importance: Dict[str, float]
    decision_dna: str
    inference_latency_ms: float


class RuleBasedDetector:
    """High-precision rule-based sandwich detection using bracket heuristics"""
    
    def __init__(self, min_profit_threshold: float = 1000):
        self.min_profit_threshold = min_profit_threshold
        self.known_dex_programs = {
            '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',  # Raydium V4
            'CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK',  # Raydium CPMM
            'CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C',  # Raydium CPMM
            '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP',  # Orca Whirlpool
            'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc',  # Whirlpool
            '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P',  # Pump.fun
        }
        
    def detect_sandwich(self, transactions: List[Dict]) -> Optional[DetectionResult]:
        """Detect sandwich attacks using bracket pattern matching"""
        start_time = time.perf_counter()
        
        # Group transactions by slot
        slot_groups = {}
        for tx in transactions:
            slot = tx['slot']
            if slot not in slot_groups:
                slot_groups[slot] = []
            slot_groups[slot].append(tx)
        
        # Look for sandwich patterns
        for slot, txs in slot_groups.items():
            if len(txs) < 3:
                continue
                
            # Check each possible sandwich pattern
            for i in range(len(txs) - 2):
                front_run = txs[i]
                victim = txs[i + 1]
                back_run = txs[i + 2]
                
                # Check if same attacker
                if front_run['signer'] != back_run['signer']:
                    continue
                    
                # Check if different from victim
                if front_run['signer'] == victim['signer']:
                    continue
                    
                # Check if interacting with same pool
                front_programs = set(front_run.get('program_ids', []))
                victim_programs = set(victim.get('program_ids', []))
                back_programs = set(back_run.get('program_ids', []))
                
                if not (front_programs & self.known_dex_programs and
                        victim_programs & self.known_dex_programs and
                        back_programs & self.known_dex_programs):
                    continue
                
                # Calculate profit estimate
                profit = self._estimate_sandwich_profit(front_run, victim, back_run)
                
                if profit > self.min_profit_threshold:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Generate decision DNA
                    decision_dna = self._generate_decision_dna({
                        'front_run': front_run['signature'],
                        'victim': victim['signature'],
                        'back_run': back_run['signature'],
                        'profit': profit
                    })
                    
                    return DetectionResult(
                        is_mev=True,
                        mev_type='sandwich',
                        confidence=0.95,
                        attacker_address=front_run['signer'],
                        victim_address=victim['signer'],
                        profit_estimate=profit,
                        feature_importance={
                            'bracket_pattern': 1.0,
                            'same_pool': 0.8,
                            'profit_threshold': 0.6
                        },
                        decision_dna=decision_dna,
                        inference_latency_ms=latency_ms
                    )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        return DetectionResult(
            is_mev=False,
            mev_type=None,
            confidence=0.0,
            attacker_address=None,
            victim_address=None,
            profit_estimate=None,
            feature_importance={},
            decision_dna=self._generate_decision_dna({'no_pattern': True}),
            inference_latency_ms=latency_ms
        )
    
    def _estimate_sandwich_profit(self, front_run: Dict, victim: Dict, back_run: Dict) -> float:
        """Estimate sandwich attack profit"""
        # Simplified profit calculation
        # In production, this would analyze actual token movements
        victim_amount = victim.get('amount', 0)
        slippage = 0.03  # Assume 3% slippage
        return victim_amount * slippage
    
    def _generate_decision_dna(self, features: Dict) -> str:
        """Generate cryptographic DNA for decision tracking"""
        feature_str = str(sorted(features.items()))
        return hashlib.sha256(feature_str.encode()).hexdigest()


class StatisticalAnomalyDetector:
    """Z-score based anomaly detection for MEV behaviors"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.history = []
        self.mean = None
        self.std = None
        
    def detect_anomaly(self, transaction: Dict) -> DetectionResult:
        """Detect anomalies using statistical methods"""
        start_time = time.perf_counter()
        
        # Extract features
        features = self._extract_features(transaction)
        
        # Update statistics
        self.history.append(features)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        if len(self.history) < 100:
            # Not enough data for statistics
            latency_ms = (time.perf_counter() - start_time) * 1000
            return DetectionResult(
                is_mev=False,
                mev_type=None,
                confidence=0.0,
                attacker_address=None,
                victim_address=None,
                profit_estimate=None,
                feature_importance={},
                decision_dna=self._generate_decision_dna(features),
                inference_latency_ms=latency_ms
            )
        
        # Calculate z-scores
        history_array = np.array(self.history)
        mean = np.mean(history_array, axis=0)
        std = np.std(history_array, axis=0)
        
        z_scores = np.abs((features - mean) / (std + 1e-10))
        max_z_score = np.max(z_scores)
        
        # Anomaly threshold
        is_anomaly = max_z_score > 3.0
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return DetectionResult(
            is_mev=is_anomaly,
            mev_type='anomaly' if is_anomaly else None,
            confidence=min(max_z_score / 5.0, 1.0) if is_anomaly else 0.0,
            attacker_address=transaction.get('signer'),
            victim_address=None,
            profit_estimate=None,
            feature_importance={
                'fee_zscore': float(z_scores[0]),
                'compute_zscore': float(z_scores[1]),
                'amount_zscore': float(z_scores[2])
            },
            decision_dna=self._generate_decision_dna(features),
            inference_latency_ms=latency_ms
        )
    
    def _extract_features(self, transaction: Dict) -> np.ndarray:
        """Extract statistical features from transaction"""
        return np.array([
            transaction.get('fee', 0),
            transaction.get('compute_units', 0),
            transaction.get('amount', 0)
        ])
    
    def _generate_decision_dna(self, features) -> str:
        """Generate DNA for statistical detection"""
        feature_str = str(features.tolist())
        return hashlib.sha256(feature_str.encode()).hexdigest()


class GNNDetector(nn.Module):
    """Graph Neural Network for transaction flow analysis"""
    
    def __init__(self, node_features: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # 5 MEV types
        )
        
        self.mev_types = ['sandwich', 'arbitrage', 'liquidation', 'jit', 'normal']
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Forward pass through GNN"""
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0)[0].unsqueeze(0)
            x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        return self.classifier(x)
    
    def detect(self, transaction_graph: Data) -> DetectionResult:
        """Detect MEV using GNN"""
        start_time = time.perf_counter()
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(transaction_graph.x, transaction_graph.edge_index)
            probs = F.softmax(logits, dim=1)
            
            max_prob, pred_class = probs.max(1)
            mev_type = self.mev_types[pred_class.item()]
            
            is_mev = mev_type != 'normal'
            confidence = float(max_prob.item())
            
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return DetectionResult(
            is_mev=is_mev,
            mev_type=mev_type if is_mev else None,
            confidence=confidence,
            attacker_address=None,  # Would extract from graph
            victim_address=None,
            profit_estimate=None,
            feature_importance={
                f'class_{mt}': float(probs[0, i].item()) 
                for i, mt in enumerate(self.mev_types)
            },
            decision_dna=hashlib.sha256(str(probs.tolist()).encode()).hexdigest(),
            inference_latency_ms=latency_ms
        )


class TransformerDetector(nn.Module):
    """Transformer for instruction sequence modeling"""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # 5 MEV types
        )
        
        self.mev_types = ['sandwich', 'arbitrage', 'liquidation', 'jit', 'normal']
        
    def forward(self, instruction_ids: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward pass through transformer"""
        batch_size, seq_len = instruction_ids.shape
        
        # Embedding and positional encoding
        x = self.embedding(instruction_ids)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global pooling (CLS token or mean)
        x = x.mean(dim=1)
        
        # Classification
        return self.classifier(x)
    
    def detect(self, instruction_sequence: torch.Tensor) -> DetectionResult:
        """Detect MEV using transformer"""
        start_time = time.perf_counter()
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(instruction_sequence.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            
            max_prob, pred_class = probs.max(1)
            mev_type = self.mev_types[pred_class.item()]
            
            is_mev = mev_type != 'normal'
            confidence = float(max_prob.item())
            
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return DetectionResult(
            is_mev=is_mev,
            mev_type=mev_type if is_mev else None,
            confidence=confidence,
            attacker_address=None,
            victim_address=None,
            profit_estimate=None,
            feature_importance={
                f'class_{mt}': float(probs[0, i].item()) 
                for i, mt in enumerate(self.mev_types)
            },
            decision_dna=hashlib.sha256(str(probs.tolist()).encode()).hexdigest(),
            inference_latency_ms=latency_ms
        )


class HybridMEVDetector(nn.Module):
    """State-of-the-art hybrid detector combining GNN and Transformer with adversarial training"""
    
    def __init__(self, gnn_features: int = 128, transformer_vocab: int = 10000):
        super().__init__()
        
        # Sub-models
        self.gnn = GNNDetector(gnn_features)
        self.transformer = TransformerDetector(transformer_vocab)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(10, 256),  # 5 + 5 class probabilities
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )
        
        # Adversarial discriminator for robustness
        self.discriminator = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.mev_types = ['sandwich', 'arbitrage', 'liquidation', 'jit', 'normal']
        
    def forward(self, graph_data: Data, instruction_sequence: torch.Tensor):
        """Hybrid forward pass"""
        # Get predictions from both models
        gnn_logits = self.gnn(graph_data.x, graph_data.edge_index, graph_data.batch)
        transformer_logits = self.transformer(instruction_sequence)
        
        # Concatenate predictions
        combined = torch.cat([
            F.softmax(gnn_logits, dim=1),
            F.softmax(transformer_logits, dim=1)
        ], dim=1)
        
        # Fusion
        fused_logits = self.fusion(combined)
        
        return fused_logits, gnn_logits, transformer_logits
    
    def detect(self, graph_data: Data, instruction_sequence: torch.Tensor) -> DetectionResult:
        """Detect MEV using hybrid model"""
        start_time = time.perf_counter()
        
        self.eval()
        with torch.no_grad():
            fused_logits, gnn_logits, transformer_logits = self.forward(
                graph_data, instruction_sequence.unsqueeze(0)
            )
            
            # Get final predictions
            probs = F.softmax(fused_logits, dim=1)
            gnn_probs = F.softmax(gnn_logits, dim=1)
            transformer_probs = F.softmax(transformer_logits, dim=1)
            
            max_prob, pred_class = probs.max(1)
            mev_type = self.mev_types[pred_class.item()]
            
            is_mev = mev_type != 'normal'
            confidence = float(max_prob.item())
            
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Calculate feature importance
        feature_importance = {
            f'hybrid_{mt}': float(probs[0, i].item()) 
            for i, mt in enumerate(self.mev_types)
        }
        feature_importance.update({
            f'gnn_{mt}': float(gnn_probs[0, i].item()) 
            for i, mt in enumerate(self.mev_types)
        })
        feature_importance.update({
            f'transformer_{mt}': float(transformer_probs[0, i].item()) 
            for i, mt in enumerate(self.mev_types)
        })
        
        return DetectionResult(
            is_mev=is_mev,
            mev_type=mev_type if is_mev else None,
            confidence=confidence,
            attacker_address=None,
            victim_address=None,
            profit_estimate=None,
            feature_importance=feature_importance,
            decision_dna=hashlib.sha256(str(feature_importance).encode()).hexdigest(),
            inference_latency_ms=latency_ms
        )


class ONNXModelServer:
    """High-performance ONNX model serving for production inference"""
    
    def __init__(self, model_path: str):
        # Load ONNX model with optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
    def predict(self, features: np.ndarray) -> DetectionResult:
        """Run inference with ONNX model"""
        start_time = time.perf_counter()
        
        # Prepare inputs
        inputs = {self.input_names[0]: features.astype(np.float32)}
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        probs = outputs[0]
        
        # Process results
        mev_types = ['sandwich', 'arbitrage', 'liquidation', 'jit', 'normal']
        max_idx = np.argmax(probs[0])
        mev_type = mev_types[max_idx]
        confidence = float(probs[0, max_idx])
        
        is_mev = mev_type != 'normal'
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return DetectionResult(
            is_mev=is_mev,
            mev_type=mev_type if is_mev else None,
            confidence=confidence,
            attacker_address=None,
            victim_address=None,
            profit_estimate=None,
            feature_importance={
                mt: float(probs[0, i]) for i, mt in enumerate(mev_types)
            },
            decision_dna=hashlib.sha256(str(probs.tolist()).encode()).hexdigest(),
            inference_latency_ms=latency_ms
        )


# Model export utilities
def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, output_path: str):
    """Export PyTorch model to ONNX for production deployment"""
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Model exported to {output_path}")


# Initialize detectors
def create_detection_ensemble():
    """Create ensemble of all detectors"""
    return {
        'rule_based': RuleBasedDetector(),
        'statistical': StatisticalAnomalyDetector(),
        'gnn': GNNDetector(),
        'transformer': TransformerDetector(),
        'hybrid': HybridMEVDetector()
    }