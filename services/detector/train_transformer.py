#!/usr/bin/env python3
"""
Transformer Model for MEV Instruction Sequence Analysis
DETECTION-ONLY: Pure pattern recognition
Target: ROC-AUC ≥0.95, Inference <100μs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import clickhouse_driver
import onnx
import torch.onnx
from typing import List, Tuple, Dict, Optional
import hashlib
import json
import logging
from datetime import datetime, timedelta
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instruction vocabulary for Solana programs
INSTRUCTION_VOCAB = {
    'PAD': 0,
    'UNK': 1,
    # System Program
    'SYSTEM_TRANSFER': 2,
    'SYSTEM_CREATE': 3,
    # Token Program
    'TOKEN_TRANSFER': 4,
    'TOKEN_APPROVE': 5,
    'TOKEN_REVOKE': 6,
    'TOKEN_MINT': 7,
    'TOKEN_BURN': 8,
    # Associated Token
    'ATA_CREATE': 9,
    # Raydium AMM
    'RAY_SWAP': 10,
    'RAY_ADD_LIQ': 11,
    'RAY_REMOVE_LIQ': 12,
    # Jupiter
    'JUP_ROUTE': 13,
    'JUP_SWAP': 14,
    # Orca
    'ORCA_SWAP': 15,
    'ORCA_COLLECT': 16,
    # Serum DEX
    'SERUM_PLACE': 17,
    'SERUM_CANCEL': 18,
    'SERUM_SETTLE': 19,
    # Pump.fun
    'PUMP_BUY': 20,
    'PUMP_SELL': 21,
    # Jito
    'JITO_TIP': 22,
    'JITO_BUNDLE': 23,
    # Metaplex
    'META_MINT_NFT': 24,
    'META_TRANSFER_NFT': 25,
    # Common MEV patterns
    'SANDWICH_FRONT': 100,
    'SANDWICH_BACK': 101,
    'BACKRUN': 102,
    'LIQUIDATION': 103,
    'ARBITRAGE': 104
}

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class MEVTransformer(nn.Module):
    """
    Transformer for MEV pattern detection in instruction sequences
    Optimized for ultra-low latency inference
    """
    
    def __init__(self, 
                 vocab_size: int = 256,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 max_seq_len: int = 128,
                 num_classes: int = 5):  # [normal, sandwich, backrun, liquidation, arbitrage]
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Pattern detection heads
        self.pattern_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Confidence scoring
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Attention aggregation
        self.attention_pool = nn.MultiheadAttention(
            d_model, 
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # MEV-specific pattern kernels (learned)
        self.pattern_kernels = nn.Parameter(torch.randn(5, d_model))
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len]
        
        # Embedding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.position_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Create attention mask for padding
        if mask is None:
            mask = (x == 0).all(dim=-1)
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=mask)
        
        # Pattern-specific attention
        pattern_attn, _ = self.attention_pool(
            self.pattern_kernels.unsqueeze(0).expand(x.size(0), -1, -1),
            encoded,
            encoded,
            key_padding_mask=mask
        )
        
        # Global pooling (mean + max)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            masked_encoded = encoded * mask_expanded
            seq_lengths = mask_expanded.sum(dim=1)
            mean_pooled = masked_encoded.sum(dim=1) / seq_lengths
        else:
            mean_pooled = encoded.mean(dim=1)
        
        max_pooled = encoded.max(dim=1)[0]
        
        # Combine pooling strategies
        pooled = mean_pooled + max_pooled + pattern_attn.mean(dim=1)
        
        # Classification
        pattern_logits = self.pattern_detector(pooled)
        confidence = self.confidence_head(pooled)
        
        return pattern_logits, confidence

class InstructionSequenceDataset(Dataset):
    """Dataset for instruction sequences from ClickHouse"""
    
    def __init__(self, sequences: List[List[int]], labels: List[int], max_len: int = 128):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Pad or truncate sequence
        if len(seq) < self.max_len:
            seq = seq + [INSTRUCTION_VOCAB['PAD']] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SequenceEncoder:
    """Encode transaction instructions into sequences"""
    
    def __init__(self):
        self.program_mappings = {
            '11111111111111111111111111111111': 'SYSTEM',
            'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA': 'TOKEN',
            'ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL': 'ATA',
            '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8': 'RAY',
            'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4': 'JUP',
            'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc': 'ORCA',
            '9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin': 'SERUM',
            '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P': 'PUMP',
            'T1pyyaTNZsKv2WcRAB8oVnk93mLJw2XzjtVYqCsaHqt': 'JITO'
        }
        
        self.ix_type_mappings = {
            ('SYSTEM', 0): 'SYSTEM_TRANSFER',
            ('SYSTEM', 1): 'SYSTEM_CREATE',
            ('TOKEN', 3): 'TOKEN_TRANSFER',
            ('TOKEN', 4): 'TOKEN_APPROVE',
            ('TOKEN', 5): 'TOKEN_REVOKE',
            ('TOKEN', 7): 'TOKEN_MINT',
            ('TOKEN', 8): 'TOKEN_BURN',
            ('ATA', 0): 'ATA_CREATE',
            ('RAY', 9): 'RAY_SWAP',
            ('JUP', 0): 'JUP_ROUTE',
            ('PUMP', 0): 'PUMP_BUY',
            ('PUMP', 1): 'PUMP_SELL',
            ('JITO', 0): 'JITO_TIP'
        }
    
    def encode_transaction(self, programs: List[str], ix_kinds: List[int]) -> List[int]:
        """Encode a transaction's instructions into sequence"""
        sequence = []
        
        for prog, ix_type in zip(programs, ix_kinds):
            # Map program to name
            prog_name = self.program_mappings.get(prog, 'UNK')
            
            # Map instruction type
            ix_key = (prog_name, ix_type)
            if ix_key in self.ix_type_mappings:
                ix_name = self.ix_type_mappings[ix_key]
                sequence.append(INSTRUCTION_VOCAB.get(ix_name, INSTRUCTION_VOCAB['UNK']))
            else:
                sequence.append(INSTRUCTION_VOCAB['UNK'])
        
        return sequence
    
    def detect_sandwich_pattern(self, sequence: List[int]) -> bool:
        """Heuristic detection of sandwich pattern"""
        # Look for swap-swap-swap pattern with same pool
        swap_indices = [10, 14, 15, 20, 21]  # All swap instructions
        
        swap_positions = []
        for i, ix in enumerate(sequence):
            if ix in swap_indices:
                swap_positions.append(i)
        
        # Check for 3 swaps in quick succession
        if len(swap_positions) >= 3:
            for i in range(len(swap_positions) - 2):
                if swap_positions[i+2] - swap_positions[i] <= 5:  # Within 5 instructions
                    return True
        
        return False

class DatasetBuilder:
    """Build training dataset from ClickHouse"""
    
    def __init__(self, ch_host='localhost', ch_port=9000):
        self.ch = clickhouse_driver.Client(
            host=ch_host,
            port=ch_port,
            settings={'use_numpy': True}
        )
        self.encoder = SequenceEncoder()
    
    def build_dataset(self, start_date: str, end_date: str, limit: int = 10000) -> Tuple[List[List[int]], List[int]]:
        """Build dataset of instruction sequences with labels"""
        
        sequences = []
        labels = []
        
        # Fetch sandwich attacks
        query = f"""
        SELECT DISTINCT
            c.slot,
            c.victim_sig,
            c.attacker_a_sig,
            c.attacker_b_sig
        FROM ch.candidates c
        WHERE c.detection_ts >= '{start_date}'
          AND c.detection_ts <= '{end_date}'
          AND c.ensemble_score > 0.8
        LIMIT {limit // 2}
        """
        
        sandwiches = self.ch.execute(query)
        logger.info(f"Processing {len(sandwiches)} sandwich attacks")
        
        for slot, victim, front, back in sandwiches:
            # Get instruction sequences for all three transactions
            query = f"""
            SELECT programs, ix_kinds
            FROM ch.raw_tx
            WHERE sig IN ('{front}', '{victim}', '{back}')
            ORDER BY slot, ts
            """
            
            txs = self.ch.execute(query)
            if len(txs) == 3:
                # Combine sequences
                combined_seq = []
                for programs, ix_kinds in txs:
                    seq = self.encoder.encode_transaction(programs, ix_kinds)
                    combined_seq.extend(seq)
                
                sequences.append(combined_seq)
                labels.append(1)  # Sandwich
        
        # Fetch normal transactions
        query = f"""
        SELECT 
            programs,
            ix_kinds
        FROM ch.raw_tx
        WHERE ts >= '{start_date}'
          AND ts <= '{end_date}'
          AND slot NOT IN (
              SELECT slot FROM ch.candidates 
              WHERE detection_ts >= '{start_date}' AND detection_ts <= '{end_date}'
          )
        ORDER BY rand()
        LIMIT {limit // 2}
        """
        
        normal_txs = self.ch.execute(query)
        logger.info(f"Processing {len(normal_txs)} normal transactions")
        
        for programs, ix_kinds in normal_txs:
            seq = self.encoder.encode_transaction(programs, ix_kinds)
            sequences.append(seq)
            labels.append(0)  # Normal
        
        return sequences, labels

def train_model():
    """Main training pipeline"""
    
    # Build dataset
    logger.info("Building dataset from ClickHouse...")
    dataset_builder = DatasetBuilder(ch_host='clickhouse')
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    sequences, labels = dataset_builder.build_dataset(start_date, end_date, limit=20000)
    
    if not sequences:
        logger.error("No data found for training")
        return
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Create datasets and loaders
    train_dataset = InstructionSequenceDataset(X_train, y_train)
    test_dataset = InstructionSequenceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MEVTransformer(
        vocab_size=256,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        max_seq_len=128,
        num_classes=5
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=50,
        steps_per_epoch=len(train_loader)
    )
    
    # Class weights for imbalanced dataset
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    logger.info("Starting transformer training...")
    best_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Create padding mask
            mask = (sequences == INSTRUCTION_VOCAB['PAD'])
            
            # Forward pass
            pattern_logits, confidence = model(sequences, mask)
            
            # Multi-class to binary for sandwich detection
            binary_labels = (labels > 0).long()
            loss = criterion(pattern_logits[:, :2], binary_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Store predictions
            probs = F.softmax(pattern_logits[:, :2], dim=1)[:, 1]
            train_preds.extend(probs.cpu().detach().numpy())
            train_true.extend(binary_labels.cpu().numpy())
        
        # Validation
        model.eval()
        test_preds = []
        test_true = []
        test_confidences = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                mask = (sequences == INSTRUCTION_VOCAB['PAD'])
                pattern_logits, confidence = model(sequences, mask)
                
                binary_labels = (labels > 0).long()
                probs = F.softmax(pattern_logits[:, :2], dim=1)[:, 1]
                
                test_preds.extend(probs.cpu().numpy())
                test_true.extend(binary_labels.cpu().numpy())
                test_confidences.extend(confidence.squeeze().cpu().numpy())
        
        # Calculate metrics
        train_auc = roc_auc_score(train_true, train_preds)
        test_auc = roc_auc_score(test_true, test_preds)
        
        # Calculate FPR at 95% recall
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(test_true, test_preds)
        idx_95 = np.where(recall >= 0.95)[0]
        if len(idx_95) > 0:
            fpr_at_95 = 1 - precision[idx_95[-1]]
        else:
            fpr_at_95 = 1.0
        
        avg_confidence = np.mean(test_confidences)
        
        logger.info(f"Epoch {epoch+1}: Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}, "
                   f"FPR@95={fpr_at_95:.4f}, Avg Confidence={avg_confidence:.3f}")
        
        # Save best model
        if test_auc > best_auc and test_auc >= 0.95 and fpr_at_95 < 0.005:
            best_auc = test_auc
            patience_counter = 0
            
            logger.info(f"New best model! Saving with AUC={test_auc:.4f}")
            
            # Export to ONNX for production inference
            dummy_input = torch.randint(0, 256, (1, 128)).to(device)
            torch.onnx.export(
                model,
                (dummy_input, None),
                "/home/kidgordones/0solana/solana2/models/mev_transformer.onnx",
                export_params=True,
                opset_version=11,
                input_names=['sequence', 'mask'],
                output_names=['pattern_logits', 'confidence'],
                dynamic_axes={
                    'sequence': {0: 'batch_size'},
                    'mask': {0: 'batch_size'}
                }
            )
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'auc': test_auc,
                'fpr': fpr_at_95
            }, "/home/kidgordones/0solana/solana2/models/mev_transformer.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
    
    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_model()