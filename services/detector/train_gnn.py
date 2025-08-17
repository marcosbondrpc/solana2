#!/usr/bin/env python3
"""
Graph Neural Network for MEV Sandwich Detection
DETECTION-ONLY: Pure observability, no execution
Target: ROC-AUC ≥0.95, FPR <0.5%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
import clickhouse_driver
import onnx
import torch.onnx
from typing import List, Tuple, Dict
import hashlib
import json
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SandwichGNN(nn.Module):
    """
    Graph Neural Network for sandwich attack detection
    Processes transaction graphs with account relationships
    """
    def __init__(self, node_features: int = 128, hidden_dim: int = 256, num_layers: int = 3):
        super(SandwichGNN, self).__init__()
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final graph layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # [not_sandwich, sandwich, confidence]
        )
        
        # Edge weight learning
        self.edge_weight_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Learn edge weights
        if edge_attr is not None:
            edge_weights = self.edge_weight_net(edge_attr).squeeze()
        else:
            edge_weights = None
        
        # Graph convolutions with residual connections
        identity = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index, edge_weights)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            
            # Residual connection every 2 layers
            if i % 2 == 1 and i < len(self.convs) - 1:
                if identity.shape == x.shape:
                    x = x + identity
                identity = x
        
        # Self-attention on node features
        if batch is not None:
            # Reshape for attention
            batch_size = batch.max().item() + 1
            x_reshaped = x.view(batch_size, -1, x.size(-1))
            attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            x = attn_out.view(-1, x.size(-1))
        
        # Global graph pooling
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0)[0].unsqueeze(0)
            x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        out = self.classifier(x)
        return out

class TransactionGraphBuilder:
    """Builds graph representations from transaction data"""
    
    def __init__(self, ch_client):
        self.ch = ch_client
        self.program_encodings = {}
        self.account_features = {}
        
    def build_graph(self, slot: int, window_size: int = 10) -> Data:
        """Build graph from transactions in slot window"""
        
        # Fetch transactions
        query = f"""
        SELECT 
            sig, payer, programs, accounts, 
            amount_in, amount_out, fee, priority_fee,
            token_in, token_out, pool_keys
        FROM ch.raw_tx
        WHERE slot >= {slot - window_size} AND slot <= {slot + window_size}
        ORDER BY slot, ts
        """
        
        txs = self.ch.execute(query)
        if not txs:
            return None
            
        # Build node features and edges
        nodes = {}
        edges = []
        
        for tx in txs:
            sig, payer, programs, accounts, amt_in, amt_out, fee, prio_fee, tok_in, tok_out, pools = tx
            
            # Add nodes for accounts
            for acc in accounts:
                if acc not in nodes:
                    nodes[acc] = self._get_account_features(acc)
            
            # Add edges for account interactions
            for i, acc1 in enumerate(accounts):
                for acc2 in accounts[i+1:]:
                    edges.append((acc1, acc2, self._compute_edge_features(acc1, acc2, tx)))
        
        # Convert to tensors
        node_list = list(nodes.keys())
        node_idx = {n: i for i, n in enumerate(node_list)}
        
        x = torch.tensor([nodes[n] for n in node_list], dtype=torch.float32)
        
        edge_index = []
        edge_attr = []
        for src, dst, feat in edges:
            if src in node_idx and dst in node_idx:
                edge_index.append([node_idx[src], node_idx[dst]])
                edge_attr.append(feat)
        
        if not edge_index:
            return None
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _get_account_features(self, account: str) -> List[float]:
        """Extract features for an account"""
        if account in self.account_features:
            return self.account_features[account]
        
        # Query account statistics
        query = f"""
        SELECT 
            count() as tx_count,
            sum(fee) as total_fees,
            avg(amount_in) as avg_amount_in,
            avg(amount_out) as avg_amount_out,
            uniqExact(token_in) as unique_tokens_in,
            uniqExact(token_out) as unique_tokens_out
        FROM ch.raw_tx
        WHERE '{account}' IN accounts
        AND ts >= now() - INTERVAL 1 DAY
        """
        
        stats = self.ch.execute(query)
        if stats and stats[0]:
            features = list(stats[0])
        else:
            features = [0] * 6
        
        # Pad to 128 dimensions
        features.extend([0] * (128 - len(features)))
        
        self.account_features[account] = features
        return features
    
    def _compute_edge_features(self, acc1: str, acc2: str, tx: tuple) -> List[float]:
        """Compute edge features between two accounts"""
        # Simple features: amount flow and fee ratio
        amt_in = tx[4] if tx[4] else 0
        fee = tx[6] if tx[6] else 0
        
        return [amt_in / 1e9, fee / 1e6]  # Normalize to SOL

class DatasetBuilder:
    """Build training dataset from ClickHouse"""
    
    def __init__(self, ch_host='localhost', ch_port=9000):
        self.ch = clickhouse_driver.Client(
            host=ch_host,
            port=ch_port,
            settings={'use_numpy': True}
        )
        self.graph_builder = TransactionGraphBuilder(self.ch)
        
    def build_dataset(self, start_date: str, end_date: str) -> Tuple[List[Data], List[int]]:
        """Build dataset of graphs with labels"""
        
        # Fetch confirmed sandwich attacks
        query = f"""
        SELECT 
            slot,
            victim_sig,
            attacker_a_sig,
            attacker_b_sig,
            ensemble_score
        FROM ch.candidates
        WHERE detection_ts >= '{start_date}'
          AND detection_ts <= '{end_date}'
          AND ensemble_score > 0.8
        """
        
        sandwiches = self.ch.execute(query)
        logger.info(f"Found {len(sandwiches)} sandwich attacks")
        
        graphs = []
        labels = []
        
        # Positive samples (sandwiches)
        for slot, victim, front, back, score in sandwiches[:1000]:  # Limit for training
            graph = self.graph_builder.build_graph(slot)
            if graph:
                graphs.append(graph)
                labels.append(1)
        
        # Negative samples (normal transactions)
        query = f"""
        SELECT DISTINCT slot
        FROM ch.raw_tx
        WHERE ts >= '{start_date}'
          AND ts <= '{end_date}'
          AND slot NOT IN (
              SELECT slot FROM ch.candidates 
              WHERE detection_ts >= '{start_date}' AND detection_ts <= '{end_date}'
          )
        ORDER BY rand()
        LIMIT {len(graphs)}
        """
        
        normal_slots = self.ch.execute(query)
        for (slot,) in normal_slots:
            graph = self.graph_builder.build_graph(slot)
            if graph:
                graphs.append(graph)
                labels.append(0)
        
        logger.info(f"Built dataset with {len(graphs)} graphs")
        return graphs, labels

def train_model():
    """Main training pipeline"""
    
    # Build dataset
    logger.info("Building dataset...")
    dataset_builder = DatasetBuilder(ch_host='clickhouse')
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    graphs, labels = dataset_builder.build_dataset(start_date, end_date)
    
    if not graphs:
        logger.error("No data found for training")
        return
    
    # Split dataset
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        graphs, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        [(g, l) for g, l in zip(train_graphs, train_labels)],
        batch_size=32,
        shuffle=True
    )
    
    test_loader = DataLoader(
        [(g, l) for g, l in zip(test_graphs, test_labels)],
        batch_size=32,
        shuffle=False
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SandwichGNN().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 1.0]).to(device))  # Weight sandwich class
    
    # Training loop
    logger.info("Starting training...")
    best_auc = 0
    
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        for batch in train_loader:
            graphs_batch, labels_batch = batch
            
            # Combine batch
            x = torch.cat([g.x for g in graphs_batch])
            edge_index = torch.cat([g.edge_index + offset for g, offset in 
                                   zip(graphs_batch, self._compute_offsets(graphs_batch))], dim=1)
            batch_idx = torch.cat([torch.full((g.x.size(0),), i) for i, g in enumerate(graphs_batch)])
            
            x = x.to(device)
            edge_index = edge_index.to(device)
            batch_idx = batch_idx.to(device)
            labels_batch = torch.tensor(labels_batch).to(device)
            
            optimizer.zero_grad()
            out = model(x, edge_index, batch=batch_idx)
            loss = criterion(out, labels_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(F.softmax(out, dim=1)[:, 1].cpu().detach().numpy())
            train_true.extend(labels_batch.cpu().numpy())
        
        scheduler.step()
        
        # Validation
        model.eval()
        test_preds = []
        test_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                graphs_batch, labels_batch = batch
                
                x = torch.cat([g.x for g in graphs_batch])
                edge_index = torch.cat([g.edge_index + offset for g, offset in 
                                       zip(graphs_batch, self._compute_offsets(graphs_batch))], dim=1)
                batch_idx = torch.cat([torch.full((g.x.size(0),), i) for i, g in enumerate(graphs_batch)])
                
                x = x.to(device)
                edge_index = edge_index.to(device)
                batch_idx = batch_idx.to(device)
                
                out = model(x, edge_index, batch=batch_idx)
                probs = F.softmax(out, dim=1)[:, 1]
                
                test_preds.extend(probs.cpu().numpy())
                test_true.extend(labels_batch)
        
        # Calculate metrics
        train_auc = roc_auc_score(train_true, train_preds)
        test_auc = roc_auc_score(test_true, test_preds)
        
        # Calculate false positive rate at 95% recall
        precision, recall, thresholds = precision_recall_curve(test_true, test_preds)
        idx_95_recall = np.where(recall >= 0.95)[0]
        if len(idx_95_recall) > 0:
            fpr_at_95_recall = 1 - precision[idx_95_recall[-1]]
        else:
            fpr_at_95_recall = 1.0
        
        logger.info(f"Epoch {epoch+1}: Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}, FPR@95={fpr_at_95_recall:.4f}")
        
        # Save best model
        if test_auc > best_auc and test_auc >= 0.95 and fpr_at_95_recall < 0.005:
            best_auc = test_auc
            logger.info(f"New best model! AUC={test_auc:.4f}, FPR={fpr_at_95_recall:.4f}")
            
            # Export to ONNX
            dummy_input = (
                torch.randn(128, 128).to(device),
                torch.randint(0, 128, (2, 256)).to(device),
                None,
                torch.zeros(128, dtype=torch.long).to(device)
            )
            
            torch.onnx.export(
                model,
                dummy_input,
                "/home/kidgordones/0solana/solana2/models/sandwich_gnn.onnx",
                export_params=True,
                opset_version=11,
                input_names=['x', 'edge_index', 'edge_attr', 'batch'],
                output_names=['output'],
                dynamic_axes={
                    'x': {0: 'num_nodes'},
                    'edge_index': {1: 'num_edges'},
                    'batch': {0: 'num_nodes'}
                }
            )
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': test_auc,
                'fpr': fpr_at_95_recall
            }, "/home/kidgordones/0solana/solana2/models/sandwich_gnn.pth")
    
    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")

def _compute_offsets(graphs):
    """Compute node offsets for batching"""
    offsets = [0]
    for g in graphs[:-1]:
        offsets.append(offsets[-1] + g.x.size(0))
    return offsets

if __name__ == "__main__":
    train_model()