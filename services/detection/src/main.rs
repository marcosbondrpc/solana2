//! GNN + Transformer Detection Service
//! DEFENSIVE-ONLY: Pure detection with <100μs inference
//! Thompson Sampling for route optimization

#![feature(portable_simd)]

use ahash::{AHashMap, AHashSet};
use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use lru::LruCache;
use mimalloc::MiMalloc;
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
use once_cell::sync::Lazy;
use ordered_float::OrderedFloat;
use parking_lot::{Mutex, RwLock};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::{dijkstra, strongly_connected_components};
use priority_queue::PriorityQueue;
use prometheus::{register_histogram_vec, register_int_counter_vec, HistogramVec, IntCounterVec};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Beta, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use statrs::distribution::{Beta as BetaDist, ContinuousCDF};
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, watch, Semaphore};
use tokio::task::{spawn_blocking, JoinHandle};
use tokio::time::{interval, sleep};
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Metrics
static INFERENCE_METRICS: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "detection_inference_us",
        "Model inference latency in microseconds",
        &["model"],
        vec![10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 500.0]
    )
    .unwrap()
});

static DETECTION_COUNTER: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec!("detection_total", "Total detections", &["type"]).unwrap()
});

/// Transaction graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxNode {
    pub tx_hash: String,
    pub block_number: u64,
    pub timestamp: u64,
    pub from_address: String,
    pub to_address: String,
    pub value: f64,
    pub gas_price: f64,
    pub features: Vec<f32>,
}

/// Edge in transaction graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxEdge {
    pub edge_type: EdgeType,
    pub weight: f32,
    pub temporal_distance: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    Transfer,
    Contract,
    Swap,
    FlashLoan,
    Liquidation,
}

/// Graph Neural Network for transaction flow analysis
pub struct TransactionGNN {
    node_embedding_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
    weights: Arc<RwLock<GNNWeights>>,
    cache: Arc<Mutex<LruCache<String, Array2<f32>>>>,
}

struct GNNWeights {
    node_transform: Array2<f32>,
    edge_transform: Array2<f32>,
    attention_weights: Array2<f32>,
    output_layer: Array2<f32>,
}

impl TransactionGNN {
    pub fn new(embedding_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        
        // Initialize weights with Xavier initialization
        let xavier_init = |rows: usize, cols: usize| -> Array2<f32> {
            let scale = (2.0 / (rows + cols) as f32).sqrt();
            Array2::from_shape_fn((rows, cols), |_| {
                rng.gen_range(-scale..scale)
            })
        };
        
        let weights = GNNWeights {
            node_transform: xavier_init(embedding_dim, hidden_dim),
            edge_transform: xavier_init(embedding_dim, hidden_dim),
            attention_weights: xavier_init(hidden_dim * 2, 1),
            output_layer: xavier_init(hidden_dim, 3), // 3 output classes
        };
        
        Self {
            node_embedding_dim: embedding_dim,
            hidden_dim,
            num_layers,
            weights: Arc::new(RwLock::new(weights)),
            cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(1000).unwrap()))),
        }
    }

    /// Ultra-fast inference (<100μs)
    #[inline(always)]
    pub fn infer(&self, graph: &DiGraph<TxNode, TxEdge>) -> DetectionResult {
        let start = Instant::now();
        
        // Check cache first
        let graph_hash = self.compute_graph_hash(graph);
        if let Some(cached) = self.cache.lock().get(&graph_hash) {
            return self.interpret_output(cached.clone());
        }
        
        // Convert graph to tensor representation
        let (node_features, adjacency) = self.graph_to_tensors(graph);
        
        // Message passing layers
        let mut hidden = node_features.clone();
        let weights = self.weights.read();
        
        for _ in 0..self.num_layers {
            // Graph convolution with attention
            hidden = self.graph_convolution(&hidden, &adjacency, &weights);
            
            // ReLU activation (vectorized)
            hidden.mapv_inplace(|x| x.max(0.0));
        }
        
        // Global pooling
        let pooled = hidden.mean_axis(Axis(0)).unwrap();
        
        // Output layer
        let output = pooled.dot(&weights.output_layer);
        
        // Cache result
        self.cache.lock().put(graph_hash, output.clone().insert_axis(Axis(0)));
        
        let elapsed = start.elapsed();
        INFERENCE_METRICS
            .with_label_values(&["gnn"])
            .observe(elapsed.as_micros() as f64);
        
        self.interpret_output(output.insert_axis(Axis(0)))
    }

    fn graph_to_tensors(&self, graph: &DiGraph<TxNode, TxEdge>) -> (Array2<f32>, Array2<f32>) {
        let n = graph.node_count();
        let mut node_features = Array2::zeros((n, self.node_embedding_dim));
        let mut adjacency = Array2::zeros((n, n));
        
        // Fill node features
        for (i, node_idx) in graph.node_indices().enumerate() {
            if let Some(node) = graph.node_weight(node_idx) {
                for (j, &feat) in node.features.iter().take(self.node_embedding_dim).enumerate() {
                    node_features[[i, j]] = feat;
                }
            }
        }
        
        // Fill adjacency matrix
        for edge in graph.edge_references() {
            let source = edge.source().index();
            let target = edge.target().index();
            adjacency[[source, target]] = edge.weight().weight;
        }
        
        (node_features, adjacency)
    }

    fn graph_convolution(
        &self,
        features: &Array2<f32>,
        adjacency: &Array2<f32>,
        weights: &GNNWeights,
    ) -> Array2<f32> {
        // Compute attention scores
        let transformed = features.dot(&weights.node_transform);
        let neighbor_sum = adjacency.dot(&transformed);
        
        // Apply attention mechanism
        let attention = self.compute_attention(&transformed, &neighbor_sum, weights);
        
        // Weighted aggregation
        transformed * (1.0 - 0.5) + neighbor_sum * 0.5 * attention
    }

    fn compute_attention(
        &self,
        node_features: &Array2<f32>,
        neighbor_features: &Array2<f32>,
        weights: &GNNWeights,
    ) -> Array2<f32> {
        let n = node_features.nrows();
        let mut attention = Array2::zeros((n, 1));
        
        for i in 0..n {
            let concat = concatenate(
                node_features.row(i).to_owned(),
                neighbor_features.row(i).to_owned(),
            );
            let score = concat.dot(&weights.attention_weights.column(0));
            attention[[i, 0]] = score.tanh(); // Squash to [-1, 1]
        }
        
        softmax_2d(&attention)
    }

    fn compute_graph_hash(&self, graph: &DiGraph<TxNode, TxEdge>) -> String {
        let mut hasher = blake3::Hasher::new();
        for node in graph.node_weights() {
            hasher.update(node.tx_hash.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    fn interpret_output(&self, output: Array2<f32>) -> DetectionResult {
        let probs = softmax_2d(&output);
        let class_idx = argmax(&probs.row(0).to_owned());
        
        let detection_type = match class_idx {
            0 => DetectionType::Normal,
            1 => DetectionType::Suspicious,
            2 => DetectionType::Malicious,
            _ => DetectionType::Unknown,
        };
        
        DetectionResult {
            detection_type,
            confidence: probs[[0, class_idx]],
            features: output.row(0).to_vec(),
        }
    }
}

/// Transformer for temporal pattern detection
pub struct TemporalTransformer {
    embed_dim: usize,
    num_heads: usize,
    ff_dim: usize,
    weights: Arc<RwLock<TransformerWeights>>,
    cache: Arc<Mutex<LruCache<String, Array3<f32>>>>,
}

struct TransformerWeights {
    q_proj: Array2<f32>,
    k_proj: Array2<f32>,
    v_proj: Array2<f32>,
    ff1: Array2<f32>,
    ff2: Array2<f32>,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,
}

impl TemporalTransformer {
    pub fn new(embed_dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        
        let weights = TransformerWeights {
            q_proj: random_matrix(embed_dim, embed_dim, &mut rng),
            k_proj: random_matrix(embed_dim, embed_dim, &mut rng),
            v_proj: random_matrix(embed_dim, embed_dim, &mut rng),
            ff1: random_matrix(embed_dim, ff_dim, &mut rng),
            ff2: random_matrix(ff_dim, embed_dim, &mut rng),
            layer_norm1: LayerNorm {
                gamma: Array1::ones(embed_dim),
                beta: Array1::zeros(embed_dim),
                eps: 1e-6,
            },
            layer_norm2: LayerNorm {
                gamma: Array1::ones(embed_dim),
                beta: Array1::zeros(embed_dim),
                eps: 1e-6,
            },
        };
        
        Self {
            embed_dim,
            num_heads,
            ff_dim,
            weights: Arc::new(RwLock::new(weights)),
            cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(1000).unwrap()))),
        }
    }

    /// Ultra-fast transformer inference
    #[inline(always)]
    pub fn infer(&self, sequence: &Array3<f32>) -> TemporalDetectionResult {
        let start = Instant::now();
        
        // Check cache
        let seq_hash = self.compute_sequence_hash(sequence);
        if let Some(cached) = self.cache.lock().get(&seq_hash) {
            return self.interpret_temporal_output(cached.clone());
        }
        
        let weights = self.weights.read();
        let batch_size = sequence.shape()[0];
        let seq_len = sequence.shape()[1];
        
        // Multi-head attention
        let q = self.project(sequence, &weights.q_proj);
        let k = self.project(sequence, &weights.k_proj);
        let v = self.project(sequence, &weights.v_proj);
        
        let attention_out = self.multi_head_attention(&q, &k, &v);
        
        // Add & norm
        let norm1 = self.layer_norm(&(sequence + &attention_out), &weights.layer_norm1);
        
        // Feed-forward
        let ff_out = self.feed_forward(&norm1, &weights.ff1, &weights.ff2);
        
        // Add & norm
        let output = self.layer_norm(&(&norm1 + &ff_out), &weights.layer_norm2);
        
        // Cache result
        self.cache.lock().put(seq_hash, output.clone());
        
        let elapsed = start.elapsed();
        INFERENCE_METRICS
            .with_label_values(&["transformer"])
            .observe(elapsed.as_micros() as f64);
        
        self.interpret_temporal_output(output)
    }

    fn project(&self, input: &Array3<f32>, weight: &Array2<f32>) -> Array3<f32> {
        let (batch, seq_len, _) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut output = Array3::zeros((batch, seq_len, self.embed_dim));
        
        for b in 0..batch {
            for s in 0..seq_len {
                let projected = input.slice(s![b, s, ..]).dot(weight);
                output.slice_mut(s![b, s, ..]).assign(&projected);
            }
        }
        
        output
    }

    fn multi_head_attention(&self, q: &Array3<f32>, k: &Array3<f32>, v: &Array3<f32>) -> Array3<f32> {
        let batch_size = q.shape()[0];
        let seq_len = q.shape()[1];
        let head_dim = self.embed_dim / self.num_heads;
        
        let mut output = Array3::zeros((batch_size, seq_len, self.embed_dim));
        
        // Simplified attention for speed
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                let start = h * head_dim;
                let end = start + head_dim;
                
                // Compute attention scores
                let q_h = q.slice(s![b, .., start..end]);
                let k_h = k.slice(s![b, .., start..end]);
                let v_h = v.slice(s![b, .., start..end]);
                
                let scores = q_h.dot(&k_h.t()) / (head_dim as f32).sqrt();
                let attention_weights = softmax_2d(&scores);
                
                let attended = attention_weights.dot(&v_h.to_owned().into_shape((seq_len, head_dim)).unwrap());
                output.slice_mut(s![b, .., start..end]).assign(&attended);
            }
        }
        
        output
    }

    fn feed_forward(&self, input: &Array3<f32>, w1: &Array2<f32>, w2: &Array2<f32>) -> Array3<f32> {
        let (batch, seq_len, _) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut output = Array3::zeros((batch, seq_len, self.embed_dim));
        
        for b in 0..batch {
            for s in 0..seq_len {
                let hidden = input.slice(s![b, s, ..]).dot(w1).mapv(|x| x.max(0.0)); // ReLU
                let out = hidden.dot(w2);
                output.slice_mut(s![b, s, ..]).assign(&out);
            }
        }
        
        output
    }

    fn layer_norm(&self, input: &Array3<f32>, norm: &LayerNorm) -> Array3<f32> {
        let mean = input.mean_axis(Axis(2)).unwrap();
        let var = input.var_axis(Axis(2), 0.0);
        
        let mut normalized = input.clone();
        for i in 0..input.shape()[0] {
            for j in 0..input.shape()[1] {
                let slice = input.slice(s![i, j, ..]);
                let m = mean[[i, j]];
                let v = var[[i, j]];
                let norm_slice = ((slice - m) / (v + norm.eps).sqrt()) * &norm.gamma + &norm.beta;
                normalized.slice_mut(s![i, j, ..]).assign(&norm_slice);
            }
        }
        
        normalized
    }

    fn compute_sequence_hash(&self, sequence: &Array3<f32>) -> String {
        let mut hasher = blake3::Hasher::new();
        for val in sequence.iter() {
            hasher.update(&val.to_le_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    fn interpret_temporal_output(&self, output: Array3<f32>) -> TemporalDetectionResult {
        // Pool over sequence dimension
        let pooled = output.mean_axis(Axis(1)).unwrap();
        let probs = softmax_2d(&pooled);
        
        TemporalDetectionResult {
            pattern_detected: probs[[0, 0]] > 0.7,
            confidence: probs[[0, 0]],
            temporal_features: pooled.row(0).to_vec(),
        }
    }
}

/// Thompson Sampling for route optimization
pub struct ThompsonSampler {
    arms: DashMap<String, BanditArm>,
    exploration_rate: f64,
    decay_factor: f64,
}

#[derive(Debug, Clone)]
struct BanditArm {
    successes: f64,
    failures: f64,
    total_reward: f64,
    last_selected: Instant,
}

impl ThompsonSampler {
    pub fn new(exploration_rate: f64) -> Self {
        Self {
            arms: DashMap::new(),
            exploration_rate,
            decay_factor: 0.99,
        }
    }

    pub fn select_route(&self, routes: &[String]) -> String {
        let mut rng = rand::thread_rng();
        let mut best_route = routes[0].clone();
        let mut best_sample = 0.0;
        
        for route in routes {
            let arm = self.arms.entry(route.clone()).or_insert(BanditArm {
                successes: 1.0,
                failures: 1.0,
                total_reward: 0.0,
                last_selected: Instant::now(),
            });
            
            // Thompson sampling from Beta distribution
            let beta = Beta::new(arm.successes, arm.failures).unwrap();
            let sample = beta.sample(&mut rng);
            
            // Add exploration bonus for rarely selected arms
            let time_bonus = arm.last_selected.elapsed().as_secs_f64() * 0.001;
            let adjusted_sample = sample + time_bonus * self.exploration_rate;
            
            if adjusted_sample > best_sample {
                best_sample = adjusted_sample;
                best_route = route.clone();
            }
        }
        
        DETECTION_COUNTER.with_label_values(&["route_selected"]).inc();
        best_route
    }

    pub fn update(&self, route: &str, reward: f64) {
        if let Some(mut arm) = self.arms.get_mut(route) {
            if reward > 0.0 {
                arm.successes += reward;
            } else {
                arm.failures += 1.0 - reward;
            }
            arm.total_reward += reward;
            arm.last_selected = Instant::now();
            
            // Apply decay to old observations
            arm.successes *= self.decay_factor;
            arm.failures *= self.decay_factor;
        }
    }
}

/// Hybrid detection engine combining GNN and Transformer
pub struct HybridDetectionEngine {
    gnn: Arc<TransactionGNN>,
    transformer: Arc<TemporalTransformer>,
    sampler: Arc<ThompsonSampler>,
    detection_threshold: f64,
}

impl HybridDetectionEngine {
    pub fn new() -> Self {
        Self {
            gnn: Arc::new(TransactionGNN::new(128, 256, 3)),
            transformer: Arc::new(TemporalTransformer::new(128, 8, 512)),
            sampler: Arc::new(ThompsonSampler::new(0.1)),
            detection_threshold: 0.75,
        }
    }

    pub async fn detect(&self, input: DetectionInput) -> DetectionOutput {
        let start = Instant::now();
        
        // Build transaction graph
        let graph = self.build_graph(&input.transactions);
        
        // Run GNN detection
        let gnn_result = self.gnn.infer(&graph);
        
        // Prepare temporal sequence
        let sequence = self.prepare_sequence(&input.transactions);
        
        // Run transformer detection
        let temporal_result = self.transformer.infer(&sequence);
        
        // Combine results
        let combined_confidence = (gnn_result.confidence + temporal_result.confidence) / 2.0;
        
        // Select optimal route using Thompson Sampling
        let routes = vec![
            "direct".to_string(),
            "batched".to_string(),
            "delayed".to_string(),
        ];
        let selected_route = self.sampler.select_route(&routes);
        
        // Update sampler based on confidence
        self.sampler.update(&selected_route, combined_confidence);
        
        let elapsed = start.elapsed();
        
        DetectionOutput {
            detected: combined_confidence > self.detection_threshold,
            confidence: combined_confidence,
            detection_type: gnn_result.detection_type,
            temporal_pattern: temporal_result.pattern_detected,
            recommended_route: selected_route,
            latency_us: elapsed.as_micros() as u64,
        }
    }

    fn build_graph(&self, transactions: &[Transaction]) -> DiGraph<TxNode, TxEdge> {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();
        
        // Add nodes
        for tx in transactions {
            let node = TxNode {
                tx_hash: tx.hash.clone(),
                block_number: tx.block_number,
                timestamp: tx.timestamp,
                from_address: tx.from.clone(),
                to_address: tx.to.clone(),
                value: tx.value,
                gas_price: tx.gas_price,
                features: tx.features.clone(),
            };
            let idx = graph.add_node(node);
            node_map.insert(tx.hash.clone(), idx);
        }
        
        // Add edges based on relationships
        for tx in transactions {
            if let Some(&from_idx) = node_map.get(&tx.hash) {
                // Find related transactions
                for other_tx in transactions {
                    if tx.hash != other_tx.hash {
                        if tx.to == other_tx.from || self.is_related(tx, other_tx) {
                            if let Some(&to_idx) = node_map.get(&other_tx.hash) {
                                let edge = TxEdge {
                                    edge_type: self.classify_edge(tx, other_tx),
                                    weight: self.compute_edge_weight(tx, other_tx),
                                    temporal_distance: other_tx.timestamp.saturating_sub(tx.timestamp),
                                };
                                graph.add_edge(from_idx, to_idx, edge);
                            }
                        }
                    }
                }
            }
        }
        
        graph
    }

    fn prepare_sequence(&self, transactions: &[Transaction]) -> Array3<f32> {
        let seq_len = transactions.len().min(100); // Max sequence length
        let embed_dim = 128;
        let mut sequence = Array3::zeros((1, seq_len, embed_dim));
        
        for (i, tx) in transactions.iter().take(seq_len).enumerate() {
            for (j, &feat) in tx.features.iter().take(embed_dim).enumerate() {
                sequence[[0, i, j]] = feat;
            }
        }
        
        sequence
    }

    fn is_related(&self, tx1: &Transaction, tx2: &Transaction) -> bool {
        // Check for various relationships
        tx1.to == tx2.from
            || tx1.from == tx2.to
            || (tx2.timestamp - tx1.timestamp) < 60 // Within 1 minute
    }

    fn classify_edge(&self, _tx1: &Transaction, _tx2: &Transaction) -> EdgeType {
        // Simplified classification
        EdgeType::Transfer
    }

    fn compute_edge_weight(&self, tx1: &Transaction, tx2: &Transaction) -> f32 {
        // Weight based on temporal proximity and value similarity
        let time_diff = (tx2.timestamp - tx1.timestamp) as f32;
        let value_sim = 1.0 / (1.0 + (tx1.value - tx2.value).abs() as f32);
        
        value_sim * (1.0 / (1.0 + time_diff / 60.0))
    }
}

// Helper functions
fn concatenate(a: Array1<f32>, b: Array1<f32>) -> Array1<f32> {
    let mut result = Array1::zeros(a.len() + b.len());
    result.slice_mut(s![..a.len()]).assign(&a);
    result.slice_mut(s![a.len()..]).assign(&b);
    result
}

fn softmax_2d(input: &Array2<f32>) -> Array2<f32> {
    let max = input.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp = input.mapv(|x| (x - max).exp());
    let sum = exp.sum();
    exp / sum
}

fn argmax(arr: &Array1<f32>) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Array2<f32> {
    let scale = (2.0 / (rows + cols) as f32).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
}

// API types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: String,
    pub block_number: u64,
    pub timestamp: u64,
    pub from: String,
    pub to: String,
    pub value: f64,
    pub gas_price: f64,
    pub features: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionInput {
    pub transactions: Vec<Transaction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionOutput {
    pub detected: bool,
    pub confidence: f64,
    pub detection_type: DetectionType,
    pub temporal_pattern: bool,
    pub recommended_route: String,
    pub latency_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionType {
    Normal,
    Suspicious,
    Malicious,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub detection_type: DetectionType,
    pub confidence: f64,
    pub features: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TemporalDetectionResult {
    pub pattern_detected: bool,
    pub confidence: f64,
    pub temporal_features: Vec<f32>,
}

// API handlers
async fn detect_handler(
    State(engine): State<Arc<HybridDetectionEngine>>,
    Json(input): Json<DetectionInput>,
) -> Json<DetectionOutput> {
    let output = engine.detect(input).await;
    DETECTION_COUNTER.with_label_values(&["api_request"]).inc();
    Json(output)
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "detection-service",
        "defensive_only": true
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info,detection=debug")
        .json()
        .init();

    info!("Starting GNN + Transformer Detection Service (DEFENSIVE-ONLY)");
    info!("Target: <100μs inference latency");

    // Initialize detection engine
    let engine = Arc::new(HybridDetectionEngine::new());

    // Build API router
    let app = Router::new()
        .route("/api/v1/detect", post(detect_handler))
        .route("/health", get(health_handler))
        .layer(CorsLayer::permissive())
        .with_state(engine);

    // Start server
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 8093));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    info!("Detection API listening on {}", addr);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_speed() {
        let gnn = TransactionGNN::new(128, 256, 3);
        let mut graph = DiGraph::new();
        
        // Add test nodes
        for i in 0..10 {
            graph.add_node(TxNode {
                tx_hash: format!("tx_{}", i),
                block_number: 1000 + i,
                timestamp: 1000000 + i * 10,
                from_address: format!("addr_{}", i),
                to_address: format!("addr_{}", i + 1),
                value: 100.0,
                gas_price: 20.0,
                features: vec![0.1; 128],
            });
        }
        
        let start = Instant::now();
        let _result = gnn.infer(&graph);
        let elapsed = start.elapsed();
        
        assert!(
            elapsed.as_micros() < 100,
            "Inference took {:?}, expected <100μs",
            elapsed
        );
    }

    #[test]
    fn test_thompson_sampling() {
        let sampler = ThompsonSampler::new(0.1);
        let routes = vec!["route1".to_string(), "route2".to_string()];
        
        // Test selection
        let selected = sampler.select_route(&routes);
        assert!(routes.contains(&selected));
        
        // Test update
        sampler.update(&selected, 0.9);
        
        // Should prefer high-reward route
        for _ in 0..10 {
            sampler.update("route1", 0.9);
            sampler.update("route2", 0.1);
        }
        
        let mut selections = HashMap::new();
        for _ in 0..100 {
            let route = sampler.select_route(&routes);
            *selections.entry(route).or_insert(0) += 1;
        }
        
        // route1 should be selected more often
        assert!(selections.get("route1").unwrap_or(&0) > selections.get("route2").unwrap_or(&0));
    }
}