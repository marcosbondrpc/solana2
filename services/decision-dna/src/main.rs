//! Decision DNA System - Cryptographic Audit Trail
//! DEFENSIVE-ONLY: Detection event tracking with Ed25519 signatures
//! Merkle tree anchoring to Solana for immutable audit trail

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use blake3::{Hash as Blake3Hash, Hasher};
use borsh::{BorshDeserialize, BorshSerialize};
use chrono::{DateTime, Utc};
use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use lru::LruCache;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use prometheus::{register_histogram_vec, register_int_counter_vec, HistogramVec, IntCounterVec};
use rand::rngs::OsRng;
use rayon::prelude::*;
use rocksdb::{Options, WriteBatch, DB};
use serde::{Deserialize, Serialize};
use solana_client::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    instruction::{AccountMeta, Instruction},
    message::Message as SolanaMessage,
    pubkey::Pubkey,
    signature::Keypair,
    signer::Signer as SolanaSigner,
    system_instruction,
    transaction::Transaction,
};
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, watch, Semaphore};
use tokio::task::{spawn_blocking, JoinHandle};
use tokio::time::{interval, sleep};
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// Metrics
static DNA_METRICS: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "decision_dna_latency_ms",
        "Decision DNA operation latency",
        &["operation"],
        vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    )
    .unwrap()
});

static DNA_COUNTER: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec!("decision_dna_total", "Total DNA operations", &["type"]).unwrap()
});

/// Decision event with full cryptographic proof
#[derive(Debug, Clone, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct DecisionEvent {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
    pub event_type: EventType,
    pub detection_data: DetectionData,
    pub signature: Vec<u8>,
    pub parent_hash: Option<Vec<u8>>,
    pub merkle_proof: Option<MerkleProof>,
}

#[derive(Debug, Clone, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub enum EventType {
    ArbitrageDetected,
    SandwichDetected,
    LiquidationDetected,
    FlashLoanDetected,
    AnomalyDetected,
    FrontRunDetected,
    WashTradeDetected,
}

#[derive(Debug, Clone, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct DetectionData {
    pub transaction_hash: String,
    pub block_number: u64,
    pub confidence_score: f64,
    pub profit_estimate: Option<f64>,
    pub affected_pools: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub root: Vec<u8>,
    pub leaf_index: u64,
    pub siblings: Vec<Vec<u8>>,
    pub anchor_tx: Option<String>, // Solana transaction hash
}

/// Hash chain for audit trail
pub struct HashChain {
    db: Arc<DB>,
    current_hash: Arc<RwLock<Vec<u8>>>,
    sequence: AtomicU64,
    chain_id: Uuid,
}

impl HashChain {
    pub fn new(db_path: &Path) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_max_open_files(10000);
        opts.set_use_fsync(false);
        opts.set_bytes_per_sync(8388608);
        opts.create_missing_column_families(true);

        let db = Arc::new(DB::open(&opts, db_path)?);
        let chain_id = Uuid::new_v4();
        
        // Initialize genesis block
        let genesis_hash = blake3::hash(b"DECISION_DNA_GENESIS").as_bytes().to_vec();
        db.put(b"genesis", &genesis_hash)?;

        Ok(Self {
            db,
            current_hash: Arc::new(RwLock::new(genesis_hash)),
            sequence: AtomicU64::new(0),
            chain_id,
        })
    }

    pub fn add_event(&self, event: &DecisionEvent) -> Result<Vec<u8>> {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
        let parent_hash = self.current_hash.read().clone();
        
        // Compute new hash
        let mut hasher = Hasher::new();
        hasher.update(&parent_hash);
        hasher.update(&event.id.as_bytes());
        hasher.update(&event.timestamp.timestamp_nanos_opt().unwrap_or(0).to_le_bytes());
        hasher.update(&borsh::to_vec(event)?);
        let new_hash = hasher.finalize().as_bytes().to_vec();
        
        // Store in database
        let key = format!("event:{}", sequence);
        let mut batch = WriteBatch::default();
        batch.put(key.as_bytes(), &borsh::to_vec(event)?);
        batch.put(format!("hash:{}", sequence).as_bytes(), &new_hash);
        self.db.write(batch)?;
        
        // Update current hash
        *self.current_hash.write() = new_hash.clone();
        
        DNA_COUNTER.with_label_values(&["chain_add"]).inc();
        
        Ok(new_hash)
    }

    pub fn verify_chain(&self, from_sequence: u64, to_sequence: u64) -> Result<bool> {
        let mut current_hash = if from_sequence == 0 {
            self.db.get(b"genesis")?.unwrap_or_default()
        } else {
            self.db
                .get(format!("hash:{}", from_sequence - 1).as_bytes())?
                .unwrap_or_default()
        };

        for seq in from_sequence..=to_sequence {
            let event_key = format!("event:{}", seq);
            let hash_key = format!("hash:{}", seq);
            
            let event_data = self.db.get(event_key.as_bytes())?.context("Event not found")?;
            let stored_hash = self.db.get(hash_key.as_bytes())?.context("Hash not found")?;
            
            // Recompute hash
            let mut hasher = Hasher::new();
            hasher.update(&current_hash);
            hasher.update(&event_data);
            let computed_hash = hasher.finalize().as_bytes().to_vec();
            
            if computed_hash != stored_hash {
                return Ok(false);
            }
            
            current_hash = computed_hash;
        }
        
        Ok(true)
    }
}

/// Merkle tree manager for batch anchoring
pub struct MerkleManager {
    pending_leaves: Arc<Mutex<Vec<Vec<u8>>>>,
    current_tree: Arc<RwLock<Option<MerkleTree>>>,
    anchor_interval: Duration,
    solana_client: Option<RpcClient>,
    anchor_keypair: Option<Keypair>,
}

#[derive(Clone)]
struct MerkleTree {
    leaves: Vec<Vec<u8>>,
    root: Vec<u8>,
    height: usize,
}

impl MerkleTree {
    fn new(leaves: Vec<Vec<u8>>) -> Self {
        if leaves.is_empty() {
            return Self {
                leaves: vec![],
                root: vec![],
                height: 0,
            };
        }

        let mut current_level = leaves.clone();
        let mut height = 0;

        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in current_level.chunks(2) {
                let mut hasher = Hasher::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    hasher.update(&chunk[0]); // Duplicate for odd number
                }
                next_level.push(hasher.finalize().as_bytes().to_vec());
            }
            
            current_level = next_level;
            height += 1;
        }

        Self {
            leaves,
            root: current_level[0].clone(),
            height,
        }
    }

    fn get_proof(&self, leaf_index: usize) -> Option<Vec<Vec<u8>>> {
        if leaf_index >= self.leaves.len() {
            return None;
        }

        let mut siblings = Vec::new();
        let mut index = leaf_index;
        let mut level_nodes = self.leaves.clone();

        for _ in 0..self.height {
            let sibling_index = if index % 2 == 0 { index + 1 } else { index - 1 };
            
            if sibling_index < level_nodes.len() {
                siblings.push(level_nodes[sibling_index].clone());
            } else {
                siblings.push(level_nodes[index].clone()); // Duplicate self
            }

            // Compute next level
            let mut next_level = Vec::new();
            for chunk in level_nodes.chunks(2) {
                let mut hasher = Hasher::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    hasher.update(&chunk[0]);
                }
                next_level.push(hasher.finalize().as_bytes().to_vec());
            }
            
            level_nodes = next_level;
            index /= 2;
        }

        Some(siblings)
    }
}

impl MerkleManager {
    pub fn new(
        anchor_interval: Duration,
        solana_rpc: Option<String>,
        anchor_keypair: Option<Keypair>,
    ) -> Self {
        let solana_client = solana_rpc.map(|url| RpcClient::new_with_commitment(url, CommitmentConfig::confirmed()));

        Self {
            pending_leaves: Arc::new(Mutex::new(Vec::new())),
            current_tree: Arc::new(RwLock::new(None)),
            anchor_interval,
            solana_client,
            anchor_keypair,
        }
    }

    pub fn add_leaf(&self, leaf: Vec<u8>) {
        self.pending_leaves.lock().push(leaf);
        DNA_COUNTER.with_label_values(&["merkle_leaf"]).inc();
    }

    pub async fn build_and_anchor(&self) -> Result<Option<String>> {
        let leaves = {
            let mut pending = self.pending_leaves.lock();
            if pending.is_empty() {
                return Ok(None);
            }
            std::mem::take(&mut *pending)
        };

        let tree = MerkleTree::new(leaves);
        let root = tree.root.clone();
        
        // Anchor to Solana if configured
        let anchor_tx = if let (Some(client), Some(keypair)) = (&self.solana_client, &self.anchor_keypair) {
            match self.anchor_to_solana(client, keypair, &root).await {
                Ok(sig) => {
                    info!("Anchored merkle root to Solana: {}", sig);
                    Some(sig)
                }
                Err(e) => {
                    error!("Failed to anchor to Solana: {}", e);
                    None
                }
            }
        } else {
            None
        };

        *self.current_tree.write() = Some(tree);
        DNA_COUNTER.with_label_values(&["merkle_anchor"]).inc();

        Ok(anchor_tx)
    }

    async fn anchor_to_solana(
        &self,
        client: &RpcClient,
        keypair: &Keypair,
        root: &[u8],
    ) -> Result<String> {
        // Create memo instruction with merkle root
        let memo = format!("DNA_MERKLE:{}", hex::encode(root));
        let memo_instruction = Instruction {
            program_id: Pubkey::from_str_const("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"),
            accounts: vec![],
            data: memo.into_bytes(),
        };

        let recent_blockhash = client.get_latest_blockhash()?;
        let message = SolanaMessage::new(&[memo_instruction], Some(&keypair.pubkey()));
        let transaction = Transaction::new(&[keypair], message, recent_blockhash);

        let signature = client.send_and_confirm_transaction(&transaction)?;
        Ok(signature.to_string())
    }
}

/// Decision DNA service
pub struct DecisionDNAService {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    hash_chain: Arc<HashChain>,
    merkle_manager: Arc<MerkleManager>,
    event_cache: Arc<Mutex<LruCache<Uuid, DecisionEvent>>>,
    metrics_tx: mpsc::UnboundedSender<MetricEvent>,
    shutdown: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
struct MetricEvent {
    timestamp_ns: u64,
    operation: String,
    latency_ns: u64,
}

impl DecisionDNAService {
    pub async fn new(
        db_path: &Path,
        solana_rpc: Option<String>,
        anchor_keypair: Option<Keypair>,
    ) -> Result<Arc<Self>> {
        // Generate signing keypair
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        // Initialize components
        let hash_chain = Arc::new(HashChain::new(db_path)?);
        let merkle_manager = Arc::new(MerkleManager::new(
            Duration::from_secs(300), // 5 minute anchoring
            solana_rpc,
            anchor_keypair,
        ));

        let (metrics_tx, mut metrics_rx) = mpsc::unbounded_channel();

        // Spawn metrics aggregator
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            let mut events = Vec::new();

            loop {
                tokio::select! {
                    Some(event) = metrics_rx.recv() => {
                        events.push(event);
                    }
                    _ = interval.tick() => {
                        if !events.is_empty() {
                            // Aggregate metrics
                            for event in &events {
                                DNA_METRICS
                                    .with_label_values(&[&event.operation])
                                    .observe(event.latency_ns as f64 / 1_000_000.0);
                            }
                            events.clear();
                        }
                    }
                }
            }
        });

        Ok(Arc::new(Self {
            signing_key,
            verifying_key,
            hash_chain,
            merkle_manager,
            event_cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(10000).unwrap()))),
            metrics_tx,
            shutdown: Arc::new(AtomicBool::new(false)),
        }))
    }

    /// Create and sign a detection event
    pub async fn create_event(
        &self,
        event_type: EventType,
        detection_data: DetectionData,
    ) -> Result<DecisionEvent> {
        let start = Instant::now();
        
        let id = Uuid::new_v4();
        let timestamp = Utc::now();
        let sequence = self.hash_chain.sequence.load(Ordering::SeqCst);
        let parent_hash = Some(self.hash_chain.current_hash.read().clone());

        // Create event without signature first
        let mut event = DecisionEvent {
            id,
            timestamp,
            sequence,
            event_type,
            detection_data,
            signature: vec![],
            parent_hash,
            merkle_proof: None,
        };

        // Sign the event
        let message = borsh::to_vec(&event)?;
        let signature = self.signing_key.sign(&message);
        event.signature = signature.to_bytes().to_vec();

        // Add to hash chain
        let hash = self.hash_chain.add_event(&event)?;

        // Add to merkle tree
        self.merkle_manager.add_leaf(hash);

        // Cache the event
        self.event_cache.lock().put(id, event.clone());

        let elapsed = start.elapsed();
        self.metrics_tx.send(MetricEvent {
            timestamp_ns: now_ns(),
            operation: "create_event".to_string(),
            latency_ns: elapsed.as_nanos() as u64,
        })?;

        DNA_COUNTER.with_label_values(&["event_created"]).inc();
        
        Ok(event)
    }

    /// Verify an event's signature
    pub fn verify_event(&self, event: &DecisionEvent) -> Result<bool> {
        let mut event_copy = event.clone();
        let signature_bytes = event_copy.signature.clone();
        event_copy.signature = vec![];

        let message = borsh::to_vec(&event_copy)?;
        let signature = Signature::from_bytes(&signature_bytes.try_into().map_err(|_| anyhow::anyhow!("Invalid signature length"))?);

        Ok(self.verifying_key.verify(&message, &signature).is_ok())
    }

    /// Start periodic merkle anchoring
    pub async fn start_anchoring(self: Arc<Self>) {
        let merkle_manager = self.merkle_manager.clone();
        let mut interval = interval(self.merkle_manager.anchor_interval);

        tokio::spawn(async move {
            loop {
                interval.tick().await;
                
                match merkle_manager.build_and_anchor().await {
                    Ok(Some(tx)) => info!("Merkle tree anchored: {}", tx),
                    Ok(None) => debug!("No leaves to anchor"),
                    Err(e) => error!("Failed to anchor merkle tree: {}", e),
                }
            }
        });
    }
}

/// Get current time in nanoseconds
fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// API handlers
async fn create_detection_event(
    State(service): State<Arc<DecisionDNAService>>,
    Json(payload): Json<CreateEventRequest>,
) -> Result<Json<DecisionEvent>, StatusCode> {
    let event = service
        .create_event(payload.event_type, payload.detection_data)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(event))
}

async fn verify_event(
    State(service): State<Arc<DecisionDNAService>>,
    Json(event): Json<DecisionEvent>,
) -> Result<Json<VerifyResponse>, StatusCode> {
    let valid = service
        .verify_event(&event)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(VerifyResponse { valid, event_id: event.id }))
}

async fn verify_chain(
    State(service): State<Arc<DecisionDNAService>>,
    Query(params): Query<VerifyChainParams>,
) -> Result<Json<VerifyChainResponse>, StatusCode> {
    let valid = service
        .hash_chain
        .verify_chain(params.from_sequence, params.to_sequence)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(VerifyChainResponse {
        valid,
        from_sequence: params.from_sequence,
        to_sequence: params.to_sequence,
    }))
}

#[derive(Deserialize)]
struct CreateEventRequest {
    event_type: EventType,
    detection_data: DetectionData,
}

#[derive(Serialize)]
struct VerifyResponse {
    valid: bool,
    event_id: Uuid,
}

#[derive(Deserialize)]
struct VerifyChainParams {
    from_sequence: u64,
    to_sequence: u64,
}

#[derive(Serialize)]
struct VerifyChainResponse {
    valid: bool,
    from_sequence: u64,
    to_sequence: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info,decision_dna=debug")
        .json()
        .init();

    info!("Starting Decision DNA Service (DEFENSIVE-ONLY)");

    // Initialize service
    let db_path = Path::new("/tmp/decision_dna");
    std::fs::create_dir_all(db_path)?;
    
    let service = DecisionDNAService::new(
        db_path,
        std::env::var("SOLANA_RPC").ok(),
        None, // Would load from file in production
    )
    .await?;

    // Start anchoring task
    service.clone().start_anchoring().await;

    // Build API router
    let app = Router::new()
        .route("/api/v1/event", post(create_detection_event))
        .route("/api/v1/verify", post(verify_event))
        .route("/api/v1/chain/verify", get(verify_chain))
        .layer(CorsLayer::permissive())
        .with_state(service);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8092));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    info!("Decision DNA API listening on {}", addr);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}