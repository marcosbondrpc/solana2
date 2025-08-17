use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use quinn::{Endpoint, ClientConfig, Connection};
use solana_sdk::signature::Keypair;
use tokio::sync::RwLock;
use dashmap::DashMap;
use tracing::{info, debug, warn, error};

use crate::core::SandwichDecision;

pub struct DualSubmitter {
    tpu_client: Arc<TPUClient>,
    jito_client: Arc<JitoClient>,
    submission_stats: Arc<RwLock<SubmissionStats>>,
    tip_ladder: Arc<TipLadder>,
}

pub struct Bundle {
    pub transactions: Vec<Vec<u8>>,
    pub tip_amount: u64,
    pub priority: u8,
}

struct TPUClient {
    endpoints: Vec<Arc<Connection>>,
    current_leader: Arc<RwLock<usize>>,
    quic_config: ClientConfig,
}

struct JitoClient {
    endpoint: Arc<Connection>,
    auth_keypair: Keypair,
    bundle_cache: Arc<DashMap<[u8; 32], Instant>>,
}

struct SubmissionStats {
    tpu_success: u64,
    tpu_failures: u64,
    jito_success: u64,
    jito_failures: u64,
    total_landed: u64,
    total_profit: u64,
}

pub struct TipLadder {
    buckets: Vec<TipBucket>,
}

struct TipBucket {
    min_profit: u64,
    max_profit: u64,
    base_tip_bps: u16,
    multiplier: f32,
}

impl DualSubmitter {
    pub async fn new(core_ids: Vec<usize>) -> Result<Self> {
        info!("Initializing Dual Submitter (TPU + Jito)");
        
        let tpu_client = Arc::new(TPUClient::new().await?);
        let jito_client = Arc::new(JitoClient::new().await?);
        
        let tip_ladder = Arc::new(TipLadder::new());
        
        Ok(Self {
            tpu_client,
            jito_client,
            submission_stats: Arc::new(RwLock::new(SubmissionStats::default())),
            tip_ladder,
        })
    }
    
    pub async fn submit_bundle(&self, decision: SandwichDecision) -> Result<()> {
        let start = Instant::now();
        
        // Create bundle from decision
        let bundle = self.create_bundle(decision.clone())?;
        
        // Submit to both paths in parallel
        let (tpu_result, jito_result) = tokio::join!(
            self.submit_to_tpu(&bundle),
            self.submit_to_jito(&bundle)
        );
        
        // Update stats
        let mut stats = self.submission_stats.write().await;
        
        if tpu_result.is_ok() {
            stats.tpu_success += 1;
        } else {
            stats.tpu_failures += 1;
        }
        
        if jito_result.is_ok() {
            stats.jito_success += 1;
        } else {
            stats.jito_failures += 1;
        }
        
        debug!("Bundle submission took {:?}", start.elapsed());
        
        Ok(())
    }
    
    fn create_bundle(&self, decision: SandwichDecision) -> Result<Bundle> {
        Ok(Bundle {
            transactions: vec![decision.front_tx, decision.back_tx],
            tip_amount: decision.tip_amount,
            priority: self.calculate_priority(decision.confidence),
        })
    }
    
    fn calculate_priority(&self, confidence: f32) -> u8 {
        ((confidence * 255.0) as u8).min(255)
    }
    
    async fn submit_to_tpu(&self, bundle: &Bundle) -> Result<()> {
        self.tpu_client.submit_bundle(bundle).await
    }
    
    async fn submit_to_jito(&self, bundle: &Bundle) -> Result<()> {
        self.jito_client.submit_bundle(bundle).await
    }
}

impl TPUClient {
    async fn new() -> Result<Self> {
        info!("Initializing TPU QUIC client with NanoBurst");
        
        // Configure QUIC for ultra-low latency
        let mut config = ClientConfig::with_native_roots();
        
        // NanoBurst settings
        let mut transport = quinn::TransportConfig::default();
        transport.initial_rtt(Duration::from_micros(500));
        transport.max_idle_timeout(Some(Duration::from_secs(5)).try_into()?);
        transport.stream_receive_window(1024 * 1024 * 8)?; // 8MB window
        transport.receive_window(1024 * 1024 * 24)?;       // 24MB window
        transport.send_window(1024 * 1024 * 24);           // 24MB send window
        transport.max_concurrent_bidi_streams(256u32.into());
        transport.datagram_receive_buffer_size(Some(1024 * 1024 * 16)); // 16MB
        
        config.transport_config(Arc::new(transport));
        
        // Connect to TPU leaders
        let endpoints = Self::connect_to_leaders(config).await?;
        
        Ok(Self {
            endpoints,
            current_leader: Arc::new(RwLock::new(0)),
            quic_config: ClientConfig::with_native_roots(),
        })
    }
    
    async fn connect_to_leaders(config: ClientConfig) -> Result<Vec<Arc<Connection>>> {
        let leader_addresses = vec![
            "147.28.145.64:8003",   // TPU leader 1
            "147.28.145.65:8003",   // TPU leader 2
            "147.28.145.66:8003",   // TPU leader 3
        ];
        
        let mut connections = Vec::new();
        
        for addr in leader_addresses {
            match Self::connect_with_retry(&config, addr).await {
                Ok(conn) => {
                    info!("Connected to TPU leader: {}", addr);
                    connections.push(Arc::new(conn));
                }
                Err(e) => {
                    warn!("Failed to connect to {}: {}", addr, e);
                }
            }
        }
        
        if connections.is_empty() {
            return Err(anyhow::anyhow!("Failed to connect to any TPU leaders"));
        }
        
        Ok(connections)
    }
    
    async fn connect_with_retry(config: &ClientConfig, addr: &str) -> Result<Connection> {
        let endpoint = Endpoint::client("0.0.0.0:0".parse()?)?;
        
        for attempt in 0..3 {
            match endpoint.connect_with(config.clone(), addr.parse()?, "solana-tpu")?.await {
                Ok(conn) => return Ok(conn),
                Err(e) if attempt < 2 => {
                    warn!("Connection attempt {} failed: {}", attempt + 1, e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(e) => return Err(e.into()),
            }
        }
        
        Err(anyhow::anyhow!("Max retries exceeded"))
    }
    
    async fn submit_bundle(&self, bundle: &Bundle) -> Result<()> {
        let leader_idx = *self.current_leader.read().await;
        let connection = &self.endpoints[leader_idx % self.endpoints.len()];
        
        // Open stream with priority
        let mut stream = connection.open_uni().await?;
        
        // Serialize bundle with MessagePack for speed
        let data = rmp_serde::to_vec(bundle)?;
        
        // Write with zero-copy
        stream.write_all(&data).await?;
        stream.finish().await?;
        
        Ok(())
    }
}

impl JitoClient {
    async fn new() -> Result<Self> {
        info!("Initializing Jito bundle client");
        
        let config = ClientConfig::with_native_roots();
        let endpoint = Endpoint::client("0.0.0.0:0".parse()?)?;
        
        let jito_addr = "frankfurt.mainnet.block-engine.jito.wtf:1002";
        let connection = endpoint.connect_with(
            config,
            jito_addr.parse()?,
            "jito-block-engine"
        )?.await?;
        
        // Load auth keypair
        let keypair = Keypair::new(); // Load from file in production
        
        Ok(Self {
            endpoint: Arc::new(connection),
            auth_keypair: keypair,
            bundle_cache: Arc::new(DashMap::new()),
        })
    }
    
    async fn submit_bundle(&self, bundle: &Bundle) -> Result<()> {
        // Check bundle cache for deduplication
        let bundle_hash = self.compute_bundle_hash(bundle);
        
        if self.bundle_cache.contains_key(&bundle_hash) {
            debug!("Bundle already submitted, skipping");
            return Ok(());
        }
        
        // Open authenticated stream
        let mut stream = self.endpoint.open_uni().await?;
        
        // Create Jito bundle packet
        let packet = self.create_jito_packet(bundle)?;
        
        // Submit with auth
        stream.write_all(&packet).await?;
        stream.finish().await?;
        
        // Cache bundle
        self.bundle_cache.insert(bundle_hash, Instant::now());
        
        // Clean old cache entries periodically
        if self.bundle_cache.len() > 10000 {
            self.clean_cache();
        }
        
        Ok(())
    }
    
    fn compute_bundle_hash(&self, bundle: &Bundle) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for tx in &bundle.transactions {
            tx.hash(&mut hasher);
        }
        
        let hash = hasher.finish();
        let mut result = [0u8; 32];
        result[..8].copy_from_slice(&hash.to_le_bytes());
        result
    }
    
    fn create_jito_packet(&self, bundle: &Bundle) -> Result<Vec<u8>> {
        // Create Jito bundle format
        let mut packet = Vec::new();
        
        // Add auth header
        packet.extend_from_slice(&self.auth_keypair.pubkey().to_bytes());
        
        // Add tip
        packet.extend_from_slice(&bundle.tip_amount.to_le_bytes());
        
        // Add transactions
        for tx in &bundle.transactions {
            packet.extend_from_slice(&(tx.len() as u32).to_le_bytes());
            packet.extend_from_slice(tx);
        }
        
        // Sign packet
        let signature = self.auth_keypair.sign_message(&packet);
        packet.extend_from_slice(&signature.to_bytes());
        
        Ok(packet)
    }
    
    fn clean_cache(&self) {
        let now = Instant::now();
        let expiry = Duration::from_secs(60);
        
        self.bundle_cache.retain(|_, timestamp| {
            now.duration_since(*timestamp) < expiry
        });
    }
}

impl TipLadder {
    fn new() -> Self {
        Self {
            buckets: vec![
                TipBucket {
                    min_profit: 0,
                    max_profit: 100_000_000,      // 0.1 SOL
                    base_tip_bps: 500,             // 5%
                    multiplier: 1.0,
                },
                TipBucket {
                    min_profit: 100_000_000,
                    max_profit: 500_000_000,       // 0.5 SOL
                    base_tip_bps: 750,             // 7.5%
                    multiplier: 1.2,
                },
                TipBucket {
                    min_profit: 500_000_000,
                    max_profit: 1_000_000_000,     // 1 SOL
                    base_tip_bps: 1000,            // 10%
                    multiplier: 1.5,
                },
                TipBucket {
                    min_profit: 1_000_000_000,
                    max_profit: u64::MAX,
                    base_tip_bps: 1500,            // 15%
                    multiplier: 2.0,
                },
            ],
        }
    }
    
    pub fn calculate_tip(&self, profit: u64, confidence: f32, network_load: f32) -> u64 {
        // Find appropriate bucket
        let bucket = self.buckets.iter()
            .find(|b| profit >= b.min_profit && profit < b.max_profit)
            .unwrap_or(&self.buckets[0]);
        
        // Calculate base tip
        let base_tip = (profit * bucket.base_tip_bps as u64) / 10000;
        
        // Apply multipliers
        let confidence_mult = 0.5 + (confidence * 0.5); // 0.5x to 1.0x
        let load_mult = 1.0 + (network_load * bucket.multiplier); // Dynamic based on load
        
        let final_tip = (base_tip as f64 * confidence_mult * load_mult) as u64;
        
        // Cap at 50% of profit
        final_tip.min(profit / 2)
    }
}

impl Default for SubmissionStats {
    fn default() -> Self {
        Self {
            tpu_success: 0,
            tpu_failures: 0,
            jito_success: 0,
            jito_failures: 0,
            total_landed: 0,
            total_profit: 0,
        }
    }
}