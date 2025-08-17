use solana_geyser_plugin_interface::geyser_plugin_interface::{
    GeyserPlugin, GeyserPluginError, ReplicaAccountInfoVersions,
    ReplicaBlockInfoVersions, ReplicaTransactionInfoVersions, Result as PluginResult,
    SlotStatus,
};
use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;
use dashmap::DashMap;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::ClientConfig;
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;
use crossbeam_channel::{bounded, Sender, Receiver};
use std::thread;

/// Configuration for the Kafka delta plugin
#[derive(Debug, Clone, Deserialize)]
pub struct KafkaDeltaConfig {
    pub kafka_brokers: String,
    pub topic_prefix: String,
    pub batch_size: usize,
    pub compression: String,
    pub linger_ms: u32,
    pub enable_pool_deltas: bool,
    pub tracked_programs: Vec<String>,
}

/// Pool state delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolDelta {
    pub slot: u64,
    pub program_id: String,
    pub pool_address: String,
    pub token_a_reserve: u64,
    pub token_b_reserve: u64,
    pub sqrt_price: u128,
    pub liquidity: u128,
    pub fee_rate: u16,
    pub timestamp_ms: u64,
    pub block_hash: String,
}

/// MEV opportunity signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MevSignal {
    pub slot: u64,
    pub signal_type: String,
    pub pool_address: String,
    pub estimated_profit: u64,
    pub gas_estimate: u64,
    pub priority: u8,
    pub timestamp_ms: u64,
}

pub struct KafkaDeltaPlugin {
    config: KafkaDeltaConfig,
    producer: Arc<FutureProducer>,
    runtime: Arc<Runtime>,
    pool_states: Arc<DashMap<String, PoolState>>,
    delta_sender: Sender<PoolDelta>,
    delta_receiver: Option<Receiver<PoolDelta>>,
    worker_handle: Option<thread::JoinHandle<()>>,
}

#[derive(Clone)]
struct PoolState {
    reserve_a: u64,
    reserve_b: u64,
    sqrt_price: u128,
    liquidity: u128,
    last_update_slot: u64,
}

impl KafkaDeltaPlugin {
    pub fn new(config: KafkaDeltaConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Create Kafka producer with optimized settings
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.kafka_brokers)
            .set("compression.type", &config.compression)
            .set("linger.ms", &config.linger_ms.to_string())
            .set("batch.size", "1000000") // 1MB batches
            .set("buffer.memory", "134217728") // 128MB buffer
            .set("max.in.flight.requests.per.connection", "5")
            .set("acks", "1") // Leader acknowledgment only for speed
            .create()?;
            
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .build()?
        );
        
        // Create channel for delta batching
        let (delta_sender, delta_receiver) = bounded(10000);
        
        Ok(Self {
            config,
            producer: Arc::new(producer),
            runtime,
            pool_states: Arc::new(DashMap::new()),
            delta_sender,
            delta_receiver: Some(delta_receiver),
            worker_handle: None,
        })
    }
    
    fn start_worker(&mut self) {
        if let Some(receiver) = self.delta_receiver.take() {
            let producer = Arc::clone(&self.producer);
            let runtime = Arc::clone(&self.runtime);
            let topic = format!("{}_pool_deltas", self.config.topic_prefix);
            let batch_size = self.config.batch_size;
            
            let handle = thread::spawn(move || {
                let mut batch = Vec::with_capacity(batch_size);
                
                loop {
                    // Collect batch
                    match receiver.recv() {
                        Ok(delta) => {
                            batch.push(delta);
                            
                            // Drain available items up to batch size
                            while batch.len() < batch_size {
                                match receiver.try_recv() {
                                    Ok(delta) => batch.push(delta),
                                    Err(_) => break,
                                }
                            }
                            
                            // Send batch if full or no more items
                            if batch.len() >= batch_size || receiver.is_empty() {
                                let batch_json = serde_json::to_vec(&batch).unwrap();
                                let producer_clone = Arc::clone(&producer);
                                let topic_clone = topic.clone();
                                
                                runtime.spawn(async move {
                                    let record = FutureRecord::to(&topic_clone)
                                        .payload(&batch_json)
                                        .key(&batch[0].slot.to_string());
                                        
                                    let _ = producer_clone.send(record, rdkafka::util::Timeout::Never).await;
                                });
                                
                                batch.clear();
                            }
                        }
                        Err(_) => break, // Channel closed
                    }
                }
            });
            
            self.worker_handle = Some(handle);
        }
    }
    
    fn process_account_update(
        &self,
        account: &ReplicaAccountInfoVersions,
        slot: u64,
        _is_startup: bool,
    ) -> PluginResult<()> {
        if !self.config.enable_pool_deltas {
            return Ok(());
        }
        
        let account_info = match account {
            ReplicaAccountInfoVersions::V0_0_3(info) => info,
            _ => return Ok(()),
        };
        
        // Check if this is a tracked program
        let owner = bs58::encode(&account_info.owner).into_string();
        if !self.config.tracked_programs.contains(&owner) {
            return Ok(());
        }
        
        let pubkey = bs58::encode(&account_info.pubkey).into_string();
        
        // Parse pool state from account data (simplified - actual parsing depends on program)
        if let Some(pool_state) = self.parse_pool_state(&account_info.data) {
            // Check for delta
            let mut should_send_delta = false;
            
            if let Some(mut existing) = self.pool_states.get_mut(&pubkey) {
                if existing.reserve_a != pool_state.reserve_a ||
                   existing.reserve_b != pool_state.reserve_b ||
                   existing.sqrt_price != pool_state.sqrt_price {
                    should_send_delta = true;
                    *existing = pool_state.clone();
                }
            } else {
                self.pool_states.insert(pubkey.clone(), pool_state.clone());
                should_send_delta = true;
            }
            
            if should_send_delta {
                let delta = PoolDelta {
                    slot,
                    program_id: owner,
                    pool_address: pubkey,
                    token_a_reserve: pool_state.reserve_a,
                    token_b_reserve: pool_state.reserve_b,
                    sqrt_price: pool_state.sqrt_price,
                    liquidity: pool_state.liquidity,
                    fee_rate: 30, // 0.3% default
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                    block_hash: String::new(), // Will be filled on block update
                };
                
                // Send to worker thread
                let _ = self.delta_sender.try_send(delta);
            }
        }
        
        Ok(())
    }
    
    fn parse_pool_state(&self, data: &[u8]) -> Option<PoolState> {
        // Simplified pool state parsing
        // In production, would parse actual Raydium/Orca/etc pool structures
        if data.len() < 64 {
            return None;
        }
        
        Some(PoolState {
            reserve_a: u64::from_le_bytes(data[0..8].try_into().ok()?),
            reserve_b: u64::from_le_bytes(data[8..16].try_into().ok()?),
            sqrt_price: u128::from_le_bytes(data[16..32].try_into().ok()?),
            liquidity: u128::from_le_bytes(data[32..48].try_into().ok()?),
            last_update_slot: 0,
        })
    }
    
    fn send_mev_signal(&self, signal: MevSignal) {
        let producer = Arc::clone(&self.producer);
        let runtime = Arc::clone(&self.runtime);
        let topic = format!("{}_mev_signals", self.config.topic_prefix);
        
        runtime.spawn(async move {
            let payload = serde_json::to_vec(&signal).unwrap();
            let record = FutureRecord::to(&topic)
                .payload(&payload)
                .key(&signal.pool_address);
                
            let _ = producer.send(record, rdkafka::util::Timeout::Never).await;
        });
    }
}

impl GeyserPlugin for KafkaDeltaPlugin {
    fn name(&self) -> &'static str {
        "KafkaDeltaPlugin"
    }
    
    fn on_load(&mut self, config_file: &str) -> PluginResult<()> {
        let config_str = std::fs::read_to_string(config_file)
            .map_err(|e| GeyserPluginError::ConfigFileReadError {
                msg: e.to_string()
            })?;
            
        let config: KafkaDeltaConfig = serde_json::from_str(&config_str)
            .map_err(|e| GeyserPluginError::ConfigFileReadError {
                msg: e.to_string()
            })?;
            
        self.config = config;
        self.start_worker();
        
        log::info!("KafkaDeltaPlugin loaded successfully");
        Ok(())
    }
    
    fn on_unload(&mut self) {
        if let Some(handle) = self.worker_handle.take() {
            drop(self.delta_sender.clone()); // Close channel
            let _ = handle.join();
        }
        log::info!("KafkaDeltaPlugin unloaded");
    }
    
    fn update_account(
        &mut self,
        account: ReplicaAccountInfoVersions,
        slot: u64,
        is_startup: bool,
    ) -> PluginResult<()> {
        self.process_account_update(&account, slot, is_startup)
    }
    
    fn update_slot_status(
        &mut self,
        slot: u64,
        _parent: Option<u64>,
        status: SlotStatus,
    ) -> PluginResult<()> {
        // Detect MEV opportunities on confirmed slots
        if matches!(status, SlotStatus::Confirmed) {
            // Analyze pool states for arbitrage
            for entry in self.pool_states.iter() {
                let (address, state) = entry.pair();
                
                // Simple arbitrage detection (production would be more sophisticated)
                let price = state.reserve_a as f64 / state.reserve_b as f64;
                if price > 1.05 || price < 0.95 {
                    let signal = MevSignal {
                        slot,
                        signal_type: "arbitrage".to_string(),
                        pool_address: address.clone(),
                        estimated_profit: ((price - 1.0).abs() * 1000000.0) as u64,
                        gas_estimate: 50000,
                        priority: if (price - 1.0).abs() > 0.1 { 9 } else { 5 },
                        timestamp_ms: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64,
                    };
                    
                    self.send_mev_signal(signal);
                }
            }
        }
        
        Ok(())
    }
    
    fn notify_transaction(
        &mut self,
        _transaction: ReplicaTransactionInfoVersions,
        _slot: u64,
    ) -> PluginResult<()> {
        // Could track DEX transactions for sandwich detection
        Ok(())
    }
    
    fn notify_block_metadata(
        &mut self,
        _blockinfo: ReplicaBlockInfoVersions,
    ) -> PluginResult<()> {
        Ok(())
    }
    
    fn account_data_notifications_enabled(&self) -> bool {
        true
    }
    
    fn transaction_notifications_enabled(&self) -> bool {
        true
    }
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "C" fn _create_plugin() -> *mut dyn GeyserPlugin {
    let config = KafkaDeltaConfig {
        kafka_brokers: "localhost:9092".to_string(),
        topic_prefix: "solana".to_string(),
        batch_size: 100,
        compression: "lz4".to_string(),
        linger_ms: 10,
        enable_pool_deltas: true,
        tracked_programs: vec![
            "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8".to_string(), // Raydium V4
            "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP".to_string(), // Orca
        ],
    };
    
    let plugin = KafkaDeltaPlugin::new(config).unwrap();
    let plugin_box = Box::new(plugin);
    Box::into_raw(plugin_box)
}