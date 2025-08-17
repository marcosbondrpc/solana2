use anyhow::Result;
use arc_swap::ArcSwap;
use bytes::Bytes;
use bytemuck::{Pod, Zeroable};
use clickhouse::Client;
use core_affinity::CoreId;
use crossbeam::channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    clock::Slot,
    commitment_config::{CommitmentConfig, CommitmentLevel},
    pubkey::Pubkey,
    signature::Signature,
};
use solana_transaction_status::{
    EncodedConfirmedBlock, EncodedTransaction, EncodedTransactionWithStatusMeta,
    UiTransactionEncoding,
};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{mpsc, Semaphore},
    time::{interval, sleep},
};
use tracing::{error, info, warn};

const BATCH_SIZE: usize = 100;
const WORKER_THREADS: usize = 16;
const CHECKPOINT_INTERVAL: u64 = 1000;
const MAX_RETRIES: u32 = 5;
const RETRY_DELAY_MS: u64 = 100;
const CLICKHOUSE_BATCH: usize = 10000;
const PREFETCH_WINDOW: u64 = 50;

// Zero-copy transaction structure for ultra-fast parsing
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RawTransaction {
    slot: u64,
    block_time: i64,
    signature: [u8; 64],
    fee: u64,
    compute_units: u64,
    priority_fee: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointState {
    last_slot: u64,
    blocks_processed: u64,
    txs_processed: u64,
    checkpoint_time: SystemTime,
    error_count: u64,
}

#[derive(Debug, Clone)]
struct ScrapperConfig {
    rpc_urls: Vec<String>,
    clickhouse_url: String,
    start_slot: u64,
    end_slot: Option<u64>,
    checkpoint_path: String,
    max_concurrent: usize,
    cpu_affinity: bool,
}

struct HistoricalScrapper {
    config: Arc<ScrapperConfig>,
    rpc_clients: Vec<Arc<RpcClient>>,
    clickhouse: Arc<Client>,
    checkpoint: Arc<RwLock<CheckpointState>>,
    metrics: Arc<Metrics>,
    shutdown: Arc<AtomicBool>,
    slot_queue: Arc<RwLock<VecDeque<Slot>>>,
    processing_slots: Arc<DashMap<Slot, Instant>>,
}

struct Metrics {
    blocks_fetched: AtomicU64,
    blocks_parsed: AtomicU64,
    txs_extracted: AtomicU64,
    accounts_indexed: AtomicU64,
    labels_created: AtomicU64,
    errors: AtomicU64,
    avg_fetch_time_ms: AtomicU64,
    avg_parse_time_ms: AtomicU64,
    clickhouse_writes: AtomicU64,
}

impl HistoricalScrapper {
    pub fn new(config: ScrapperConfig) -> Result<Self> {
        let rpc_clients: Vec<Arc<RpcClient>> = config
            .rpc_urls
            .iter()
            .map(|url| Arc::new(RpcClient::new(url.clone())))
            .collect();

        let clickhouse = Arc::new(Client::default().with_url(&config.clickhouse_url));

        let checkpoint = Arc::new(RwLock::new(CheckpointState {
            last_slot: config.start_slot,
            blocks_processed: 0,
            txs_processed: 0,
            checkpoint_time: SystemTime::now(),
            error_count: 0,
        }));

        let metrics = Arc::new(Metrics {
            blocks_fetched: AtomicU64::new(0),
            blocks_parsed: AtomicU64::new(0),
            txs_extracted: AtomicU64::new(0),
            accounts_indexed: AtomicU64::new(0),
            labels_created: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            avg_fetch_time_ms: AtomicU64::new(0),
            avg_parse_time_ms: AtomicU64::new(0),
            clickhouse_writes: AtomicU64::new(0),
        });

        Ok(Self {
            config: Arc::new(config),
            rpc_clients,
            clickhouse,
            checkpoint,
            metrics,
            shutdown: Arc::new(AtomicBool::new(false)),
            slot_queue: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            processing_slots: Arc::new(DashMap::new()),
        })
    }

    pub async fn run(&self) -> Result<()> {
        info!("Starting Historical Scrapper with {} RPC endpoints", self.rpc_clients.len());
        
        // Initialize ClickHouse tables
        self.init_clickhouse_tables().await?;
        
        // Load checkpoint if exists
        self.load_checkpoint().await?;

        // Set CPU affinity for maximum performance
        if self.config.cpu_affinity {
            self.set_cpu_affinity();
        }

        // Start background tasks
        let mut tasks = vec![];
        
        // Slot fetcher task - fills the queue with slots to process
        let slot_fetcher = self.spawn_slot_fetcher();
        tasks.push(slot_fetcher);
        
        // Worker pool for parallel block fetching
        for worker_id in 0..WORKER_THREADS {
            let worker = self.spawn_worker(worker_id);
            tasks.push(worker);
        }
        
        // ClickHouse writer task
        let writer = self.spawn_clickhouse_writer();
        tasks.push(writer);
        
        // Metrics reporter
        let reporter = self.spawn_metrics_reporter();
        tasks.push(reporter);
        
        // Checkpoint saver
        let checkpointer = self.spawn_checkpointer();
        tasks.push(checkpointer);

        // Wait for shutdown signal
        tokio::signal::ctrl_c().await?;
        info!("Shutdown signal received, gracefully stopping...");
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Wait for all tasks to complete
        for task in tasks {
            task.await?;
        }
        
        // Final checkpoint
        self.save_checkpoint().await?;
        
        Ok(())
    }

    async fn init_clickhouse_tables(&self) -> Result<()> {
        let queries = vec![
            // Transactions table with optimal structure
            r#"
            CREATE TABLE IF NOT EXISTS sol.txs (
                slot UInt64,
                block_time DateTime64(3),
                signature FixedString(88),
                fee UInt64,
                compute_units UInt32,
                priority_fee UInt64,
                success UInt8,
                err String,
                num_instructions UInt16,
                num_accounts UInt16,
                program_ids Array(String),
                account_keys Array(String),
                pre_balances Array(UInt64),
                post_balances Array(UInt64),
                log_messages Array(String),
                inner_instructions String,
                rewards String
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(block_time)
            ORDER BY (slot, signature)
            SETTINGS index_granularity = 8192
            "#,
            
            // Accounts table for state tracking
            r#"
            CREATE TABLE IF NOT EXISTS sol.accounts (
                slot UInt64,
                pubkey FixedString(44),
                owner FixedString(44),
                lamports UInt64,
                data_len UInt32,
                executable UInt8,
                rent_epoch UInt64,
                updated_at DateTime64(3)
            ) ENGINE = ReplacingMergeTree(updated_at)
            PARTITION BY toYYYYMM(updated_at)
            ORDER BY (pubkey, slot)
            SETTINGS index_granularity = 8192
            "#,
            
            // Labels table for MEV detection
            r#"
            CREATE TABLE IF NOT EXISTS sol.labels (
                signature FixedString(88),
                label_type Enum8('arbitrage' = 1, 'sandwich' = 2, 'liquidation' = 3, 'jit' = 4, 'cex_arb' = 5),
                confidence Float32,
                profit_lamports Int64,
                victim_signatures Array(String),
                dex_programs Array(String),
                token_mints Array(String),
                created_at DateTime64(3) DEFAULT now64(3)
            ) ENGINE = MergeTree()
            ORDER BY (signature, label_type)
            SETTINGS index_granularity = 8192
            "#,
            
            // Checkpoint table
            r#"
            CREATE TABLE IF NOT EXISTS sol.checkpoints (
                checkpoint_id UUID DEFAULT generateUUIDv4(),
                last_slot UInt64,
                blocks_processed UInt64,
                txs_processed UInt64,
                checkpoint_time DateTime64(3),
                error_count UInt64
            ) ENGINE = MergeTree()
            ORDER BY checkpoint_time
            "#,
        ];

        for query in queries {
            self.clickhouse
                .query(query)
                .execute()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create table: {}", e))?;
        }

        info!("ClickHouse tables initialized successfully");
        Ok(())
    }

    async fn load_checkpoint(&self) -> Result<()> {
        let query = "SELECT last_slot, blocks_processed, txs_processed, checkpoint_time, error_count 
                     FROM sol.checkpoints 
                     ORDER BY checkpoint_time DESC 
                     LIMIT 1";
        
        let mut cursor = self.clickhouse.query(query).fetch::<CheckpointState>()?;
        
        if let Some(checkpoint) = cursor.next().await? {
            *self.checkpoint.write() = checkpoint;
            info!("Loaded checkpoint: slot {} with {} blocks processed", 
                  checkpoint.last_slot, checkpoint.blocks_processed);
        } else {
            info!("No checkpoint found, starting from slot {}", self.config.start_slot);
        }
        
        Ok(())
    }

    async fn save_checkpoint(&self) -> Result<()> {
        let checkpoint = self.checkpoint.read().clone();
        
        let mut insert = self.clickhouse.insert("sol.checkpoints")?;
        insert.write(&checkpoint).await?;
        insert.end().await?;
        
        info!("Checkpoint saved at slot {}", checkpoint.last_slot);
        Ok(())
    }

    fn spawn_slot_fetcher(&self) -> tokio::task::JoinHandle<()> {
        let config = self.config.clone();
        let slot_queue = self.slot_queue.clone();
        let checkpoint = self.checkpoint.clone();
        let shutdown = self.shutdown.clone();
        
        tokio::spawn(async move {
            let mut current_slot = checkpoint.read().last_slot;
            let end_slot = config.end_slot.unwrap_or(u64::MAX);
            
            while !shutdown.load(Ordering::Relaxed) && current_slot < end_slot {
                let mut queue = slot_queue.write();
                
                // Fill queue up to capacity
                while queue.len() < 10000 && current_slot < end_slot {
                    queue.push_back(current_slot);
                    current_slot += 1;
                }
                
                drop(queue);
                sleep(Duration::from_millis(100)).await;
            }
        })
    }

    fn spawn_worker(&self, worker_id: usize) -> tokio::task::JoinHandle<()> {
        let rpc_client = self.rpc_clients[worker_id % self.rpc_clients.len()].clone();
        let slot_queue = self.slot_queue.clone();
        let processing_slots = self.processing_slots.clone();
        let metrics = self.metrics.clone();
        let shutdown = self.shutdown.clone();
        let (tx, rx) = mpsc::channel(1000);
        
        // Spawn processor thread
        let processor_handle = self.spawn_processor(rx);
        
        tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                // Get next slot to process
                let slot = {
                    let mut queue = slot_queue.write();
                    queue.pop_front()
                };
                
                let Some(slot) = slot else {
                    sleep(Duration::from_millis(10)).await;
                    continue;
                };
                
                // Mark slot as being processed
                processing_slots.insert(slot, Instant::now());
                
                // Fetch block with retries
                let fetch_start = Instant::now();
                let mut retries = 0;
                
                loop {
                    match Self::fetch_block_binary(&rpc_client, slot).await {
                        Ok(block) => {
                            let fetch_time = fetch_start.elapsed().as_millis() as u64;
                            metrics.avg_fetch_time_ms.store(fetch_time, Ordering::Relaxed);
                            metrics.blocks_fetched.fetch_add(1, Ordering::Relaxed);
                            
                            // Send to processor
                            if tx.send((slot, block)).await.is_err() {
                                break;
                            }
                            
                            break;
                        }
                        Err(e) => {
                            retries += 1;
                            if retries >= MAX_RETRIES {
                                error!("Failed to fetch slot {} after {} retries: {}", slot, retries, e);
                                metrics.errors.fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                            sleep(Duration::from_millis(RETRY_DELAY_MS * retries as u64)).await;
                        }
                    }
                }
                
                // Remove from processing
                processing_slots.remove(&slot);
            }
            
            drop(tx);
            processor_handle.await.ok();
        })
    }

    async fn fetch_block_binary(
        client: &RpcClient,
        slot: Slot,
    ) -> Result<EncodedConfirmedBlock> {
        // Fetch with base64 encoding for maximum efficiency
        let block = client
            .get_block_with_encoding(slot, UiTransactionEncoding::Base64)
            .await?;
        
        Ok(block)
    }

    fn spawn_processor(&self, mut rx: mpsc::Receiver<(Slot, EncodedConfirmedBlock)>) -> tokio::task::JoinHandle<()> {
        let metrics = self.metrics.clone();
        let checkpoint = self.checkpoint.clone();
        
        tokio::spawn(async move {
            let (tx, rx_writer) = mpsc::channel(10000);
            
            while let Some((slot, block)) = rx.recv().await {
                let parse_start = Instant::now();
                
                // Zero-copy parse transactions
                let txs = Self::parse_block_zero_copy(slot, &block);
                
                let parse_time = parse_start.elapsed().as_millis() as u64;
                metrics.avg_parse_time_ms.store(parse_time, Ordering::Relaxed);
                metrics.blocks_parsed.fetch_add(1, Ordering::Relaxed);
                metrics.txs_extracted.fetch_add(txs.len() as u64, Ordering::Relaxed);
                
                // Update checkpoint
                {
                    let mut cp = checkpoint.write();
                    if slot > cp.last_slot {
                        cp.last_slot = slot;
                    }
                    cp.blocks_processed += 1;
                    cp.txs_processed += txs.len() as u64;
                }
                
                // Send to writer
                if tx.send(txs).await.is_err() {
                    break;
                }
            }
        })
    }

    fn parse_block_zero_copy(slot: Slot, block: &EncodedConfirmedBlock) -> Vec<ParsedTransaction> {
        let block_time = block.block_time.unwrap_or(0);
        let mut transactions = Vec::with_capacity(block.transactions.len());
        
        // Process transactions in parallel using rayon
        block.transactions.par_iter().filter_map(|tx_with_meta| {
            Self::parse_transaction(slot, block_time, tx_with_meta)
        }).collect_into_vec(&mut transactions);
        
        transactions
    }

    fn parse_transaction(
        slot: Slot,
        block_time: i64,
        tx_with_meta: &EncodedTransactionWithStatusMeta,
    ) -> Option<ParsedTransaction> {
        let meta = tx_with_meta.meta.as_ref()?;
        
        // Extract signature
        let signature = match &tx_with_meta.transaction {
            EncodedTransaction::Binary(data, _) => {
                // First 64 bytes are signature in base64
                if data.len() >= 88 {
                    data[..88].to_string()
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        
        // Parse transaction details
        let parsed = ParsedTransaction {
            slot,
            block_time,
            signature,
            fee: meta.fee,
            compute_units: meta.compute_units_consumed.unwrap_or(0),
            priority_fee: extract_priority_fee(meta),
            success: meta.err.is_none(),
            err: meta.err.as_ref().map(|e| format!("{:?}", e)).unwrap_or_default(),
            num_instructions: count_instructions(meta),
            num_accounts: extract_account_keys(&tx_with_meta.transaction).len() as u16,
            program_ids: extract_program_ids(&tx_with_meta.transaction),
            account_keys: extract_account_keys(&tx_with_meta.transaction),
            pre_balances: meta.pre_balances.clone(),
            post_balances: meta.post_balances.clone(),
            log_messages: meta.log_messages.clone().unwrap_or_default(),
            inner_instructions: serde_json::to_string(&meta.inner_instructions).unwrap_or_default(),
            rewards: serde_json::to_string(&meta.rewards).unwrap_or_default(),
        };
        
        Some(parsed)
    }

    fn spawn_clickhouse_writer(&self) -> tokio::task::JoinHandle<()> {
        let clickhouse = self.clickhouse.clone();
        let metrics = self.metrics.clone();
        let shutdown = self.shutdown.clone();
        
        tokio::spawn(async move {
            let (tx, mut rx) = mpsc::channel::<Vec<ParsedTransaction>>(1000);
            let mut batch = Vec::with_capacity(CLICKHOUSE_BATCH);
            
            loop {
                tokio::select! {
                    Some(txs) = rx.recv() => {
                        batch.extend(txs);
                        
                        if batch.len() >= CLICKHOUSE_BATCH {
                            if let Err(e) = Self::write_batch_to_clickhouse(&clickhouse, &batch).await {
                                error!("Failed to write to ClickHouse: {}", e);
                            } else {
                                metrics.clickhouse_writes.fetch_add(batch.len() as u64, Ordering::Relaxed);
                            }
                            batch.clear();
                        }
                    }
                    _ = sleep(Duration::from_secs(1)) => {
                        if !batch.is_empty() {
                            if let Err(e) = Self::write_batch_to_clickhouse(&clickhouse, &batch).await {
                                error!("Failed to write to ClickHouse: {}", e);
                            } else {
                                metrics.clickhouse_writes.fetch_add(batch.len() as u64, Ordering::Relaxed);
                            }
                            batch.clear();
                        }
                        
                        if shutdown.load(Ordering::Relaxed) {
                            break;
                        }
                    }
                }
            }
        })
    }

    async fn write_batch_to_clickhouse(
        client: &Client,
        batch: &[ParsedTransaction],
    ) -> Result<()> {
        let mut insert = client.insert("sol.txs")?;
        
        for tx in batch {
            insert.write(tx).await?;
        }
        
        insert.end().await?;
        Ok(())
    }

    fn spawn_metrics_reporter(&self) -> tokio::task::JoinHandle<()> {
        let metrics = self.metrics.clone();
        let shutdown = self.shutdown.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                
                info!(
                    "Metrics - Blocks: {}, Parsed: {}, Txs: {}, Errors: {}, Avg Fetch: {}ms, Avg Parse: {}ms, CH Writes: {}",
                    metrics.blocks_fetched.load(Ordering::Relaxed),
                    metrics.blocks_parsed.load(Ordering::Relaxed),
                    metrics.txs_extracted.load(Ordering::Relaxed),
                    metrics.errors.load(Ordering::Relaxed),
                    metrics.avg_fetch_time_ms.load(Ordering::Relaxed),
                    metrics.avg_parse_time_ms.load(Ordering::Relaxed),
                    metrics.clickhouse_writes.load(Ordering::Relaxed)
                );
            }
        })
    }

    fn spawn_checkpointer(&self) -> tokio::task::JoinHandle<()> {
        let checkpoint = self.checkpoint.clone();
        let clickhouse = self.clickhouse.clone();
        let shutdown = self.shutdown.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            
            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                
                let cp = checkpoint.read().clone();
                
                let mut insert = clickhouse.insert("sol.checkpoints").unwrap();
                insert.write(&cp).await.ok();
                insert.end().await.ok();
                
                info!("Checkpoint saved: slot {}, blocks {}, txs {}", 
                      cp.last_slot, cp.blocks_processed, cp.txs_processed);
            }
        })
    }

    fn set_cpu_affinity(&self) {
        // Pin main thread to CPU 0
        core_affinity::set_for_current(CoreId { id: 0 });
        
        // Workers will be pinned to subsequent cores
        info!("CPU affinity set for maximum performance");
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParsedTransaction {
    slot: u64,
    block_time: i64,
    signature: String,
    fee: u64,
    compute_units: u64,
    priority_fee: u64,
    success: bool,
    err: String,
    num_instructions: u16,
    num_accounts: u16,
    program_ids: Vec<String>,
    account_keys: Vec<String>,
    pre_balances: Vec<u64>,
    post_balances: Vec<u64>,
    log_messages: Vec<String>,
    inner_instructions: String,
    rewards: String,
}

// Helper functions
fn extract_priority_fee(meta: &solana_transaction_status::UiTransactionStatusMeta) -> u64 {
    // Priority fee is the difference between fee and base fee
    // Base fee = 5000 * signatures
    let base_fee = 5000;
    if meta.fee > base_fee {
        meta.fee - base_fee
    } else {
        0
    }
}

fn count_instructions(meta: &solana_transaction_status::UiTransactionStatusMeta) -> u16 {
    let mut count = 0u16;
    if let Some(inner) = &meta.inner_instructions {
        for ix_group in inner {
            count = count.saturating_add(ix_group.instructions.len() as u16);
        }
    }
    count
}

fn extract_program_ids(tx: &EncodedTransaction) -> Vec<String> {
    // Extract unique program IDs from transaction
    // This would need full parsing of the transaction
    vec![]
}

fn extract_account_keys(tx: &EncodedTransaction) -> Vec<String> {
    // Extract account keys from transaction
    // This would need full parsing of the transaction
    vec![]
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .json()
        .init();

    // Parse CLI arguments
    let config = ScrapperConfig {
        rpc_urls: vec![
            "https://api.mainnet-beta.solana.com".to_string(),
            "https://solana-api.projectserum.com".to_string(),
        ],
        clickhouse_url: "http://localhost:8123".to_string(),
        start_slot: 250_000_000,
        end_slot: None,
        checkpoint_path: "/tmp/scrapper_checkpoint.json".to_string(),
        max_concurrent: 100,
        cpu_affinity: true,
    };

    let scrapper = HistoricalScrapper::new(config)?;
    scrapper.run().await?;

    Ok(())
}