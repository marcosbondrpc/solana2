use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::time;
use anyhow::Result;
use tracing::{debug, error, info, warn};
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_client::rpc_config::{RpcBlockConfig, RpcTransactionConfig};
use solana_sdk::{
    commitment_config::{CommitmentConfig, CommitmentLevel},
    signature::Signature,
};
use solana_transaction_status::{
    EncodedConfirmedBlock, EncodedTransaction, UiTransactionEncoding,
};
use dashmap::DashMap;
use crossbeam::queue::ArrayQueue;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::storage::ClickHouseStorage;
use crate::kafka_producer::{KafkaProducer, ProgressUpdate};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapperStatus {
    pub is_running: bool,
    pub current_slot: u64,
    pub target_slot: u64,
    pub blocks_processed: u64,
    pub transactions_processed: u64,
    pub accounts_discovered: u64,
    pub programs_discovered: u64,
    pub start_time: Option<i64>,
    pub estimated_completion: Option<i64>,
    pub errors: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockData {
    pub slot: u64,
    pub blockhash: String,
    pub parent_slot: u64,
    pub block_time: Option<i64>,
    pub block_height: Option<u64>,
    pub transaction_count: u32,
    pub rewards: Vec<RewardData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardData {
    pub pubkey: String,
    pub lamports: i64,
    pub reward_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionData {
    pub signature: String,
    pub slot: u64,
    pub block_time: Option<i64>,
    pub fee: u64,
    pub compute_units_consumed: Option<u64>,
    pub status: bool,
    pub accounts: Vec<String>,
    pub program_ids: Vec<String>,
    pub instructions_count: u32,
    pub logs: Option<Vec<String>>,
}

pub struct DataScrapper {
    config: Config,
    storage: Arc<ClickHouseStorage>,
    kafka_producer: Arc<KafkaProducer>,
    rpc_client: Arc<RpcClient>,
    
    // State management
    is_running: Arc<AtomicBool>,
    current_slot: Arc<AtomicU64>,
    target_slot: Arc<AtomicU64>,
    blocks_processed: Arc<AtomicU64>,
    transactions_processed: Arc<AtomicU64>,
    
    // Discovery tracking
    accounts_discovered: Arc<DashMap<String, AccountInfo>>,
    programs_discovered: Arc<DashMap<String, ProgramInfo>>,
    
    // Performance metrics
    processing_queue: Arc<ArrayQueue<u64>>,
    error_count: Arc<AtomicU64>,
    start_time: Arc<RwLock<Option<Instant>>>,
}

#[derive(Debug, Clone)]
struct AccountInfo {
    pub pubkey: String,
    pub first_seen_slot: u64,
    pub last_seen_slot: u64,
    pub transaction_count: u64,
}

#[derive(Debug, Clone)]
struct ProgramInfo {
    pub program_id: String,
    pub first_seen_slot: u64,
    pub invocation_count: u64,
}

impl DataScrapper {
    pub async fn new(
        config: Config,
        storage: Arc<ClickHouseStorage>,
        kafka_producer: Arc<KafkaProducer>,
    ) -> Result<Self> {
        let rpc_client = Arc::new(RpcClient::new_with_commitment(
            config.rpc_endpoint.clone(),
            CommitmentConfig {
                commitment: CommitmentLevel::Confirmed,
            },
        ));
        
        // Get current slot
        let current_slot = rpc_client.get_slot().await?;
        
        Ok(Self {
            config,
            storage,
            kafka_producer,
            rpc_client,
            is_running: Arc::new(AtomicBool::new(false)),
            current_slot: Arc::new(AtomicU64::new(current_slot)),
            target_slot: Arc::new(AtomicU64::new(current_slot)),
            blocks_processed: Arc::new(AtomicU64::new(0)),
            transactions_processed: Arc::new(AtomicU64::new(0)),
            accounts_discovered: Arc::new(DashMap::new()),
            programs_discovered: Arc::new(DashMap::new()),
            processing_queue: Arc::new(ArrayQueue::new(10000)),
            error_count: Arc::new(AtomicU64::new(0)),
            start_time: Arc::new(RwLock::new(None)),
        })
    }
    
    pub async fn start_scrapping(&self) -> Result<()> {
        if self.is_running.load(Ordering::Acquire) {
            warn!("Scrapper is already running");
            return Ok(());
        }
        
        self.is_running.store(true, Ordering::Release);
        *self.start_time.write() = Some(Instant::now());
        
        info!("Starting data scrapping from slot {}", self.current_slot.load(Ordering::Acquire));
        
        // Get target slot (latest)
        let target_slot = self.rpc_client.get_slot().await?;
        self.target_slot.store(target_slot, Ordering::Release);
        
        // Start multiple worker tasks for parallel processing
        let worker_count = self.config.scrapper_workers;
        let mut workers = Vec::new();
        
        for worker_id in 0..worker_count {
            let scrapper = self.clone();
            workers.push(tokio::spawn(async move {
                scrapper.worker_task(worker_id).await
            }));
        }
        
        // Start slot producer task
        let scrapper = self.clone();
        let producer_task = tokio::spawn(async move {
            scrapper.slot_producer_task().await
        });
        
        // Start progress reporter task
        let scrapper = self.clone();
        let reporter_task = tokio::spawn(async move {
            scrapper.progress_reporter_task().await
        });
        
        // Wait for all tasks
        producer_task.await?;
        for worker in workers {
            let _ = worker.await;
        }
        reporter_task.abort();
        
        self.is_running.store(false, Ordering::Release);
        info!("Data scrapping completed");
        
        Ok(())
    }
    
    async fn slot_producer_task(&self) {
        let mut current = self.current_slot.load(Ordering::Acquire);
        let target = self.target_slot.load(Ordering::Acquire);
        
        while current <= target && self.is_running.load(Ordering::Acquire) {
            // Add slots to processing queue
            for _ in 0..100 {
                if current > target {
                    break;
                }
                
                // Try to push to queue, if full wait a bit
                while self.processing_queue.push(current).is_err() {
                    if !self.is_running.load(Ordering::Acquire) {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                
                current += 1;
            }
            
            self.current_slot.store(current, Ordering::Release);
            
            // Update target periodically
            if current % 1000 == 0 {
                if let Ok(new_target) = self.rpc_client.get_slot().await {
                    self.target_slot.store(new_target, Ordering::Release);
                }
            }
        }
    }
    
    async fn worker_task(&self, worker_id: usize) {
        info!("Worker {} started", worker_id);
        
        while self.is_running.load(Ordering::Acquire) {
            // Get slot from queue
            let slot = match self.processing_queue.pop() {
                Some(slot) => slot,
                None => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
            };
            
            // Process the slot
            if let Err(e) = self.process_slot(slot).await {
                error!("Worker {} failed to process slot {}: {}", worker_id, slot, e);
                self.error_count.fetch_add(1, Ordering::Relaxed);
                
                // Re-queue the slot for retry
                let _ = self.processing_queue.push(slot);
            }
        }
        
        info!("Worker {} stopped", worker_id);
    }
    
    async fn process_slot(&self, slot: u64) -> Result<()> {
        let start = Instant::now();
        
        // Fetch block with full transaction details
        let block_config = RpcBlockConfig {
            encoding: Some(UiTransactionEncoding::JsonParsed),
            transaction_details: Some(solana_client::rpc_config::TransactionDetails::Full),
            rewards: Some(true),
            commitment: Some(CommitmentConfig::confirmed()),
            max_supported_transaction_version: Some(0),
        };
        
        let block = match self.rpc_client.get_block_with_config(slot, block_config).await {
            Ok(block) => block,
            Err(e) => {
                // Skip slots that don't exist (gaps in the chain)
                debug!("Slot {} not found: {}", slot, e);
                return Ok(());
            }
        };
        
        // Extract block data
        let block_data = self.extract_block_data(slot, &block)?;
        
        // Store block data
        self.storage.store_block(&block_data).await?;
        
        // Process transactions
        if let Some(transactions) = &block.transactions {
            let mut tx_batch = Vec::new();
            
            for tx in transactions {
                if let Some(tx_data) = self.extract_transaction_data(slot, &block, tx)? {
                    // Track accounts and programs
                    for account in &tx_data.accounts {
                        self.track_account(account, slot);
                    }
                    for program in &tx_data.program_ids {
                        self.track_program(program, slot);
                    }
                    
                    tx_batch.push(tx_data);
                    
                    // Batch insert for efficiency
                    if tx_batch.len() >= 100 {
                        self.storage.store_transactions(&tx_batch).await?;
                        self.transactions_processed.fetch_add(tx_batch.len() as u64, Ordering::Relaxed);
                        tx_batch.clear();
                    }
                }
            }
            
            // Store remaining transactions
            if !tx_batch.is_empty() {
                self.storage.store_transactions(&tx_batch).await?;
                self.transactions_processed.fetch_add(tx_batch.len() as u64, Ordering::Relaxed);
            }
        }
        
        self.blocks_processed.fetch_add(1, Ordering::Relaxed);
        
        debug!("Processed slot {} in {:?}", slot, start.elapsed());
        
        Ok(())
    }
    
    fn extract_block_data(&self, slot: u64, block: &EncodedConfirmedBlock) -> Result<BlockData> {
        let transaction_count = block.transactions.as_ref().map(|txs| txs.len() as u32).unwrap_or(0);
        
        let rewards = block.rewards.as_ref().map(|rewards| {
            rewards.iter().map(|r| RewardData {
                pubkey: r.pubkey.clone(),
                lamports: r.lamports,
                reward_type: format!("{:?}", r.reward_type),
            }).collect()
        }).unwrap_or_default();
        
        Ok(BlockData {
            slot,
            blockhash: block.blockhash.clone(),
            parent_slot: block.parent_slot,
            block_time: block.block_time,
            block_height: block.block_height,
            transaction_count,
            rewards,
        })
    }
    
    fn extract_transaction_data(
        &self,
        slot: u64,
        block: &EncodedConfirmedBlock,
        tx: &solana_transaction_status::EncodedTransactionWithStatusMeta,
    ) -> Result<Option<TransactionData>> {
        let transaction = match &tx.transaction {
            EncodedTransaction::Json(tx) => tx,
            _ => return Ok(None),
        };
        
        let meta = tx.meta.as_ref();
        
        // Extract signature
        let signature = transaction.signatures.first()
            .ok_or_else(|| anyhow::anyhow!("No signature found"))?
            .clone();
        
        // Extract accounts
        let accounts: Vec<String> = transaction.message.account_keys.iter()
            .map(|k| k.to_string())
            .collect();
        
        // Extract program IDs from instructions
        let mut program_ids = Vec::new();
        if let Some(instructions) = &transaction.message.instructions {
            for instruction in instructions {
                if let Some(program_id_index) = instruction.program_id_index {
                    if let Some(program_id) = accounts.get(program_id_index as usize) {
                        if !program_ids.contains(program_id) {
                            program_ids.push(program_id.clone());
                        }
                    }
                }
            }
        }
        
        let instructions_count = transaction.message.instructions
            .as_ref()
            .map(|i| i.len() as u32)
            .unwrap_or(0);
        
        // Extract status and compute units
        let status = meta.map(|m| m.err.is_none()).unwrap_or(false);
        let fee = meta.map(|m| m.fee).unwrap_or(0);
        let compute_units_consumed = meta.and_then(|m| m.compute_units_consumed);
        let logs = meta.and_then(|m| m.log_messages.clone());
        
        Ok(Some(TransactionData {
            signature,
            slot,
            block_time: block.block_time,
            fee,
            compute_units_consumed,
            status,
            accounts,
            program_ids,
            instructions_count,
            logs,
        }))
    }
    
    fn track_account(&self, account: &str, slot: u64) {
        self.accounts_discovered
            .entry(account.to_string())
            .and_modify(|info| {
                info.last_seen_slot = slot;
                info.transaction_count += 1;
            })
            .or_insert(AccountInfo {
                pubkey: account.to_string(),
                first_seen_slot: slot,
                last_seen_slot: slot,
                transaction_count: 1,
            });
    }
    
    fn track_program(&self, program_id: &str, slot: u64) {
        self.programs_discovered
            .entry(program_id.to_string())
            .and_modify(|info| {
                info.invocation_count += 1;
            })
            .or_insert(ProgramInfo {
                program_id: program_id.to_string(),
                first_seen_slot: slot,
                invocation_count: 1,
            });
    }
    
    async fn progress_reporter_task(&self) {
        let mut interval = time::interval(Duration::from_secs(1));
        
        while self.is_running.load(Ordering::Acquire) {
            interval.tick().await;
            
            let status = self.get_status();
            
            // Send progress update via Kafka
            let update = ProgressUpdate {
                timestamp: chrono::Utc::now().timestamp_millis(),
                current_slot: status.current_slot,
                target_slot: status.target_slot,
                blocks_processed: status.blocks_processed,
                transactions_processed: status.transactions_processed,
                percentage: if status.target_slot > 0 {
                    (status.current_slot as f64 / status.target_slot as f64 * 100.0) as u32
                } else {
                    0
                },
            };
            
            if let Err(e) = self.kafka_producer.send_progress(update).await {
                error!("Failed to send progress update: {}", e);
            }
            
            // Log progress
            if status.blocks_processed % 100 == 0 {
                info!(
                    "Progress: {}/{} slots ({:.2}%), {} transactions, {} accounts, {} programs",
                    status.current_slot,
                    status.target_slot,
                    (status.current_slot as f64 / status.target_slot as f64 * 100.0),
                    status.transactions_processed,
                    status.accounts_discovered,
                    status.programs_discovered
                );
            }
        }
    }
    
    pub fn get_status(&self) -> ScrapperStatus {
        let current_slot = self.current_slot.load(Ordering::Acquire);
        let target_slot = self.target_slot.load(Ordering::Acquire);
        let blocks_processed = self.blocks_processed.load(Ordering::Acquire);
        
        let start_time = self.start_time.read().map(|t| {
            chrono::Utc::now().timestamp_millis() - t.elapsed().as_millis() as i64
        });
        
        let estimated_completion = if blocks_processed > 0 && current_slot < target_slot {
            start_time.map(|start| {
                let elapsed = chrono::Utc::now().timestamp_millis() - start;
                let rate = blocks_processed as f64 / elapsed as f64;
                let remaining = (target_slot - current_slot) as f64;
                let eta_ms = (remaining / rate) as i64;
                chrono::Utc::now().timestamp_millis() + eta_ms
            })
        } else {
            None
        };
        
        ScrapperStatus {
            is_running: self.is_running.load(Ordering::Acquire),
            current_slot,
            target_slot,
            blocks_processed,
            transactions_processed: self.transactions_processed.load(Ordering::Acquire),
            accounts_discovered: self.accounts_discovered.len() as u64,
            programs_discovered: self.programs_discovered.len() as u64,
            start_time,
            estimated_completion,
            errors: self.error_count.load(Ordering::Acquire) as u32,
        }
    }
    
    pub fn stop(&self) {
        info!("Stopping data scrapper");
        self.is_running.store(false, Ordering::Release);
    }
}

impl Clone for DataScrapper {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            storage: self.storage.clone(),
            kafka_producer: self.kafka_producer.clone(),
            rpc_client: self.rpc_client.clone(),
            is_running: self.is_running.clone(),
            current_slot: self.current_slot.clone(),
            target_slot: self.target_slot.clone(),
            blocks_processed: self.blocks_processed.clone(),
            transactions_processed: self.transactions_processed.clone(),
            accounts_discovered: self.accounts_discovered.clone(),
            programs_discovered: self.programs_discovered.clone(),
            processing_queue: self.processing_queue.clone(),
            error_count: self.error_count.clone(),
            start_time: self.start_time.clone(),
        }
    }
}