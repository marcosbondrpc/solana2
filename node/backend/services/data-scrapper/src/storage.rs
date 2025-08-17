use std::sync::Arc;
use anyhow::Result;
use clickhouse::{Client, Row};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info};
use chrono::{DateTime, Utc};

use crate::config::Config;
use crate::scrapper::{BlockData, TransactionData};

pub struct ClickHouseStorage {
    client: Client,
    database: String,
}

impl ClickHouseStorage {
    pub async fn new(config: &Config) -> Result<Self> {
        let client = Client::default()
            .with_url(&config.clickhouse_url)
            .with_user(&config.clickhouse_username)
            .with_password(&config.clickhouse_password)
            .with_database(&config.clickhouse_database);
        
        let storage = Self {
            client,
            database: config.clickhouse_database.clone(),
        };
        
        // Initialize database and tables
        storage.init_database().await?;
        
        Ok(storage)
    }
    
    async fn init_database(&self) -> Result<()> {
        info!("Initializing ClickHouse database: {}", self.database);
        
        // Create database if not exists
        self.client
            .query(&format!("CREATE DATABASE IF NOT EXISTS {}", self.database))
            .execute()
            .await?;
        
        // Create blocks table
        self.client
            .query(&format!(
                "CREATE TABLE IF NOT EXISTS {}.blocks (
                    slot UInt64,
                    blockhash String,
                    parent_slot UInt64,
                    block_time Nullable(DateTime64(3)),
                    block_height Nullable(UInt64),
                    transaction_count UInt32,
                    created_at DateTime64(3) DEFAULT now64(3),
                    INDEX idx_slot slot TYPE minmax GRANULARITY 1000,
                    INDEX idx_block_time block_time TYPE minmax GRANULARITY 1000
                ) ENGINE = MergeTree()
                ORDER BY (slot)
                PARTITION BY toYYYYMM(created_at)
                SETTINGS index_granularity = 8192",
                self.database
            ))
            .execute()
            .await?;
        
        // Create transactions table
        self.client
            .query(&format!(
                "CREATE TABLE IF NOT EXISTS {}.transactions (
                    signature String,
                    slot UInt64,
                    block_time Nullable(DateTime64(3)),
                    fee UInt64,
                    compute_units_consumed Nullable(UInt64),
                    status Bool,
                    accounts Array(String),
                    program_ids Array(String),
                    instructions_count UInt32,
                    logs Nullable(Array(String)),
                    created_at DateTime64(3) DEFAULT now64(3),
                    INDEX idx_signature signature TYPE bloom_filter(0.01) GRANULARITY 1,
                    INDEX idx_slot slot TYPE minmax GRANULARITY 1000,
                    INDEX idx_programs program_ids TYPE bloom_filter(0.01) GRANULARITY 1
                ) ENGINE = MergeTree()
                ORDER BY (slot, signature)
                PARTITION BY toYYYYMM(created_at)
                SETTINGS index_granularity = 8192",
                self.database
            ))
            .execute()
            .await?;
        
        // Create accounts table
        self.client
            .query(&format!(
                "CREATE TABLE IF NOT EXISTS {}.accounts (
                    pubkey String,
                    first_seen_slot UInt64,
                    last_seen_slot UInt64,
                    transaction_count UInt64,
                    updated_at DateTime64(3) DEFAULT now64(3),
                    INDEX idx_pubkey pubkey TYPE bloom_filter(0.01) GRANULARITY 1
                ) ENGINE = ReplacingMergeTree(updated_at)
                ORDER BY (pubkey)
                SETTINGS index_granularity = 8192",
                self.database
            ))
            .execute()
            .await?;
        
        // Create programs table
        self.client
            .query(&format!(
                "CREATE TABLE IF NOT EXISTS {}.programs (
                    program_id String,
                    first_seen_slot UInt64,
                    invocation_count UInt64,
                    updated_at DateTime64(3) DEFAULT now64(3),
                    INDEX idx_program_id program_id TYPE bloom_filter(0.01) GRANULARITY 1
                ) ENGINE = ReplacingMergeTree(updated_at)
                ORDER BY (program_id)
                SETTINGS index_granularity = 8192",
                self.database
            ))
            .execute()
            .await?;
        
        // Create datasets table
        self.client
            .query(&format!(
                "CREATE TABLE IF NOT EXISTS {}.datasets (
                    id String,
                    name String,
                    description String,
                    slot_start UInt64,
                    slot_end UInt64,
                    filters String,
                    format String,
                    size_bytes UInt64,
                    row_count UInt64,
                    created_at DateTime64(3) DEFAULT now64(3),
                    status String,
                    INDEX idx_id id TYPE bloom_filter(0.01) GRANULARITY 1
                ) ENGINE = MergeTree()
                ORDER BY (created_at, id)
                SETTINGS index_granularity = 8192",
                self.database
            ))
            .execute()
            .await?;
        
        // Create models table
        self.client
            .query(&format!(
                "CREATE TABLE IF NOT EXISTS {}.ml_models (
                    id String,
                    name String,
                    type String,
                    dataset_id String,
                    parameters String,
                    metrics String,
                    training_duration_ms UInt64,
                    created_at DateTime64(3) DEFAULT now64(3),
                    status String,
                    INDEX idx_id id TYPE bloom_filter(0.01) GRANULARITY 1
                ) ENGINE = MergeTree()
                ORDER BY (created_at, id)
                SETTINGS index_granularity = 8192",
                self.database
            ))
            .execute()
            .await?;
        
        info!("ClickHouse database initialized successfully");
        
        Ok(())
    }
    
    pub async fn store_block(&self, block: &BlockData) -> Result<()> {
        let block_time = block.block_time.map(|t| {
            DateTime::<Utc>::from_timestamp(t, 0)
                .unwrap_or_else(|| Utc::now())
        });
        
        self.client
            .query(&format!(
                "INSERT INTO {}.blocks (slot, blockhash, parent_slot, block_time, block_height, transaction_count) 
                 VALUES (?, ?, ?, ?, ?, ?)",
                self.database
            ))
            .bind(block.slot)
            .bind(&block.blockhash)
            .bind(block.parent_slot)
            .bind(block_time)
            .bind(block.block_height)
            .bind(block.transaction_count)
            .execute()
            .await?;
        
        Ok(())
    }
    
    pub async fn store_transactions(&self, transactions: &[TransactionData]) -> Result<()> {
        if transactions.is_empty() {
            return Ok(());
        }
        
        let mut insert = self.client.insert(&format!("{}.transactions", self.database))?;
        
        for tx in transactions {
            let block_time = tx.block_time.map(|t| {
                DateTime::<Utc>::from_timestamp(t, 0)
                    .unwrap_or_else(|| Utc::now())
            });
            
            insert.write(&TransactionRow {
                signature: tx.signature.clone(),
                slot: tx.slot,
                block_time,
                fee: tx.fee,
                compute_units_consumed: tx.compute_units_consumed,
                status: tx.status,
                accounts: tx.accounts.clone(),
                program_ids: tx.program_ids.clone(),
                instructions_count: tx.instructions_count,
                logs: tx.logs.clone(),
            }).await?;
        }
        
        insert.end().await?;
        
        Ok(())
    }
    
    pub async fn get_blocks(&self, start_slot: u64, end_slot: u64, limit: u32) -> Result<Vec<BlockData>> {
        let rows = self.client
            .query(&format!(
                "SELECT slot, blockhash, parent_slot, block_time, block_height, transaction_count 
                 FROM {}.blocks 
                 WHERE slot >= ? AND slot <= ?
                 ORDER BY slot DESC
                 LIMIT ?",
                self.database
            ))
            .bind(start_slot)
            .bind(end_slot)
            .bind(limit)
            .fetch_all::<BlockRow>()
            .await?;
        
        Ok(rows.into_iter().map(|r| r.into()).collect())
    }
    
    pub async fn get_transactions(&self, slot: Option<u64>, limit: u32) -> Result<Vec<TransactionData>> {
        let query = if let Some(slot) = slot {
            format!(
                "SELECT signature, slot, block_time, fee, compute_units_consumed, status, 
                        accounts, program_ids, instructions_count, logs
                 FROM {}.transactions 
                 WHERE slot = ?
                 ORDER BY signature
                 LIMIT ?",
                self.database
            )
        } else {
            format!(
                "SELECT signature, slot, block_time, fee, compute_units_consumed, status, 
                        accounts, program_ids, instructions_count, logs
                 FROM {}.transactions 
                 ORDER BY slot DESC, signature
                 LIMIT ?",
                self.database
            )
        };
        
        let mut query = self.client.query(&query);
        
        if let Some(slot) = slot {
            query = query.bind(slot);
        }
        
        let rows = query.bind(limit).fetch_all::<TransactionRow>().await?;
        
        Ok(rows.into_iter().map(|r| r.into()).collect())
    }
    
    pub async fn get_storage_stats(&self) -> Result<StorageStats> {
        let block_count: u64 = self.client
            .query(&format!("SELECT count() FROM {}.blocks", self.database))
            .fetch_one()
            .await?;
        
        let transaction_count: u64 = self.client
            .query(&format!("SELECT count() FROM {}.transactions", self.database))
            .fetch_one()
            .await?;
        
        let account_count: u64 = self.client
            .query(&format!("SELECT count() FROM {}.accounts", self.database))
            .fetch_one()
            .await?;
        
        let program_count: u64 = self.client
            .query(&format!("SELECT count() FROM {}.programs", self.database))
            .fetch_one()
            .await?;
        
        Ok(StorageStats {
            block_count,
            transaction_count,
            account_count,
            program_count,
        })
    }
}

#[derive(Debug, Clone, Row, Serialize, Deserialize)]
struct BlockRow {
    slot: u64,
    blockhash: String,
    parent_slot: u64,
    block_time: Option<DateTime<Utc>>,
    block_height: Option<u64>,
    transaction_count: u32,
}

impl From<BlockRow> for BlockData {
    fn from(row: BlockRow) -> Self {
        BlockData {
            slot: row.slot,
            blockhash: row.blockhash,
            parent_slot: row.parent_slot,
            block_time: row.block_time.map(|t| t.timestamp()),
            block_height: row.block_height,
            transaction_count: row.transaction_count,
            rewards: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Row, Serialize, Deserialize)]
struct TransactionRow {
    signature: String,
    slot: u64,
    block_time: Option<DateTime<Utc>>,
    fee: u64,
    compute_units_consumed: Option<u64>,
    status: bool,
    accounts: Vec<String>,
    program_ids: Vec<String>,
    instructions_count: u32,
    logs: Option<Vec<String>>,
}

impl From<TransactionRow> for TransactionData {
    fn from(row: TransactionRow) -> Self {
        TransactionData {
            signature: row.signature,
            slot: row.slot,
            block_time: row.block_time.map(|t| t.timestamp()),
            fee: row.fee,
            compute_units_consumed: row.compute_units_consumed,
            status: row.status,
            accounts: row.accounts,
            program_ids: row.program_ids,
            instructions_count: row.instructions_count,
            logs: row.logs,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub block_count: u64,
    pub transaction_count: u64,
    pub account_count: u64,
    pub program_count: u64,
}