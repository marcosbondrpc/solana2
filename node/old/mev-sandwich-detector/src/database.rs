use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use clickhouse_rs::{Pool, Client, Block, row};
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::ClientConfig;
use tokio::sync::mpsc;
use serde_json::json;
use tracing::{info, debug, error};

use crate::core::SandwichDecision;

pub struct ClickHouseWriter {
    pool: Pool,
    kafka_producer: Arc<FutureProducer>,
    write_buffer: Arc<tokio::sync::Mutex<Vec<SandwichDecision>>>,
    batch_size: usize,
}

impl ClickHouseWriter {
    pub async fn new() -> Result<Self> {
        info!("Initializing ClickHouse writer for 200k+ rows/s");
        
        // Create connection pool
        let pool = Pool::new(
            "tcp://localhost:9000?compression=lz4&connection_timeout=10&query_timeout=30"
        );
        
        // Initialize Kafka producer for streaming
        let kafka_producer = ClientConfig::new()
            .set("bootstrap.servers", "localhost:9092")
            .set("batch.size", "1048576")              // 1MB batches
            .set("linger.ms", "10")                    // 10ms linger
            .set("compression.type", "zstd")
            .set("acks", "1")                          // Leader ack only for speed
            .set("request.timeout.ms", "5000")
            .create::<FutureProducer>()?;
        
        Ok(Self {
            pool,
            kafka_producer: Arc::new(kafka_producer),
            write_buffer: Arc::new(tokio::sync::Mutex::new(Vec::with_capacity(10000))),
            batch_size: 1000,
        })
    }
    
    pub async fn record_decision(&self, decision: SandwichDecision) -> Result<()> {
        // Send to Kafka for immediate streaming
        self.send_to_kafka(&decision).await?;
        
        // Buffer for batch insert
        let mut buffer = self.write_buffer.lock().await;
        buffer.push(decision);
        
        if buffer.len() >= self.batch_size {
            let batch = buffer.drain(..).collect::<Vec<_>>();
            drop(buffer); // Release lock
            
            // Async batch insert
            let pool = self.pool.clone();
            tokio::spawn(async move {
                if let Err(e) = Self::batch_insert(pool, batch).await {
                    error!("Batch insert failed: {}", e);
                }
            });
        }
        
        Ok(())
    }
    
    async fn send_to_kafka(&self, decision: &SandwichDecision) -> Result<()> {
        let payload = json!({
            "timestamp": decision.timestamp.elapsed().as_micros(),
            "target_tx": hex::encode(&decision.target_tx),
            "expected_profit": decision.expected_profit,
            "gas_cost": decision.gas_cost,
            "confidence": decision.confidence,
            "tip_amount": decision.tip_amount,
            "features": {
                "input_amount": decision.features.input_amount,
                "output_amount": decision.features.output_amount,
                "pool_reserves": decision.features.pool_reserves,
                "gas_price": decision.features.gas_price,
                "slippage": decision.features.slippage_tolerance,
                "mempool_depth": decision.features.mempool_depth,
            }
        });
        
        let record = FutureRecord::to("sandwich-decisions")
            .key(&decision.target_tx[..8])
            .payload(&serde_json::to_vec(&payload)?);
        
        // Fire and forget for speed
        let producer = self.kafka_producer.clone();
        tokio::spawn(async move {
            if let Err((e, _)) = producer.send(record, Duration::from_millis(100)).await {
                debug!("Kafka send failed: {}", e);
            }
        });
        
        Ok(())
    }
    
    async fn batch_insert(pool: Pool, decisions: Vec<SandwichDecision>) -> Result<()> {
        let start = Instant::now();
        let mut client = pool.get_handle().await?;
        
        // Prepare block
        let mut block = Block::new();
        
        for decision in &decisions {
            let features_json = serde_json::to_string(&json!({
                "input": decision.features.input_amount,
                "output": decision.features.output_amount,
                "reserves": decision.features.pool_reserves,
                "gas_price": decision.features.gas_price,
                "slippage": decision.features.slippage_tolerance,
                "depth": decision.features.mempool_depth,
                "time_features": decision.features.time_features,
                "pool_features": decision.features.pool_features,
                "market_features": decision.features.market_features,
            }))?;
            
            block.push(row! {
                timestamp: chrono::Utc::now(),
                decision_time_us: decision.timestamp.elapsed().as_micros() as u32,
                target_tx: hex::encode(&decision.target_tx),
                front_tx: hex::encode(&decision.front_tx),
                back_tx: hex::encode(&decision.back_tx),
                expected_profit: decision.expected_profit,
                gas_cost: decision.gas_cost,
                net_profit: decision.expected_profit.saturating_sub(decision.gas_cost),
                confidence: decision.confidence,
                tip_amount: decision.tip_amount,
                features: features_json,
            })?;
        }
        
        // Execute insert
        client.insert("mev_sandwich", block).await?;
        
        debug!(
            "Inserted {} decisions in {:?} ({:.0} rows/s)",
            decisions.len(),
            start.elapsed(),
            decisions.len() as f64 / start.elapsed().as_secs_f64()
        );
        
        Ok(())
    }
    
    pub async fn record_outcome(
        &self,
        tx_hash: &[u8; 64],
        landed: bool,
        actual_profit: Option<u64>,
    ) -> Result<()> {
        let mut client = self.pool.get_handle().await?;
        
        client.execute(
            "INSERT INTO bundle_outcomes (tx_hash, landed, actual_profit, timestamp) VALUES (?, ?, ?, ?)",
            &[
                &hex::encode(tx_hash),
                &landed,
                &actual_profit,
                &chrono::Utc::now(),
            ]
        ).await?;
        
        Ok(())
    }
}