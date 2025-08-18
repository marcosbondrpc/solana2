use anyhow::Result;
use chrono::{DateTime, Utc};
use clickhouse::Client as ClickHouseClient;
use dashmap::DashMap;
use futures::StreamExt;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use solana_client::rpc_client::RpcClient;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Signature;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use solana_transaction_status::option_serializer::OptionSerializer;

// DEX Program IDs
const RAYDIUM_V4: &str = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8";
const ORCA_WHIRLPOOL: &str = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc";
const PHOENIX: &str = "PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY";
const METEORA_POOLS: &str = "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo";
const OPENBOOK_V2: &str = "opnb2LAfJYbRMAHHvqjCwQxanZn7ReEHp1k81EohpZb";
const LIFINITY: &str = "2wT8Yq49kHgDzXuPxZSaeLaH1qbmGXtEyPy64bL7aD3c";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArbitrageTransaction {
    tx_signature: String,
    block_time: DateTime<Utc>,
    slot: u64,
    block_height: u64,
    signer: String,
    program: String,
    instruction_type: String,
    path: Vec<String>,
    dex_sequence: Vec<String>,
    token_in: String,
    token_out: String,
    amount_in: Decimal,
    amount_out: Decimal,
    revenue_sol: Decimal,
    costs_gas_sol: Decimal,
    costs_priority_fee_sol: Decimal,
    costs_total_sol: Decimal,
    net_profit_sol: Decimal,
    roi: f32,
    mev_type: u8,
    bundle_id: String,
    bundle_landed: u8,
    jito_tip: Decimal,
    market_volatility: f32,
    liquidity_depth: Decimal,
    slippage_pct: f32,
    latency_ms: u32,
    parse_time_us: u32,
    execution_time_ms: u32,
    risk_score: f32,
    risk_factors: String,
    label_is_arb: u8,
    confidence_score: f32,
    strategy_type: String,
}

struct DataCapturePipeline {
    rpc_client: Arc<RpcClient>,
    kafka_producer: Arc<FutureProducer>,
    redis_conn: Arc<RwLock<ConnectionManager>>,
    clickhouse_client: Arc<ClickHouseClient>,
    cache: Arc<DashMap<String, ArbitrageTransaction>>,
}

impl DataCapturePipeline {
    async fn new() -> Result<Self> {
        // Initialize Solana RPC client
        let rpc_client = Arc::new(RpcClient::new_with_commitment(
            "http://localhost:8899".to_string(),
            CommitmentConfig::confirmed(),
        ));

        // Initialize Kafka producer
        let kafka_producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", "localhost:9092")
            .set("message.timeout.ms", "5000")
            .set("compression.type", "lz4")
            .set("batch.size", "65536")
            .set("linger.ms", "10")
            .create()?;

        // Initialize Redis connection
        let redis_client = redis::Client::open("redis://127.0.0.1/")?;
        let redis_conn = ConnectionManager::new(redis_client).await?;

        // Initialize ClickHouse client
        let clickhouse_client = ClickHouseClient::default()
            .with_url("http://localhost:8123")
            .with_database("solana_arbitrage");

        Ok(Self {
            rpc_client,
            kafka_producer: Arc::new(kafka_producer),
            redis_conn: Arc::new(RwLock::new(redis_conn)),
            clickhouse_client: Arc::new(clickhouse_client),
            cache: Arc::new(DashMap::new()),
        })
    }

    async fn process_transaction(&self, signature: &Signature) -> Result<Option<ArbitrageTransaction>> {
        let start = Instant::now();
        // Fetch transaction from Solana
        let tx = match self.rpc_client.get_transaction(signature, solana_transaction_status::UiTransactionEncoding::Json) {
            Ok(tx) => tx,
            Err(e) => {
                warn!("Failed to fetch transaction {}: {}", signature, e);
                return Ok(None);
            }
        };
        // Parse transaction for arbitrage patterns
        let parse_start = Instant::now();
        let arb_data = self.parse_arbitrage_data(tx).await?;
        let parse_time_us = parse_start.elapsed().as_micros() as u32;

        if let Some(mut arb) = arb_data {
            arb.parse_time_us = parse_time_us;
            arb.latency_ms = start.elapsed().as_millis() as u32;
            self.cache_transaction(&arb).await?;
            self.send_to_kafka(&arb).await?;
            self.store_in_clickhouse(&arb).await?;
            Ok(Some(arb))
        } else {
            Ok(None)
        }
    }

    async fn parse_arbitrage_data(&self, tx: solana_transaction_status::EncodedConfirmedTransactionWithStatusMeta) -> Result<Option<ArbitrageTransaction>> {
        // Check if transaction involves DEX programs
        let mut dex_programs = Vec::new();
        let mut is_potential_arb = false;

        if let Some(meta) = &tx.transaction.meta {
            if let OptionSerializer::Some(inner_sets) = meta.inner_instructions.clone() {
                for instruction_set in inner_sets {
                    for instruction in &instruction_set.instructions {
                        // For UiInstruction, we must inspect either parsed program or programIdIndex via message
                        if let Some(decoded) = tx.transaction.transaction.decode() {
                            let msg = &decoded.message;
                            let program_str = match instruction {
                                solana_transaction_status::UiInstruction::Compiled(ci) => {
                                    let pidx = ci.program_id_index as usize;
                                    let keys = msg.static_account_keys();
                                    if pidx < keys.len() { keys[pidx].to_string() } else { continue; }
                                }
                                solana_transaction_status::UiInstruction::Parsed(pi) => match &pi {
                                    solana_transaction_status::UiParsedInstruction::Parsed(p) => p.program.clone(),
                                    solana_transaction_status::UiParsedInstruction::PartiallyDecoded(pd) => pd.program_id.clone(),
                                }
                            };
                            if self.is_dex_program(&program_str) {
                                dex_programs.push(self.get_dex_name(&program_str));
                                if dex_programs.len() >= 2 { is_potential_arb = true; }
                            }
                        }
                    }
                }
            }
        }

        if !is_potential_arb {
            return Ok(None);
        }

        // Calculate arbitrage metrics
        let (profit, roi) = self.calculate_profit(&tx)?;
        // Build arbitrage transaction data
        let decoded = tx.transaction.transaction.decode().ok_or_else(|| anyhow::anyhow!("decode failed"))?;
        let arb_tx = ArbitrageTransaction {
            tx_signature: decoded.signatures[0].to_string(),
            block_time: Utc::now(),
            slot: tx.slot,
            block_height: tx.slot,
            signer: decoded.message.static_account_keys()[0].to_string(),
            program: dex_programs[0].clone(),
            instruction_type: "swap".to_string(),
            path: dex_programs.clone(),
            dex_sequence: dex_programs,
            token_in: "SOL".to_string(),
            token_out: "USDC".to_string(),
            amount_in: Decimal::from(1),
            amount_out: Decimal::from(1),
            revenue_sol: profit,
            costs_gas_sol: Decimal::from_str("0.00005")?,
            costs_priority_fee_sol: Decimal::from_str("0.0001")?,
            costs_total_sol: Decimal::from_str("0.00015")?,
            net_profit_sol: profit - Decimal::from_str("0.00015")?,
            roi,
            mev_type: 1,
            bundle_id: "".to_string(),
            bundle_landed: 0,
            jito_tip: Decimal::ZERO,
            market_volatility: 0.0,
            liquidity_depth: Decimal::from(100000),
            slippage_pct: 0.1,
            latency_ms: 0,
            parse_time_us: 0,
            execution_time_ms: 0,
            risk_score: 0.2,
            risk_factors: "{}".to_string(),
            label_is_arb: 1,
            confidence_score: 0.85,
            strategy_type: "cross_dex_arb".to_string(),
        };
        Ok(Some(arb_tx))
    }

    fn is_dex_program(&self, program_id: &str) -> bool {
        matches!(
            program_id,
            RAYDIUM_V4 | ORCA_WHIRLPOOL | PHOENIX | METEORA_POOLS | OPENBOOK_V2 | LIFINITY
        )
    }

    fn get_dex_name(&self, program_id: &str) -> String {
        match program_id {
            RAYDIUM_V4 => "Raydium",
            ORCA_WHIRLPOOL => "Orca",
            PHOENIX => "Phoenix",
            METEORA_POOLS => "Meteora",
            OPENBOOK_V2 => "OpenBook",
            LIFINITY => "Lifinity",
            _ => "Unknown",
        }.to_string()
    }

    fn calculate_profit(&self, _tx: &solana_transaction_status::EncodedConfirmedTransactionWithStatusMeta) -> Result<(Decimal, f32)> {
        let profit = Decimal::from_str("0.05")?;
        let roi = 5.0;
        Ok((profit, roi))
    }

    async fn cache_transaction(&self, arb: &ArbitrageTransaction) -> Result<()> {
        let mut conn = self.redis_conn.write().await;
        let key = format!("arb:{}", arb.tx_signature);
        let value = serde_json::to_string(arb)?;
        conn.set_ex::<_, _, ()>(&key, value, 3600).await?;
        let score = arb.net_profit_sol.to_string().parse::<f64>()?;
        conn.zadd::<_, _, _, ()>("recent_arbs", &arb.tx_signature, score).await?;
        conn.zremrangebyrank::<_, ()>("recent_arbs", 0, -1001).await?;
        Ok(())
    }

    async fn send_to_kafka(&self, arb: &ArbitrageTransaction) -> Result<()> {
        let payload = serde_json::to_string(arb)?;
        let record = FutureRecord::to("solana-transactions")
            .key(&arb.tx_signature)
            .payload(&payload);
        match self.kafka_producer.send(record, Duration::from_secs(0)).await {
            Ok(_) => info!("Sent to Kafka: {}", arb.tx_signature),
            Err((e, _)) => error!("Kafka send error: {}", e),
        }
        Ok(())
    }

    async fn store_in_clickhouse(&self, arb: &ArbitrageTransaction) -> Result<()> {
        let query = r#"
            INSERT INTO transactions (
                tx_signature, block_time, slot, signer, program,
                net_profit_sol, roi, latency_ms, label_is_arb
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;
        self.clickhouse_client
            .query(query)
            .bind(&arb.tx_signature)
            .bind(arb.block_time)
            .bind(arb.slot)
            .bind(&arb.signer)
            .bind(&arb.program)
            .bind(arb.net_profit_sol.to_string())
            .bind(arb.roi)
            .bind(arb.latency_ms)
            .bind(arb.label_is_arb)
            .execute()
            .await?;
        Ok(())
    }

    async fn run(&self) -> Result<()> {
        info!("Starting Solana Arbitrage Data Capture Pipeline");
        loop {
            match self.rpc_client.get_slot() {
                Ok(slot) => {
                    info!("Processing slot: {}", slot);
                    match self.rpc_client.get_block(slot) {
                        Ok(block) => {
                            for tx in block.transactions {
                                if let Some(decoded) = tx.transaction.decode() { if let Some(signature) = decoded.signatures.first() {
                                    tokio::spawn({
                                        let pipeline = self.clone();
                                        let sig = *signature;
                                        async move {
                                            if let Err(e) = pipeline.process_transaction(&sig).await {
                                                error!("Error processing transaction: {}", e);
                                            }
                                        }
                                    });
                                } }
                            }
                        }
                        Err(e) => warn!("Failed to get block {}: {}", slot, e),
                    }
                }
                Err(e) => error!("Failed to get slot: {}", e),
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
        }
    }
}

impl Clone for DataCapturePipeline {
    fn clone(&self) -> Self {
        Self {
            rpc_client: self.rpc_client.clone(),
            kafka_producer: self.kafka_producer.clone(),
            redis_conn: self.redis_conn.clone(),
            clickhouse_client: self.clickhouse_client.clone(),
            cache: self.cache.clone(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();
    info!("Initializing Solana Arbitrage Data Capture System");
    let pipeline = DataCapturePipeline::new().await?;
    pipeline.run().await?;
    Ok(())
}