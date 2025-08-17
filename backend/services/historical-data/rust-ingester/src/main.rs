use anyhow::{Context, Result};
use backoff::{backoff::Backoff, ExponentialBackoff};
use chrono::Utc;
use clap::Parser;
use dashmap::DashMap;
use futures::stream::StreamExt;
use metrics::{counter, gauge, histogram};
use parking_lot::RwLock;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use serde::{Deserialize, Serialize};
use solana_sdk::signature::Signature;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::{interval, sleep};
use tonic::transport::channel::ClientTlsConfig;
use tracing::{debug, error, info, warn};
use yellowstone_grpc_client::{GeyserGrpcClient, GeyserGrpcClientError, Interceptor};
use yellowstone_grpc_proto::prelude::*;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// Ultra-high-performance Solana historical data ingester
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Yellowstone gRPC endpoint
    #[clap(long, env = "YELLOWSTONE_ENDPOINT", default_value = "https://grpc.mainnet.rpcpool.com")]
    yellowstone_endpoint: String,

    /// Yellowstone auth token
    #[clap(long, env = "YELLOWSTONE_TOKEN")]
    yellowstone_token: Option<String>,

    /// Kafka brokers
    #[clap(long, env = "KAFKA_BROKERS", default_value = "localhost:19092")]
    kafka_brokers: String,

    /// Kafka compression
    #[clap(long, env = "KAFKA_COMPRESSION", default_value = "snappy")]
    kafka_compression: String,

    /// Kafka batch size
    #[clap(long, env = "KAFKA_BATCH_SIZE", default_value = "1000000")]
    kafka_batch_size: usize,

    /// Kafka linger ms
    #[clap(long, env = "KAFKA_LINGER_MS", default_value = "10")]
    kafka_linger_ms: u64,

    /// Buffer size for each stream
    #[clap(long, env = "BUFFER_SIZE", default_value = "100000")]
    buffer_size: usize,

    /// Number of Kafka producer threads
    #[clap(long, env = "PRODUCER_THREADS", default_value = "8")]
    producer_threads: usize,

    /// Metrics port
    #[clap(long, env = "METRICS_PORT", default_value = "9090")]
    metrics_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlotMessage {
    slot: u64,
    parent_slot: u64,
    block_height: u64,
    block_time: i64,
    leader: String,
    rewards_json: String,
    block_hash: String,
    parent_hash: String,
    transaction_count: u32,
    entry_count: u32,
    tick_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlockMessage {
    slot: u64,
    block_height: u64,
    block_hash: String,
    parent_hash: String,
    block_time: i64,
    leader: String,
    rewards_json: String,
    block_cost: u64,
    max_supported_transaction_version: u8,
    transaction_count: u32,
    executed_transaction_count: u32,
    entries_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransactionMessage {
    signature: String,
    slot: u64,
    block_time: i64,
    block_index: u32,
    transaction_index: u32,
    is_vote: bool,
    success: bool,
    fee: u64,
    compute_units_consumed: u64,
    err: String,
    memo: String,
    signer: String,
    signers: Vec<String>,
    account_keys: Vec<String>,
    pre_balances: Vec<u64>,
    post_balances: Vec<u64>,
    pre_token_balances_json: String,
    post_token_balances_json: String,
    instructions_json: String,
    inner_instructions_json: String,
    log_messages: Vec<String>,
    rewards_json: String,
    loaded_addresses_json: String,
    return_data_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AccountMessage {
    pubkey: String,
    slot: u64,
    write_version: u64,
    lamports: u64,
    owner: String,
    executable: bool,
    rent_epoch: u64,
    data_len: u32,
    data_hash: String,
    update_type: String,
}

struct IngesterState {
    producer: FutureProducer,
    slots_processed: AtomicU64,
    transactions_processed: AtomicU64,
    accounts_processed: AtomicU64,
    last_slot: AtomicU64,
    is_running: AtomicBool,
    dedup_cache: Arc<DashMap<String, Instant>>,
}

impl IngesterState {
    fn new(kafka_config: ClientConfig) -> Result<Self> {
        let producer: FutureProducer = kafka_config
            .create()
            .context("Failed to create Kafka producer")?;

        Ok(Self {
            producer,
            slots_processed: AtomicU64::new(0),
            transactions_processed: AtomicU64::new(0),
            accounts_processed: AtomicU64::new(0),
            last_slot: AtomicU64::new(0),
            is_running: AtomicBool::new(true),
            dedup_cache: Arc::new(DashMap::new()),
        })
    }

    async fn send_to_kafka<T: Serialize>(&self, topic: &str, key: &str, message: &T) -> Result<()> {
        // Check dedup cache
        let cache_key = format!("{}-{}", topic, key);
        if let Some(entry) = self.dedup_cache.get(&cache_key) {
            if entry.elapsed() < Duration::from_secs(60) {
                return Ok(()); // Skip duplicate within 60 seconds
            }
        }

        let payload = serde_json::to_vec(message)?;
        let record = FutureRecord::to(topic)
            .key(key)
            .payload(&payload);

        let start = Instant::now();
        
        match self.producer.send(record, Timeout::After(Duration::from_secs(10))).await {
            Ok(_) => {
                histogram!("kafka_send_duration_ms", start.elapsed().as_millis() as f64, "topic" => topic);
                counter!("kafka_messages_sent", 1, "topic" => topic);
                
                // Update dedup cache
                self.dedup_cache.insert(cache_key, Instant::now());
                
                // Clean old entries periodically
                if self.dedup_cache.len() > 100000 {
                    let now = Instant::now();
                    self.dedup_cache.retain(|_, v| now.duration_since(*v) < Duration::from_secs(300));
                }
                
                Ok(())
            }
            Err((e, _)) => {
                counter!("kafka_send_errors", 1, "topic" => topic);
                Err(anyhow::anyhow!("Kafka send error: {}", e))
            }
        }
    }
}

async fn process_slot_update(state: Arc<IngesterState>, update: SubscribeUpdateSlot) -> Result<()> {
    let slot = update.slot;
    
    if let Some(block_time) = update.block_time {
        let message = SlotMessage {
            slot,
            parent_slot: update.parent.unwrap_or(0),
            block_height: update.slot, // Approximation
            block_time: block_time.timestamp,
            leader: String::new(), // Will be filled from block
            rewards_json: "[]",
            block_hash: String::new(),
            parent_hash: String::new(),
            transaction_count: 0,
            entry_count: 0,
            tick_count: 0,
        };

        state.send_to_kafka("solana.slots", &slot.to_string(), &message).await?;
        state.slots_processed.fetch_add(1, Ordering::Relaxed);
        state.last_slot.store(slot, Ordering::Relaxed);
        gauge!("ingester_last_slot", slot as f64);
    }
    
    Ok(())
}

async fn process_block_update(state: Arc<IngesterState>, update: SubscribeUpdateBlock) -> Result<()> {
    let block_msg = update.block.ok_or_else(|| anyhow::anyhow!("Block message missing"))?;
    let slot = update.slot;
    
    let message = BlockMessage {
        slot,
        block_height: block_msg.block_height.map(|h| h.block_height).unwrap_or(slot),
        block_hash: bs58::encode(&block_msg.blockhash).into_string(),
        parent_hash: bs58::encode(&block_msg.previous_blockhash).into_string(),
        block_time: block_msg.block_time.map(|t| t.timestamp).unwrap_or(0),
        leader: block_msg.rewards.first()
            .and_then(|r| r.pubkey.clone())
            .map(|p| bs58::encode(p).into_string())
            .unwrap_or_default(),
        rewards_json: serde_json::to_string(&block_msg.rewards).unwrap_or_else(|_| "[]".to_string()),
        block_cost: 0, // Would need to calculate from transactions
        max_supported_transaction_version: 0,
        transaction_count: block_msg.transactions.len() as u32,
        executed_transaction_count: block_msg.transactions.iter()
            .filter(|tx| tx.meta.as_ref().map(|m| m.err.is_none()).unwrap_or(false))
            .count() as u32,
        entries_json: "[]",
    };

    state.send_to_kafka("solana.blocks", &slot.to_string(), &message).await?;

    // Process transactions in the block
    for (index, tx) in block_msg.transactions.into_iter().enumerate() {
        if let Err(e) = process_transaction(state.clone(), slot, index as u32, tx).await {
            warn!("Failed to process transaction: {}", e);
        }
    }

    Ok(())
}

async fn process_transaction(
    state: Arc<IngesterState>,
    slot: u64,
    index: u32,
    tx: SubscribeUpdateTransactionInfo,
) -> Result<()> {
    let signature = tx.signature
        .map(|s| bs58::encode(s).into_string())
        .unwrap_or_default();
    
    if signature.is_empty() {
        return Ok(());
    }

    let meta = tx.meta.unwrap_or_default();
    let transaction = tx.transaction.unwrap_or_default();
    
    // Decode message
    let message = transaction.message.unwrap_or_default();
    let account_keys: Vec<String> = message.account_keys
        .iter()
        .map(|k| bs58::encode(k).into_string())
        .collect();
    
    // Check if it's a vote transaction
    let is_vote = message.instructions.iter().any(|ix| {
        ix.program_id_index as usize == 0 && 
        account_keys.get(0).map(|k| k.starts_with("Vote")).unwrap_or(false)
    });

    let tx_message = TransactionMessage {
        signature: signature.clone(),
        slot,
        block_time: 0, // Will be filled from block
        block_index: index,
        transaction_index: index,
        is_vote,
        success: meta.err.is_none(),
        fee: meta.fee,
        compute_units_consumed: meta.compute_units_consumed.unwrap_or(0),
        err: meta.err.as_ref()
            .and_then(|e| e.err.as_ref())
            .map(|e| format!("{:?}", e))
            .unwrap_or_default(),
        memo: String::new(),
        signer: account_keys.first().cloned().unwrap_or_default(),
        signers: account_keys.clone(),
        account_keys: account_keys.clone(),
        pre_balances: meta.pre_balances,
        post_balances: meta.post_balances,
        pre_token_balances_json: serde_json::to_string(&meta.pre_token_balances)
            .unwrap_or_else(|_| "[]".to_string()),
        post_token_balances_json: serde_json::to_string(&meta.post_token_balances)
            .unwrap_or_else(|_| "[]".to_string()),
        instructions_json: serde_json::to_string(&message.instructions)
            .unwrap_or_else(|_| "[]".to_string()),
        inner_instructions_json: serde_json::to_string(&meta.inner_instructions)
            .unwrap_or_else(|_| "[]".to_string()),
        log_messages: meta.log_messages,
        rewards_json: serde_json::to_string(&meta.rewards)
            .unwrap_or_else(|_| "[]".to_string()),
        loaded_addresses_json: serde_json::to_string(&meta.loaded_addresses)
            .unwrap_or_else(|_| "{}".to_string()),
        return_data_json: serde_json::to_string(&meta.return_data)
            .unwrap_or_else(|_| "null".to_string()),
    };

    state.send_to_kafka("solana.transactions", &signature, &tx_message).await?;
    state.transactions_processed.fetch_add(1, Ordering::Relaxed);
    
    Ok(())
}

async fn process_account_update(state: Arc<IngesterState>, update: SubscribeUpdateAccount) -> Result<()> {
    let account = update.account.ok_or_else(|| anyhow::anyhow!("Account missing"))?;
    let pubkey = bs58::encode(&account.pubkey).into_string();
    
    let message = AccountMessage {
        pubkey: pubkey.clone(),
        slot: update.slot,
        write_version: account.write_version,
        lamports: account.lamports,
        owner: bs58::encode(&account.owner).into_string(),
        executable: account.executable,
        rent_epoch: account.rent_epoch,
        data_len: account.data.len() as u32,
        data_hash: format!("{:x}", md5::compute(&account.data)),
        update_type: if update.is_startup { "create" } else { "update" }.to_string(),
    };

    state.send_to_kafka("solana.accounts", &pubkey, &message).await?;
    state.accounts_processed.fetch_add(1, Ordering::Relaxed);
    
    Ok(())
}

async fn connect_yellowstone(
    endpoint: String,
    token: Option<String>,
) -> Result<GeyserGrpcClient<impl Interceptor + Clone>> {
    let mut backoff = ExponentialBackoff::default();
    backoff.max_elapsed_time = Some(Duration::from_secs(300));

    loop {
        match GeyserGrpcClient::build_from_shared(endpoint.clone())? 
            .x_token(token.clone())?
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(10))
            .tls_config(ClientTlsConfig::new().with_native_roots())?
            .connect()
            .await
        {
            Ok(client) => {
                info!("Connected to Yellowstone gRPC");
                return Ok(client);
            }
            Err(e) => {
                let delay = backoff.next_backoff()
                    .ok_or_else(|| anyhow::anyhow!("Max reconnection attempts reached"))?;
                error!("Failed to connect to Yellowstone: {}. Retrying in {:?}", e, delay);
                sleep(delay).await;
            }
        }
    }
}

async fn run_ingester(args: Args) -> Result<()> {
    // Initialize Kafka producer
    let mut kafka_config = ClientConfig::new();
    kafka_config
        .set("bootstrap.servers", &args.kafka_brokers)
        .set("compression.type", &args.kafka_compression)
        .set("batch.size", &args.kafka_batch_size.to_string())
        .set("linger.ms", &args.kafka_linger_ms.to_string())
        .set("message.max.bytes", "10485760")
        .set("queue.buffering.max.messages", "1000000")
        .set("queue.buffering.max.kbytes", "1048576")
        .set("socket.keepalive.enable", "true")
        .set("request.timeout.ms", "30000");

    let state = Arc::new(IngesterState::new(kafka_config)?);

    // Connect to Yellowstone
    let mut client = connect_yellowstone(args.yellowstone_endpoint.clone(), args.yellowstone_token.clone()).await?;

    // Create subscription
    let mut slots_filter = HashMap::new();
    slots_filter.insert(
        "all".to_string(),
        SubscribeRequestFilterSlots {
            filter_by_commitment: Some(true),
        },
    );

    let mut blocks_filter = HashMap::new();
    blocks_filter.insert(
        "all".to_string(),
        SubscribeRequestFilterBlocks {
            account_include: vec![],
            include_transactions: Some(true),
            include_accounts: Some(false),
            include_entries: Some(false),
        },
    );

    let mut transactions_filter = HashMap::new();
    transactions_filter.insert(
        "all".to_string(),
        SubscribeRequestFilterTransactions {
            vote: None,
            failed: None,
            signature: None,
            account_include: vec![],
            account_exclude: vec![],
            account_required: vec![],
        },
    );

    let mut accounts_filter = HashMap::new();
    accounts_filter.insert(
        "all".to_string(),
        SubscribeRequestFilterAccounts {
            account: vec![],
            owner: vec![],
            filters: vec![],
        },
    );

    let request = SubscribeRequest {
        slots: slots_filter,
        blocks: blocks_filter,
        transactions: transactions_filter,
        accounts: accounts_filter,
        blocks_meta: HashMap::new(),
        entry: HashMap::new(),
        commitment: Some(CommitmentLevel::Confirmed as i32),
        accounts_data_slice: vec![],
        ping: None,
    };

    // Subscribe to updates
    let (_, mut stream) = client.subscribe_with_request(Some(request)).await?;
    info!("Subscribed to Yellowstone updates");

    // Process stream
    while let Some(message) = stream.next().await {
        match message {
            Ok(msg) => {
                for filter in msg.filters {
                    let state_clone = state.clone();
                    
                    tokio::spawn(async move {
                        for update in filter.1 {
                            match update.update_oneof {
                                Some(UpdateOneof::Slot(slot)) => {
                                    if let Err(e) = process_slot_update(state_clone.clone(), slot).await {
                                        error!("Failed to process slot: {}", e);
                                    }
                                }
                                Some(UpdateOneof::Block(block)) => {
                                    if let Err(e) = process_block_update(state_clone.clone(), block).await {
                                        error!("Failed to process block: {}", e);
                                    }
                                }
                                Some(UpdateOneof::Account(account)) => {
                                    if let Err(e) = process_account_update(state_clone.clone(), account).await {
                                        error!("Failed to process account: {}", e);
                                    }
                                }
                                _ => {}
                            }
                        }
                    });
                }
                
                // Update metrics
                counter!("yellowstone_messages_received", 1);
                gauge!("ingester_slots_processed", state.slots_processed.load(Ordering::Relaxed) as f64);
                gauge!("ingester_transactions_processed", state.transactions_processed.load(Ordering::Relaxed) as f64);
                gauge!("ingester_accounts_processed", state.accounts_processed.load(Ordering::Relaxed) as f64);
            }
            Err(e) => {
                error!("Stream error: {}", e);
                counter!("yellowstone_stream_errors", 1);
                
                // Reconnect on error
                warn!("Reconnecting to Yellowstone...");
                client = connect_yellowstone(args.yellowstone_endpoint.clone(), args.yellowstone_token.clone()).await?;
                let (_, new_stream) = client.subscribe_with_request(Some(request.clone())).await?;
                stream = new_stream;
                info!("Reconnected to Yellowstone");
            }
        }
    }

    Ok(())
}

async fn metrics_server(port: u16) {
    let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    let handle = builder.install_recorder().expect("Failed to install Prometheus recorder");

    let app = axum::Router::new()
        .route("/metrics", axum::routing::get(move || async move {
            handle.render()
        }));

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    info!("Metrics server listening on {}", addr);
    
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .expect("Failed to start metrics server");
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .json()
        .init();

    let args = Args::parse();
    info!("Starting Solana Historical Ingester");

    // Start metrics server
    let metrics_port = args.metrics_port;
    tokio::spawn(async move {
        metrics_server(metrics_port).await;
    });

    // Run ingester with retry logic
    loop {
        match run_ingester(args.clone()).await {
            Ok(_) => {
                warn!("Ingester stopped unexpectedly");
            }
            Err(e) => {
                error!("Ingester error: {}", e);
            }
        }
        
        warn!("Restarting ingester in 5 seconds...");
        sleep(Duration::from_secs(5)).await;
    }
}