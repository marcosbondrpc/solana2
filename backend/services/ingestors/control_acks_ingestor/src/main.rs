// LEGENDARY Control ACKs Ingestor - Kafka to ClickHouse Pipeline
// Ingests hash-chained control acknowledgments with cryptographic verification
// Optimized for ultra-high throughput with batching and compression

use anyhow::{Result, Context};
use clickhouse::{Client, Row};
use rdkafka::{
    consumer::{Consumer, StreamConsumer, CommitMode},
    ClientConfig,
    Message,
    TopicPartitionList,
    Offset,
};
use futures::StreamExt;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, NaiveDateTime};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use tokio::sync::RwLock;
use std::sync::Arc;
use tracing::{info, warn, error, debug};
use prometheus::{Counter, Histogram, Gauge, register_counter, register_histogram, register_gauge};

// Protobuf message definition (matches ack_writer.rs)
#[derive(Debug, Clone, Serialize, Deserialize, prost::Message)]
pub struct Ack {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(int32, tag = "2")]
    pub module: i32,
    #[prost(string, tag = "3")]
    pub agent_id: String,
    #[prost(uint64, tag = "4")]
    pub ts: u64,
    #[prost(string, tag = "5")]
    pub status: String,
    #[prost(string, tag = "6")]
    pub reason: String,
    #[prost(bytes = "vec", tag = "7")]
    pub hash: Vec<u8>,
    #[prost(bytes = "vec", tag = "8")]
    pub prev_hash: Vec<u8>,
}

// ClickHouse row structure
#[derive(Row, Serialize, Deserialize, Debug)]
struct AckRow {
    dt: DateTime<Utc>,
    ts: DateTime<Utc>,
    request_id: String,
    module: String,
    agent_id: String,
    status: String,
    reason: String,
    hash: Vec<u8>,
    prev_hash: Vec<u8>,
    sequence: u64,
    verified: bool,
}

// Metrics
lazy_static::lazy_static! {
    static ref ACKS_PROCESSED: Counter = register_counter!(
        "control_acks_processed_total",
        "Total number of control ACKs processed"
    ).unwrap();
    
    static ref ACKS_VERIFIED: Counter = register_counter!(
        "control_acks_verified_total",
        "Total number of control ACKs successfully verified"
    ).unwrap();
    
    static ref ACKS_FAILED: Counter = register_counter!(
        "control_acks_failed_total",
        "Total number of control ACKs that failed verification"
    ).unwrap();
    
    static ref BATCH_SIZE: Histogram = register_histogram!(
        "control_acks_batch_size",
        "Size of batches written to ClickHouse"
    ).unwrap();
    
    static ref WRITE_LATENCY: Histogram = register_histogram!(
        "control_acks_write_latency_seconds",
        "Latency of writing to ClickHouse"
    ).unwrap();
    
    static ref KAFKA_LAG: Gauge = register_gauge!(
        "control_acks_kafka_lag",
        "Current Kafka consumer lag"
    ).unwrap();
}

// Configuration
#[derive(Debug, Deserialize)]
struct Config {
    kafka_brokers: String,
    kafka_topic: String,
    kafka_group_id: String,
    clickhouse_url: String,
    clickhouse_database: String,
    clickhouse_table: String,
    batch_size: usize,
    batch_timeout_ms: u64,
    verify_chain: bool,
    metrics_port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            kafka_brokers: "localhost:9092".to_string(),
            kafka_topic: "control-acks-proto".to_string(),
            kafka_group_id: "ch-ack-ingestor".to_string(),
            clickhouse_url: "http://localhost:8123".to_string(),
            clickhouse_database: "default".to_string(),
            clickhouse_table: "control_acks".to_string(),
            batch_size: 1000,
            batch_timeout_ms: 100,
            verify_chain: true,
            metrics_port: 9091,
        }
    }
}

// Hash chain verifier
struct ChainVerifier {
    last_hash: RwLock<Option<Vec<u8>>>,
    sequence: RwLock<u64>,
}

impl ChainVerifier {
    fn new() -> Self {
        Self {
            last_hash: RwLock::new(None),
            sequence: RwLock::new(0),
        }
    }

    async fn verify(&self, ack: &Ack) -> bool {
        let mut last_hash = self.last_hash.write().await;
        
        // First ACK or chain reset
        if last_hash.is_none() {
            *last_hash = Some(ack.hash.clone());
            return true;
        }

        // Verify chain continuity
        let expected_prev = last_hash.as_ref().unwrap();
        if &ack.prev_hash != expected_prev {
            warn!(
                "Chain broken: expected prev_hash {}, got {}",
                hex::encode(&expected_prev[..8]),
                hex::encode(&ack.prev_hash[..8])
            );
            return false;
        }

        // Verify hash computation
        let computed_hash = Self::compute_hash(ack);
        if computed_hash != ack.hash {
            warn!(
                "Hash mismatch: computed {}, stored {}",
                hex::encode(&computed_hash[..8]),
                hex::encode(&ack.hash[..8])
            );
            return false;
        }

        // Update state
        *last_hash = Some(ack.hash.clone());
        true
    }

    fn compute_hash(ack: &Ack) -> Vec<u8> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(ack.request_id.as_bytes());
        hasher.update(&ack.module.to_le_bytes());
        hasher.update(ack.agent_id.as_bytes());
        hasher.update(&ack.ts.to_le_bytes());
        hasher.update(ack.status.as_bytes());
        hasher.update(ack.reason.as_bytes());
        hasher.update(&ack.prev_hash);
        
        hasher.finalize().as_bytes().to_vec()
    }

    async fn get_sequence(&self) -> u64 {
        let mut seq = self.sequence.write().await;
        let current = *seq;
        *seq += 1;
        current
    }
}

// Main ingestor
struct ControlAcksIngestor {
    config: Config,
    consumer: StreamConsumer,
    clickhouse: Client,
    verifier: Arc<ChainVerifier>,
    batch: VecDeque<AckRow>,
    last_flush: Instant,
}

impl ControlAcksIngestor {
    async fn new(config: Config) -> Result<Self> {
        // Create Kafka consumer with optimized settings
        let consumer: StreamConsumer = ClientConfig::new()
            .set("bootstrap.servers", &config.kafka_brokers)
            .set("group.id", &config.kafka_group_id)
            .set("enable.auto.commit", "false")
            .set("auto.offset.reset", "earliest")
            .set("session.timeout.ms", "6000")
            .set("heartbeat.interval.ms", "2000")
            .set("max.poll.interval.ms", "300000")
            .set("fetch.min.bytes", "1024")
            .set("fetch.wait.max.ms", "10")
            .set("enable.partition.eof", "false")
            .set("compression.type", "zstd")
            .create()
            .context("Failed to create Kafka consumer")?;

        // Subscribe to topic
        consumer
            .subscribe(&[&config.kafka_topic])
            .context("Failed to subscribe to Kafka topic")?;

        // Create ClickHouse client with connection pooling
        let clickhouse = Client::default()
            .with_url(&config.clickhouse_url)
            .with_database(&config.clickhouse_database)
            .with_compression(clickhouse::Compression::Lz4Hc)
            .with_option("max_threads", "8")
            .with_option("max_block_size", "65536")
            .with_option("max_insert_threads", "4");

        Ok(Self {
            config,
            consumer,
            clickhouse,
            verifier: Arc::new(ChainVerifier::new()),
            batch: VecDeque::with_capacity(1000),
            last_flush: Instant::now(),
        })
    }

    async fn run(&mut self) -> Result<()> {
        info!("Starting Control ACKs Ingestor");
        info!("Kafka: {} -> ClickHouse: {}/{}", 
            self.config.kafka_topic,
            self.config.clickhouse_database,
            self.config.clickhouse_table
        );

        let mut stream = self.consumer.stream();
        
        while let Some(message_result) = stream.next().await {
            match message_result {
                Ok(message) => {
                    if let Some(payload) = message.payload() {
                        self.process_message(payload).await?;
                        
                        // Commit offset after processing
                        self.consumer.commit_message(&message, CommitMode::Async)?;
                        
                        // Update lag metric
                        if let Ok(watermarks) = self.consumer.fetch_watermarks(
                            message.topic(),
                            message.partition(),
                            Duration::from_millis(100)
                        ) {
                            let lag = watermarks.high() - message.offset();
                            KAFKA_LAG.set(lag as f64);
                        }
                    }
                    
                    // Check if batch should be flushed
                    if self.should_flush() {
                        self.flush_batch().await?;
                    }
                }
                Err(e) => {
                    error!("Kafka error: {}", e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }

        Ok(())
    }

    async fn process_message(&mut self, payload: &[u8]) -> Result<()> {
        // Decode protobuf
        let ack = match prost::Message::decode(payload) {
            Ok(ack) => ack,
            Err(e) => {
                warn!("Failed to decode ACK: {}", e);
                ACKS_FAILED.inc();
                return Ok(());
            }
        };

        // Verify hash chain if enabled
        let verified = if self.config.verify_chain {
            let is_valid = self.verifier.verify(&ack).await;
            if is_valid {
                ACKS_VERIFIED.inc();
            } else {
                ACKS_FAILED.inc();
            }
            is_valid
        } else {
            true
        };

        // Convert module enum
        let module_str = match ack.module {
            0 => "ARBITRAGE",
            1 => "MEV",
            _ => "UNKNOWN",
        }.to_string();

        // Create ClickHouse row
        let row = AckRow {
            dt: Utc::now(),
            ts: DateTime::from_timestamp(ack.ts as i64, 0).unwrap_or(Utc::now()),
            request_id: ack.request_id,
            module: module_str,
            agent_id: ack.agent_id,
            status: ack.status,
            reason: ack.reason,
            hash: ack.hash,
            prev_hash: ack.prev_hash,
            sequence: self.verifier.get_sequence().await,
            verified,
        };

        // Add to batch
        self.batch.push_back(row);
        ACKS_PROCESSED.inc();

        Ok(())
    }

    fn should_flush(&self) -> bool {
        self.batch.len() >= self.config.batch_size ||
        self.last_flush.elapsed() > Duration::from_millis(self.config.batch_timeout_ms)
    }

    async fn flush_batch(&mut self) -> Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }

        let batch_size = self.batch.len();
        let start = Instant::now();

        debug!("Flushing batch of {} ACKs to ClickHouse", batch_size);

        // Build insert query with all rows
        let mut insert = self.clickhouse.insert(&self.config.clickhouse_table)?;
        
        while let Some(row) = self.batch.pop_front() {
            insert = insert.write(&row).await?;
        }

        // Execute insert
        insert.end().await.context("Failed to write to ClickHouse")?;

        // Update metrics
        let elapsed = start.elapsed();
        BATCH_SIZE.observe(batch_size as f64);
        WRITE_LATENCY.observe(elapsed.as_secs_f64());

        info!(
            "Flushed {} ACKs to ClickHouse in {:.2}ms",
            batch_size,
            elapsed.as_secs_f64() * 1000.0
        );

        self.last_flush = Instant::now();
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down ingestor...");
        
        // Flush remaining batch
        if !self.batch.is_empty() {
            self.flush_batch().await?;
        }

        // Commit final offsets
        self.consumer.commit_consumer_state(CommitMode::Sync)?;
        
        info!("Shutdown complete");
        Ok(())
    }
}

// Metrics HTTP server
async fn serve_metrics(port: u16) {
    use axum::{Router, routing::get, response::Response};
    use prometheus::{Encoder, TextEncoder};

    let app = Router::new()
        .route("/metrics", get(|| async {
            let encoder = TextEncoder::new();
            let metric_families = prometheus::gather();
            let mut buffer = vec![];
            encoder.encode(&metric_families, &mut buffer).unwrap();
            Response::builder()
                .header("Content-Type", encoder.format_type())
                .body(String::from_utf8(buffer).unwrap())
                .unwrap()
        }))
        .route("/health", get(|| async { "OK" }));

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .unwrap();
    
    info!("Metrics server listening on :{}", port);
    
    axum::serve(listener, app).await.unwrap();
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("control_acks_ingestor=debug".parse()?)
        )
        .json()
        .init();

    // Load configuration
    let config = Config {
        kafka_brokers: std::env::var("KAFKA_BROKERS")
            .unwrap_or_else(|_| "localhost:9092".to_string()),
        kafka_topic: std::env::var("KAFKA_TOPIC")
            .unwrap_or_else(|_| "control-acks-proto".to_string()),
        kafka_group_id: std::env::var("KAFKA_GROUP_ID")
            .unwrap_or_else(|_| "ch-ack-ingestor".to_string()),
        clickhouse_url: std::env::var("CLICKHOUSE_URL")
            .unwrap_or_else(|_| "http://localhost:8123".to_string()),
        clickhouse_database: std::env::var("CLICKHOUSE_DATABASE")
            .unwrap_or_else(|_| "default".to_string()),
        clickhouse_table: std::env::var("CLICKHOUSE_TABLE")
            .unwrap_or_else(|_| "control_acks".to_string()),
        batch_size: std::env::var("BATCH_SIZE")
            .unwrap_or_else(|_| "1000".to_string())
            .parse()?,
        batch_timeout_ms: std::env::var("BATCH_TIMEOUT_MS")
            .unwrap_or_else(|_| "100".to_string())
            .parse()?,
        verify_chain: std::env::var("VERIFY_CHAIN")
            .unwrap_or_else(|_| "true".to_string())
            .parse()?,
        metrics_port: std::env::var("METRICS_PORT")
            .unwrap_or_else(|_| "9091".to_string())
            .parse()?,
    };

    info!("Starting with config: {:?}", config);

    // Start metrics server
    tokio::spawn(serve_metrics(config.metrics_port));

    // Create and run ingestor
    let mut ingestor = ControlAcksIngestor::new(config).await?;

    // Handle shutdown signals
    let shutdown = tokio::signal::ctrl_c();
    
    tokio::select! {
        result = ingestor.run() => {
            if let Err(e) = result {
                error!("Ingestor failed: {}", e);
            }
        }
        _ = shutdown => {
            info!("Received shutdown signal");
            ingestor.shutdown().await?;
        }
    }

    Ok(())
}