use std::time::Duration;
use anyhow::Result;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::config::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    pub timestamp: i64,
    pub current_slot: u64,
    pub target_slot: u64,
    pub blocks_processed: u64,
    pub transactions_processed: u64,
    pub percentage: u32,
}

pub struct KafkaProducer {
    producer: FutureProducer,
    progress_topic: String,
    metrics_topic: String,
}

impl KafkaProducer {
    pub async fn new(config: &Config) -> Result<Self> {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.kafka_brokers)
            .set("message.timeout.ms", "5000")
            .set("compression.type", "snappy")
            .set("batch.size", "65536")
            .set("linger.ms", "10")
            .create()?;
        
        Ok(Self {
            producer,
            progress_topic: config.kafka_topic_progress.clone(),
            metrics_topic: config.kafka_topic_metrics.clone(),
        })
    }
    
    pub async fn send_progress(&self, update: ProgressUpdate) -> Result<()> {
        let payload = serde_json::to_string(&update)?;
        
        let record = FutureRecord::to(&self.progress_topic)
            .payload(&payload)
            .key(&update.current_slot.to_string());
        
        match self.producer.send(record, Duration::from_secs(1)).await {
            Ok(_) => {
                debug!("Progress update sent: slot {}", update.current_slot);
                Ok(())
            }
            Err((e, _)) => {
                error!("Failed to send progress update: {}", e);
                Err(anyhow::anyhow!("Kafka send failed: {}", e))
            }
        }
    }
}