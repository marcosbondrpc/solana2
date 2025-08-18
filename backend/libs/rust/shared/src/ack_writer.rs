// LEGENDARY Hash-Chained ACK Writer for Control Command Auditing
// Provides cryptographic tamper-evidence for all control operations
// Uses BLAKE3 for ultra-fast hashing with 256-bit security

use anyhow::{Result, Context};
use blake3::Hasher;
use rdkafka::{
    producer::{FutureProducer, FutureRecord},
    ClientConfig,
    util::Timeout,
};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use tokio::sync::RwLock;
use std::sync::Arc;
use prost::Message as ProstMessage;

/// Control module types from protobuf
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i32)]
pub enum Module {
    Arbitrage = 0,
    MEV = 1,
    Unknown = 99,
}

impl From<i32> for Module {
    fn from(value: i32) -> Self {
        match value {
            0 => Module::Arbitrage,
            1 => Module::MEV,
            _ => Module::Unknown,
        }
    }
}

/// Control ACK with hash chaining for tamper evidence
#[derive(Debug, Clone, Serialize, Deserialize, ProstMessage)]
pub struct Ack {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(enumeration = "Module", tag = "2")]
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

/// Thread-safe hash chain state
#[derive(Debug)]
struct HashChainState {
    prev_hash: Vec<u8>,
    sequence: u64,
    last_persist_seq: u64,
}

/// ACK writer with built-in hash chaining and Kafka publishing
pub struct AckWriter {
    producer: FutureProducer,
    topic: String,
    state: Arc<RwLock<HashChainState>>,
    agent_id: String,
    persist_path: Option<String>,
}

impl AckWriter {
    /// Create new ACK writer with Kafka configuration
    pub fn new(brokers: &str, agent_id: impl Into<String>) -> Result<Self> {
        let producer = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .set("compression.type", "zstd")
            .set("compression.level", "3")
            .set("linger.ms", "1")
            .set("batch.size", "32768")
            .set("request.timeout.ms", "5000")
            .set("delivery.timeout.ms", "10000")
            .set("enable.idempotence", "true")
            .set("max.in.flight.requests.per.connection", "5")
            .set("retries", "10")
            .create::<FutureProducer>()
            .context("Failed to create Kafka producer")?;

        // Initialize with genesis hash (all zeros)
        let genesis_hash = vec![0u8; 32];
        
        Ok(Self {
            producer,
            topic: "control-acks-proto".to_string(),
            state: Arc::new(RwLock::new(HashChainState {
                prev_hash: genesis_hash,
                sequence: 0,
                last_persist_seq: 0,
            })),
            agent_id: agent_id.into(),
            persist_path: None,
        })
    }

    /// Enable hash chain persistence to disk for recovery
    pub fn with_persistence(mut self, path: impl Into<String>) -> Self {
        self.persist_path = Some(path.into());
        self
    }

    /// Load hash chain state from disk if available
    pub async fn load_state(&self) -> Result<()> {
        if let Some(ref path) = self.persist_path {
            if std::path::Path::new(path).exists() {
                let data = tokio::fs::read(path).await?;
                if data.len() >= 40 {  // 32 bytes hash + 8 bytes sequence
                    let mut state = self.state.write().await;
                    state.prev_hash = data[..32].to_vec();
                    state.sequence = u64::from_le_bytes(
                        data[32..40].try_into().unwrap_or([0u8; 8])
                    );
                    state.last_persist_seq = state.sequence;
                    
                    log::info!(
                        "Loaded hash chain state: seq={}, hash={}",
                        state.sequence,
                        hex::encode(&state.prev_hash[..8])
                    );
                }
            }
        }
        Ok(())
    }

    /// Persist current hash chain state to disk
    async fn persist_state(&self, state: &HashChainState) -> Result<()> {
        if let Some(ref path) = self.persist_path {
            // Only persist every 100 sequences to reduce I/O
            if state.sequence - state.last_persist_seq >= 100 {
                let mut data = Vec::with_capacity(40);
                data.extend_from_slice(&state.prev_hash);
                data.extend_from_slice(&state.sequence.to_le_bytes());
                
                // Atomic write with rename
                let tmp_path = format!("{}.tmp", path);
                tokio::fs::write(&tmp_path, &data).await?;
                tokio::fs::rename(&tmp_path, path).await?;
                
                log::debug!("Persisted hash chain state at seq {}", state.sequence);
            }
        }
        Ok(())
    }

    /// Emit an ACK with hash chaining
    pub async fn emit(
        &self,
        request_id: &str,
        module: Module,
        status: &str,
        reason: &str,
    ) -> Result<Vec<u8>> {
        // Get current timestamp
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Acquire write lock for hash chain
        let mut state = self.state.write().await;
        
        // Compute hash chain
        let hash = self.compute_hash(
            request_id,
            module,
            &self.agent_id,
            ts,
            status,
            reason,
            &state.prev_hash,
        );

        // Create ACK message
        let ack = Ack {
            request_id: request_id.to_string(),
            module: module as i32,
            agent_id: self.agent_id.clone(),
            ts,
            status: status.to_string(),
            reason: reason.to_string(),
            hash: hash.clone(),
            prev_hash: state.prev_hash.clone(),
        };

        // Encode to protobuf
        let mut buf = Vec::with_capacity(256);
        ack.encode(&mut buf)?;

        // Send to Kafka with retry logic
        let record = FutureRecord::to(&self.topic)
            .key(request_id)
            .payload(&buf);

        // Use timeout for send operation
        let timeout = Timeout::After(Duration::from_secs(5));
        
        match self.producer.send(record, timeout).await {
            Ok((partition, offset)) => {
                log::debug!(
                    "ACK sent: request_id={}, partition={}, offset={}, hash={}",
                    request_id,
                    partition,
                    offset,
                    hex::encode(&hash[..8])
                );
            }
            Err((e, _)) => {
                log::error!("Failed to send ACK to Kafka: {}", e);
                return Err(anyhow::anyhow!("Kafka send failed: {}", e));
            }
        }

        // Update state
        state.prev_hash = hash.clone();
        state.sequence += 1;

        // Persist state periodically
        if let Err(e) = self.persist_state(&state).await {
            log::warn!("Failed to persist hash chain state: {}", e);
        }

        Ok(hash)
    }

    /// Emit ACK for successful command application
    pub async fn emit_success(
        &self,
        request_id: &str,
        module: Module,
    ) -> Result<Vec<u8>> {
        self.emit(request_id, module, "applied", "").await
    }

    /// Emit ACK for command rejection
    pub async fn emit_rejection(
        &self,
        request_id: &str,
        module: Module,
        reason: &str,
    ) -> Result<Vec<u8>> {
        self.emit(request_id, module, "rejected", reason).await
    }

    /// Emit ACK for command receipt (before processing)
    pub async fn emit_received(
        &self,
        request_id: &str,
        module: Module,
    ) -> Result<Vec<u8>> {
        self.emit(request_id, module, "received", "").await
    }

    /// Compute BLAKE3 hash for the ACK
    fn compute_hash(
        &self,
        request_id: &str,
        module: Module,
        agent_id: &str,
        ts: u64,
        status: &str,
        reason: &str,
        prev_hash: &[u8],
    ) -> Vec<u8> {
        let mut hasher = Hasher::new();
        
        // Hash all fields in deterministic order
        hasher.update(request_id.as_bytes());
        hasher.update(&(module as i32).to_le_bytes());
        hasher.update(agent_id.as_bytes());
        hasher.update(&ts.to_le_bytes());
        hasher.update(status.as_bytes());
        hasher.update(reason.as_bytes());
        hasher.update(prev_hash);
        
        // Add sequence number for additional ordering guarantee
        let state = futures::executor::block_on(self.state.read());
        hasher.update(&state.sequence.to_le_bytes());
        
        hasher.finalize().as_bytes().to_vec()
    }

    /// Verify hash chain integrity for a sequence of ACKs
    pub fn verify_chain(acks: &[Ack]) -> Result<bool> {
        if acks.is_empty() {
            return Ok(true);
        }

        // Start with genesis or first prev_hash
        let mut expected_prev = if acks[0].prev_hash.is_empty() {
            vec![0u8; 32]
        } else {
            acks[0].prev_hash.clone()
        };

        for (i, ack) in acks.iter().enumerate() {
            // Verify prev_hash matches
            if ack.prev_hash != expected_prev {
                log::error!(
                    "Chain broken at index {}: expected prev_hash {}, got {}",
                    i,
                    hex::encode(&expected_prev[..8]),
                    hex::encode(&ack.prev_hash[..8])
                );
                return Ok(false);
            }

            // Recompute hash and verify
            let mut hasher = Hasher::new();
            hasher.update(ack.request_id.as_bytes());
            hasher.update(&ack.module.to_le_bytes());
            hasher.update(ack.agent_id.as_bytes());
            hasher.update(&ack.ts.to_le_bytes());
            hasher.update(ack.status.as_bytes());
            hasher.update(ack.reason.as_bytes());
            hasher.update(&ack.prev_hash);
            
            let computed_hash = hasher.finalize().as_bytes().to_vec();
            
            if computed_hash != ack.hash {
                log::error!(
                    "Hash mismatch at index {}: computed {}, stored {}",
                    i,
                    hex::encode(&computed_hash[..8]),
                    hex::encode(&ack.hash[..8])
                );
                return Ok(false);
            }

            // This hash becomes next prev_hash
            expected_prev = ack.hash.clone();
        }

        Ok(true)
    }

    /// Get current hash chain head
    pub async fn get_chain_head(&self) -> (Vec<u8>, u64) {
        let state = self.state.read().await;
        (state.prev_hash.clone(), state.sequence)
    }

    /// Reset hash chain (use with caution!)
    pub async fn reset_chain(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.prev_hash = vec![0u8; 32];
        state.sequence = 0;
        state.last_persist_seq = 0;
        
        if let Some(ref path) = self.persist_path {
            if std::path::Path::new(path).exists() {
                tokio::fs::remove_file(path).await?;
            }
        }
        
        log::warn!("Hash chain reset to genesis");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hash_chain_integrity() {
        // Create mock ACKs
        let mut acks = vec![];
        let mut prev_hash = vec![0u8; 32];

        for i in 0..10 {
            let request_id = format!("req_{}", i);
            let module = if i % 2 == 0 { Module::MEV } else { Module::Arbitrage };
            
            let mut hasher = Hasher::new();
            hasher.update(request_id.as_bytes());
            hasher.update(&(module as i32).to_le_bytes());
            hasher.update(b"test-agent");
            hasher.update(&(i as u64).to_le_bytes());
            hasher.update(b"applied");
            hasher.update(b"");
            hasher.update(&prev_hash);
            
            let hash = hasher.finalize().as_bytes().to_vec();
            
            acks.push(Ack {
                request_id,
                module: module as i32,
                agent_id: "test-agent".to_string(),
                ts: i as u64,
                status: "applied".to_string(),
                reason: "".to_string(),
                hash: hash.clone(),
                prev_hash: prev_hash.clone(),
            });
            
            prev_hash = hash;
        }

        // Verify chain integrity
        assert!(AckWriter::verify_chain(&acks).unwrap());

        // Tamper with one ACK
        acks[5].status = "tampered".to_string();
        
        // Chain should now be invalid
        assert!(!AckWriter::verify_chain(&acks).unwrap());
    }
}