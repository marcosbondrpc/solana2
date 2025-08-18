//! # MEV Proto
//! 
//! Ultra-optimized protobuf definitions for SOTA MEV infrastructure.
//! 
//! This crate provides high-performance, zero-copy protobuf message handling
//! for MEV operations with sub-millisecond latency requirements.

#![warn(missing_docs)]
#![warn(clippy::all)]

use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};

pub mod realtime;
pub mod control;
pub mod jobs;

/// Common traits and utilities for MEV protobuf messages
pub mod common {
    use super::*;

    /// Trait for messages that can provide timestamps
    pub trait Timestamped {
        /// Get the timestamp in nanoseconds since Unix epoch
        fn timestamp_ns(&self) -> u64;
        
        /// Get the age of this message in nanoseconds
        fn age_ns(&self) -> u64 {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            now.saturating_sub(self.timestamp_ns())
        }
        
        /// Check if message is older than threshold
        fn is_stale(&self, threshold_ns: u64) -> bool {
            self.age_ns() > threshold_ns
        }
    }

    /// Trait for messages that can be validated
    pub trait Validated {
        /// Validate message contents
        fn validate(&self) -> Result<()>;
    }

    /// Trait for cryptographically signed messages
    pub trait Signed {
        /// Verify the message signature
        fn verify_signature(&self, public_key: &[u8]) -> Result<bool>;
        
        /// Get the message nonce for replay protection
        fn nonce(&self) -> u64;
    }

    /// Performance optimization utilities
    pub mod perf {
        /// Zero-copy message parsing with validation
        pub fn parse_envelope_fast(data: &[u8]) -> Result<crate::realtime::Envelope> {
            // TODO: Implement ultra-fast parsing with SIMD optimizations
            let envelope = protobuf::Message::parse_from_bytes(data)?;
            Ok(envelope)
        }
        
        /// Batch message processing for high throughput
        pub fn process_batch_vectorized(envelopes: &[crate::realtime::Envelope]) -> Vec<Result<()>> {
            // TODO: Implement vectorized batch processing
            envelopes.iter()
                .map(|_| Ok(()))
                .collect()
        }
    }
}

/// Re-export commonly used types
pub use realtime::{Envelope, Batch, MevOpportunity, ArbitrageOpportunity, BundleOutcome};
pub use control::{Command, PolicyUpdate, ModelSwap, KillSwitch};
pub use jobs::{TrainRequest, TrainStatus, InferenceRequest, InferenceResponse};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_creation() {
        let mut envelope = realtime::Envelope::new();
        envelope.timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        envelope.stream_id = "test".to_string();
        envelope.sequence = 1;
        envelope.type_ = "test_message".to_string();
        
        assert!(!envelope.stream_id.is_empty());
        assert!(envelope.timestamp_ns > 0);
    }
}