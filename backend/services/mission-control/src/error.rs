use thiserror::Error;

#[derive(Error, Debug)]
pub enum MissionControlError {
    #[error("RPC error: {0}")]
    RpcError(String),
    
    #[error("Jito connection error: {0}")]
    JitoError(String),
    
    #[error("Cache error: {0}")]
    CacheError(#[from] redis::RedisError),
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] clickhouse::error::Error),
    
    #[error("Kafka error: {0}")]
    KafkaError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    
    #[error("WebSocket error: {0}")]
    WebSocketError(String),
    
    #[error("Circuit breaker open")]
    CircuitBreakerOpen,
    
    #[error("Timeout exceeded")]
    Timeout,
    
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    
    #[error("Metrics collection error: {0}")]
    MetricsError(String),
}

pub type Result<T> = std::result::Result<T, MissionControlError>;