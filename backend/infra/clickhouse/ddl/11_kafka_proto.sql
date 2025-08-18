-- ClickHouse Kafka Engine with Protobuf Support
-- Ultra-optimized for 200k+ rows/second ingestion

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS legendary_mev;
USE legendary_mev;

-- Drop existing Kafka tables if they exist (for clean setup)
DROP TABLE IF EXISTS kafka_realtime_proto;
DROP TABLE IF EXISTS kafka_control_proto;
DROP TABLE IF EXISTS kafka_bandit_proto;
DROP TABLE IF EXISTS kafka_mev_opportunities_proto;
DROP TABLE IF EXISTS kafka_arb_opportunities_proto;

-- Realtime events Kafka consumer with Protobuf
CREATE TABLE kafka_realtime_proto
(
    -- DecisionDNA fields
    decision_id String,
    model_version String,
    features_hash String,
    parent_hash String,
    decision_timestamp DateTime64(9),
    
    -- Event type discriminator
    event_type Enum8('mev' = 1, 'arb' = 2, 'bandit' = 3),
    
    -- MevEvent fields
    mev_tx_hash String,
    mev_victim_tx String,
    mev_sandwich_type String,
    mev_expected_profit Decimal64(8),
    mev_gas_price Decimal64(8),
    mev_priority_fee Decimal64(8),
    mev_block_number UInt64,
    mev_confidence Float32,
    
    -- ArbEvent fields
    arb_opportunity_id String,
    arb_source_dex String,
    arb_target_dex String,
    arb_token_in String,
    arb_token_out String,
    arb_amount_in Decimal64(8),
    arb_expected_out Decimal64(8),
    arb_profit_usd Decimal64(8),
    arb_route Array(String),
    
    -- BanditEvent fields
    bandit_arm_id String,
    bandit_reward Float64,
    bandit_confidence Float64,
    bandit_exploration_bonus Float64,
    bandit_context Map(String, String),
    
    -- Metadata
    received_at DateTime64(9) DEFAULT now64(9)
)
ENGINE = Kafka
SETTINGS 
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'realtime-proto',
    kafka_group_name = 'clickhouse-realtime-consumer',
    kafka_format = 'Protobuf',
    kafka_schema = 'protocol/realtime.proto:RealtimeEvent',
    kafka_num_consumers = 3,
    kafka_max_block_size = 65536,
    kafka_skip_broken_messages = 100,
    kafka_commit_every_batch = 1,
    kafka_handle_error_mode = 'stream';

-- Control commands Kafka consumer
CREATE TABLE kafka_control_proto
(
    command_id String,
    command_type String,
    payload String, -- JSON encoded command data
    signature String,
    signer_pubkey String,
    nonce UInt64,
    timestamp DateTime64(9),
    received_at DateTime64(9) DEFAULT now64(9)
)
ENGINE = Kafka
SETTINGS 
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'control-commands-proto',
    kafka_group_name = 'clickhouse-control-consumer',
    kafka_format = 'Protobuf',
    kafka_schema = 'protocol/control.proto:Command',
    kafka_num_consumers = 1,
    kafka_max_block_size = 1000,
    kafka_skip_broken_messages = 10;

-- Bandit events specialized consumer
CREATE TABLE kafka_bandit_proto
(
    arm_id String,
    decision_id String,
    reward Float64,
    cost Float64,
    confidence Float64,
    exploration_bonus Float64,
    total_budget Float64,
    remaining_budget Float64,
    context_features Map(String, Float32),
    metadata Map(String, String),
    timestamp DateTime64(9),
    received_at DateTime64(9) DEFAULT now64(9)
)
ENGINE = Kafka
SETTINGS 
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'bandit-events-proto',
    kafka_group_name = 'clickhouse-bandit-consumer',
    kafka_format = 'Protobuf',
    kafka_schema = 'protocol/realtime.proto:BanditEvent',
    kafka_num_consumers = 2,
    kafka_max_block_size = 32768;

-- MEV opportunities high-throughput consumer
CREATE TABLE kafka_mev_opportunities_proto
(
    opportunity_id String,
    block_number UInt64,
    transaction_hash String,
    victim_tx String,
    opportunity_type Enum8('sandwich' = 1, 'frontrun' = 2, 'backrun' = 3, 'liquidation' = 4),
    token_in String,
    token_out String,
    amount_in Decimal128(18),
    expected_profit Decimal128(18),
    gas_estimate UInt64,
    priority_fee Decimal64(8),
    confidence_score Float32,
    features Array(Float32), -- Feature vector for ML
    decision_dna_hash String,
    created_at DateTime64(9),
    received_at DateTime64(9) DEFAULT now64(9)
)
ENGINE = Kafka
SETTINGS 
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'mev-opportunities',
    kafka_group_name = 'clickhouse-mev-consumer',
    kafka_format = 'Protobuf',
    kafka_schema = 'protocol/realtime.proto:MevOpportunity',
    kafka_num_consumers = 4,
    kafka_max_block_size = 100000,
    kafka_poll_timeout_ms = 100,
    kafka_flush_interval_ms = 100;

-- Arbitrage opportunities consumer
CREATE TABLE kafka_arb_opportunities_proto
(
    opportunity_id String,
    source_pool String,
    target_pool String,
    token_path Array(String),
    dex_path Array(String),
    input_amount Decimal128(18),
    output_amount Decimal128(18),
    profit_amount Decimal128(18),
    profit_usd Decimal64(8),
    gas_cost Decimal64(8),
    net_profit Decimal64(8),
    price_impact Float32,
    slippage_tolerance Float32,
    confidence_score Float32,
    features Array(Float32),
    decision_dna_hash String,
    created_at DateTime64(9),
    received_at DateTime64(9) DEFAULT now64(9)
)
ENGINE = Kafka
SETTINGS 
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'arbitrage-paths',
    kafka_group_name = 'clickhouse-arb-consumer',
    kafka_format = 'Protobuf',
    kafka_schema = 'protocol/realtime.proto:ArbOpportunity',
    kafka_num_consumers = 4,
    kafka_max_block_size = 100000,
    kafka_poll_timeout_ms = 100;

-- Control ACKs consumer for audit trail
CREATE TABLE kafka_control_acks_proto
(
    ack_id String,
    command_id String,
    status Enum8('success' = 1, 'failed' = 2, 'timeout' = 3),
    message String,
    hash_chain String, -- Previous ACK hash for chain verification
    executor String,
    execution_time_ms UInt32,
    timestamp DateTime64(9),
    received_at DateTime64(9) DEFAULT now64(9)
)
ENGINE = Kafka
SETTINGS 
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'control-acks',
    kafka_group_name = 'clickhouse-acks-consumer',
    kafka_format = 'Protobuf',
    kafka_schema = 'protocol/control.proto:Ack',
    kafka_num_consumers = 1;

-- Performance metrics ingestion (high frequency)
CREATE TABLE kafka_performance_metrics_proto
(
    metric_name String,
    metric_value Float64,
    labels Map(String, String),
    timestamp DateTime64(9)
)
ENGINE = Kafka
SETTINGS 
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'performance-metrics',
    kafka_group_name = 'clickhouse-metrics-consumer',
    kafka_format = 'JSONEachRow', -- Metrics stay in JSON for flexibility
    kafka_num_consumers = 2,
    kafka_max_block_size = 50000,
    kafka_poll_timeout_ms = 50,
    kafka_flush_interval_ms = 50;

-- Create views to monitor Kafka consumer lag
CREATE OR REPLACE VIEW kafka_consumer_status AS
SELECT 
    'kafka_realtime_proto' as consumer_table,
    (SELECT count() FROM kafka_realtime_proto) as messages_consumed,
    (SELECT max(received_at) FROM kafka_realtime_proto) as last_message_time
UNION ALL
SELECT 
    'kafka_mev_opportunities_proto' as consumer_table,
    (SELECT count() FROM kafka_mev_opportunities_proto) as messages_consumed,
    (SELECT max(received_at) FROM kafka_mev_opportunities_proto) as last_message_time
UNION ALL
SELECT 
    'kafka_arb_opportunities_proto' as consumer_table,
    (SELECT count() FROM kafka_arb_opportunities_proto) as messages_consumed,
    (SELECT max(received_at) FROM kafka_arb_opportunities_proto) as last_message_time;

-- Optimize table settings for ultra-high ingestion
ALTER TABLE kafka_realtime_proto MODIFY SETTING max_insert_block_size = 1048576;
ALTER TABLE kafka_mev_opportunities_proto MODIFY SETTING max_insert_block_size = 1048576;
ALTER TABLE kafka_arb_opportunities_proto MODIFY SETTING max_insert_block_size = 1048576;