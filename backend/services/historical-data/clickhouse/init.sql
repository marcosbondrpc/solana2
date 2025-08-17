-- ClickHouse DDL for Solana Historical Data Spine
-- Optimized for billions of transactions with sub-millisecond query performance

CREATE DATABASE IF NOT EXISTS solana_history;
USE solana_history;

-- ====================================
-- KAFKA STAGING TABLES (JSONEachRow)
-- ====================================

-- Slots staging from Kafka
CREATE TABLE IF NOT EXISTS kafka_slots_staging (
    slot UInt64,
    parent_slot UInt64,
    block_height UInt64,
    block_time DateTime64(3),
    leader String,
    rewards_json String,
    block_hash String,
    parent_hash String,
    transaction_count UInt32,
    entry_count UInt32,
    tick_count UInt32,
    ingestion_timestamp DateTime64(3) DEFAULT now64(3)
) ENGINE = Kafka()
SETTINGS
    kafka_broker_list = 'redpanda:9092',
    kafka_topic_list = 'solana.slots',
    kafka_group_name = 'clickhouse-slots-consumer',
    kafka_format = 'JSONEachRow',
    kafka_num_consumers = 4,
    kafka_max_block_size = 100000,
    kafka_skip_broken_messages = 100,
    kafka_commit_every_batch = 1;

-- Blocks staging from Kafka
CREATE TABLE IF NOT EXISTS kafka_blocks_staging (
    slot UInt64,
    block_height UInt64,
    block_hash String,
    parent_hash String,
    block_time DateTime64(3),
    leader String,
    rewards_json String,
    block_cost UInt64,
    max_supported_transaction_version UInt8,
    transaction_count UInt32,
    executed_transaction_count UInt32,
    entries_json String,
    ingestion_timestamp DateTime64(3) DEFAULT now64(3)
) ENGINE = Kafka()
SETTINGS
    kafka_broker_list = 'redpanda:9092',
    kafka_topic_list = 'solana.blocks',
    kafka_group_name = 'clickhouse-blocks-consumer',
    kafka_format = 'JSONEachRow',
    kafka_num_consumers = 4,
    kafka_max_block_size = 100000,
    kafka_skip_broken_messages = 100,
    kafka_commit_every_batch = 1;

-- Transactions staging from Kafka
CREATE TABLE IF NOT EXISTS kafka_transactions_staging (
    signature String,
    slot UInt64,
    block_time DateTime64(3),
    block_index UInt32,
    transaction_index UInt32,
    is_vote Bool,
    success Bool,
    fee UInt64,
    compute_units_consumed UInt64,
    err String,
    memo String,
    signer String,
    signers Array(String),
    account_keys Array(String),
    pre_balances Array(UInt64),
    post_balances Array(UInt64),
    pre_token_balances_json String,
    post_token_balances_json String,
    instructions_json String,
    inner_instructions_json String,
    log_messages Array(String),
    rewards_json String,
    loaded_addresses_json String,
    return_data_json String,
    ingestion_timestamp DateTime64(3) DEFAULT now64(3)
) ENGINE = Kafka()
SETTINGS
    kafka_broker_list = 'redpanda:9092',
    kafka_topic_list = 'solana.transactions',
    kafka_group_name = 'clickhouse-transactions-consumer',
    kafka_format = 'JSONEachRow',
    kafka_num_consumers = 8,
    kafka_max_block_size = 50000,
    kafka_skip_broken_messages = 100,
    kafka_commit_every_batch = 1;

-- Account updates staging from Kafka
CREATE TABLE IF NOT EXISTS kafka_accounts_staging (
    pubkey String,
    slot UInt64,
    write_version UInt64,
    lamports UInt64,
    owner String,
    executable Bool,
    rent_epoch UInt64,
    data_len UInt32,
    data_hash String,
    update_type Enum8('create' = 1, 'update' = 2, 'delete' = 3),
    ingestion_timestamp DateTime64(3) DEFAULT now64(3)
) ENGINE = Kafka()
SETTINGS
    kafka_broker_list = 'redpanda:9092',
    kafka_topic_list = 'solana.accounts',
    kafka_group_name = 'clickhouse-accounts-consumer',
    kafka_format = 'JSONEachRow',
    kafka_num_consumers = 6,
    kafka_max_block_size = 100000,
    kafka_skip_broken_messages = 100,
    kafka_commit_every_batch = 1;

-- ====================================
-- PERSISTENT TABLES (ReplacingMergeTree)
-- ====================================

-- Slots table with deduplication
CREATE TABLE IF NOT EXISTS slots (
    slot UInt64,
    parent_slot UInt64,
    block_height UInt64,
    block_time DateTime64(3),
    leader String,
    rewards_json String,
    block_hash String,
    parent_hash String,
    transaction_count UInt32,
    entry_count UInt32,
    tick_count UInt32,
    ingestion_timestamp DateTime64(3),
    _version UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))
) ENGINE = ReplacingMergeTree(_version)
PARTITION BY toYYYYMM(block_time)
ORDER BY (slot, block_time)
PRIMARY KEY slot
SETTINGS 
    index_granularity = 8192,
    merge_with_ttl_timeout = 3600,
    min_bytes_for_wide_part = 10485760;

-- Blocks table with deduplication
CREATE TABLE IF NOT EXISTS blocks (
    slot UInt64,
    block_height UInt64,
    block_hash String,
    parent_hash String,
    block_time DateTime64(3),
    leader String,
    rewards_json String,
    block_cost UInt64,
    max_supported_transaction_version UInt8,
    transaction_count UInt32,
    executed_transaction_count UInt32,
    entries_json String,
    ingestion_timestamp DateTime64(3),
    _version UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))
) ENGINE = ReplacingMergeTree(_version)
PARTITION BY toYYYYMM(block_time)
ORDER BY (slot, block_hash)
PRIMARY KEY slot
SETTINGS 
    index_granularity = 8192,
    merge_with_ttl_timeout = 3600,
    min_bytes_for_wide_part = 10485760;

-- Transactions table with deduplication
CREATE TABLE IF NOT EXISTS transactions (
    signature String,
    slot UInt64,
    block_time DateTime64(3),
    block_index UInt32,
    transaction_index UInt32,
    is_vote Bool,
    success Bool,
    fee UInt64,
    compute_units_consumed UInt64,
    err String,
    memo String,
    signer String,
    signers Array(String),
    account_keys Array(String),
    pre_balances Array(UInt64),
    post_balances Array(UInt64),
    pre_token_balances_json String,
    post_token_balances_json String,
    instructions_json String,
    inner_instructions_json String,
    log_messages Array(String),
    rewards_json String,
    loaded_addresses_json String,
    return_data_json String,
    ingestion_timestamp DateTime64(3),
    _version UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))
) ENGINE = ReplacingMergeTree(_version)
PARTITION BY toYYYYMMDD(block_time)
ORDER BY (signature, slot, block_time)
PRIMARY KEY signature
SETTINGS 
    index_granularity = 8192,
    merge_with_ttl_timeout = 3600,
    min_bytes_for_wide_part = 10485760;

-- Create bloom filter index for signature lookups
ALTER TABLE transactions ADD INDEX idx_signature_bloom signature TYPE bloom_filter(0.01) GRANULARITY 1;

-- Account updates table with deduplication
CREATE TABLE IF NOT EXISTS account_updates (
    pubkey String,
    slot UInt64,
    write_version UInt64,
    lamports UInt64,
    owner String,
    executable Bool,
    rent_epoch UInt64,
    data_len UInt32,
    data_hash String,
    update_type Enum8('create' = 1, 'update' = 2, 'delete' = 3),
    ingestion_timestamp DateTime64(3),
    _version UInt64 DEFAULT toUnixTimestamp64Milli(now64(3))
) ENGINE = ReplacingMergeTree(_version)
PARTITION BY intDiv(slot, 1000000)
ORDER BY (pubkey, slot, write_version)
PRIMARY KEY (pubkey, slot)
SETTINGS 
    index_granularity = 8192,
    merge_with_ttl_timeout = 3600,
    min_bytes_for_wide_part = 10485760;

-- ====================================
-- MATERIALIZED VIEWS (Kafka → MergeTree)
-- ====================================

-- MV for slots
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_slots_ingestion
TO slots
AS SELECT
    slot,
    parent_slot,
    block_height,
    block_time,
    leader,
    rewards_json,
    block_hash,
    parent_hash,
    transaction_count,
    entry_count,
    tick_count,
    ingestion_timestamp
FROM kafka_slots_staging
WHERE slot > 0;

-- MV for blocks
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_blocks_ingestion
TO blocks
AS SELECT
    slot,
    block_height,
    block_hash,
    parent_hash,
    block_time,
    leader,
    rewards_json,
    block_cost,
    max_supported_transaction_version,
    transaction_count,
    executed_transaction_count,
    entries_json,
    ingestion_timestamp
FROM kafka_blocks_staging
WHERE slot > 0;

-- MV for transactions
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_transactions_ingestion
TO transactions
AS SELECT
    signature,
    slot,
    block_time,
    block_index,
    transaction_index,
    is_vote,
    success,
    fee,
    compute_units_consumed,
    err,
    memo,
    signer,
    signers,
    account_keys,
    pre_balances,
    post_balances,
    pre_token_balances_json,
    post_token_balances_json,
    instructions_json,
    inner_instructions_json,
    log_messages,
    rewards_json,
    loaded_addresses_json,
    return_data_json,
    ingestion_timestamp
FROM kafka_transactions_staging
WHERE length(signature) > 0;

-- MV for account updates
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_accounts_ingestion
TO account_updates
AS SELECT
    pubkey,
    slot,
    write_version,
    lamports,
    owner,
    executable,
    rent_epoch,
    data_len,
    data_hash,
    update_type,
    ingestion_timestamp
FROM kafka_accounts_staging
WHERE length(pubkey) > 0;

-- ====================================
-- MONITORING & ANALYTICS TABLES
-- ====================================

-- Ingestion metrics table
CREATE TABLE IF NOT EXISTS ingestion_metrics (
    timestamp DateTime64(3),
    table_name String,
    rows_ingested UInt64,
    bytes_ingested UInt64,
    lag_ms UInt32,
    errors UInt32
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (table_name, timestamp)
TTL timestamp + INTERVAL 30 DAY;

-- Consumer progress tracking
CREATE TABLE IF NOT EXISTS consumer_progress (
    consumer_group String,
    topic String,
    partition UInt32,
    current_offset UInt64,
    committed_offset UInt64,
    lag UInt64,
    update_time DateTime64(3) DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(update_time)
ORDER BY (consumer_group, topic, partition);

-- ====================================
-- ANALYTICAL VIEWS
-- ====================================

-- Daily transaction statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_stats
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(day)
ORDER BY (day, is_vote)
AS SELECT
    toDate(block_time) as day,
    is_vote,
    count() as tx_count,
    sum(fee) as total_fees,
    sum(compute_units_consumed) as total_compute_units,
    avg(fee) as avg_fee,
    max(fee) as max_fee,
    min(fee) as min_fee
FROM transactions
GROUP BY day, is_vote;

-- Account activity summary
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_account_activity
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(toDate(slot / 2))
ORDER BY pubkey
AS SELECT
    pubkey,
    max(slot) as last_active_slot,
    count() as update_count,
    max(lamports) as max_lamports,
    argMax(owner, slot) as current_owner,
    argMax(data_len, slot) as current_data_len
FROM account_updates
GROUP BY pubkey;

-- ====================================
-- OPTIMIZATION SETTINGS
-- ====================================

-- Enable async inserts for better throughput
SYSTEM RELOAD CONFIG;

-- Set merge settings for optimal performance
ALTER TABLE slots MODIFY SETTING merge_max_block_size = 1048576;
ALTER TABLE blocks MODIFY SETTING merge_max_block_size = 1048576;
ALTER TABLE transactions MODIFY SETTING merge_max_block_size = 1048576;
ALTER TABLE account_updates MODIFY SETTING merge_max_block_size = 1048576;

-- Create indexes for common queries
ALTER TABLE transactions ADD INDEX idx_slot slot TYPE minmax GRANULARITY 1;
ALTER TABLE transactions ADD INDEX idx_block_time block_time TYPE minmax GRANULARITY 1;
ALTER TABLE transactions ADD INDEX idx_signer signer TYPE bloom_filter(0.01) GRANULARITY 1;
ALTER TABLE account_updates ADD INDEX idx_owner owner TYPE bloom_filter(0.01) GRANULARITY 1;