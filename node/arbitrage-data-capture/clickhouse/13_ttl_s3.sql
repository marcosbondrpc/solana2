-- TTL Configuration with S3 Cold Storage
-- Moves old data to S3 after 14 days for cost optimization

USE legendary_mev;

-- S3 Storage Policy Configuration
-- Note: Requires ClickHouse S3 storage configuration in config.xml

-- Create storage policies for hot/cold data tiering
-- This assumes S3 is configured in ClickHouse config as 'cold_s3'

-- Alter MEV opportunities table for S3 tiering
ALTER TABLE mev_opportunities MODIFY TTL 
    date + INTERVAL 3 DAY TO VOLUME 'hot',  -- Keep recent 3 days on fast SSD
    date + INTERVAL 14 DAY TO VOLUME 'cold', -- Move to slower disk after 14 days
    date + INTERVAL 90 DAY DELETE;          -- Delete after 90 days

-- Alter Arbitrage opportunities table for S3 tiering  
ALTER TABLE arb_opportunities MODIFY TTL
    date + INTERVAL 3 DAY TO VOLUME 'hot',
    date + INTERVAL 14 DAY TO VOLUME 'cold',
    date + INTERVAL 90 DAY DELETE;

-- Alter Bandit events table for S3 tiering
ALTER TABLE bandit_events MODIFY TTL
    date + INTERVAL 7 DAY TO VOLUME 'hot',
    date + INTERVAL 30 DAY TO VOLUME 'cold',
    date + INTERVAL 180 DAY DELETE;

-- Control commands need longer retention for audit
ALTER TABLE control_commands MODIFY TTL
    date + INTERVAL 30 DAY TO VOLUME 'hot',
    date + INTERVAL 90 DAY TO VOLUME 'cold',
    date + INTERVAL 365 DAY DELETE; -- 1 year retention

-- Decision lineage needs permanent storage for compliance
ALTER TABLE decision_lineage MODIFY TTL
    date + INTERVAL 30 DAY TO VOLUME 'hot',
    date + INTERVAL 365 DAY TO VOLUME 'cold';
    -- No deletion for decision lineage

-- Performance metrics have shortest retention
ALTER TABLE performance_metrics MODIFY TTL
    date + INTERVAL 1 DAY TO VOLUME 'hot',
    date + INTERVAL 7 DAY TO VOLUME 'cold',
    date + INTERVAL 30 DAY DELETE;

-- Create aggregated tables for long-term statistics
DROP TABLE IF EXISTS daily_mev_summary;
CREATE TABLE daily_mev_summary
(
    date Date,
    opportunity_type Enum8('sandwich' = 1, 'frontrun' = 2, 'backrun' = 3, 'liquidation' = 4),
    
    -- Counts
    total_opportunities UInt64,
    executed_opportunities UInt64,
    successful_opportunities UInt64,
    
    -- Financial metrics
    total_expected_profit Decimal128(18),
    total_actual_profit Decimal128(18),
    total_gas_cost Decimal128(18),
    net_profit Decimal128(18),
    
    -- Performance metrics
    avg_confidence Float32,
    avg_execution_time_ms Float32,
    success_rate Float32,
    
    -- Statistical metrics
    p50_profit Decimal64(8),
    p95_profit Decimal64(8),
    p99_profit Decimal64(8),
    
    -- Model performance
    model_versions Array(String),
    avg_model_accuracy Float32
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, opportunity_type)
PRIMARY KEY date
TTL date + INTERVAL 2 YEAR TO VOLUME 'cold'; -- Keep summaries for 2 years

-- Materialized view to populate daily summary
CREATE MATERIALIZED VIEW daily_mev_summary_mv TO daily_mev_summary AS
SELECT 
    toDate(created_at) as date,
    opportunity_type,
    count() as total_opportunities,
    countIf(executed = true) as executed_opportunities,
    countIf(actual_profit > 0) as successful_opportunities,
    sum(expected_profit) as total_expected_profit,
    sum(actual_profit) as total_actual_profit,
    sum(gas_estimate * priority_fee) as total_gas_cost,
    sum(actual_profit) - sum(gas_estimate * priority_fee) as net_profit,
    avg(confidence_score) as avg_confidence,
    avg(execution_time_ms) as avg_execution_time_ms,
    countIf(actual_profit > 0) / countIf(executed = true) as success_rate,
    quantile(0.5)(expected_profit) as p50_profit,
    quantile(0.95)(expected_profit) as p95_profit,
    quantile(0.99)(expected_profit) as p99_profit,
    groupUniqArray(model_version) as model_versions,
    0.0 as avg_model_accuracy -- To be updated by separate process
FROM mev_opportunities
WHERE created_at >= today()
GROUP BY date, opportunity_type;

-- Daily arbitrage summary
DROP TABLE IF EXISTS daily_arb_summary;
CREATE TABLE daily_arb_summary
(
    date Date,
    
    -- Counts
    total_opportunities UInt64,
    executed_opportunities UInt64,
    successful_opportunities UInt64,
    
    -- Path metrics
    avg_hop_count Float32,
    unique_pools UInt64,
    unique_tokens UInt64,
    
    -- Financial metrics
    total_input Decimal128(18),
    total_output Decimal128(18),
    total_profit Decimal128(18),
    total_gas_cost Decimal128(18),
    net_profit Decimal128(18),
    
    -- Performance metrics
    avg_confidence Float32,
    avg_execution_latency_ms Float32,
    success_rate Float32,
    
    -- Route selection metrics
    bandit_selections UInt64,
    static_selections UInt64,
    ml_selections UInt64,
    
    -- Statistical metrics
    p50_profit Decimal64(8),
    p95_profit Decimal64(8),
    p99_profit Decimal64(8)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY date
PRIMARY KEY date
TTL date + INTERVAL 2 YEAR TO VOLUME 'cold';

-- Create system tables for monitoring storage usage

CREATE OR REPLACE VIEW storage_usage AS
SELECT 
    table,
    partition,
    disk_name,
    formatReadableSize(sum(bytes_on_disk)) as size_on_disk,
    sum(rows) as total_rows,
    max(modification_time) as last_modified
FROM system.parts
WHERE database = 'legendary_mev' AND active
GROUP BY table, partition, disk_name
ORDER BY sum(bytes_on_disk) DESC;

CREATE OR REPLACE VIEW ttl_status AS
SELECT 
    database,
    table,
    partition,
    min_time,
    max_time,
    delete_ttl_info.expression as ttl_expression,
    formatReadableSize(bytes_on_disk) as size
FROM system.parts
WHERE database = 'legendary_mev' AND active
ORDER BY table, partition;

-- Function to manually trigger TTL cleanup
-- Run this periodically or on-demand
-- clickhouse-client --query "OPTIMIZE TABLE legendary_mev.mev_opportunities FINAL"
-- clickhouse-client --query "OPTIMIZE TABLE legendary_mev.arb_opportunities FINAL"

-- Create settings for automatic TTL merge optimization
ALTER TABLE mev_opportunities MODIFY SETTING merge_with_ttl_timeout = 3600; -- 1 hour
ALTER TABLE arb_opportunities MODIFY SETTING merge_with_ttl_timeout = 3600;
ALTER TABLE bandit_events MODIFY SETTING merge_with_ttl_timeout = 7200; -- 2 hours
ALTER TABLE performance_metrics MODIFY SETTING merge_with_ttl_timeout = 1800; -- 30 minutes