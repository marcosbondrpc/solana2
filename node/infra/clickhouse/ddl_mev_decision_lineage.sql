CREATE TABLE IF NOT EXISTS mev_decision_lineage
(
    ts                    DateTime64(6, 'UTC'),
    request_id            UUID,
    agent_id              LowCardinality(String),
    model_version         LowCardinality(String),
    slot                  UInt64,
    leader                FixedString(44),
    route                 LowCardinality(String),
    tip_lamports          UInt64,
    lamports_per_cu       Float64,
    expected_cu           Float64,
    ev_estimate           Float64,
    p_land_prior          Float64,
    decision_ms           Float64,
    submit_ms             Float64,
    land_ms               Float64,
    tx_signature          FixedString(88),
    bundle_hash           FixedString(64),
    jito_code             LowCardinality(String),
    dedupe_key_b3         FixedString(64),
    features              JSON,
    control_actor         LowCardinality(String),
    node_id               LowCardinality(String)
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (ts, request_id)
TTL ts + INTERVAL 30 DAY
SETTINGS index_granularity = 8192, allow_nullable_key = 1;

CREATE MATERIALIZED VIEW IF NOT EXISTS mev_decision_daily
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (toStartOfDay(ts), route)
AS
SELECT toStartOfDay(ts) AS day, route, avg(decision_ms) AS p50_decision_ms, count() AS n,
       sumIf(1, jito_code = 'Ok') AS ok_cnt
FROM mev_decision_lineage GROUP BY day, route;