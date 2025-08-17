CREATE TABLE IF NOT EXISTS mev_counterfactuals
(
    ts                    DateTime64(6, 'UTC'),
    request_id            UUID,
    slot                  UInt64,
    leader                FixedString(44),
    chosen_route          LowCardinality(String),
    shadow_route          LowCardinality(String),
    predicted_land_prob   Float64,
    predicted_ev          Float64,
    realized_land         UInt8,
    realized_ev           Float64
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (ts, request_id, shadow_route)
TTL ts + INTERVAL 30 DAY;