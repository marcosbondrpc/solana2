ALTER TABLE solana_rt.raw_tx
MODIFY TTL ts + INTERVAL 14 DAY;

ALTER TABLE solana_rt.detections
MODIFY TTL ts + INTERVAL 30 DAY;