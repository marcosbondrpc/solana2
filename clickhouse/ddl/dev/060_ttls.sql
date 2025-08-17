ALTER TABLE solana_rt_dev.raw_tx
MODIFY TTL ts + INTERVAL 7 DAY;

ALTER TABLE solana_rt_dev.detections
MODIFY TTL ts + INTERVAL 7 DAY;