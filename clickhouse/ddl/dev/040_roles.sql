CREATE ROLE IF NOT EXISTS api_ro;
GRANT SELECT ON solana_rt_dev.* TO api_ro;