CREATE ROLE IF NOT EXISTS api_ro;
GRANT SELECT ON solana_rt.* TO api_ro;