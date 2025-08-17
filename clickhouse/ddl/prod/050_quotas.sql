CREATE QUOTA IF NOT EXISTS api_ro_quota
FOR INTERVAL 1 minute
MAX queries 5000, read_rows 200000000, read_bytes 4294967296
TO api_ro;