CREATE QUOTA IF NOT EXISTS api_ro_quota
FOR INTERVAL 1 minute
MAX queries 1000, read_rows 50000000, read_bytes 1073741824
TO api_ro;