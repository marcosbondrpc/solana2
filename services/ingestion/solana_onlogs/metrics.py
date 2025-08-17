from prometheus_client import Counter, Gauge, Histogram

batches_ok = Counter("ingest_batches_total", "Batches processed", ["status"])
batch_size = Histogram("ingest_batch_size", "Rows per batch", buckets=(10,50,100,200,500,1000))
rows_dropped = Counter("ingest_rows_dropped_total", "Rows dropped due to queue backpressure")
reconnects = Counter("ingest_reconnects_total", "Reconnect attempts")
ch_insert_latency_ms = Histogram("ch_insert_latency_ms", "ClickHouse insert latency (ms)", buckets=(5,10,20,50,100,200,400,800,1600))
ingest_lag_ms = Gauge("ingest_lag_ms", "Ingestion lag in milliseconds")