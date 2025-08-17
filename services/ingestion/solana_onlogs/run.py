import asyncio
import json
import random
import time
import os
from typing import Any, Dict

import websockets
from prometheus_client import start_http_server

from .config import settings
from .decoder import decode_log_notification
from .batching import Batcher
from .writer import ClickHouseWriter
from . import metrics as m

from backend.app.ch.client import CH

SUBSCRIBE_REQ = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "logsSubscribe",
    "params": [
        "all",
        {"commitment": "confirmed"}
    ]
}

async def producer(queue: asyncio.Queue[Dict[str, Any]]) -> None:
    backoff_ms = settings.RECONNECT_MIN_MS
    while True:
        try:
            async with websockets.connect(str(settings.SOLANA_WS), max_queue=None) as ws:
                await ws.send(json.dumps(SUBSCRIBE_REQ, separators=(",", ":")))
                try:
                    _ = await asyncio.wait_for(ws.recv(), timeout=5)
                except Exception:
                    pass
                backoff_ms = settings.RECONNECT_MIN_MS
                while True:
                    msg = await ws.recv()
                    obj = json.loads(msg)
                    if obj.get("method") == "logsNotification":
                        row = decode_log_notification(obj)
                        try:
                            m.ingest_lag_ms.set(0)
                        except Exception:
                            pass
                        try:
                            queue.put_nowait(row)
                        except asyncio.QueueFull:
                            try:
                                _ = queue.get_nowait()
                            except Exception:
                                pass
                            m.rows_dropped.inc()
                            try:
                                queue.put_nowait(row)
                            except Exception:
                                pass
        except Exception:
            m.reconnects.inc()
            jitter = 1.0 + (random.random() * 0.2 - 0.1)
            await asyncio.sleep((backoff_ms / 1000.0) * jitter)
            backoff_ms = min(int(backoff_ms * 2), settings.RECONNECT_MAX_MS)

async def consumer(queue: asyncio.Queue[Dict[str, Any]]) -> None:
    ch = CH(url=settings.CH_URL, db=settings.CH_DB, timeout_s=settings.CH_TIMEOUT_S, user=settings.CH_USER, password=settings.CH_PASS)
    table = "solana_rt_dev.raw_tx" if settings.CH_DB.endswith("_dev") else "solana_rt.raw_tx"
    writer = ClickHouseWriter(ch, table=table)
    batcher = Batcher(settings.BATCH_MAX_ROWS, settings.BATCH_MAX_MS)
    try:
        while True:
            try:
                row = await asyncio.wait_for(queue.get(), timeout=settings.BATCH_MAX_MS / 1000.0)
                flushed, out = batcher.add(row)
                if flushed:
                    t0 = time.perf_counter()
                    n = await writer.write_rows(out)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    m.ch_insert_latency_ms.observe(dt_ms)
                    m.batches_ok.labels(status="ok").inc()
                    m.batch_size.observe(n)
            except asyncio.TimeoutError:
                flushed, out = await batcher.maybe_flush_on_time()
                if flushed:
                    t0 = time.perf_counter()
                    n = await writer.write_rows(out)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    m.ch_insert_latency_ms.observe(dt_ms)
                    m.batches_ok.labels(status="ok").inc()
                    m.batch_size.observe(n)
            except Exception:
                m.batches_ok.labels(status="err").inc()
    finally:
        flushed, out = batcher.flush_now()
        if flushed:
            try:
                t0 = time.perf_counter()
                n = await writer.write_rows(out)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                m.ch_insert_latency_ms.observe(dt_ms)
                m.batches_ok.labels(status="ok").inc()
                m.batch_size.observe(n)
            except Exception:
                m.batches_ok.labels(status="err").inc()
        await ch.close()

async def main() -> None:
    start_http_server(int(os.getenv("INGEST_METRICS_PORT", "9108")))
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=settings.QUEUE_MAX)
    prod = asyncio.create_task(producer(queue))
    cons = asyncio.create_task(consumer(queue))
    await asyncio.gather(prod, cons)

if __name__ == "__main__":
    asyncio.run(main())