import asyncio
import json
import random
from typing import Any, Dict

import websockets

from .config import settings
from .decoder import decode_log_notification
from .batching import Batcher
from .writer import ClickHouseWriter
from .metrics import counters

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

async def producer(queue: asyncio.Queue) -> None:
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
                            queue.put_nowait(row)
                        except asyncio.QueueFull:
                            try:
                                _ = queue.get_nowait()
                            except Exception:
                                pass
                            counters.rows_dropped += 1
                            try:
                                queue.put_nowait(row)
                            except Exception:
                                pass
        except Exception:
            counters.reconnects += 1
            jitter = 1.0 + (random.random() * 0.2 - 0.1)
            await asyncio.sleep((backoff_ms / 1000.0) * jitter)
            backoff_ms = min(int(backoff_ms * 2), settings.RECONNECT_MAX_MS)

async def consumer(queue: asyncio.Queue) -> None:
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
                    n = await writer.write_rows(out)
                    counters.batches_ok += 1
                    counters.rows_ok += n
            except asyncio.TimeoutError:
                flushed, out = await batcher.maybe_flush_on_time()
                if flushed:
                    n = await writer.write_rows(out)
                    counters.batches_ok += 1
                    counters.rows_ok += n
            except Exception:
                counters.batches_err += 1
    finally:
        flushed, out = batcher.flush_now()
        if flushed:
            try:
                n = await writer.write_rows(out)
                counters.batches_ok += 1
                counters.rows_ok += n
            except Exception:
                counters.batches_err += 1
        await ch.close()

async def main() -> None:
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=settings.QUEUE_MAX)
    prod = asyncio.create_task(producer(queue))
    cons = asyncio.create_task(consumer(queue))
    await asyncio.gather(prod, cons)

if __name__ == "__main__":
    asyncio.run(main())