from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import anyio
import json
import time
from typing import AsyncIterator

router = APIRouter()

async def _sse_stream() -> AsyncIterator[bytes]:
    last_keepalive = time.monotonic()
    eid = 0
    while True:
        now_ms = int(time.time() * 1000)
        eid += 1
        payload = {"t": "stats_update", "ts": now_ms}
        chunk = f"id: {eid}\nretry: 1500\nevent: stats_update\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
        yield chunk.encode("utf-8")
        if time.monotonic() - last_keepalive >= 20:
            last_keepalive = time.monotonic()
            yield b": keepalive\n\n"
        await anyio.sleep(5)

@router.get("/events")
async def events():
    return StreamingResponse(_sse_stream(), media_type="text/event-stream")