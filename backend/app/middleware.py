import time
from typing import Callable, Awaitable
from fastapi import Request
from starlette.responses import Response

async def timing_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
	t0 = time.perf_counter()
	resp: Response = await call_next(request)
	dt_ms = (time.perf_counter() - t0) * 1000.0
	resp.headers["X-Request-Latency-ms"] = f"{dt_ms:.2f}"
	return resp