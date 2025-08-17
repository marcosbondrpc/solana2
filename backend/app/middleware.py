import time
from typing import Callable, Awaitable
from fastapi import Request
from starlette.responses import Response
from .metrics import API_LATENCY_MS

async def timing_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
	t0 = time.perf_counter()
	resp: Response = await call_next(request)
	dt_ms = (time.perf_counter() - t0) * 1000.0
	resp.headers["X-Request-Latency-ms"] = f"{dt_ms:.2f}"
	route = getattr(getattr(request, "scope", {}), "get", lambda *_: None)("path") if hasattr(request, "scope") else None
	route_label = request.scope.get("route").path if request.scope.get("route") else (route or request.url.path)
	try:
		API_LATENCY_MS.labels(route=route_label, method=request.method, status=str(resp.status_code)).observe(dt_ms)
	except Exception:
		pass
	return resp