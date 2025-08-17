import time
import asyncio
from typing import Callable, Dict, Tuple
from fastapi import Request
from starlette.responses import Response
from starlette.middleware.cors import CORSMiddleware
from .metrics import API_LATENCY_MS
from .settings import settings
from .security.auth import extract_bearer, verify_jwt, AuthError

async def timing_middleware(request: Request, call_next: Callable):
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

_rate_state: Dict[str, Tuple[int, int]] = {}

async def rate_limit_middleware(request: Request, call_next: Callable):
	if settings.RATE_LIMIT_PER_MIN <= 0:
		return await call_next(request)
	path = request.url.path
	if path.startswith("/metrics") or path == "/health" or path == "/events":
		return await call_next(request)
	key = request.headers.get("authorization") or request.client.host
	now_min = int(time.time() // 60)
	w, used = _rate_state.get(key, (now_min, 0))
	if w != now_min:
		w, used = now_min, 0
	_rate_state[key] = (w, used)
	if used >= settings.RATE_LIMIT_PER_MIN:
		return Response(status_code=429)
	_rate_state[key] = (w, used + 1)
	return await call_next(request)

async def auth_middleware(request: Request, call_next: Callable):
	if not settings.API_REQUIRE_AUTH:
		return await call_next(request)
	path = request.url.path
	if path == "/health" or path.startswith("/metrics") or path == "/events" or path == "/ws":
		return await call_next(request)
	if path.startswith("/api/"):
		token = extract_bearer(request)
		if not token:
			return Response(status_code=401)
		try:
			_ = verify_jwt(token)
		except AuthError:
			return Response(status_code=401)
	return await call_next(request)

def apply_cors(app):
	if settings.CORS_ORIGINS:
		app.add_middleware(
			CORSMiddleware,
			allow_origins=settings.CORS_ORIGINS,
			allow_credentials=False,
			allow_methods=["GET"],
			allow_headers=["Authorization", "Content-Type"],
			max_age=600,
		)