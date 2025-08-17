import time
from fastapi import APIRouter, Depends, Request
from ..deps import get_ch
from ..ch.client import CH

router = APIRouter()

@router.get("/health")
async def health(request: Request, ch: CH = Depends(get_ch)):
	try:
		ms = await ch.ping_latency_ms()
		uptime = time.perf_counter() - request.app.state.start_time
		return {"status": "ok", "uptime_s": round(uptime, 3), "db_latency_ms": round(ms, 2)}
	except Exception:
		uptime = time.perf_counter() - request.app.state.start_time
		return {"status": "degraded", "uptime_s": round(uptime, 3), "db_latency_ms": None}