import anyio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .settings import settings
from .middleware import timing_middleware, rate_limit_middleware, auth_middleware, apply_cors
from .routers import health as health_router
from .routers import snapshot as snapshot_router
from .routers import detections as detections_router
from .routers import entities as entities_router
from .routers import metrics as metrics_router
from . import ws as ws_router
from . import sse as sse_router
from .metrics import router as prom_router
from .deps import get_ch

@asynccontextmanager
async def lifespan(app: FastAPI):
	ch = await get_ch()
	yield
	await ch.close()

app = FastAPI(lifespan=lifespan)

apply_cors(app)
app.middleware("http")(auth_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(timing_middleware)

app.include_router(health_router.router)
app.include_router(snapshot_router.router)
app.include_router(detections_router.router)
app.include_router(entities_router.router)
app.include_router(metrics_router.router)
app.include_router(ws_router.router)
app.include_router(sse_router.router)
app.include_router(prom_router)