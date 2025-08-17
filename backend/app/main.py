import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .settings import settings
from .middleware import timing_middleware
from .routers import health as health_router
from .routers import snapshot as snapshot_router
from .routers import detections as detections_router
from .routers import entities as entities_router
from .routers import metrics as metrics_router
from . import ws as ws_router
from . import sse as sse_router
from .deps import get_ch

@asynccontextmanager
async def lifespan(app: FastAPI):
	app.state.start_time = time.perf_counter()
	ch = await get_ch()
	yield
	await ch.close()

app = FastAPI(lifespan=lifespan)
app.middleware("http")(timing_middleware)
if settings.CORS_ORIGINS:
	app.add_middleware(
		CORSMiddleware,
		allow_origins=settings.CORS_ORIGINS,
		allow_methods=["*"],
		allow_headers=["*"],
		allow_credentials=False,
	)
app.include_router(health_router.router)
app.include_router(snapshot_router.router)
app.include_router(detections_router.router)
app.include_router(entities_router.router)
app.include_router(metrics_router.router)
app.include_router(ws_router.router)
app.include_router(sse_router.router)