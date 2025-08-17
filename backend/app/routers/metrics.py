from fastapi import APIRouter
from ..ch.schemas import MetricsJSON

router = APIRouter()

@router.get("/api/metrics", response_model=MetricsJSON)
async def metrics():
	return MetricsJSON()