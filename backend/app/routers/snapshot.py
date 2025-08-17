from fastapi import APIRouter, Depends, Query
from typing import List
from ..deps import get_ch
from ..ch.client import CH
from ..ch.schemas import Detection, SnapshotResponse

router = APIRouter(prefix="/api")

@router.get("/snapshot", response_model=SnapshotResponse)
async def snapshot(limit: int = Query(200, ge=1, le=500), ch: CH = Depends(get_ch)):
	sql = """
	SELECT seq, ts, slot, kind, sig, address, score
	FROM detections
	ORDER BY seq DESC
	LIMIT {limit:UInt32}
	"""
	rows = await ch.select(sql, {"limit": limit})
	detections: List[Detection] = [Detection(**r) for r in rows]
	as_of = detections[0].seq if detections else 0
	return {"as_of_seq": as_of, "detections": detections}