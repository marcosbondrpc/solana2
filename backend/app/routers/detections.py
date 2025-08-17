from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from ..deps import get_ch
from ..ch.client import CH
from ..ch.schemas import Detection
from datetime import datetime, timedelta, timezone

router = APIRouter(prefix="/api")

@router.get("/detections", response_model=List[Detection])
async def list_detections(
	since: Optional[str] = Query(None),
	limit: int = Query(200, ge=1, le=1000),
	addr: Optional[str] = Query(None),
	ch: CH = Depends(get_ch),
):
	if since is None:
		since_dt = datetime.now(timezone.utc) - timedelta(hours=1)
	else:
		since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
	addr_val = addr or ""
	sql = """
	SELECT seq, ts, slot, kind, sig, address, score
	FROM detections
	WHERE ts >= {since:DateTime64(3)}
	  AND ({addr:String} = '' OR address = {addr:String})
	ORDER BY seq DESC
	LIMIT {limit:UInt32}
	"""
	rows = await ch.select(sql, {"since": since_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "addr": addr_val, "limit": limit})
	return [Detection(**r) for r in rows]

@router.get("/detections/range", response_model=List[Detection])
async def detections_range(
	from_seq: int = Query(..., ge=0),
	to_seq: int = Query(..., gt=0),
	ch: CH = Depends(get_ch),
):
	if not (from_seq < to_seq):
		return []
	if to_seq - from_seq > 5000:
		to_seq = from_seq + 5000
	sql = """
	SELECT seq, ts, slot, kind, sig, address, score
	FROM detections
	WHERE seq > {from_seq:UInt64} AND seq <= {to_seq:UInt64}
	ORDER BY seq ASC
	"""
	rows = await ch.select(sql, {"from_seq": from_seq, "to_seq": to_seq})
	return [Detection(**r) for r in rows]