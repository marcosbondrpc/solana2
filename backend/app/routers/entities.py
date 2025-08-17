from fastapi import APIRouter, Depends, Query, Path
from typing import List
from ..deps import get_ch
from ..ch.client import CH
from ..ch.schemas import EntitySummary, EntityDetail, Detection

router = APIRouter(prefix="/api")

@router.get("/entities", response_model=List[EntitySummary])
async def entities(limit: int = Query(100, ge=1, le=1000), ch: CH = Depends(get_ch)):
	sql = """
	SELECT
	  address,
	  count() AS cnt,
	  max(ts) AS last_ts,
	  sum(score) AS score_sum
	FROM detections
	GROUP BY address
	ORDER BY cnt DESC
	LIMIT {limit:UInt32}
	"""
	rows = await ch.select(sql, {"limit": limit})
	return [EntitySummary(**r) for r in rows]

@router.get("/entities/{address}", response_model=EntityDetail)
async def entity_detail(address: str = Path(..., min_length=1), ch: CH = Depends(get_ch)):
	profile_sql = """
	SELECT
	  address,
	  count() AS cnt,
	  max(ts) AS last_ts,
	  min(ts) AS first_ts,
	  sum(score) AS score_sum
	FROM detections
	WHERE address = {addr:String}
	GROUP BY address
	"""
	dets_sql = """
	SELECT seq, ts, slot, kind, sig, address, score
	FROM detections
	WHERE address = {addr:String}
	ORDER BY seq DESC
	LIMIT 200
	"""
	prof_rows = await ch.select(profile_sql, {"addr": address})
	if not prof_rows:
		return {"address": address, "cnt": 0, "last_ts": "", "first_ts": "", "score_sum": 0.0, "recent": []}
	det_rows = await ch.select(dets_sql, {"addr": address})
	return {
		**prof_rows[0],
		"recent": [Detection(**r) for r in det_rows]
	}