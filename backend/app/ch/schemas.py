from pydantic import BaseModel
from typing import List, Optional

class Detection(BaseModel):
	seq: int
	ts: str
	slot: int
	kind: str
	sig: str
	address: str
	score: float

class SnapshotResponse(BaseModel):
	as_of_seq: int
	detections: List[Detection]

class EntitySummary(BaseModel):
	address: str
	cnt: int
	last_ts: str
	score_sum: float

class EntityDetail(BaseModel):
	address: str
	cnt: int
	last_ts: str
	first_ts: str
	score_sum: float
	recent: List[Detection]

class MetricsJSON(BaseModel):
	api_lat_p50_ms: Optional[float] = None
	api_lat_p95_ms: Optional[float] = None
	ch_query_p95_ms: Optional[float] = None
	ws_clients: int = 0
	ws_backlog: int = 0
	ingest_lag_ms: int = 0