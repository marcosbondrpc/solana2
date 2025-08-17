import anyio, httpx, json, time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from ..settings import settings

class CH:
	def __init__(
		self,
		url: Optional[str] = None,
		db: Optional[str] = None,
		timeout_s: Optional[float] = None,
		user: Optional[str] = None,
		password: Optional[str] = None
	) -> None:
		self.base: str = str(url or settings.CH_URL)
		self.db: str = db or settings.CH_DB
		self.timeout_s: float = float(timeout_s or settings.CH_TIMEOUT_S)
		auth_user = user if user is not None else settings.CH_USER
		auth_pass = password if password is not None else settings.CH_PASS
		self.auth: Optional[Tuple[str, str]] = (auth_user, auth_pass) if auth_user else None
		self.client = httpx.AsyncClient(
			base_url=self.base,
			timeout=httpx.Timeout(self.timeout_s)
		)

	async def close(self) -> None:
		await self.client.aclose()

	async def select(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		qp: Dict[str, Any] = {"database": self.db, "default_format": "JSONEachRow"}
		if params:
			for k, v in params.items():
				qp[f"param_{k}"] = v
		async with anyio.move_on_after(self.timeout_s):
			r = await self.client.post("/", params=qp, content=sql.encode("utf-8"), auth=self.auth)
			r.raise_for_status()
			return [json.loads(line) for line in r.text.splitlines() if line]
		raise TimeoutError("ClickHouse SELECT timeout")

	async def insert_json_each_row(self, table: str, rows: Iterable[Dict[str, Any]]) -> int:
		ndjson = "\n".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) for r in rows)
		data = ndjson.encode("utf-8")
		if len(data) > 5 * 1024 * 1024:
			raise ValueError("Insert payload exceeds 5MB guard")
		qp = {"database": self.db, "query": f"INSERT INTO {table} FORMAT JSONEachRow"}
		async with anyio.move_on_after(self.timeout_s):
			r = await self.client.post("/", params=qp, content=data, auth=self.auth)
			r.raise_for_status()
			return len(ndjson.splitlines())
		raise TimeoutError("ClickHouse INSERT timeout")

	async def ping_latency_ms(self) -> float:
		t0 = time.perf_counter()
		_ = await self.select("SELECT 1")
		return (time.perf_counter() - t0) * 1000.0