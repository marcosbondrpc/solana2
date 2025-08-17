from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import anyio
import msgpack
import time
from .metrics import WS_CLIENTS, WS_BACKLOG, WS_DROPS
from .security.ws_tokens import verify_ws_token
from .settings import settings

router = APIRouter()
Seq = int

SEQ_OFFSET = int(time.time() * 1000) << 16
_seq: Seq = SEQ_OFFSET
_seq_lock = anyio.Lock()

RING_CAP = 65536
_ring: Deque[Tuple[Seq, bytes]] = deque(maxlen=RING_CAP)

ClientQ = Deque[bytes]
CLIENTS: Dict[WebSocket, ClientQ] = {}
MAXQ = 4096

async def _next_seq() -> Seq:
	global _seq
	async with _seq_lock:
		_seq += 1
		return _seq

def _pack(event: Dict) -> bytes:
	return msgpack.packb(event, use_bin_type=True)

async def _enqueue(pkt: bytes, seq: Seq) -> None:
	_ring.append((seq, pkt))
	for ws, q in list(CLIENTS.items()):
		if len(q) >= MAXQ:
			try:
				q.popleft()
			except Exception:
				pass
			notice = _pack({"t": "notice", "ts": int(time.time() * 1000), "seq": seq, "data": {"code": "backpressure_drop"}})
			q.append(notice)
			try:
				WS_DROPS.inc()
			except Exception:
				pass
		q.append(pkt)
		try:
			WS_BACKLOG.labels(client_id=str(id(ws))).set(len(q))
		except Exception:
			pass

def _replay_from(last_seq: Seq) -> List[bytes]:
	out: List[bytes] = []
	for s, pkt in _ring:
		if s > last_seq:
			out.append(pkt)
	return out

@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: Optional[str] = Query(default=None)):
	if settings.WS_REQUIRE_TOKEN and not verify_ws_token(token):
		await ws.close()
		return
	await ws.accept()
	q: ClientQ = deque()
	CLIENTS[ws] = q
	try:
		WS_CLIENTS.inc()
		WS_BACKLOG.labels(client_id=str(id(ws))).set(0)
		async with anyio.create_task_group() as tg:
			tg.start_soon(_sender_loop, ws, q)
			tg.start_soon(_heartbeat_loop, ws)
			while True:
				msg = await ws.receive_bytes()
				try:
					obj = msgpack.unpackb(msg, raw=False)
				except Exception:
					continue
				t = obj.get("t")
				if t == "subscribe":
					last_seq = int(obj.get("last_seq") or 0)
					replay = _replay_from(last_seq) if last_seq > 0 else []
					for pkt in replay:
						if len(q) >= MAXQ:
							try:
								WS_BACKLOG.labels(client_id=str(id(ws))).set(len(q))
							except Exception:
								pass
							try:
								q.popleft()
							except Exception:
								pass
						q.append(pkt)
					hello = _pack({"t": "hello", "ts": int(time.time() * 1000), "seq": 0, "data": {"heartbeat_ms": 10000}})
					q.append(hello)
				elif t == "pong":
					pass
				else:
					pass
	except WebSocketDisconnect:
		pass
	finally:
		CLIENTS.pop(ws, None)
		try:
			WS_CLIENTS.dec()
			WS_BACKLOG.remove(str(id(ws)))
		except Exception:
			pass

async def _sender_loop(ws: WebSocket, q: ClientQ):
	while True:
		if q:
			pkt = q.popleft()
			await ws.send_bytes(pkt)
		else:
			await anyio.sleep(0.01)

async def _heartbeat_loop(ws: WebSocket):
	while True:
		hb = _pack({"t": "heartbeat", "ts": int(time.time() * 1000), "seq": 0})
		await ws.send_bytes(hb)
		await anyio.sleep(10)

async def emit(event_type: str, data: Dict) -> Seq:
	seq = await _next_seq()
	pkt = _pack({"t": event_type, "ts": int(time.time() * 1000), "seq": seq, "data": data})
	await _enqueue(pkt, seq)
	return seq

broadcast = emit