from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import anyio
import msgpack
import time
from .security.ws_tokens import verify_ws_token
from .metrics import WS_CLIENTS, WS_BACKLOG, WS_DROPS

router = APIRouter()
Seq = int

SEQ_OFFSET = int(time.time() * 1000) << 16
_seq: Seq = SEQ_OFFSET
_seq_lock = anyio.Lock()

RING_CAP = 65536
_ring: Deque[Tuple[Seq, bytes]] = deque(maxlen=RING_CAP)

ClientQ = Deque[bytes]
CLIENTS: Dict[WebSocket, ClientQ] = {}
NOTICE_TS: Dict[WebSocket, float] = {}
MAXQ = 4096

async def _next_seq() -> Seq:
	global _seq
	async with _seq_lock:
		_seq += 1
		return _seq

def _pack(event: Dict) -> bytes:
	return msgpack.packb(event, use_bin_type=True)

def _ring_bounds() -> Tuple[Optional[Seq], Optional[Seq]]:
	if not _ring:
		return None, None
	first = _ring[0][0]
	last = _ring[-1][0]
	return first, last

async def _enqueue(pkt: bytes, seq: Seq) -> None:
	_ring.append((seq, pkt))
	now = time.time()
	for ws, q in list(CLIENTS.items()):
		if len(q) >= MAXQ:
			last_notice = NOTICE_TS.get(ws, 0.0)
			if now - last_notice > 10.0:
				NOTICE_TS[ws] = now
				notice = _pack({"t": "notice", "ts": int(time.time() * 1000), "seq": seq, "data": {"code": "backpressure_drop"}})
				try:
					q.append(notice)
				except Exception:
					pass
			try:
				q.popleft()
			except Exception:
				pass
			try:
				WS_DROPS.inc()
			except Exception:
				pass
		q.append(pkt)
		try:
			WS_BACKLOG.labels(client_id=str(id(ws))).set(len(q))
		except Exception:
			pass

def _replay_from(last_seq: Seq) -> Tuple[bool, List[bytes]]:
	if not _ring:
		return False, []
	first, last = _ring_bounds()
	if first is None or last is None:
		return False, []
	if last_seq < first - 1 or last_seq > last:
		return False, []
	out: List[bytes] = []
	for s, pkt in _ring:
		if s > last_seq:
			out.append(pkt)
	return True, out

@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: Optional[str] = Query(default=None)):
	h = ws.headers
	hdr_token = h.get("x-ws-token") or None
	if not token and hdr_token:
		token = hdr_token
	if not token:
		auth = h.get("authorization") or h.get("Authorization")
		if auth and auth.lower().startswith("bearer "):
			token = auth.split(" ", 1)[1].strip()
	if token:
		verify_ws_token(token)
	await ws.accept()
	q: ClientQ = deque()
	CLIENTS[ws] = q
	NOTICE_TS[ws] = 0.0
	try:
		try:
			WS_CLIENTS.inc()
			WS_BACKLOG.labels(client_id=str(id(ws))).set(0)
		except Exception:
			pass
		async with anyio.create_task_group() as tg:
			tg.start_soon(_sender_loop, ws, q)
			tg.start_soon(_heartbeat_loop, ws)
			subscribed = False
			while True:
				msg = await ws.receive_bytes()
				try:
					obj = msgpack.unpackb(msg, raw=False)
				except Exception:
					continue
				t = obj.get("t")
				if not subscribed and t == "subscribe":
					last_seq = obj.get("last_seq")
					if last_seq is not None:
						try:
							last_seq = int(last_seq)
						except Exception:
							last_seq = 0
					else:
						last_seq = 0
					ok, replay = _replay_from(last_seq) if last_seq > 0 else (True, [])
					if not ok and last_seq > 0:
						q.append(_pack({"t": "needs_snapshot", "ts": int(time.time() * 1000), "seq": 0}))
					for pkt in replay:
						if len(q) >= MAXQ:
							now = time.time()
							last_notice = NOTICE_TS.get(ws, 0.0)
							if now - last_notice > 10.0:
								NOTICE_TS[ws] = now
								q.append(_pack({"t": "notice", "ts": int(time.time() * 1000), "seq": 0, "data": {"code": "backpressure_drop"}}))
							try:
								q.popleft()
							except Exception:
								pass
							try:
								WS_DROPS.inc()
							except Exception:
								pass
						q.append(pkt)
						try:
							WS_BACKLOG.labels(client_id=str(id(ws))).set(len(q))
						except Exception:
							pass
					hello = _pack({"t": "hello", "ts": int(time.time() * 1000), "seq": 0, "data": {"heartbeat_ms": 10000}})
					q.append(hello)
					subscribed = True
				elif t == "pong":
					pass
				else:
					pass
	except WebSocketDisconnect:
		pass
	finally:
		CLIENTS.pop(ws, None)
		NOTICE_TS.pop(ws, None)
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