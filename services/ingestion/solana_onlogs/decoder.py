import json
import time
from typing import Any, Dict

def decode_log_notification(msg: Dict[str, Any]) -> Dict[str, Any]:
    now_ms = int(time.time() * 1000)
    params = msg.get("params") or {}
    result = params.get("result") or {}
    context = result.get("context") or {}
    value = result.get("value") or {}
    slot = int(context.get("slot") or 0)
    sig = value.get("signature") or ""
    err = value.get("err", None)
    status = "ok" if err is None else "err"
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return {
        "ts": now_ms,
        "slot": slot,
        "sig": sig,
        "program": "",
        "accounts": [],
        "status": status,
        "fee": 0,
        "payload": payload,
    }