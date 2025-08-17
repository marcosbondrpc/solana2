from typing import Optional
import time
import jwt
from ..settings import settings

def verify_ws_token(token: Optional[str]) -> bool:
	if not settings.WS_REQUIRE_TOKEN:
		return True
	if not token or not settings.WS_TOKEN_SECRET:
		return False
	try:
		payload = jwt.decode(token, settings.WS_TOKEN_SECRET, algorithms=["HS256"])
		if "exp" in payload:
			if time.time() > float(payload["exp"]):
				return False
		else:
			if "iat" in payload and (time.time() - float(payload["iat"]) > settings.WS_TOKEN_MAX_AGE_S):
				return False
		return True
	except Exception:
		return False