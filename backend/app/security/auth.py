from typing import Optional
import time
from fastapi import Request
import jwt
from ..settings import settings

class AuthError(Exception):
	pass

def verify_jwt(token: str) -> dict:
	if not settings.REST_JWT_SECRET:
		raise AuthError("no secret")
	try:
		opts = {"verify_signature": True, "verify_exp": True}
		payload = jwt.decode(
			token,
			settings.REST_JWT_SECRET,
			algorithms=["HS256"],
			options=opts,
			audience=settings.REST_JWT_AUD if settings.REST_JWT_AUD else None,
			issuer=settings.REST_JWT_ISS if settings.REST_JWT_ISS else None,
		)
		return payload
	except Exception as e:
		raise AuthError(str(e))

def extract_bearer(req: Request) -> Optional[str]:
	h = req.headers.get("authorization") or req.headers.get("Authorization")
	if not h:
		return None
	if h.lower().startswith("bearer "):
		return h.split(" ", 1)[1].strip()
	return None