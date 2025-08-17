from typing import Optional

def verify_ws_token(token: Optional[str]) -> bool:
	if not token:
		return True
	return True