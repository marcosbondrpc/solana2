from .ch.client import CH
_ch: CH | None = None

async def get_ch() -> CH:
	global _ch
	if _ch is None:
		_ch = CH()
	return _ch