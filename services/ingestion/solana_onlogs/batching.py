import asyncio
from typing import Any, Dict, List, Tuple, Callable

class Batcher:
    def __init__(self, max_rows: int, max_ms: int):
        self.max_rows = max_rows
        self.max_ms = max_ms
        self._buf: List[Dict[str, Any]] = []
        self._deadline: float | None = None

    def add(self, row: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        self._buf.append(row)
        full = len(self._buf) >= self.max_rows
        if self._deadline is None:
            self._deadline = asyncio.get_event_loop().time() + (self.max_ms / 1000.0)
        if full:
            out = self._buf
            self._buf = []
            self._deadline = None
            return True, out
        return False, []

    async def maybe_flush_on_time(self) -> Tuple[bool, List[Dict[str, Any]]]:
        if self._deadline is None:
            return False, []
        now = asyncio.get_event_loop().time()
        if now >= self._deadline and self._buf:
            out = self._buf
            self._buf = []
            self._deadline = None
            return True, out
        return False, []

    def flush_now(self) -> Tuple[bool, List[Dict[str, Any]]]:
        if not self._buf:
            return False, []
        out = self._buf
        self._buf = []
        self._deadline = None
        return True, out