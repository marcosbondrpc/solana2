from typing import Any, Dict, Iterable
from backend.app.ch.client import CH

class ClickHouseWriter:
    def __init__(self, ch: CH, table: str):
        self._ch = ch
        self._table = table

    async def write_rows(self, rows: Iterable[Dict[str, Any]]) -> int:
        return await self._ch.insert_json_each_row(self._table, rows)