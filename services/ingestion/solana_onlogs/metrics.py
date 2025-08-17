from dataclasses import dataclass, field

@dataclass
class Counters:
    batches_ok: int = 0
    batches_err: int = 0
    rows_ok: int = 0
    rows_dropped: int = 0
    reconnects: int = 0

counters = Counters()