"""
API route modules
"""

from .datasets import router as datasets_router
from .clickhouse import router as clickhouse_router
from .training import router as training_router
from .control import router as control_router
from .realtime import router as realtime_router

__all__ = [
    "datasets_router",
    "clickhouse_router",
    "training_router",
    "control_router",
    "realtime_router"
]