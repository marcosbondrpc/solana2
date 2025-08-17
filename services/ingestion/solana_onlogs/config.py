from pydantic import BaseSettings, AnyUrl
from typing import Optional

class Settings(BaseSettings):
    SOLANA_WS: AnyUrl = "wss://api.mainnet-beta.solana.com"
    CH_URL: str = "http://localhost:8123"
    CH_DB: str = "solana_rt_dev"
    CH_USER: Optional[str] = None
    CH_PASS: Optional[str] = None
    CH_TIMEOUT_S: float = 0.8

    BATCH_MAX_ROWS: int = 500
    BATCH_MAX_MS: int = 25
    QUEUE_MAX: int = 10000
    RECONNECT_MIN_MS: int = 100
    RECONNECT_MAX_MS: int = 5000

    class Config:
        env_prefix = "INGEST_"
        env_file = ".env"
        case_sensitive = True

settings = Settings()