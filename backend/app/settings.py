from pydantic import BaseSettings, AnyHttpUrl
from typing import List, Optional

class Settings(BaseSettings):
	CH_URL: AnyHttpUrl = "http://localhost:8123"
	CH_DB: str = "default"
	CH_USER: Optional[str] = None
	CH_PASS: Optional[str] = None
	CH_TIMEOUT_S: float = 1.0
	CORS_ORIGINS: List[str] = []

	class Config:
		env_file = ".env"
		case_sensitive = True

settings = Settings()