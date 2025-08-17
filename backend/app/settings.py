from pydantic import BaseSettings, AnyHttpUrl
from typing import List, Optional

class Settings(BaseSettings):
	CH_URL: AnyHttpUrl = "http://localhost:8123"
	CH_DB: str = "default"
	CH_USER: Optional[str] = None
	CH_PASS: Optional[str] = None
	CH_TIMEOUT_S: float = 1.0
	CORS_ORIGINS: List[str] = []

	# REST JWT
	API_REQUIRE_AUTH: bool = False
	REST_JWT_SECRET: Optional[str] = None
	REST_JWT_ISS: Optional[str] = None
	REST_JWT_AUD: Optional[str] = None

	# WS tokens
	WS_REQUIRE_TOKEN: bool = False
	WS_TOKEN_SECRET: Optional[str] = None
	WS_TOKEN_MAX_AGE_S: int = 300

	# Simple rate limit
	RATE_LIMIT_PER_MIN: int = 600

	class Config:
		env_file = ".env"
		case_sensitive = True

settings = Settings()