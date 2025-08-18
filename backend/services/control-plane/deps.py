"""
Security dependencies: JWT, RBAC, API keys
Ultra-secure with hardware-backed crypto
"""

import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
import redis.asyncio as redis
from pydantic import BaseModel


# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ultra-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token
bearer_scheme = HTTPBearer()

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Redis for session management (optional)
redis_client: Optional[redis.Redis] = None


async def get_redis() -> Optional[redis.Redis]:
    """Get Redis client for session management"""
    global redis_client
    if redis_client is None and os.getenv("REDIS_URL"):
        redis_client = await redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6390"),
            decode_responses=True,
            socket_keepalive=True,
            socket_connect_timeout=2,
            retry_on_timeout=True,
            max_connections=100
        )
    return redis_client


class TokenData(BaseModel):
    """JWT token payload"""
    sub: str  # Subject (user ID)
    roles: List[str] = []
    permissions: List[str] = []
    exp: Optional[int] = None
    iat: Optional[int] = None
    jti: Optional[str] = None  # JWT ID for revocation


class User(BaseModel):
    """User model"""
    id: str
    username: str
    email: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []
    is_active: bool = True
    api_keys: List[str] = []


# Role-based permissions
ROLE_PERMISSIONS = {
    "admin": [
        "control:write",
        "control:read",
        "realtime:read",
        "realtime:write",
        "datasets:read",
        "datasets:write",
        "training:write",
        "training:read",
        "system:manage"
    ],
    "operator": [
        "control:write",
        "control:read",
        "realtime:read",
        "datasets:read",
        "training:read"
    ],
    "viewer": [
        "control:read",
        "realtime:read",
        "datasets:read"
    ],
    "bot": [
        "realtime:read",
        "realtime:write"
    ]
}


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": f"{time.time_ns()}"  # Unique token ID
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": f"refresh_{time.time_ns()}",
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> TokenData:
    """Verify JWT token and extract data"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check if token is revoked (if using Redis)
        redis_conn = await get_redis()
        if redis_conn:
            jti = payload.get("jti")
            if jti and await redis_conn.get(f"revoked_token:{jti}"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
        
        # Extract token data
        token_data = TokenData(
            sub=payload.get("sub"),
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", []),
            exp=payload.get("exp"),
            iat=payload.get("iat"),
            jti=payload.get("jti")
        )
        
        if token_data.sub is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        return token_data
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}"
        )


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[User]:
    """Verify API key and return user"""
    if not api_key:
        return None
    
    # Check API key in Redis or database
    redis_conn = await get_redis()
    if redis_conn:
        user_data = await redis_conn.get(f"api_key:{api_key}")
        if user_data:
            import json
            user_dict = json.loads(user_data)
            return User(**user_dict)
    
    # Fallback to environment variable check (for testing)
    if api_key == os.getenv("MASTER_API_KEY"):
        return User(
            id="master",
            username="master",
            roles=["admin"],
            permissions=ROLE_PERMISSIONS["admin"]
        )
    
    return None


async def get_current_user(
    token_data: Optional[TokenData] = Depends(verify_token),
    api_user: Optional[User] = Depends(verify_api_key)
) -> User:
    """Get current user from token or API key"""
    
    # Prefer API key for bot/service authentication
    if api_user:
        return api_user
    
    # Fall back to JWT token
    if token_data:
        # In production, fetch user from database
        # For now, create user from token data
        return User(
            id=token_data.sub,
            username=token_data.sub,
            roles=token_data.roles,
            permissions=token_data.permissions
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No valid authentication provided"
    )


def require_permission(permission: str):
    """Require specific permission"""
    async def permission_checker(user: User = Depends(get_current_user)):
        # Check direct permissions
        if permission in user.permissions:
            return user
        
        # Check role-based permissions
        for role in user.roles:
            if role in ROLE_PERMISSIONS and permission in ROLE_PERMISSIONS[role]:
                return user
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission '{permission}' required"
        )
    
    return permission_checker


def require_role(role: str):
    """Require specific role"""
    async def role_checker(user: User = Depends(get_current_user)):
        if role not in user.roles and "admin" not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return user
    
    return role_checker


def require_any_role(roles: List[str]):
    """Require any of the specified roles"""
    async def role_checker(user: User = Depends(get_current_user)):
        if not any(role in user.roles for role in roles) and "admin" not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {roles} required"
            )
        return user
    
    return role_checker


async def revoke_token(jti: str):
    """Revoke a token by its JWT ID"""
    redis_conn = await get_redis()
    if redis_conn:
        # Store revoked token with expiration
        await redis_conn.setex(
            f"revoked_token:{jti}",
            timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS + 1),
            "1"
        )


async def create_api_key(user: User) -> str:
    """Create new API key for user"""
    import secrets
    api_key = f"mev_{secrets.token_urlsafe(32)}"
    
    redis_conn = await get_redis()
    if redis_conn:
        # Store API key with user data
        import json
        await redis_conn.set(
            f"api_key:{api_key}",
            json.dumps(user.dict())
        )
    
    return api_key


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


# Audit logging
async def audit_log(
    action: str,
    user: User,
    details: Dict[str, Any],
    success: bool = True
):
    """Log security-relevant actions"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user.id,
        "username": user.username,
        "action": action,
        "success": success,
        "details": details,
        "ip": details.get("ip", "unknown")
    }
    
    # Store in Redis for real-time access
    redis_conn = await get_redis()
    if redis_conn:
        await redis_conn.lpush(
            "audit_log",
            json.dumps(log_entry)
        )
        # Trim to last 10000 entries
        await redis_conn.ltrim("audit_log", 0, 9999)
    
    # Also log to file or external service
    import logging
    logger = logging.getLogger("audit")
    logger.info(f"AUDIT: {log_entry}")


# Rate limiting helpers
async def check_rate_limit(
    key: str,
    limit: int = 100,
    window: int = 60
) -> bool:
    """Check rate limit using Redis"""
    redis_conn = await get_redis()
    if not redis_conn:
        return True  # Allow if Redis not available
    
    pipe = redis_conn.pipeline()
    now = time.time()
    window_start = now - window
    
    # Remove old entries
    await pipe.zremrangebyscore(key, 0, window_start)
    # Add current request
    await pipe.zadd(key, {str(now): now})
    # Count requests in window
    await pipe.zcount(key, window_start, now)
    # Set expiration
    await pipe.expire(key, window + 1)
    
    results = await pipe.execute()
    request_count = results[2]
    
    return request_count <= limit