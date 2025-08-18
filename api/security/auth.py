"""
JWT authentication and authorization middleware
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError
import hashlib
import hmac
import time
from models.schemas import UserRole, TokenResponse
from security.policy import check_permission, Permission, get_role_permissions


# Configuration
SECRET_KEY = "LEGENDARY-MEV-SYSTEM-SECRET-KEY-CHANGE-IN-PRODUCTION"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token security
security = HTTPBearer()


class AuthenticationError(Exception):
    """Custom authentication error"""
    pass


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


def verify_ed25519_signature(
    message: bytes,
    signature: bytes,
    public_key: bytes
) -> bool:
    """Verify Ed25519 signature for critical operations"""
    try:
        VerifyKey(public_key).verify(message, signature)
        return True
    except BadSignatureError:
        return False
    except Exception:
        return False


def generate_ack_hash(command_id: str, timestamp: int, result: str) -> str:
    """Generate ACK hash for audit trail chain"""
    data = f"{command_id}:{timestamp}:{result}".encode()
    return hashlib.sha256(data).hexdigest()


class TokenData:
    """Token payload data"""
    def __init__(self, username: str, user_id: str, role: UserRole, permissions: list):
        self.username = username
        self.user_id = user_id
        self.role = role
        self.permissions = permissions


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Extract and validate current user from JWT token"""
    token = credentials.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        role: str = payload.get("role")
        
        if username is None or user_id is None or role is None:
            raise credentials_exception
        
        # Convert role string to enum
        try:
            user_role = UserRole(role)
        except ValueError:
            raise credentials_exception
        
        # Get permissions for role
        permissions = get_role_permissions(user_role)
        
        return TokenData(
            username=username,
            user_id=user_id,
            role=user_role,
            permissions=permissions
        )
        
    except AuthenticationError:
        raise credentials_exception


def require_permission(permission: Permission):
    """Dependency to require specific permission"""
    async def permission_checker(current_user: TokenData = Depends(get_current_user)):
        if not check_permission(current_user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission.value}"
            )
        return current_user
    return permission_checker


def require_role(min_role: UserRole):
    """Dependency to require minimum role level"""
    role_hierarchy = {
        UserRole.VIEWER: 0,
        UserRole.ANALYST: 1,
        UserRole.OPERATOR: 2,
        UserRole.ML_ENGINEER: 2,  # Same level as operator
        UserRole.ADMIN: 3
    }
    
    async def role_checker(current_user: TokenData = Depends(get_current_user)):
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(min_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {min_role.value}"
            )
        return current_user
    return role_checker


class RateLimiter:
    """Simple in-memory rate limiter"""
    def __init__(self):
        self.requests: Dict[str, list] = {}
    
    def check_rate_limit(self, user_id: str, role: UserRole, window_seconds: int = 60) -> bool:
        """Check if user is within rate limit"""
        from security.policy import get_rate_limit
        
        now = time.time()
        limit = get_rate_limit(role)
        
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if now - req_time < window_seconds
        ]
        
        # Check if limit exceeded
        if len(self.requests[user_id]) >= limit:
            return False
        
        # Add current request
        self.requests[user_id].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(current_user: TokenData = Depends(get_current_user)):
    """Check rate limit for current user"""
    if not rate_limiter.check_rate_limit(current_user.user_id, current_user.role):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return current_user


# Multisig verification for critical operations
class MultisigVerifier:
    """2-of-3 multisig verification for critical operations"""
    
    def __init__(self):
        self.pending_operations: Dict[str, Dict] = {}
        self.authorized_keys = {
            "key1": "PUBLIC_KEY_1_PLACEHOLDER",
            "key2": "PUBLIC_KEY_2_PLACEHOLDER",
            "key3": "PUBLIC_KEY_3_PLACEHOLDER"
        }
    
    def initiate_operation(self, operation_id: str, operation_data: Dict) -> Dict:
        """Start a multisig operation"""
        self.pending_operations[operation_id] = {
            "data": operation_data,
            "signatures": {},
            "created_at": datetime.now(timezone.utc),
            "status": "pending"
        }
        return self.pending_operations[operation_id]
    
    def add_signature(self, operation_id: str, key_id: str, signature: str) -> bool:
        """Add a signature to pending operation"""
        if operation_id not in self.pending_operations:
            return False
        
        if key_id not in self.authorized_keys:
            return False
        
        self.pending_operations[operation_id]["signatures"][key_id] = signature
        
        # Check if we have enough signatures (2 of 3)
        if len(self.pending_operations[operation_id]["signatures"]) >= 2:
            self.pending_operations[operation_id]["status"] = "approved"
            return True
        
        return False
    
    def verify_operation(self, operation_id: str) -> bool:
        """Check if operation is approved"""
        if operation_id not in self.pending_operations:
            return False
        
        return self.pending_operations[operation_id]["status"] == "approved"


# Global multisig verifier
multisig_verifier = MultisigVerifier()