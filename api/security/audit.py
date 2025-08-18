"""
Audit logging middleware and utilities
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from enum import Enum
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from models.schemas import AuditLogEntry, UserRole
import asyncio
import aiofiles
from pathlib import Path


class AuditLogger:
    """Centralized audit logging system"""
    
    def __init__(self, log_dir: str = "/var/log/mev-api"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.hash_chain = []
        self.lock = asyncio.Lock()
    
    def _to_serializable(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.astimezone(timezone.utc).isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except Exception:
                return obj.hex()
        if isinstance(obj, (set, tuple)):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        return obj

    def _canonicalize(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._canonicalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._canonicalize(v) for v in obj]
        return self._to_serializable(obj)

    def _canonical_dumps(self, obj: Any) -> str:
        return json.dumps(self._canonicalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def generate_entry_hash(self, entry: Dict[str, Any], previous_hash: str = "") -> str:
        """Generate hash for audit entry (for immutable chain)"""
        entry_str = self._canonical_dumps(entry)
        combined = f"{previous_hash}{entry_str}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    async def log_entry(
        self,
        user_id: str,
        role: UserRole,
        action: str,
        resource: str,
        details: Dict[str, Any],
        ip_address: str,
        user_agent: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> str:
        """Log an audit entry"""
        async with self.lock:
            entry = AuditLogEntry(
                id=hashlib.sha256(f"{user_id}{datetime.now().timestamp()}".encode()).hexdigest()[:16],
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                role=role,
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                error_message=error_message
            )
            
            # Convert to dict for hashing
            entry_dict = entry.dict()
            
            # Generate hash with chain
            previous_hash = self.hash_chain[-1] if self.hash_chain else ""
            entry_hash = self.generate_entry_hash(entry_dict, previous_hash)
            entry_dict["hash"] = entry_hash
            entry_dict["previous_hash"] = previous_hash
            
            # Add to chain
            self.hash_chain.append(entry_hash)
            
            # Write to file
            async with aiofiles.open(self.current_log_file, "a") as f:
                await f.write(self._canonical_dumps(entry_dict) + "\n")
            
            return entry_hash
    
    async def verify_chain(self, start_date: Optional[datetime] = None) -> bool:
        """Verify the integrity of the audit chain"""
        log_file = self.current_log_file
        if start_date:
            log_file = self.log_dir / f"audit_{start_date.strftime('%Y%m%d')}.jsonl"
        
        if not log_file.exists():
            return True  # No logs to verify
        
        previous_hash = ""
        async with aiofiles.open(log_file, "r") as f:
            async for line in f:
                try:
                    entry = json.loads(line.strip())
                    stored_hash = entry.pop("hash", "")
                    stored_previous = entry.pop("previous_hash", "")
                    
                    # Verify previous hash matches
                    if stored_previous != previous_hash:
                        return False
                    
                    # Verify entry hash
                    calculated_hash = self.generate_entry_hash(entry, previous_hash)
                    if calculated_hash != stored_hash:
                        return False
                    
                    previous_hash = stored_hash
                except Exception:
                    return False
        
        return True
    
    async def get_recent_entries(
        self,
        limit: int = 100,
        user_id: Optional[str] = None,
        action: Optional[str] = None
    ) -> list:
        """Retrieve recent audit entries"""
        entries = []
        
        async with aiofiles.open(self.current_log_file, "r") as f:
            async for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    # Apply filters
                    if user_id and entry.get("user_id") != user_id:
                        continue
                    if action and entry.get("action") != action:
                        continue
                    
                    entries.append(entry)
                    
                    if len(entries) >= limit:
                        break
                except Exception:
                    continue
        
        return entries[-limit:]  # Return most recent


# Global audit logger instance
audit_logger = AuditLogger()


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically log all API requests"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip health checks and metrics endpoints
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        # Extract request details
        start_time = datetime.now(timezone.utc)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Extract user info if authenticated
        user_id = "anonymous"
        role = UserRole.VIEWER
        
        try:
            # Get auth header
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                from security.auth import decode_token
                token = auth_header.split(" ")[1]
                try:
                    payload = decode_token(token)
                    user_id = payload.get("user_id", "anonymous")
                    role = UserRole(payload.get("role", "viewer"))
                except Exception:
                    pass
        except Exception:
            pass
        
        # Capture request body for POST/PUT
        request_body = {}
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    request_body = json.loads(body)
                    # Remove sensitive fields
                    request_body.pop("password", None)
                    request_body.pop("signature", None)
                    request_body.pop("private_key", None)
            except Exception:
                pass
        
        # Process request
        response = None
        error_message = None
        success = False
        
        try:
            response = await call_next(request)
            success = response.status_code < 400
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            # Log the request
            if response:
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                await audit_logger.log_entry(
                    user_id=user_id,
                    role=role,
                    action=f"{request.method} {request.url.path}",
                    resource=str(request.url),
                    details={
                        "method": request.method,
                        "path": request.url.path,
                        "query_params": dict(request.query_params),
                        "request_body": request_body,
                        "status_code": response.status_code if response else 500,
                        "duration_ms": duration_ms
                    },
                    ip_address=client_ip,
                    user_agent=user_agent,
                    success=success,
                    error_message=error_message
                )
        
        return response


class SecurityEventLogger:
    """Log security-specific events"""
    
    @staticmethod
    async def log_authentication_attempt(
        username: str,
        ip_address: str,
        success: bool,
        reason: Optional[str] = None
    ):
        """Log authentication attempts"""
        await audit_logger.log_entry(
            user_id=username,
            role=UserRole.VIEWER,
            action="AUTHENTICATION_ATTEMPT",
            resource="/auth/token",
            details={
                "username": username,
                "success": success,
                "reason": reason
            },
            ip_address=ip_address,
            user_agent="",
            success=success,
            error_message=reason if not success else None
        )
    
    @staticmethod
    async def log_permission_denied(
        user_id: str,
        role: UserRole,
        resource: str,
        required_permission: str,
        ip_address: str
    ):
        """Log permission denied events"""
        await audit_logger.log_entry(
            user_id=user_id,
            role=role,
            action="PERMISSION_DENIED",
            resource=resource,
            details={
                "required_permission": required_permission,
                "user_role": role.value
            },
            ip_address=ip_address,
            user_agent="",
            success=False,
            error_message=f"Permission {required_permission} required"
        )
    
    @staticmethod
    async def log_critical_operation(
        user_id: str,
        role: UserRole,
        operation: str,
        details: Dict[str, Any],
        ip_address: str,
        signature_verified: bool
    ):
        """Log critical operations like kill-switch activation"""
        await audit_logger.log_entry(
            user_id=user_id,
            role=role,
            action=f"CRITICAL_OPERATION_{operation.upper()}",
            resource=f"/control/{operation}",
            details={
                **details,
                "signature_verified": signature_verified,
                "multisig_required": operation in ["kill-switch", "policy-update"]
            },
            ip_address=ip_address,
            user_agent="",
            success=signature_verified,
            error_message=None if signature_verified else "Invalid signature"
        )