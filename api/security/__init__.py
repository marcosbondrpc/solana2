"""
Security and authentication modules
"""

from .auth import (
    get_current_user,
    require_permission,
    require_role,
    check_rate_limit,
    TokenData,
    create_access_token,
    create_refresh_token,
    verify_password,
    get_password_hash,
    verify_ed25519_signature,
    generate_ack_hash,
    multisig_verifier
)

from .policy import (
    UserRole,
    Permission,
    ROLE_PERMISSIONS,
    ENDPOINT_PERMISSIONS,
    check_permission,
    check_endpoint_permission,
    get_role_permissions,
    get_rate_limit,
    get_query_limits
)

from .audit import (
    AuditLogger,
    AuditMiddleware,
    SecurityEventLogger,
    audit_logger
)

__all__ = [
    # Auth
    "get_current_user",
    "require_permission",
    "require_role",
    "check_rate_limit",
    "TokenData",
    "create_access_token",
    "create_refresh_token",
    "verify_password",
    "get_password_hash",
    "verify_ed25519_signature",
    "generate_ack_hash",
    "multisig_verifier",
    
    # Policy
    "UserRole",
    "Permission",
    "ROLE_PERMISSIONS",
    "ENDPOINT_PERMISSIONS",
    "check_permission",
    "check_endpoint_permission",
    "get_role_permissions",
    "get_rate_limit",
    "get_query_limits",
    
    # Audit
    "AuditLogger",
    "AuditMiddleware",
    "SecurityEventLogger",
    "audit_logger"
]