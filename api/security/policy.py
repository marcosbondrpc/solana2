"""
Role-based access control policies and permissions
"""

from typing import Dict, List, Set
from enum import Enum
from models.schemas import UserRole


class Permission(str, Enum):
    """Fine-grained permissions"""
    # Read permissions
    VIEW_METRICS = "view_metrics"
    VIEW_ALERTS = "view_alerts"
    VIEW_MODELS = "view_models"
    VIEW_AUDIT_LOG = "view_audit_log"
    
    # Query permissions
    QUERY_CLICKHOUSE = "query_clickhouse"
    EXPORT_DATASETS = "export_datasets"
    
    # ML permissions
    TRAIN_MODELS = "train_models"
    DEPLOY_MODELS = "deploy_models"
    DELETE_MODELS = "delete_models"
    
    # Control permissions
    ACTIVATE_KILL_SWITCH = "activate_kill_switch"
    UPDATE_CONFIG = "update_config"
    MANAGE_POLICIES = "manage_policies"
    
    # Admin permissions
    MANAGE_USERS = "manage_users"
    VIEW_ALL_AUDIT_LOGS = "view_all_audit_logs"
    SYSTEM_ADMIN = "system_admin"


# Role-permission mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.VIEWER: {
        Permission.VIEW_METRICS,
        Permission.VIEW_ALERTS,
    },
    
    UserRole.ANALYST: {
        Permission.VIEW_METRICS,
        Permission.VIEW_ALERTS,
        Permission.VIEW_MODELS,
        Permission.QUERY_CLICKHOUSE,
        Permission.EXPORT_DATASETS,
    },
    
    UserRole.OPERATOR: {
        Permission.VIEW_METRICS,
        Permission.VIEW_ALERTS,
        Permission.VIEW_MODELS,
        Permission.VIEW_AUDIT_LOG,
        Permission.QUERY_CLICKHOUSE,
        Permission.ACTIVATE_KILL_SWITCH,
        Permission.UPDATE_CONFIG,
    },
    
    UserRole.ML_ENGINEER: {
        Permission.VIEW_METRICS,
        Permission.VIEW_ALERTS,
        Permission.VIEW_MODELS,
        Permission.QUERY_CLICKHOUSE,
        Permission.EXPORT_DATASETS,
        Permission.TRAIN_MODELS,
        Permission.DEPLOY_MODELS,
        Permission.DELETE_MODELS,
    },
    
    UserRole.ADMIN: {
        # Admins get all permissions
        Permission.VIEW_METRICS,
        Permission.VIEW_ALERTS,
        Permission.VIEW_MODELS,
        Permission.VIEW_AUDIT_LOG,
        Permission.QUERY_CLICKHOUSE,
        Permission.EXPORT_DATASETS,
        Permission.TRAIN_MODELS,
        Permission.DEPLOY_MODELS,
        Permission.DELETE_MODELS,
        Permission.ACTIVATE_KILL_SWITCH,
        Permission.UPDATE_CONFIG,
        Permission.MANAGE_POLICIES,
        Permission.MANAGE_USERS,
        Permission.VIEW_ALL_AUDIT_LOGS,
        Permission.SYSTEM_ADMIN,
    }
}


# API endpoint permission requirements
ENDPOINT_PERMISSIONS: Dict[str, Set[Permission]] = {
    # Dataset endpoints
    "POST:/datasets/export": {Permission.EXPORT_DATASETS},
    
    # ClickHouse endpoints
    "POST:/clickhouse/query": {Permission.QUERY_CLICKHOUSE},
    
    # Training endpoints
    "POST:/training/train": {Permission.TRAIN_MODELS},
    "GET:/training/models": {Permission.VIEW_MODELS},
    "POST:/training/models/{id}/deploy": {Permission.DEPLOY_MODELS},
    "DELETE:/training/models/{id}": {Permission.DELETE_MODELS},
    
    # Control endpoints
    "POST:/control/kill-switch": {Permission.ACTIVATE_KILL_SWITCH},
    "GET:/control/audit-log": {Permission.VIEW_AUDIT_LOG},
    "GET:/control/audit-log/all": {Permission.VIEW_ALL_AUDIT_LOGS},
    
    # Admin endpoints
    "POST:/admin/users": {Permission.MANAGE_USERS},
    "PUT:/admin/policies": {Permission.MANAGE_POLICIES},
}


# Rate limiting by role (requests per minute)
ROLE_RATE_LIMITS: Dict[UserRole, int] = {
    UserRole.VIEWER: 60,
    UserRole.ANALYST: 120,
    UserRole.OPERATOR: 180,
    UserRole.ML_ENGINEER: 240,
    UserRole.ADMIN: 600,
}


# Query limits by role
ROLE_QUERY_LIMITS: Dict[UserRole, Dict[str, int]] = {
    UserRole.VIEWER: {
        "max_rows": 1000,
        "timeout_seconds": 10,
    },
    UserRole.ANALYST: {
        "max_rows": 100000,
        "timeout_seconds": 60,
    },
    UserRole.OPERATOR: {
        "max_rows": 500000,
        "timeout_seconds": 120,
    },
    UserRole.ML_ENGINEER: {
        "max_rows": 10000000,
        "timeout_seconds": 300,
    },
    UserRole.ADMIN: {
        "max_rows": 100000000,
        "timeout_seconds": 600,
    },
}


def check_permission(role: UserRole, permission: Permission) -> bool:
    """Check if a role has a specific permission"""
    return permission in ROLE_PERMISSIONS.get(role, set())


def check_endpoint_permission(role: UserRole, method: str, path: str) -> bool:
    """Check if a role can access an endpoint"""
    endpoint_key = f"{method}:{path}"
    required_permissions = ENDPOINT_PERMISSIONS.get(endpoint_key, set())
    
    if not required_permissions:
        # No specific permissions required
        return True
    
    user_permissions = ROLE_PERMISSIONS.get(role, set())
    return bool(required_permissions & user_permissions)


def get_role_permissions(role: UserRole) -> List[str]:
    """Get all permissions for a role"""
    return [p.value for p in ROLE_PERMISSIONS.get(role, set())]


def get_rate_limit(role: UserRole) -> int:
    """Get rate limit for a role"""
    return ROLE_RATE_LIMITS.get(role, 60)


def get_query_limits(role: UserRole) -> Dict[str, int]:
    """Get query limits for a role"""
    return ROLE_QUERY_LIMITS.get(role, ROLE_QUERY_LIMITS[UserRole.VIEWER])