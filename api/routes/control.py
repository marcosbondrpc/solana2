"""
Control plane endpoints - kill-switch and audit
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException

from models.schemas import (
    KillSwitchRequest,
    KillSwitchResponse,
    AuditLogEntry,
    UserRole
)
from security.auth import (
    require_role,
    get_current_user,
    TokenData,
    verify_ed25519_signature,
    generate_ack_hash,
    multisig_verifier
)
from security.audit import audit_logger, SecurityEventLogger
from security.policy import Permission

router = APIRouter()


class ControlPlane:
    """Control plane manager for critical operations"""
    
    def __init__(self):
        self.kill_switches: Dict[str, Dict] = {}
        self.active_throttles: Dict[str, float] = {}
        self.ack_chain: List[str] = []
    
    async def activate_kill_switch(
        self,
        target: str,
        reason: str,
        duration_seconds: Optional[int],
        user_id: str,
        signature_verified: bool
    ) -> Dict:
        """Activate kill switch for target system"""
        
        # Generate ACK hash
        command_id = f"kill_{target}_{datetime.now().timestamp()}"
        
        # Create kill switch entry
        self.kill_switches[target] = {
            "target": target,
            "reason": reason,
            "activated_at": datetime.now(),
            "activated_by": user_id,
            "expires_at": datetime.now() + timedelta(seconds=duration_seconds) if duration_seconds else None,
            "signature_verified": signature_verified,
            "active": True
        }
        
        # Add to ACK chain
        previous_hash = self.ack_chain[-1] if self.ack_chain else ""
        ack_hash = generate_ack_hash(command_id, int(datetime.now().timestamp()), "activated")
        self.ack_chain.append(ack_hash)
        
        # Send command to Kafka for other services
        from services.kafka_bridge import get_kafka_bridge
        bridge = await get_kafka_bridge()
        
        await bridge.send_command(
            "control.kill_switch",
            {
                "command_id": command_id,
                "target": target,
                "action": "activate",
                "reason": reason,
                "duration_seconds": duration_seconds,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "command_id": command_id,
            "ack_hash": ack_hash,
            "previous_hash": previous_hash
        }
    
    def get_active_kill_switches(self) -> List[Dict]:
        """Get currently active kill switches"""
        
        active = []
        now = datetime.now()
        
        for target, info in self.kill_switches.items():
            if info["active"]:
                # Check if expired
                if info["expires_at"] and info["expires_at"] < now:
                    info["active"] = False
                else:
                    active.append({
                        "target": target,
                        **info
                    })
        
        return active
    
    def deactivate_kill_switch(self, target: str, user_id: str) -> bool:
        """Deactivate kill switch"""
        
        if target in self.kill_switches:
            self.kill_switches[target]["active"] = False
            self.kill_switches[target]["deactivated_at"] = datetime.now()
            self.kill_switches[target]["deactivated_by"] = user_id
            return True
        
        return False


# Global control plane instance
control_plane = ControlPlane()


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def activate_kill_switch(
    request: KillSwitchRequest,
    current_user: TokenData = Depends(require_role(UserRole.OPERATOR))
) -> KillSwitchResponse:
    """
    Activate emergency kill switch
    Requires OPERATOR role and Ed25519 signature
    For critical operations, requires 2-of-3 multisig
    """
    
    # Verify signature
    message = f"{request.target}:{request.reason}:{request.duration_seconds}".encode()
    signature_valid = verify_ed25519_signature(
        message,
        bytes.fromhex(request.signature),
        bytes.fromhex(request.public_key)
    )
    
    if not signature_valid:
        # Log failed attempt
        await SecurityEventLogger.log_critical_operation(
            user_id=current_user.user_id,
            role=current_user.role,
            operation="kill_switch",
            details=request.dict(),
            ip_address="",  # Would get from request
            signature_verified=False
        )
        
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    # For "all" target, require multisig
    multisig_status = None
    if request.target == "all":
        operation_id = f"kill_all_{datetime.now().timestamp()}"
        
        # Check if this is part of multisig
        if not request.force:
            # Initiate or update multisig
            multisig_verifier.initiate_operation(
                operation_id,
                request.dict()
            )
            
            # Add this signature
            multisig_verifier.add_signature(
                operation_id,
                request.public_key,
                request.signature
            )
            
            # Check if we have enough signatures
            if not multisig_verifier.verify_operation(operation_id):
                return KillSwitchResponse(
                    success=False,
                    activated=False,
                    target=request.target,
                    message="Waiting for additional signatures (1 of 2 required)",
                    ack_hash="",
                    multisig_status={"signatures_received": 1, "signatures_required": 2}
                )
            
            multisig_status = {"signatures_received": 2, "signatures_required": 2}
    
    # Activate kill switch
    result = await control_plane.activate_kill_switch(
        target=request.target,
        reason=request.reason,
        duration_seconds=request.duration_seconds,
        user_id=current_user.user_id,
        signature_verified=True
    )
    
    # Log operation
    await SecurityEventLogger.log_critical_operation(
        user_id=current_user.user_id,
        role=current_user.role,
        operation="kill_switch",
        details={**request.dict(), "multisig": multisig_status is not None},
        ip_address="",  # Would get from request
        signature_verified=True
    )
    
    return KillSwitchResponse(
        success=True,
        activated=True,
        target=request.target,
        expires_at=datetime.now() + timedelta(seconds=request.duration_seconds) if request.duration_seconds else None,
        ack_hash=result["ack_hash"],
        multisig_status=multisig_status,
        message=f"Kill switch activated for {request.target}"
    )


@router.delete("/kill-switch/{target}")
async def deactivate_kill_switch(
    target: str,
    current_user: TokenData = Depends(require_role(UserRole.OPERATOR))
) -> Dict[str, Any]:
    """Deactivate kill switch"""
    
    success = control_plane.deactivate_kill_switch(target, current_user.user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Kill switch not found")
    
    # Send deactivation command
    from services.kafka_bridge import get_kafka_bridge
    bridge = await get_kafka_bridge()
    
    await bridge.send_command(
        "control.kill_switch",
        {
            "target": target,
            "action": "deactivate",
            "timestamp": datetime.now().isoformat()
        }
    )
    
    return {
        "success": True,
        "message": f"Kill switch deactivated for {target}"
    }


@router.get("/kill-switch/status")
async def get_kill_switch_status(
    current_user: TokenData = Depends(require_role(UserRole.OPERATOR))
) -> Dict[str, Any]:
    """Get active kill switches"""
    
    active = control_plane.get_active_kill_switches()
    
    return {
        "success": True,
        "active_switches": active,
        "total_active": len(active)
    }


@router.get("/audit-log")
async def get_audit_log(
    limit: int = 100,
    action: Optional[str] = None,
    current_user: TokenData = Depends(require_role(UserRole.OPERATOR))
) -> Dict[str, Any]:
    """
    Get audit log entries
    Regular users see their own entries
    Admins see all entries
    """
    
    # Determine which entries to show
    if current_user.role == UserRole.ADMIN:
        user_filter = None  # Show all
    else:
        user_filter = current_user.user_id  # Show only user's entries
    
    # Get entries
    entries = await audit_logger.get_recent_entries(
        limit=limit,
        user_id=user_filter,
        action=action
    )
    
    return {
        "success": True,
        "entries": entries,
        "count": len(entries)
    }


@router.get("/audit-log/verify")
async def verify_audit_chain(
    date: Optional[str] = None,
    current_user: TokenData = Depends(require_role(UserRole.OPERATOR))
) -> Dict[str, Any]:
    """Verify audit log chain integrity"""
    
    start_date = None
    if date:
        try:
            start_date = datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    
    is_valid = await audit_logger.verify_chain(start_date)
    
    return {
        "success": True,
        "chain_valid": is_valid,
        "date": date or "current",
        "message": "Audit chain is valid" if is_valid else "Audit chain integrity check failed"
    }


@router.get("/ack-chain")
async def get_ack_chain(
    limit: int = 100,
    current_user: TokenData = Depends(require_role(UserRole.OPERATOR))
) -> Dict[str, Any]:
    """Get ACK hash chain for command verification"""
    
    chain = control_plane.ack_chain[-limit:] if control_plane.ack_chain else []
    
    return {
        "success": True,
        "chain": chain,
        "length": len(control_plane.ack_chain),
        "latest": chain[-1] if chain else None
    }


@router.post("/throttle")
async def set_throttle(
    target: str,
    percent: float,
    current_user: TokenData = Depends(require_role(UserRole.OPERATOR))
) -> Dict[str, Any]:
    """Set throttle percentage for a system"""
    
    if percent < 0 or percent > 100:
        raise HTTPException(status_code=400, detail="Percent must be between 0 and 100")
    
    control_plane.active_throttles[target] = percent
    
    # Send throttle command
    from services.kafka_bridge import get_kafka_bridge
    bridge = await get_kafka_bridge()
    
    await bridge.send_command(
        "control.throttle",
        {
            "target": target,
            "percent": percent,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    return {
        "success": True,
        "message": f"Throttle set to {percent}% for {target}"
    }


@router.get("/system-status")
async def get_system_status(
    current_user: TokenData = Depends(require_role(UserRole.VIEWER))
) -> Dict[str, Any]:
    """Get overall system status"""
    
    from services.clickhouse_client import get_clickhouse_pool
    from services.kafka_bridge import get_kafka_bridge
    
    # Get latest metrics
    pool = await get_clickhouse_pool()
    
    try:
        metrics_query = """
            SELECT 
                avg(latency_p50_ms) as avg_p50,
                avg(latency_p99_ms) as avg_p99,
                avg(bundle_land_rate) as avg_land_rate,
                avg(ingestion_rate) as avg_ingestion,
                max(timestamp) as latest_update
            FROM mev.system_metrics
            WHERE timestamp >= now() - INTERVAL 5 MINUTE
        """
        
        data, _ = await pool.execute_query(metrics_query, use_cache=False)
        metrics = data[0] if data else {}
        
    except Exception:
        metrics = {}
    
    # Get Kafka lag
    bridge = await get_kafka_bridge()
    lag = await bridge.get_consumer_lag()
    
    # Check SLOs
    slo_violations = []
    
    if metrics.get("avg_p50") and metrics["avg_p50"] > 8:
        slo_violations.append("P50 latency exceeds 8ms")
    
    if metrics.get("avg_p99") and metrics["avg_p99"] > 20:
        slo_violations.append("P99 latency exceeds 20ms")
    
    if metrics.get("avg_land_rate") and metrics["avg_land_rate"] < 0.65:
        slo_violations.append("Bundle land rate below 65%")
    
    return {
        "success": True,
        "status": "healthy" if not slo_violations else "degraded",
        "metrics": metrics,
        "kafka_lag": sum(lag.values()),
        "active_kill_switches": len(control_plane.get_active_kill_switches()),
        "active_throttles": control_plane.active_throttles,
        "slo_violations": slo_violations,
        "timestamp": datetime.now().isoformat()
    }