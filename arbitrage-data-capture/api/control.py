"""
Control Plane API: Command publishing with Ed25519 signatures
Ultra-low-latency Kafka publishing with protobuf support
"""

import os
import time
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
import aiokafka
from aiokafka import AIOKafkaProducer
import nacl.signing
import nacl.encoding
from google.protobuf.json_format import MessageToJson, Parse

from .deps import User, get_current_user, require_permission, audit_log
from .proto_gen import control_pb2


router = APIRouter()

# Global Kafka producer (singleton)
kafka_producer: Optional[AIOKafkaProducer] = None
kafka_lock = asyncio.Lock()

# Ed25519 signing key
signing_key: Optional[nacl.signing.SigningKey] = None


class CommandRequest(BaseModel):
    """Control command request"""
    module: str = Field(..., description="Target module")
    action: str = Field(..., description="Action to perform")
    params: Dict[str, Any] = Field(default_factory=dict, description="Command parameters")
    priority: int = Field(default=0, description="Command priority (0=normal, 1=high, 2=critical)")


class PolicyUpdateRequest(BaseModel):
    """Policy update request"""
    policy_id: str
    policy_type: str
    thresholds: Dict[str, float] = Field(default_factory=dict)
    rules: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    effective_from: Optional[int] = None


class ModelSwapRequest(BaseModel):
    """Model swap request"""
    model_id: str
    model_path: str
    model_type: str = "xgboost"
    version: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class KillSwitchRequest(BaseModel):
    """Kill switch request"""
    target: str
    reason: str
    duration_ms: int = 5000
    force: bool = False


class ConfigUpdateRequest(BaseModel):
    """Configuration update request"""
    config_key: str
    config_value: str
    config_type: str = "string"
    hot_reload: bool = True


async def get_kafka_producer() -> AIOKafkaProducer:
    """Get or create Kafka producer with ultra-low latency settings"""
    global kafka_producer
    
    async with kafka_lock:
        if kafka_producer is None:
            kafka_producer = AIOKafkaProducer(
                bootstrap_servers=os.getenv("KAFKA_BROKERS", "localhost:9092"),
                # Performance optimizations
                linger_ms=0,  # Send immediately (no batching delay)
                acks=1,  # Leader acknowledgment only for speed
                compression_type="lz4",  # Fast compression
                max_batch_size=16384,
                max_request_size=1048576,
                request_timeout_ms=10000,
                retry_backoff_ms=10,
                # Connection pooling
                connections_max_idle_ms=60000,
                metadata_max_age_ms=60000,
                # Idempotence for exactly-once semantics
                enable_idempotence=True,
                max_in_flight_requests_per_connection=5,
                # Custom partitioner for load balancing
                partitioner=lambda key, all_parts, available_parts: \
                    hash(key) % len(available_parts) if available_parts else 0
            )
            await kafka_producer.start()
    
    return kafka_producer


def get_signing_key() -> nacl.signing.SigningKey:
    """Get Ed25519 signing key"""
    global signing_key
    
    if signing_key is None:
        # Load from environment or generate
        key_hex = os.getenv("CTRL_SIGN_SK_HEX")
        if key_hex:
            signing_key = nacl.signing.SigningKey(
                bytes.fromhex(key_hex),
                encoder=nacl.encoding.RawEncoder
            )
        else:
            # Generate new key (for development)
            signing_key = nacl.signing.SigningKey.generate()
            print(f"Generated signing key: {signing_key.encode(nacl.encoding.HexEncoder).decode()}")
            print(f"Public key: {signing_key.verify_key.encode(nacl.encoding.HexEncoder).decode()}")
    
    return signing_key


def sign_command(command: control_pb2.Command) -> bytes:
    """Sign command with Ed25519"""
    key = get_signing_key()
    
    # Create signing payload (excluding signature field)
    payload = f"{command.id}:{command.module}:{command.action}:{command.nonce}:{command.timestamp_ns}"
    
    # Sign the payload
    signed = key.sign(payload.encode())
    
    return signed.signature


@router.post("/command", dependencies=[Depends(require_permission("control:write"))])
async def publish_command(
    request: CommandRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    req: Request = None
) -> Dict[str, Any]:
    """
    Publish control command to Kafka with Ed25519 signature
    
    Ultra-low-latency publishing with sub-millisecond response times
    """
    start_ns = time.perf_counter_ns()
    
    # Create protobuf command
    command = control_pb2.Command()
    command.id = f"cmd_{time.time_ns()}"
    command.module = request.module
    command.action = request.action
    command.nonce = time.time_ns()
    command.timestamp_ns = time.time_ns()
    
    # Add parameters
    for key, value in request.params.items():
        command.params[key] = str(value)
    
    # Sign the command
    key = get_signing_key()
    command.pubkey_id = os.getenv("CTRL_PUBKEY_ID", "default")
    command.signature = sign_command(command)
    
    # Get Kafka producer
    producer = await get_kafka_producer()
    
    # Determine topic based on priority
    topic = "control-commands-proto"
    if request.priority == 2:
        topic = "control-commands-critical"
    elif request.priority == 1:
        topic = "control-commands-high"
    
    # Publish to Kafka with module-based partitioning
    try:
        # Send with key for ordering guarantees
        future = await producer.send(
            topic,
            value=command.SerializeToString(),
            key=request.module.encode(),
            timestamp_ms=int(time.time() * 1000)
        )
        
        # Wait for acknowledgment
        metadata = await future
        
        # Calculate latency
        latency_ns = time.perf_counter_ns() - start_ns
        latency_us = latency_ns / 1000
        
        # Audit log in background
        background_tasks.add_task(
            audit_log,
            "control_command",
            user,
            {
                "command_id": command.id,
                "module": request.module,
                "action": request.action,
                "params": request.params,
                "ip": req.client.host if req and req.client else "unknown"
            }
        )
        
        return {
            "success": True,
            "command_id": command.id,
            "topic": metadata.topic,
            "partition": metadata.partition,
            "offset": metadata.offset,
            "timestamp": metadata.timestamp,
            "latency_us": latency_us,
            "signature": command.signature.hex()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish command: {str(e)}")


@router.post("/policy", dependencies=[Depends(require_permission("control:write"))])
async def update_policy(
    request: PolicyUpdateRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update system policy with immediate effect"""
    
    # Create policy update command
    policy = control_pb2.PolicyUpdate()
    policy.policy_id = request.policy_id
    policy.policy_type = request.policy_type
    policy.enabled = request.enabled
    
    if request.effective_from:
        policy.effective_from = request.effective_from
    else:
        policy.effective_from = int(time.time() * 1000)
    
    # Add thresholds and rules
    for key, value in request.thresholds.items():
        policy.thresholds[key] = value
    for key, value in request.rules.items():
        policy.rules[key] = value
    
    # Wrap in command
    command = control_pb2.Command()
    command.id = f"policy_{time.time_ns()}"
    command.module = "policy_manager"
    command.action = "update_policy"
    command.nonce = time.time_ns()
    command.timestamp_ns = time.time_ns()
    command.params["policy_data"] = policy.SerializeToString().hex()
    
    # Sign and publish
    command.pubkey_id = os.getenv("CTRL_PUBKEY_ID", "default")
    command.signature = sign_command(command)
    
    producer = await get_kafka_producer()
    future = await producer.send(
        "control-commands-proto",
        value=command.SerializeToString(),
        key=b"policy_manager"
    )
    
    metadata = await future
    
    # Audit log
    background_tasks.add_task(
        audit_log,
        "policy_update",
        user,
        {
            "policy_id": request.policy_id,
            "policy_type": request.policy_type,
            "enabled": request.enabled
        }
    )
    
    return {
        "success": True,
        "command_id": command.id,
        "policy_id": request.policy_id,
        "offset": metadata.offset
    }


@router.post("/model-swap", dependencies=[Depends(require_permission("control:write"))])
async def swap_model(
    request: ModelSwapRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Hot-swap ML model without restart"""
    
    # Create model swap command
    swap = control_pb2.ModelSwap()
    swap.model_id = request.model_id
    swap.model_path = request.model_path
    swap.model_type = request.model_type
    swap.version = request.version
    
    for key, value in request.metadata.items():
        swap.metadata[key] = value
    
    # Wrap in command
    command = control_pb2.Command()
    command.id = f"model_{time.time_ns()}"
    command.module = request.model_type
    command.action = "swap_model"
    command.nonce = time.time_ns()
    command.timestamp_ns = time.time_ns()
    command.params["model_data"] = swap.SerializeToString().hex()
    
    # Sign and publish
    command.pubkey_id = os.getenv("CTRL_PUBKEY_ID", "default")
    command.signature = sign_command(command)
    
    producer = await get_kafka_producer()
    future = await producer.send(
        "control-commands-proto",
        value=command.SerializeToString(),
        key=request.model_type.encode()
    )
    
    metadata = await future
    
    # Audit log
    background_tasks.add_task(
        audit_log,
        "model_swap",
        user,
        {
            "model_id": request.model_id,
            "version": request.version,
            "path": request.model_path
        }
    )
    
    return {
        "success": True,
        "command_id": command.id,
        "model_id": request.model_id,
        "version": request.version,
        "offset": metadata.offset
    }


@router.post("/kill-switch", dependencies=[Depends(require_permission("control:write"))])
async def activate_kill_switch(
    request: KillSwitchRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Activate emergency kill switch"""
    
    # Create kill switch command
    kill = control_pb2.KillSwitch()
    kill.target = request.target
    kill.reason = request.reason
    kill.duration_ms = request.duration_ms
    kill.force = request.force
    
    # Wrap in critical command
    command = control_pb2.Command()
    command.id = f"kill_{time.time_ns()}"
    command.module = request.target
    command.action = "kill_switch"
    command.nonce = time.time_ns()
    command.timestamp_ns = time.time_ns()
    command.params["kill_data"] = kill.SerializeToString().hex()
    
    # Sign and publish to critical topic
    command.pubkey_id = os.getenv("CTRL_PUBKEY_ID", "default")
    command.signature = sign_command(command)
    
    producer = await get_kafka_producer()
    future = await producer.send(
        "control-commands-critical",  # High priority topic
        value=command.SerializeToString(),
        key=request.target.encode()
    )
    
    metadata = await future
    
    # Audit log
    background_tasks.add_task(
        audit_log,
        "kill_switch",
        user,
        {
            "target": request.target,
            "reason": request.reason,
            "duration_ms": request.duration_ms,
            "force": request.force
        }
    )
    
    return {
        "success": True,
        "command_id": command.id,
        "target": request.target,
        "activated_at": datetime.utcnow().isoformat(),
        "duration_ms": request.duration_ms,
        "offset": metadata.offset
    }


@router.post("/config", dependencies=[Depends(require_permission("control:write"))])
async def update_config(
    request: ConfigUpdateRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update system configuration with hot reload"""
    
    # Create config update command
    config = control_pb2.ConfigUpdate()
    config.config_key = request.config_key
    config.config_value = request.config_value
    config.config_type = request.config_type
    config.hot_reload = request.hot_reload
    
    # Wrap in command
    command = control_pb2.Command()
    command.id = f"config_{time.time_ns()}"
    command.module = "config_manager"
    command.action = "update_config"
    command.nonce = time.time_ns()
    command.timestamp_ns = time.time_ns()
    command.params["config_data"] = config.SerializeToString().hex()
    
    # Sign and publish
    command.pubkey_id = os.getenv("CTRL_PUBKEY_ID", "default")
    command.signature = sign_command(command)
    
    producer = await get_kafka_producer()
    future = await producer.send(
        "control-commands-proto",
        value=command.SerializeToString(),
        key=b"config_manager"
    )
    
    metadata = await future
    
    # Audit log
    background_tasks.add_task(
        audit_log,
        "config_update",
        user,
        {
            "key": request.config_key,
            "value": request.config_value,
            "hot_reload": request.hot_reload
        }
    )
    
    return {
        "success": True,
        "command_id": command.id,
        "config_key": request.config_key,
        "hot_reload": request.hot_reload,
        "offset": metadata.offset
    }


@router.get("/status/{command_id}")
async def get_command_status(
    command_id: str,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get command execution status"""
    
    # In production, query from ClickHouse or state store
    # For now, return mock status
    return {
        "command_id": command_id,
        "status": "executed",
        "executed_at": datetime.utcnow().isoformat(),
        "results": {
            "affected_modules": 1,
            "execution_time_ms": 2.3
        }
    }


@router.get("/audit-log", dependencies=[Depends(require_permission("control:read"))])
async def get_audit_log(
    limit: int = 100,
    user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get recent audit log entries"""
    
    from .deps import get_redis
    redis_conn = await get_redis()
    
    if redis_conn:
        entries = await redis_conn.lrange("audit_log", 0, limit - 1)
        return [json.loads(entry) for entry in entries]
    
    return []


# Cleanup on shutdown
async def cleanup():
    """Clean up Kafka producer"""
    global kafka_producer
    if kafka_producer:
        await kafka_producer.stop()
        kafka_producer = None