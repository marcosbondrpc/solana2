"""
WebSocket endpoint for real-time data streaming
"""

import asyncio
import json
from typing import Dict, Set, List, Any
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.security import HTTPBearer
import logging

from models.schemas import WebSocketMessage, WebSocketSubscription
from services.kafka_bridge import get_kafka_bridge
from security.auth import decode_token

router = APIRouter()
logger = logging.getLogger(__name__)

# Bearer token security for WebSocket
security = HTTPBearer()


class ConnectionManager:
    """Manage WebSocket connections and subscriptions"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.batch_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """Accept WebSocket connection"""
        try:
            await websocket.accept()
            self.active_connections[client_id] = websocket
            self.message_queues[client_id] = asyncio.Queue(maxsize=10000)
            
            # Update metrics
            from main import active_connections
            active_connections.set(len(self.active_connections))
            
            logger.info(f"Client {client_id} connected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to accept connection: {e}")
            return False
    
    def disconnect(self, client_id: str):
        """Remove connection and clean up"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # Clean up subscriptions
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]
            
            # Cancel batch task
            if client_id in self.batch_tasks:
                self.batch_tasks[client_id].cancel()
                del self.batch_tasks[client_id]
            
            # Clean up queue
            if client_id in self.message_queues:
                del self.message_queues[client_id]
            
            # Update metrics
            from main import active_connections
            active_connections.set(len(self.active_connections))
            
            logger.info(f"Client {client_id} disconnected")
    
    async def subscribe(self, client_id: str, topics: List[str]):
        """Subscribe client to topics"""
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = set()
        
        self.subscriptions[client_id].update(topics)
        
        logger.info(f"Client {client_id} subscribed to: {topics}")
    
    async def unsubscribe(self, client_id: str, topics: List[str]):
        """Unsubscribe client from topics"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id] -= set(topics)
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                # Convert to protobuf if needed
                if "payload" in message and isinstance(message["payload"], dict):
                    message["payload"] = json.dumps(message["payload"]).encode()
                
                await websocket.send_bytes(
                    json.dumps(message).encode() if isinstance(message, dict) else message
                )
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any]):
        """Broadcast message to all clients subscribed to topic"""
        for client_id, topics in self.subscriptions.items():
            if topic in topics:
                if client_id in self.message_queues:
                    try:
                        await self.message_queues[client_id].put(message)
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full for client {client_id}")
    
    async def start_batch_sender(
        self,
        client_id: str,
        batch_window_ms: int = 15,
        max_batch_size: int = 256
    ):
        """Start batched message sender for client"""
        
        async def batch_sender():
            """Send messages in batches"""
            queue = self.message_queues.get(client_id)
            if not queue:
                return
            
            while client_id in self.active_connections:
                try:
                    batch = []
                    deadline = asyncio.get_event_loop().time() + (batch_window_ms / 1000)
                    
                    # Collect messages until deadline or batch full
                    while len(batch) < max_batch_size:
                        remaining = deadline - asyncio.get_event_loop().time()
                        if remaining <= 0:
                            break
                        
                        try:
                            message = await asyncio.wait_for(
                                queue.get(),
                                timeout=remaining
                            )
                            batch.append(message)
                        except asyncio.TimeoutError:
                            break
                    
                    # Send batch if not empty
                    if batch:
                        await self.send_batch(client_id, batch)
                    
                except Exception as e:
                    logger.error(f"Batch sender error for {client_id}: {e}")
                    await asyncio.sleep(1)
        
        # Start batch task
        self.batch_tasks[client_id] = asyncio.create_task(batch_sender())
    
    async def send_batch(self, client_id: str, messages: List[Dict[str, Any]]):
        """Send batch of messages"""
        if client_id not in self.active_connections:
            return
        
        websocket = self.active_connections[client_id]
        
        try:
            # Create batch envelope
            batch_envelope = {
                "type": "batch",
                "count": len(messages),
                "messages": messages,
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_json(batch_envelope)
            
        except Exception as e:
            logger.error(f"Failed to send batch to {client_id}: {e}")
            self.disconnect(client_id)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(None)
):
    """
    WebSocket endpoint for real-time data streaming
    Supports protobuf messages with batching
    """
    
    # Validate token
    client_id = None
    user_info = None
    
    if token:
        try:
            payload = decode_token(token)
            client_id = payload.get("user_id", "anonymous")
            user_info = {
                "user_id": payload.get("user_id"),
                "role": payload.get("role"),
                "username": payload.get("sub")
            }
        except Exception:
            # Allow anonymous connections with limited access
            client_id = f"anonymous_{datetime.now().timestamp()}"
            user_info = {"role": "viewer"}
    else:
        client_id = f"anonymous_{datetime.now().timestamp()}"
        user_info = {"role": "viewer"}
    
    # Accept connection
    if not await manager.connect(websocket, client_id):
        return
    
    try:
        # Send welcome message
        await manager.send_message(client_id, {
            "type": "welcome",
            "client_id": client_id,
            "user_info": user_info,
            "timestamp": datetime.now().isoformat(),
            "topics_available": [
                "node.health",
                "arb.alert",
                "sandwich.alert",
                "mev.opportunity",
                "system.metrics",
                "job.progress"
            ]
        })
        
        # Start message handler
        bridge = await get_kafka_bridge()
        
        # Register Kafka message handler
        async def kafka_handler(message: Any, metadata: Dict):
            """Handle messages from Kafka"""
            topic = metadata.get("topic", "unknown")
            
            # Check if client is subscribed
            if client_id in manager.subscriptions:
                if any(topic.startswith(sub) for sub in manager.subscriptions[client_id]):
                    # Queue message for batching
                    await manager.message_queues[client_id].put({
                        "seq": metadata.get("offset", 0),
                        "topic": topic,
                        "payload": message,
                        "ts_ns": metadata.get("timestamp", 0) * 1000000,
                        "node_id": "api-server"
                    })
        
        # Register handler for all topics
        for topic in ["mev.opportunities", "mev.alerts.arbitrage", "mev.alerts.sandwich"]:
            bridge.register_handler(topic, kafka_handler)
        
        # Process client messages
        while True:
            data = await websocket.receive_json()
            
            # Handle subscription messages
            if data.get("type") == "subscribe":
                topics = data.get("topics", [])
                
                # Filter topics based on user role
                if user_info.get("role") == "viewer":
                    # Viewers can only subscribe to basic topics
                    topics = [t for t in topics if t in ["node.health", "system.metrics"]]
                
                await manager.subscribe(client_id, topics)
                
                # Start batch sender if not already running
                if client_id not in manager.batch_tasks:
                    batch_config = data.get("batch", {})
                    await manager.start_batch_sender(
                        client_id,
                        batch_config.get("window_ms", 15),
                        batch_config.get("max_size", 256)
                    )
                
                # Send confirmation
                await manager.send_message(client_id, {
                    "type": "subscribed",
                    "topics": list(manager.subscriptions.get(client_id, set())),
                    "timestamp": datetime.now().isoformat()
                })
            
            elif data.get("type") == "unsubscribe":
                topics = data.get("topics", [])
                await manager.unsubscribe(client_id, topics)
                
                # Send confirmation
                await manager.send_message(client_id, {
                    "type": "unsubscribed",
                    "topics": topics,
                    "timestamp": datetime.now().isoformat()
                })
            
            elif data.get("type") == "ping":
                # Send pong
                await manager.send_message(client_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


@router.get("/connections")
async def get_active_connections() -> Dict[str, Any]:
    """Get active WebSocket connections (admin only)"""
    
    connections = []
    for client_id in manager.active_connections:
        connections.append({
            "client_id": client_id,
            "subscriptions": list(manager.subscriptions.get(client_id, set())),
            "queue_size": manager.message_queues.get(client_id, asyncio.Queue()).qsize()
        })
    
    return {
        "success": True,
        "total_connections": len(connections),
        "connections": connections
    }