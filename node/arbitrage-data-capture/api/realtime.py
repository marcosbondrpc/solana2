"""
Real-time WebSocket API with Kafka bridge
Ultra-low-latency streaming with micro-batching and backpressure
"""

import os
import time
import json
import asyncio
import zstandard as zstd
from typing import Dict, Set, List, Optional, Any
from datetime import datetime
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import HTMLResponse
import aiokafka
from aiokafka import AIOKafkaConsumer
from google.protobuf.json_format import MessageToJson

from .deps import verify_api_key, User
from .proto_gen import realtime_pb2


router = APIRouter()

# Global state
active_connections: Set[WebSocket] = set()
connection_lock = asyncio.Lock()

# Kafka consumers (one per topic)
kafka_consumers: Dict[str, AIOKafkaConsumer] = {}
consumer_lock = asyncio.Lock()

# Compression
compressor = zstd.ZstdCompressor(level=3, threads=2)
decompressor = zstd.ZstdDecompressor()

# Micro-batching configuration
BATCH_WINDOW_MS = int(os.getenv("BATCH_WINDOW_MS", "15"))  # 10-25ms window
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "256"))  # Max messages per batch
BACKPRESSURE_THRESHOLD = int(os.getenv("BACKPRESSURE_THRESHOLD", "1000"))  # Queue size threshold


class ConnectionManager:
    """Manages WebSocket connections with backpressure"""
    
    def __init__(self, websocket: WebSocket, mode: str = "proto"):
        self.websocket = websocket
        self.mode = mode
        self.queue = deque(maxlen=BACKPRESSURE_THRESHOLD)
        self.batch_buffer = []
        self.last_batch_time = time.perf_counter()
        self.stats = {
            "messages_sent": 0,
            "batches_sent": 0,
            "messages_dropped": 0,
            "bytes_sent": 0
        }
        self.closed = False
    
    async def send_batch(self):
        """Send accumulated batch to client"""
        if not self.batch_buffer or self.closed:
            return
        
        try:
            if self.mode == "proto":
                # Create protobuf batch
                batch = realtime_pb2.Batch()
                batch.batch_id = time.time_ns()
                batch.created_at_ns = time.time_ns()
                batch.compression_type = 1  # zstd
                
                for msg in self.batch_buffer:
                    envelope = batch.envelopes.add()
                    envelope.CopyFrom(msg)
                
                batch.batch_size = len(self.batch_buffer)
                
                # Serialize and compress
                data = batch.SerializeToString()
                compressed = compressor.compress(data)
                
                # Send binary frame
                await self.websocket.send_bytes(compressed)
                
                self.stats["bytes_sent"] += len(compressed)
                
            else:  # JSON mode
                # Send as NDJSON
                for msg in self.batch_buffer:
                    json_line = json.dumps(MessageToJson(msg, preserving_proto_field_name=True))
                    await self.websocket.send_text(json_line + "\n")
            
            self.stats["batches_sent"] += 1
            self.stats["messages_sent"] += len(self.batch_buffer)
            self.batch_buffer.clear()
            self.last_batch_time = time.perf_counter()
            
        except Exception as e:
            print(f"Error sending batch: {e}")
            self.closed = True
    
    async def queue_message(self, message: realtime_pb2.Envelope):
        """Queue message for batching"""
        if self.closed:
            return
        
        # Check backpressure
        if len(self.queue) >= BACKPRESSURE_THRESHOLD:
            # Drop oldest message
            self.queue.popleft()
            self.stats["messages_dropped"] += 1
        
        # Add to queue
        self.queue.append(message)
        
        # Process queue into batch
        while self.queue and len(self.batch_buffer) < BATCH_MAX_SIZE:
            self.batch_buffer.append(self.queue.popleft())
        
        # Check if batch should be sent
        elapsed_ms = (time.perf_counter() - self.last_batch_time) * 1000
        if len(self.batch_buffer) >= BATCH_MAX_SIZE or elapsed_ms >= BATCH_WINDOW_MS:
            await self.send_batch()
    
    async def close(self):
        """Close connection and cleanup"""
        self.closed = True
        if self.batch_buffer:
            await self.send_batch()


async def get_kafka_consumer(topic: str) -> AIOKafkaConsumer:
    """Get or create Kafka consumer for topic"""
    async with consumer_lock:
        if topic not in kafka_consumers:
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=os.getenv("KAFKA_BROKERS", "localhost:9092"),
                group_id=f"ws-bridge-{topic}",
                # Performance settings
                fetch_min_bytes=1,  # Don't wait for batching
                fetch_max_wait_ms=10,  # Max 10ms wait
                max_poll_records=500,  # Fetch many records at once
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                auto_offset_reset="latest",  # Start from latest
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                # Deserialization
                value_deserializer=lambda m: m  # Raw bytes
            )
            await consumer.start()
            kafka_consumers[topic] = consumer
    
    return kafka_consumers[topic]


async def kafka_rx_loop(manager: ConnectionManager, topics: List[str]):
    """
    Kafka receive loop with micro-batching
    Consumes from Kafka and sends to WebSocket with batching
    """
    consumers = []
    try:
        # Start consumers for all topics
        for topic in topics:
            consumer = await get_kafka_consumer(topic)
            consumers.append(consumer)
        
        # Batch timer task
        async def batch_timer():
            while not manager.closed:
                await asyncio.sleep(BATCH_WINDOW_MS / 1000.0)
                if manager.batch_buffer:
                    await manager.send_batch()
        
        timer_task = asyncio.create_task(batch_timer())
        
        # Consume messages
        while not manager.closed:
            for consumer in consumers:
                try:
                    # Poll with timeout
                    records = await consumer.getmany(timeout_ms=100, max_records=100)
                    
                    for topic_partition, messages in records.items():
                        for msg in messages:
                            # Parse protobuf message
                            envelope = realtime_pb2.Envelope()
                            
                            if msg.value.startswith(b'{'):  # JSON fallback
                                # Parse JSON to protobuf
                                data = json.loads(msg.value)
                                envelope.timestamp_ns = data.get("timestamp_ns", time.time_ns())
                                envelope.stream_id = data.get("stream_id", topic_partition.topic)
                                envelope.sequence = data.get("sequence", msg.offset)
                                envelope.type = data.get("type", "unknown")
                                envelope.payload = json.dumps(data.get("payload", {})).encode()
                            else:
                                # Parse protobuf directly
                                envelope.ParseFromString(msg.value)
                            
                            # Queue for batching
                            await manager.queue_message(envelope)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error consuming from Kafka: {e}")
                    await asyncio.sleep(1)
        
        timer_task.cancel()
        
    except Exception as e:
        print(f"Kafka RX loop error: {e}")
    finally:
        # Don't close shared consumers here
        pass


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    mode: str = Query("proto", description="Transport mode: json or proto"),
    topics: str = Query("mev-opportunities-proto,arbitrage-opportunities-proto", description="Comma-separated topic list"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    """
    WebSocket endpoint with micro-batching and backpressure
    
    Supports both JSON and Protobuf modes with Zstd compression
    Implements 10-25ms micro-batching with 256 message cap
    """
    await websocket.accept()
    
    # Verify authentication
    if token:
        # Verify JWT or API key
        pass  # Implement based on your auth system
    
    # Create connection manager
    manager = ConnectionManager(websocket, mode=mode)
    
    # Add to active connections
    async with connection_lock:
        active_connections.add(websocket)
    
    # Parse topics
    topic_list = [t.strip() for t in topics.split(",")]
    
    # Start Kafka receive loop
    kafka_task = asyncio.create_task(kafka_rx_loop(manager, topic_list))
    
    try:
        # Handle incoming messages (commands, ping/pong, etc.)
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle control messages
                if message == "ping":
                    await websocket.send_text("pong")
                elif message == "stats":
                    await websocket.send_json(manager.stats)
                elif message.startswith("subscribe:"):
                    # Dynamic subscription (implement if needed)
                    pass
                    
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_text("ping")
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup
        manager.closed = True
        kafka_task.cancel()
        
        async with connection_lock:
            active_connections.discard(websocket)
        
        await manager.close()
        await websocket.close()


@router.get("/connections")
async def get_connections():
    """Get active WebSocket connection count"""
    return {
        "active_connections": len(active_connections),
        "consumers": len(kafka_consumers),
        "topics": list(kafka_consumers.keys())
    }


@router.get("/")
async def websocket_test_page():
    """Simple test page for WebSocket"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MEV WebSocket Test</title>
    </head>
    <body>
        <h1>MEV Real-time WebSocket</h1>
        <div id="messages"></div>
        <script>
            const ws = new WebSocket("ws://localhost:8000/api/realtime/ws?mode=json");
            
            ws.onopen = () => {
                console.log("Connected");
                document.getElementById("messages").innerHTML += "<p>Connected</p>";
            };
            
            ws.onmessage = (event) => {
                console.log("Message:", event.data);
                document.getElementById("messages").innerHTML += "<p>" + event.data + "</p>";
            };
            
            ws.onerror = (error) => {
                console.error("Error:", error);
            };
            
            ws.onclose = () => {
                console.log("Disconnected");
                document.getElementById("messages").innerHTML += "<p>Disconnected</p>";
            };
            
            // Send ping every 10 seconds
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send("ping");
                }
            }, 10000);
        </script>
    </body>
    </html>
    """)


async def start_realtime_server():
    """Start real-time server background tasks"""
    print("🚀 Real-time server started")
    print(f"📊 Micro-batching: {BATCH_WINDOW_MS}ms window, {BATCH_MAX_SIZE} max size")
    print(f"🔒 Backpressure threshold: {BACKPRESSURE_THRESHOLD}")
    
    # Start any background tasks
    while True:
        await asyncio.sleep(60)
        # Periodic cleanup
        async with connection_lock:
            print(f"Active connections: {len(active_connections)}")


# Cleanup on shutdown
async def cleanup():
    """Clean up Kafka consumers"""
    async with consumer_lock:
        for consumer in kafka_consumers.values():
            await consumer.stop()
        kafka_consumers.clear()