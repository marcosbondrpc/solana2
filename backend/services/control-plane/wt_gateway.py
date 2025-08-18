"""
WebTransport Gateway: QUIC/HTTP3 for ultra-low latency
Lossy link optimization with datagram support
"""

import os
import time
import asyncio
import json
from typing import Dict, Set, Optional, Any
from datetime import datetime
from collections import deque

try:
    from aioquic.asyncio import QuicConnectionProtocol, serve
    from aioquic.quic.configuration import QuicConfiguration
    from aioquic.quic.events import DatagramFrameReceived, StreamDataReceived
    from aioquic.h3.connection import H3_ALPN, H3Connection
    from aioquic.h3.events import H3Event, HeadersReceived, DataReceived, DatagramReceived
    from aioquic.h3.exceptions import NoAvailablePushIDError
    from aioquic.tls import SessionTicket
    WEBTRANSPORT_AVAILABLE = True
except ImportError:
    WEBTRANSPORT_AVAILABLE = False
    print("⚠️ WebTransport not available. Install aioquic: pip install aioquic")

import zstandard as zstd
from proto_gen import realtime_pb2


# Compression
compressor = zstd.ZstdCompressor(level=3, threads=2)
decompressor = zstd.ZstdDecompressor()

# Active sessions
wt_sessions: Dict[str, 'WebTransportSession'] = {}
session_lock = asyncio.Lock()


class WebTransportSession:
    """WebTransport session handler"""
    
    def __init__(self, session_id: str, h3: H3Connection, stream_id: int):
        self.session_id = session_id
        self.h3 = h3
        self.stream_id = stream_id
        self.authenticated = False
        self.user_id: Optional[str] = None
        self.subscriptions: Set[str] = set()
        self.queue = deque(maxlen=1000)
        self.stats = {
            "datagrams_sent": 0,
            "datagrams_dropped": 0,
            "bytes_sent": 0,
            "latency_us_avg": 0
        }
        self.closed = False
    
    async def send_datagram(self, data: bytes):
        """Send datagram with best-effort delivery"""
        if self.closed:
            return
        
        try:
            # Datagrams are unreliable but fast
            self.h3.send_datagram(self.stream_id, data)
            self.stats["datagrams_sent"] += 1
            self.stats["bytes_sent"] += len(data)
        except Exception as e:
            print(f"Failed to send datagram: {e}")
            self.stats["datagrams_dropped"] += 1
    
    async def send_stream(self, data: bytes):
        """Send data over reliable stream"""
        if self.closed:
            return
        
        try:
            # Streams are reliable but may have higher latency
            self.h3.send_data(self.stream_id, data, end_stream=False)
            self.stats["bytes_sent"] += len(data)
        except Exception as e:
            print(f"Failed to send stream data: {e}")
    
    async def close(self):
        """Close session"""
        self.closed = True
        if self.h3:
            self.h3.send_data(self.stream_id, b"", end_stream=True)


class WebTransportProtocol(QuicConnectionProtocol):
    """WebTransport protocol handler"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h3: Optional[H3Connection] = None
        self.sessions: Dict[int, WebTransportSession] = {}
    
    def quic_event_received(self, event):
        """Handle QUIC events"""
        
        if isinstance(event, DatagramFrameReceived):
            # Handle datagram
            asyncio.create_task(self.handle_datagram(event.data))
        
        elif isinstance(event, StreamDataReceived):
            # Pass to H3
            if self.h3:
                for h3_event in self.h3.handle_event(event):
                    asyncio.create_task(self.handle_h3_event(h3_event))
    
    async def handle_h3_event(self, event: H3Event):
        """Handle HTTP/3 events"""
        
        if isinstance(event, HeadersReceived):
            # New WebTransport session request
            headers = dict(event.headers)
            
            if headers.get(b":method") == b"CONNECT" and \
               headers.get(b":protocol") == b"webtransport":
                # Accept WebTransport session
                session_id = f"wt_{time.time_ns()}"
                session = WebTransportSession(session_id, self.h3, event.stream_id)
                
                self.sessions[event.stream_id] = session
                
                async with session_lock:
                    wt_sessions[session_id] = session
                
                # Send acceptance headers
                self.h3.send_headers(
                    event.stream_id,
                    [(b":status", b"200")]
                )
                
                print(f"✅ WebTransport session established: {session_id}")
        
        elif isinstance(event, DataReceived):
            # Handle stream data
            if event.stream_id in self.sessions:
                session = self.sessions[event.stream_id]
                await self.handle_session_data(session, event.data)
        
        elif isinstance(event, DatagramReceived):
            # Handle datagram from client
            await self.handle_datagram(event.data)
    
    async def handle_session_data(self, session: WebTransportSession, data: bytes):
        """Handle data from WebTransport session"""
        
        try:
            # Parse control message
            message = json.loads(data.decode())
            
            if message.get("type") == "auth":
                # Authenticate session
                token = message.get("token")
                if token:  # Validate token
                    session.authenticated = True
                    session.user_id = message.get("user_id", "anonymous")
                    
                    # Send confirmation
                    response = json.dumps({"type": "auth_ok"}).encode()
                    await session.send_stream(response)
            
            elif message.get("type") == "subscribe":
                # Subscribe to topics
                topics = message.get("topics", [])
                session.subscriptions.update(topics)
                
                # Start streaming data
                asyncio.create_task(self.stream_to_session(session))
            
            elif message.get("type") == "ping":
                # Respond to ping
                pong = json.dumps({"type": "pong", "timestamp": time.time_ns()}).encode()
                await session.send_datagram(pong)
            
        except Exception as e:
            print(f"Error handling session data: {e}")
    
    async def handle_datagram(self, data: bytes):
        """Handle incoming datagram"""
        # Process high-frequency data like heartbeats
        pass
    
    async def stream_to_session(self, session: WebTransportSession):
        """Stream data to WebTransport session"""
        
        from .realtime import get_kafka_consumer
        
        # Get Kafka consumers for subscribed topics
        consumers = []
        for topic in session.subscriptions:
            consumer = await get_kafka_consumer(topic)
            consumers.append(consumer)
        
        batch = []
        last_send = time.perf_counter()
        
        while not session.closed:
            try:
                # Collect messages from Kafka
                for consumer in consumers:
                    records = await consumer.getmany(timeout_ms=10, max_records=10)
                    
                    for topic_partition, messages in records.items():
                        for msg in messages:
                            # Parse message
                            envelope = realtime_pb2.Envelope()
                            if msg.value.startswith(b'{'):
                                # JSON to proto conversion
                                data = json.loads(msg.value)
                                envelope.timestamp_ns = time.time_ns()
                                envelope.stream_id = topic_partition.topic
                                envelope.sequence = msg.offset
                                envelope.type = "data"
                                envelope.payload = msg.value
                            else:
                                envelope.ParseFromString(msg.value)
                            
                            batch.append(envelope)
                
                # Send batch if ready
                elapsed = time.perf_counter() - last_send
                if batch and (len(batch) >= 10 or elapsed > 0.01):  # 10ms or 10 messages
                    # Create batch message
                    batch_msg = realtime_pb2.Batch()
                    batch_msg.batch_id = time.time_ns()
                    batch_msg.created_at_ns = time.time_ns()
                    batch_msg.compression_type = 1  # zstd
                    
                    for envelope in batch:
                        batch_msg.envelopes.add().CopyFrom(envelope)
                    
                    batch_msg.batch_size = len(batch)
                    
                    # Serialize and compress
                    data = batch_msg.SerializeToString()
                    compressed = compressor.compress(data)
                    
                    # Send as datagram for low latency (unreliable)
                    # or as stream for reliability
                    if len(compressed) < 1200:  # MTU safe for datagram
                        await session.send_datagram(compressed)
                    else:
                        await session.send_stream(compressed)
                    
                    batch.clear()
                    last_send = time.perf_counter()
                
                # Small delay to prevent busy loop
                await asyncio.sleep(0.001)
                
            except Exception as e:
                print(f"Error streaming to session: {e}")
                await asyncio.sleep(1)


async def run_wt_gateway(
    host: str = "0.0.0.0",
    port: int = 4433,
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None
):
    """Run WebTransport gateway server"""
    
    if not WEBTRANSPORT_AVAILABLE:
        print("❌ WebTransport not available. Please install aioquic.")
        return
    
    # Configure QUIC
    configuration = QuicConfiguration(
        alpn_protocols=H3_ALPN,
        is_client=False,
        max_datagram_frame_size=65536,
    )
    
    # Load certificates
    if cert_file and key_file:
        configuration.load_cert_chain(cert_file, key_file)
    else:
        # Generate self-signed certificate for development
        from aioquic.tls import generate_certificate
        cert, key = generate_certificate(host)
        configuration.load_cert_chain(cert, key)
    
    # Start server
    print(f"🚀 Starting WebTransport gateway on {host}:{port}")
    print("📡 QUIC/HTTP3 with datagram support enabled")
    print("⚡ Ultra-low latency mode active")
    
    await serve(
        host,
        port,
        configuration=configuration,
        create_protocol=WebTransportProtocol,
        session_ticket_fetcher=lambda x: None,  # Disable session tickets for now
        session_ticket_handler=lambda x: None,
        retry=True
    )


async def start_wt_gateway():
    """Start WebTransport gateway as background task"""
    
    host = os.getenv("WT_HOST", "0.0.0.0")
    port = int(os.getenv("WT_PORT", "4433"))
    cert_file = os.getenv("WT_CERT")
    key_file = os.getenv("WT_KEY")
    
    try:
        await run_wt_gateway(host, port, cert_file, key_file)
    except Exception as e:
        print(f"WebTransport gateway error: {e}")


# API endpoints for WebTransport management
from fastapi import APIRouter

wt_router = APIRouter()


@wt_router.get("/sessions")
async def get_wt_sessions():
    """Get active WebTransport sessions"""
    
    async with session_lock:
        sessions = []
        for session_id, session in wt_sessions.items():
            sessions.append({
                "session_id": session_id,
                "authenticated": session.authenticated,
                "user_id": session.user_id,
                "subscriptions": list(session.subscriptions),
                "stats": session.stats
            })
    
    return {"sessions": sessions, "count": len(sessions)}


@wt_router.delete("/sessions/{session_id}")
async def close_wt_session(session_id: str):
    """Close WebTransport session"""
    
    async with session_lock:
        if session_id in wt_sessions:
            session = wt_sessions[session_id]
            await session.close()
            del wt_sessions[session_id]
            return {"status": "closed", "session_id": session_id}
    
    return {"error": "Session not found"}


# Cleanup
async def cleanup():
    """Clean up WebTransport sessions"""
    async with session_lock:
        for session in wt_sessions.values():
            await session.close()
        wt_sessions.clear()