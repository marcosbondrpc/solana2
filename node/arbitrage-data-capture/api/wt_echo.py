#!/usr/bin/env python3
"""
LEGENDARY WebTransport Echo Server with H3/QUIC Datagrams
Ultra-low latency bidirectional communication for MEV infrastructure
Supports both datagram and stream modes with nanosecond timestamp precision
"""

import asyncio
import os
import struct
import time
import json
import hashlib
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass, field
from collections import deque
import signal

try:
    from aioquic.asyncio import serve
    from aioquic.quic.configuration import QuicConfiguration
    from aioquic.h3.connection import H3_ALPN
    from aioquic.asyncio.protocol import QuicConnectionProtocol
    from aioquic.h3.events import HeadersReceived, DatagramReceived, WebTransportStreamDataReceived
    from aioquic.h3.connection import H3Connection
    from aioquic.h3.exceptions import NoAvailablePushIDError
except ImportError:
    print("ERROR: aioquic not installed. Run: pip install aioquic")
    exit(1)

@dataclass
class SessionMetrics:
    """Per-session metrics for monitoring"""
    session_id: str
    datagrams_sent: int = 0
    datagrams_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    rtt_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_rtt_us(self) -> float:
        """Average RTT in microseconds"""
        if not self.rtt_samples:
            return 0.0
        return sum(self.rtt_samples) / len(self.rtt_samples)
    
    @property
    def p99_rtt_us(self) -> float:
        """P99 RTT in microseconds"""
        if not self.rtt_samples:
            return 0.0
        sorted_samples = sorted(self.rtt_samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples)-1)]

class WebTransportEchoProtocol(QuicConnectionProtocol):
    """
    WebTransport protocol handler with echo capabilities
    Optimized for minimum latency MEV operations
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sessions: Dict[int, SessionMetrics] = {}
        self.h3: Optional[H3Connection] = None
        self.total_datagrams = 0
        self.total_bytes = 0
        
    def quic_event_received(self, event):
        """Handle QUIC events"""
        if self.h3 is None:
            self.h3 = H3Connection(self._quic, enable_webtransport=True)
            
        # Process H3 events
        for h3_event in self.h3.handle_event(event):
            if isinstance(h3_event, HeadersReceived):
                self._handle_headers(h3_event)
            elif isinstance(h3_event, DatagramReceived):
                self._handle_datagram(h3_event)
            elif isinstance(h3_event, WebTransportStreamDataReceived):
                self._handle_stream_data(h3_event)
    
    def _handle_headers(self, event: HeadersReceived):
        """Handle HTTP/3 headers for WebTransport session establishment"""
        headers = dict(event.headers)
        
        # Check for WebTransport upgrade
        if headers.get(b':method') == b'CONNECT' and \
           headers.get(b':protocol') == b'webtransport':
            # Accept WebTransport session
            session_id = event.stream_id
            
            # Create session metrics
            self.sessions[session_id] = SessionMetrics(
                session_id=hashlib.blake3(str(session_id).encode()).hexdigest()[:16]
            )
            
            # Send 200 response to establish session
            self.h3.send_headers(
                stream_id=event.stream_id,
                headers=[
                    (b':status', b'200'),
                    (b'sec-webtransport-http3-draft', b'draft02'),
                ],
            )
            self.transmit()
            
            print(f"[WT] Session established: {self.sessions[session_id].session_id}")
    
    def _handle_datagram(self, event: DatagramReceived):
        """
        Handle incoming datagram with ultra-low latency echo
        Implements nanosecond precision timestamping
        """
        data = event.data
        session_id = event.flow_id if hasattr(event, 'flow_id') else 0
        
        # Update metrics
        if session_id in self.sessions:
            metrics = self.sessions[session_id]
            metrics.datagrams_received += 1
            metrics.bytes_received += len(data)
            metrics.last_activity = time.time()
        
        self.total_datagrams += 1
        self.total_bytes += len(data)
        
        # Decode timestamp if present (first 8 bytes)
        client_ts = None
        if len(data) >= 8:
            try:
                client_ts = struct.unpack('>Q', data[:8])[0]
            except:
                pass
        
        # Create echo response with server timestamp
        server_ts = time.time_ns()
        response = struct.pack('>Q', server_ts) + data
        
        # Send echo immediately (critical path optimization)
        try:
            self.h3.send_datagram(flow_id=session_id, data=response)
            self.transmit()
            
            # Update sent metrics
            if session_id in self.sessions:
                metrics = self.sessions[session_id]
                metrics.datagrams_sent += 1
                metrics.bytes_sent += len(response)
                
                # Calculate RTT if we have client timestamp
                if client_ts:
                    rtt_ns = server_ts - client_ts
                    metrics.rtt_samples.append(rtt_ns / 1000)  # Store in microseconds
            
        except Exception as e:
            print(f"[ERROR] Failed to send datagram: {e}")
    
    def _handle_stream_data(self, event: WebTransportStreamDataReceived):
        """Handle WebTransport stream data (for reliable delivery)"""
        # Echo stream data back
        self.h3.send_data(
            stream_id=event.stream_id,
            data=event.data,
            end_stream=event.stream_ended
        )
        self.transmit()

class WebTransportEchoServer:
    """
    Production-grade WebTransport echo server
    Designed for MEV infrastructure testing and monitoring
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 4433,
                 cert_path: str = "cert.pem", key_path: str = "key.pem"):
        self.host = host
        self.port = port
        self.cert_path = cert_path
        self.key_path = key_path
        self.server = None
        self.protocols: Set[WebTransportEchoProtocol] = set()
        self.running = False
        
    async def start(self):
        """Start the WebTransport echo server"""
        
        # Configure QUIC with optimal settings for low latency
        configuration = QuicConfiguration(
            is_client=False,
            alpn_protocols=H3_ALPN,
            # Optimize for low latency
            idle_timeout=30.0,  # 30 seconds idle timeout
            max_datagram_size=65536,  # Maximum datagram size
            # Enable 0-RTT for returning clients
            server_name=self.host,
        )
        
        # Load TLS certificate and key
        configuration.load_cert_chain(self.cert_path, self.key_path)
        
        # Enable datagrams (critical for low-latency MEV)
        configuration.max_datagram_frame_size = 65536
        
        print(f"[WT-ECHO] Starting WebTransport Echo Server on {self.host}:{self.port}")
        
        # Create and start server
        self.server = await serve(
            self.host,
            self.port,
            configuration=configuration,
            create_protocol=self._create_protocol,
        )
        
        self.running = True
        print(f"[WT-ECHO] Server running - Ready for connections")
        print(f"[WT-ECHO] Test URL: https://{self.host}:{self.port}/echo")
        
        # Start metrics reporter
        asyncio.create_task(self._report_metrics())
        
    def _create_protocol(self, *args, **kwargs):
        """Factory for creating protocol instances"""
        protocol = WebTransportEchoProtocol(*args, **kwargs)
        self.protocols.add(protocol)
        return protocol
        
    async def _report_metrics(self):
        """Periodically report server metrics"""
        while self.running:
            await asyncio.sleep(10)
            
            total_sessions = sum(len(p.sessions) for p in self.protocols)
            total_datagrams = sum(p.total_datagrams for p in self.protocols)
            total_bytes = sum(p.total_bytes for p in self.protocols)
            
            # Calculate aggregate RTT statistics
            all_rtts = []
            for protocol in self.protocols:
                for session in protocol.sessions.values():
                    all_rtts.extend(session.rtt_samples)
            
            if all_rtts:
                avg_rtt = sum(all_rtts) / len(all_rtts)
                sorted_rtts = sorted(all_rtts)
                p50_rtt = sorted_rtts[len(sorted_rtts)//2]
                p99_rtt = sorted_rtts[int(len(sorted_rtts)*0.99)]
                
                print(f"[METRICS] Sessions: {total_sessions} | "
                      f"Datagrams: {total_datagrams} | "
                      f"Bytes: {total_bytes/1024/1024:.2f}MB | "
                      f"RTT(μs) avg:{avg_rtt:.1f} p50:{p50_rtt:.1f} p99:{p99_rtt:.1f}")
            else:
                print(f"[METRICS] Sessions: {total_sessions} | "
                      f"Datagrams: {total_datagrams} | "
                      f"Bytes: {total_bytes/1024/1024:.2f}MB")
    
    async def stop(self):
        """Gracefully stop the server"""
        print("[WT-ECHO] Shutting down server...")
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        print("[WT-ECHO] Server stopped")

async def generate_self_signed_cert():
    """Generate self-signed certificate for testing"""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import datetime
    
    # Generate private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Generate certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]),
        critical=False,
    ).sign(key, hashes.SHA256())
    
    # Write certificate
    with open("cert.pem", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    # Write private key
    with open("key.pem", "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    print("[CERT] Generated self-signed certificate: cert.pem, key.pem")

async def main():
    """Main entry point"""
    
    # Check for certificate files
    if not os.path.exists("cert.pem") or not os.path.exists("key.pem"):
        print("[CERT] Certificate files not found, generating self-signed cert...")
        try:
            import ipaddress
            await generate_self_signed_cert()
        except ImportError:
            print("[ERROR] cryptography package required for cert generation")
            print("[ERROR] Run: pip install cryptography")
            return
    
    # Create and start server
    server = WebTransportEchoServer()
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))
    
    try:
        await server.start()
        # Keep server running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        await server.stop()

if __name__ == "__main__":
    asyncio.run(main())