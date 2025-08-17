use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use std::os::unix::io::AsRawFd;
use tokio_uring::net::UdpSocket;
use crossbeam::channel::Sender;
use bytes::{Bytes, BytesMut};
use socket2::{Domain, Socket, Type, Protocol};
use nix::sys::socket::{setsockopt, sockopt};
use libc::{SO_TIMESTAMPING, SOF_TIMESTAMPING_RX_HARDWARE, SOF_TIMESTAMPING_RX_SOFTWARE};
use quinn::{Endpoint, ClientConfig, Connection};
use rustls::RootCertStore;
use tracing::{info, debug, error};

pub struct NetworkProcessor {
    sockets: Vec<Arc<UdpSocket>>,
    packet_tx: Sender<PacketBatch>,
    core_ids: Vec<usize>,
    ring_buffers: Vec<RingBuffer>,
}

pub struct PacketBatch {
    pub packets: Vec<Packet>,
    pub received_at: Instant,
    pub hardware_timestamp: Option<u64>,
}

pub struct Packet {
    pub data: Bytes,
    pub source: std::net::SocketAddr,
    pub timestamp: Instant,
}

struct RingBuffer {
    buffer: BytesMut,
    read_pos: usize,
    write_pos: usize,
    capacity: usize,
}

impl NetworkProcessor {
    pub async fn new(packet_tx: Sender<PacketBatch>, core_ids: Vec<usize>) -> Result<Self> {
        info!("Initializing NetworkProcessor with {} cores", core_ids.len());
        
        let mut sockets = Vec::new();
        let mut ring_buffers = Vec::new();
        
        for (idx, &core_id) in core_ids.iter().enumerate() {
            // Create socket with SO_REUSEPORT for multi-queue
            let socket = Self::create_optimized_socket(58000 + idx)?;
            
            // Pin to CPU core
            Self::pin_to_core(core_id)?;
            
            // Create ring buffer for zero-copy
            let ring = RingBuffer::new(1024 * 1024 * 16); // 16MB per ring
            
            sockets.push(Arc::new(socket));
            ring_buffers.push(ring);
        }
        
        Ok(Self {
            sockets,
            packet_tx,
            core_ids,
            ring_buffers,
        })
    }
    
    fn create_optimized_socket(port: u16) -> Result<UdpSocket> {
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        
        // Enable SO_REUSEPORT for multi-queue
        socket.set_reuse_port(true)?;
        
        // Set receive buffer to 128MB
        socket.set_recv_buffer_size(128 * 1024 * 1024)?;
        
        // Enable hardware timestamping
        let timestamping_flags = SOF_TIMESTAMPING_RX_HARDWARE | SOF_TIMESTAMPING_RX_SOFTWARE;
        unsafe {
            let optval = &timestamping_flags as *const _ as *const libc::c_void;
            libc::setsockopt(
                socket.as_raw_fd(),
                libc::SOL_SOCKET,
                SO_TIMESTAMPING,
                optval,
                std::mem::size_of_val(&timestamping_flags) as libc::socklen_t,
            );
        }
        
        // Bind to address
        let addr = format!("0.0.0.0:{}", port).parse::<std::net::SocketAddr>()?;
        socket.bind(&addr.into())?;
        
        // Convert to tokio-uring socket
        let std_socket = std::net::UdpSocket::from(socket);
        std_socket.set_nonblocking(true)?;
        
        Ok(UdpSocket::from_std(std_socket))
    }
    
    fn pin_to_core(core_id: usize) -> Result<()> {
        use nix::sched::{CpuSet, sched_setaffinity};
        use nix::unistd::Pid;
        
        let mut cpu_set = CpuSet::new();
        cpu_set.set(core_id)?;
        sched_setaffinity(Pid::from_raw(0), &cpu_set)?;
        
        Ok(())
    }
    
    pub async fn start_processing(self: Arc<Self>) -> Result<()> {
        info!("Starting network processing with io_uring");
        
        // Spawn processor for each socket/core
        for (idx, socket) in self.sockets.iter().enumerate() {
            let socket = socket.clone();
            let tx = self.packet_tx.clone();
            let core_id = self.core_ids[idx];
            
            tokio_uring::spawn(async move {
                Self::process_socket(socket, tx, core_id).await
            });
        }
        
        Ok(())
    }
    
    async fn process_socket(
        socket: Arc<UdpSocket>,
        tx: Sender<PacketBatch>,
        core_id: usize,
    ) -> Result<()> {
        // Pre-allocate buffers for zero-copy
        const BATCH_SIZE: usize = 64;
        let mut buffers: Vec<BytesMut> = (0..BATCH_SIZE)
            .map(|_| BytesMut::with_capacity(65536))
            .collect();
        
        let mut msg_vecs = vec![libc::mmsghdr::default(); BATCH_SIZE];
        let mut iovecs = vec![libc::iovec::default(); BATCH_SIZE];
        
        loop {
            let start = Instant::now();
            
            // Use recvmmsg for batch receive
            let received = unsafe {
                // Setup iovecs
                for i in 0..BATCH_SIZE {
                    iovecs[i].iov_base = buffers[i].as_mut_ptr() as *mut libc::c_void;
                    iovecs[i].iov_len = buffers[i].capacity();
                    
                    msg_vecs[i].msg_hdr.msg_iov = &mut iovecs[i];
                    msg_vecs[i].msg_hdr.msg_iovlen = 1;
                }
                
                libc::recvmmsg(
                    socket.as_raw_fd(),
                    msg_vecs.as_mut_ptr(),
                    BATCH_SIZE as u32,
                    libc::MSG_DONTWAIT,
                    std::ptr::null_mut(),
                )
            };
            
            if received > 0 {
                let mut packets = Vec::with_capacity(received as usize);
                
                for i in 0..received as usize {
                    let len = msg_vecs[i].msg_len as usize;
                    let data = buffers[i].split_to(len).freeze();
                    
                    packets.push(Packet {
                        data,
                        source: "0.0.0.0:0".parse()?, // Extract from msg_hdr
                        timestamp: Instant::now(),
                    });
                }
                
                let batch = PacketBatch {
                    packets,
                    received_at: start,
                    hardware_timestamp: Self::extract_hw_timestamp(&msg_vecs[0]),
                };
                
                // Send batch for processing
                if let Err(e) = tx.send(batch) {
                    error!("Failed to send packet batch: {}", e);
                }
                
                debug!("Core {} processed {} packets in {:?}", 
                    core_id, received, start.elapsed());
            }
            
            // Yield to prevent spinning
            tokio::task::yield_now().await;
        }
    }
    
    fn extract_hw_timestamp(msg: &libc::mmsghdr) -> Option<u64> {
        // Extract hardware timestamp from ancillary data
        // This would parse SO_TIMESTAMPING data from msg_hdr
        None // Placeholder
    }
}

impl PacketBatch {
    pub fn compute_hash(&self) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for packet in &self.packets {
            packet.data.hash(&mut hasher);
        }
        
        let hash = hasher.finish();
        let mut result = [0u8; 32];
        result[..8].copy_from_slice(&hash.to_le_bytes());
        result
    }
    
    pub fn target_signature(&self) -> [u8; 64] {
        // Extract target transaction signature
        let mut sig = [0u8; 64];
        if let Some(packet) = self.packets.first() {
            if packet.data.len() >= 64 {
                sig.copy_from_slice(&packet.data[..64]);
            }
        }
        sig
    }
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: BytesMut::with_capacity(capacity),
            read_pos: 0,
            write_pos: 0,
            capacity,
        }
    }
}

// NanoBurst QUIC implementation with custom congestion control
pub struct NanoBurstQuic {
    endpoint: Arc<Endpoint>,
    connections: Arc<dashmap::DashMap<String, Arc<Connection>>>,
    config: Arc<ClientConfig>,
}

impl NanoBurstQuic {
    pub async fn new(bind_addr: &str) -> Result<Self> {
        info!("Initializing NanoBurst QUIC with ultra-low latency settings");
        
        // Configure TLS with minimal overhead
        let mut roots = RootCertStore::empty();
        roots.add_server_trust_anchors(
            webpki_roots::TLS_SERVER_ROOTS.0.iter().map(|ta| {
                rustls::OwnedTrustAnchor::from_subject_spki_name_constraints(
                    ta.subject,
                    ta.spki,
                    ta.name_constraints,
                )
            })
        );
        
        let mut crypto = rustls::ClientConfig::builder()
            .with_safe_defaults()
            .with_root_certificates(roots)
            .with_no_client_auth();
        
        // Optimize TLS for speed
        crypto.alpn_protocols = vec![b"h3".to_vec()];
        crypto.enable_early_data = true;
        
        let mut config = ClientConfig::new(Arc::new(crypto));
        
        // NanoBurst transport config - ultra aggressive settings
        let mut transport = quinn::TransportConfig::default();
        
        // Timing optimizations
        transport.initial_rtt(std::time::Duration::from_micros(250)); // Assume 250μs RTT
        transport.max_idle_timeout(Some(std::time::Duration::from_secs(10)).try_into()?);
        transport.keep_alive_interval(Some(std::time::Duration::from_secs(2)));
        
        // Window sizes - large for burst traffic
        transport.stream_receive_window(16 * 1024 * 1024)?;  // 16MB
        transport.receive_window(32 * 1024 * 1024)?;         // 32MB  
        transport.send_window(32 * 1024 * 1024);             // 32MB
        
        // Concurrent streams - high for parallel bundles
        transport.max_concurrent_bidi_streams(512u32.into());
        transport.max_concurrent_uni_streams(512u32.into());
        
        // Datagram support for fast path
        transport.datagram_receive_buffer_size(Some(8 * 1024 * 1024)); // 8MB
        transport.datagram_send_buffer_size(8 * 1024 * 1024);
        
        // Custom congestion control parameters
        transport.congestion_controller_factory(Arc::new(NanoBurstCongestionController));
        
        config.transport_config(Arc::new(transport));
        
        // Create endpoint
        let mut endpoint = Endpoint::client(bind_addr.parse()?)?;
        endpoint.set_default_client_config(config.clone());
        
        Ok(Self {
            endpoint: Arc::new(endpoint),
            connections: Arc::new(dashmap::DashMap::new()),
            config: Arc::new(config),
        })
    }
    
    pub async fn get_connection(&self, addr: &str) -> Result<Arc<Connection>> {
        // Check cache first
        if let Some(conn) = self.connections.get(addr) {
            if !conn.close_reason().is_some() {
                return Ok(conn.clone());
            }
        }
        
        // Create new connection with 0-RTT if possible
        let conn = self.endpoint.connect(addr.parse()?, "localhost")?
            .into_0rtt()
            .map_or_else(|(conn, _)| conn, |conn| conn);
        
        let conn = Arc::new(conn.await?);
        self.connections.insert(addr.to_string(), conn.clone());
        
        Ok(conn)
    }
    
    pub async fn send_bundle(&self, addr: &str, data: &[u8]) -> Result<()> {
        let conn = self.get_connection(addr).await?;
        
        // Try datagram first for lowest latency
        if conn.max_datagram_size().is_some() {
            conn.send_datagram(data.into())?;
        } else {
            // Fall back to stream
            let mut stream = conn.open_uni().await?;
            stream.write_all(data).await?;
            stream.finish().await?;
        }
        
        Ok(())
    }
}

// Custom congestion controller for ultra-low latency
struct NanoBurstCongestionController;

impl quinn::congestion::ControllerFactory for NanoBurstCongestionController {
    fn build(
        self: Arc<Self>,
        _now: std::time::Instant,
        _current_rtt: std::time::Duration,
    ) -> Box<dyn quinn::congestion::Controller> {
        Box::new(NanoBurstController {
            window: 32 * 1024 * 1024, // Start with 32MB window
            ssthresh: u64::MAX,
            bytes_in_flight: 0,
            rtts: Vec::with_capacity(100),
        })
    }
}

struct NanoBurstController {
    window: u64,
    ssthresh: u64,
    bytes_in_flight: u64,
    rtts: Vec<std::time::Duration>,
}

impl quinn::congestion::Controller for NanoBurstController {
    fn on_sent(&mut self, _now: std::time::Instant, bytes: u64, _last_packet: bool) {
        self.bytes_in_flight += bytes;
    }
    
    fn on_ack(
        &mut self,
        _now: std::time::Instant,
        sent: std::time::Instant,
        bytes: u64,
        _app_limited: bool,
        rtt: &quinn::congestion::RttEstimator,
    ) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes);
        
        // Ultra aggressive window growth
        if self.window < self.ssthresh {
            self.window += bytes * 2; // Double aggressive slow start
        } else {
            self.window += (bytes * bytes) / self.window; // Cubic-like growth
        }
        
        // Cap at 64MB
        self.window = self.window.min(64 * 1024 * 1024);
        
        // Track RTT for adaptive behavior
        self.rtts.push(rtt.get());
        if self.rtts.len() > 100 {
            self.rtts.remove(0);
        }
    }
    
    fn on_congestion_event(
        &mut self,
        _now: std::time::Instant,
        sent: std::time::Instant,
        _is_persistent_congestion: bool,
    ) {
        // Minimal backoff - we prioritize speed over fairness
        self.ssthresh = (self.window * 3) / 4; // Only 25% reduction
        self.window = self.ssthresh;
    }
    
    fn window(&self) -> u64 {
        self.window.saturating_sub(self.bytes_in_flight)
    }
    
    fn clone_box(&self) -> Box<dyn quinn::congestion::Controller> {
        Box::new(Self {
            window: self.window,
            ssthresh: self.ssthresh,
            bytes_in_flight: 0,
            rtts: Vec::with_capacity(100),
        })
    }
    
    fn initial_window(&self) -> u64 {
        32 * 1024 * 1024 // 32MB initial window
    }
    
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}