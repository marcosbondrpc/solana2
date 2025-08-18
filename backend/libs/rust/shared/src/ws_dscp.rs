// WebSocket with DSCP Marking for Ultra-Low-Latency Network Priority
// Implements EF (Expedited Forwarding) DSCP marking and SO_TXTIME scheduling

use std::net::{SocketAddr, TcpStream};
use std::os::unix::io::{AsRawFd, RawFd};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::TcpSocket;
use tokio_tungstenite::{WebSocketStream, MaybeTlsStream};
use tungstenite::{Message, Error as WsError};
use anyhow::{Result, Context};

// DSCP values for QoS marking
const DSCP_EF: u8 = 46;      // Expedited Forwarding (voice/realtime)
const DSCP_AF41: u8 = 34;    // Assured Forwarding 4:1 (video)
const DSCP_AF31: u8 = 26;    // Assured Forwarding 3:1 (critical data)
const DSCP_CS5: u8 = 40;     // Class Selector 5 (signaling)
const DSCP_DEFAULT: u8 = 0;  // Best effort

// Socket options for Linux
const SOL_SOCKET: libc::c_int = 1;
const SO_PRIORITY: libc::c_int = 12;
const SO_TXTIME: libc::c_int = 61;
const SCM_TXTIME: libc::c_int = SO_TXTIME;

// TC priorities
const TC_PRIO_CONTROL: u32 = 7;    // Highest priority
const TC_PRIO_INTERACTIVE: u32 = 6;
const TC_PRIO_BULK: u32 = 2;

/// Priority levels for MEV traffic
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrafficPriority {
    Critical,    // MEV bundle submission
    High,        // Arbitrage execution
    Normal,      // Monitoring/telemetry
    Low,         // Background tasks
}

impl TrafficPriority {
    pub fn to_dscp(&self) -> u8 {
        match self {
            TrafficPriority::Critical => DSCP_EF,
            TrafficPriority::High => DSCP_AF41,
            TrafficPriority::Normal => DSCP_AF31,
            TrafficPriority::Low => DSCP_DEFAULT,
        }
    }
    
    pub fn to_tc_priority(&self) -> u32 {
        match self {
            TrafficPriority::Critical => TC_PRIO_CONTROL,
            TrafficPriority::High => TC_PRIO_INTERACTIVE,
            TrafficPriority::Normal => 4,
            TrafficPriority::Low => TC_PRIO_BULK,
        }
    }
}

/// Set DSCP marking on a socket
pub fn set_dscp(fd: RawFd, dscp: u8) -> Result<()> {
    let tos = dscp << 2;  // DSCP uses upper 6 bits of TOS field
    
    unsafe {
        // Set IPv4 TOS
        let ret = libc::setsockopt(
            fd,
            libc::IPPROTO_IP,
            libc::IP_TOS,
            &tos as *const _ as *const libc::c_void,
            std::mem::size_of_val(&tos) as libc::socklen_t,
        );
        
        if ret != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        
        // Set IPv6 Traffic Class
        let ret = libc::setsockopt(
            fd,
            libc::IPPROTO_IPV6,
            libc::IPV6_TCLASS,
            &tos as *const _ as *const libc::c_void,
            std::mem::size_of_val(&tos) as libc::socklen_t,
        );
        
        // IPv6 might fail if not available, that's OK
        if ret != 0 && std::io::Error::last_os_error().kind() != std::io::ErrorKind::InvalidInput {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    
    Ok(())
}

/// Set socket priority for traffic control
pub fn set_socket_priority(fd: RawFd, priority: u32) -> Result<()> {
    unsafe {
        let ret = libc::setsockopt(
            fd,
            SOL_SOCKET,
            SO_PRIORITY,
            &priority as *const _ as *const libc::c_void,
            std::mem::size_of_val(&priority) as libc::socklen_t,
        );
        
        if ret != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    
    Ok(())
}

/// Set TCP nodelay for lowest latency
pub fn set_tcp_nodelay(fd: RawFd) -> Result<()> {
    let nodelay: libc::c_int = 1;
    
    unsafe {
        let ret = libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_NODELAY,
            &nodelay as *const _ as *const libc::c_void,
            std::mem::size_of_val(&nodelay) as libc::socklen_t,
        );
        
        if ret != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    
    Ok(())
}

/// Set TCP quickack for faster ACKs
pub fn set_tcp_quickack(fd: RawFd) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        let quickack: libc::c_int = 1;
        
        unsafe {
            let ret = libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_QUICKACK,
                &quickack as *const _ as *const libc::c_void,
                std::mem::size_of_val(&quickack) as libc::socklen_t,
            );
            
            if ret != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
    }
    
    Ok(())
}

/// Configure SO_TXTIME for precise packet scheduling
pub fn set_txtime(fd: RawFd, clockid: libc::clockid_t) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        #[repr(C)]
        struct sock_txtime {
            clockid: libc::clockid_t,
            flags: u32,
        }
        
        const SOF_TXTIME_DEADLINE_MODE: u32 = 1 << 0;
        const SOF_TXTIME_REPORT_ERRORS: u32 = 1 << 1;
        
        let txtime_cfg = sock_txtime {
            clockid,
            flags: SOF_TXTIME_DEADLINE_MODE | SOF_TXTIME_REPORT_ERRORS,
        };
        
        unsafe {
            let ret = libc::setsockopt(
                fd,
                SOL_SOCKET,
                SO_TXTIME,
                &txtime_cfg as *const _ as *const libc::c_void,
                std::mem::size_of_val(&txtime_cfg) as libc::socklen_t,
            );
            
            if ret != 0 {
                // SO_TXTIME might not be available on all kernels
                eprintln!("Warning: SO_TXTIME not available: {}", std::io::Error::last_os_error());
            }
        }
    }
    
    Ok(())
}

/// Create optimized TCP socket for MEV traffic
pub async fn create_mev_socket(addr: SocketAddr, priority: TrafficPriority) -> Result<TcpStream> {
    let socket = if addr.is_ipv4() {
        TcpSocket::new_v4()?
    } else {
        TcpSocket::new_v6()?
    };
    
    let fd = socket.as_raw_fd();
    
    // Set DSCP marking
    set_dscp(fd, priority.to_dscp())
        .context("Failed to set DSCP")?;
    
    // Set socket priority
    set_socket_priority(fd, priority.to_tc_priority())
        .context("Failed to set socket priority")?;
    
    // Set TCP options
    set_tcp_nodelay(fd)
        .context("Failed to set TCP_NODELAY")?;
    set_tcp_quickack(fd)
        .context("Failed to set TCP_QUICKACK")?;
    
    // Set SO_TXTIME for precise scheduling
    set_txtime(fd, libc::CLOCK_TAI)
        .unwrap_or_else(|e| eprintln!("Warning: Failed to set SO_TXTIME: {}", e));
    
    // Set socket buffer sizes for low latency
    set_socket_buffers(fd)?;
    
    // Connect
    let stream = socket.connect(addr).await?;
    Ok(stream.into_std()?)
}

/// Set optimized socket buffer sizes
fn set_socket_buffers(fd: RawFd) -> Result<()> {
    // Small buffers for low latency (64KB)
    let buf_size: libc::c_int = 65536;
    
    unsafe {
        // Set send buffer
        libc::setsockopt(
            fd,
            SOL_SOCKET,
            libc::SO_SNDBUF,
            &buf_size as *const _ as *const libc::c_void,
            std::mem::size_of_val(&buf_size) as libc::socklen_t,
        );
        
        // Set receive buffer
        libc::setsockopt(
            fd,
            SOL_SOCKET,
            libc::SO_RCVBUF,
            &buf_size as *const _ as *const libc::c_void,
            std::mem::size_of_val(&buf_size) as libc::socklen_t,
        );
    }
    
    Ok(())
}

/// WebSocket client with DSCP marking
pub struct PriorityWebSocket {
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
    priority: TrafficPriority,
    fd: RawFd,
}

impl PriorityWebSocket {
    pub async fn connect(url: &str, priority: TrafficPriority) -> Result<Self> {
        let uri: tungstenite::http::Uri = url.parse()?;
        let host = uri.host().context("No host in URL")?;
        let port = uri.port_u16().unwrap_or(if uri.scheme_str() == Some("wss") { 443 } else { 80 });
        let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
        
        // Create optimized socket
        let stream = create_mev_socket(addr, priority).await?;
        let fd = stream.as_raw_fd();
        
        // Upgrade to WebSocket
        let (ws, _) = tokio_tungstenite::client_async(url, stream).await?;
        
        Ok(Self {
            ws,
            priority,
            fd,
        })
    }
    
    pub async fn send_with_deadline(&mut self, msg: Message, deadline: SystemTime) -> Result<()> {
        // Calculate transmission time
        let now = SystemTime::now();
        if deadline <= now {
            return Err(anyhow::anyhow!("Deadline already passed"));
        }
        
        let delay = deadline.duration_since(now)?;
        
        // Set SO_TXTIME deadline if available
        #[cfg(target_os = "linux")]
        {
            let deadline_ns = deadline.duration_since(UNIX_EPOCH)?.as_nanos() as u64;
            self.set_packet_deadline(deadline_ns)?;
        }
        
        // Send with timeout
        tokio::time::timeout(delay, self.send(msg)).await??;
        
        Ok(())
    }
    
    #[cfg(target_os = "linux")]
    fn set_packet_deadline(&self, deadline_ns: u64) -> Result<()> {
        // This would require cmsg support in Rust, simplified here
        // In production, use sendmsg with SCM_TXTIME control message
        Ok(())
    }
    
    pub async fn send(&mut self, msg: Message) -> Result<()> {
        use futures_util::SinkExt;
        self.ws.send(msg).await?;
        Ok(())
    }
    
    pub async fn recv(&mut self) -> Result<Option<Message>> {
        use futures_util::StreamExt;
        Ok(self.ws.next().await.transpose()?)
    }
    
    pub fn set_priority(&mut self, priority: TrafficPriority) -> Result<()> {
        self.priority = priority;
        set_dscp(self.fd, priority.to_dscp())?;
        set_socket_priority(self.fd, priority.to_tc_priority())?;
        Ok(())
    }
}

/// Leader phase timing gate
pub struct LeaderPhaseGate {
    leader_schedule: Vec<(SystemTime, SocketAddr)>,
    current_leader_idx: usize,
}

impl LeaderPhaseGate {
    pub fn new(leader_schedule: Vec<(SystemTime, SocketAddr)>) -> Self {
        Self {
            leader_schedule,
            current_leader_idx: 0,
        }
    }
    
    /// Get optimal send time for current leader
    pub fn get_send_time(&self) -> Option<(SystemTime, SocketAddr)> {
        if self.current_leader_idx >= self.leader_schedule.len() {
            return None;
        }
        
        let (leader_time, addr) = &self.leader_schedule[self.current_leader_idx];
        
        // Send 50ms before leader slot for propagation
        let send_time = *leader_time - Duration::from_millis(50);
        
        Some((send_time, *addr))
    }
    
    /// Advance to next leader
    pub fn next_leader(&mut self) {
        self.current_leader_idx += 1;
    }
    
    /// Wait until optimal send time
    pub async fn wait_for_slot(&mut self) -> Option<SocketAddr> {
        if let Some((send_time, addr)) = self.get_send_time() {
            let now = SystemTime::now();
            if send_time > now {
                let delay = send_time.duration_since(now).ok()?;
                tokio::time::sleep(delay).await;
            }
            self.next_leader();
            Some(addr)
        } else {
            None
        }
    }
}

/// Canary transaction sender
pub struct CanaryProbe {
    interval: Duration,
    last_sent: SystemTime,
}

impl CanaryProbe {
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval: Duration::from_millis(interval_ms),
            last_sent: SystemTime::UNIX_EPOCH,
        }
    }
    
    pub fn should_send(&self) -> bool {
        SystemTime::now().duration_since(self.last_sent)
            .map(|d| d >= self.interval)
            .unwrap_or(true)
    }
    
    pub async fn send_canary<F>(&mut self, send_fn: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        if self.should_send() {
            send_fn()?;
            self.last_sent = SystemTime::now();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dscp_values() {
        assert_eq!(TrafficPriority::Critical.to_dscp(), 46);
        assert_eq!(TrafficPriority::High.to_dscp(), 34);
        assert_eq!(TrafficPriority::Normal.to_dscp(), 26);
        assert_eq!(TrafficPriority::Low.to_dscp(), 0);
    }
    
    #[test]
    fn test_tc_priorities() {
        assert_eq!(TrafficPriority::Critical.to_tc_priority(), 7);
        assert_eq!(TrafficPriority::High.to_tc_priority(), 6);
        assert_eq!(TrafficPriority::Low.to_tc_priority(), 2);
    }
    
    #[tokio::test]
    async fn test_leader_phase_gate() {
        let now = SystemTime::now();
        let schedule = vec![
            (now + Duration::from_secs(1), "127.0.0.1:8000".parse().unwrap()),
            (now + Duration::from_secs(2), "127.0.0.1:8001".parse().unwrap()),
        ];
        
        let mut gate = LeaderPhaseGate::new(schedule);
        
        let (send_time, _) = gate.get_send_time().unwrap();
        assert!(send_time < now + Duration::from_secs(1));
        
        gate.next_leader();
        let (send_time2, _) = gate.get_send_time().unwrap();
        assert!(send_time2 > send_time);
    }
    
    #[test]
    fn test_canary_probe() {
        let mut probe = CanaryProbe::new(750);
        assert!(probe.should_send());
        
        probe.last_sent = SystemTime::now();
        assert!(!probe.should_send());
        
        probe.last_sent = SystemTime::now() - Duration::from_secs(1);
        assert!(probe.should_send());
    }
}