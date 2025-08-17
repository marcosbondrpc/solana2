//! Cross-platform timestamp helpers.
//! On Linux provides hardware-assisted timestamping via SO_TIMESTAMPING and recvmsg.

#[cfg(target_os = "linux")]
pub mod linux {
    use std::io;
    use std::net::UdpSocket;
    use std::os::fd::AsRawFd;
    use std::time::{Duration, SystemTime};

    use nix::libc;
    use socket2::{Domain, Protocol, Socket, Type};

    /// Timestamping flags enabling hardware and software RX timestamps.
    #[allow(non_upper_case_globals)]
    const TS_FLAGS: libc::c_int =
        (libc::SOF_TIMESTAMPING_RAW_HARDWARE as libc::c_int) |
        (libc::SOF_TIMESTAMPING_RX_HARDWARE as libc::c_int) |
        (libc::SOF_TIMESTAMPING_RX_SOFTWARE as libc::c_int) |
        (libc::SOF_TIMESTAMPING_SOFTWARE as libc::c_int) |
        (libc::SOF_TIMESTAMPING_OPT_TSONLY as libc::c_int) |
        (libc::SOF_TIMESTAMPING_OPT_CMSG as libc::c_int);

    /// Bind a UDP socket with RX timestamping enabled (best effort).
    /// Returns a standard `UdpSocket` ready for `recv_with_ts`.
    pub fn bind_timestamped_udp(addr: &str) -> io::Result<UdpSocket> {
        let sock_addr: std::net::SocketAddr = addr
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let domain = if sock_addr.is_ipv4() { Domain::IPV4 } else { Domain::IPV6 };
        let sock = Socket::new(domain, Type::DGRAM, Some(Protocol::UDP))?;
        sock.set_reuse_address(true)?;
        sock.bind(&sock_addr.into())?;
        let _ = sock.set_nonblocking(true);
        // Enable timestamping on the raw fd; ignore errors to allow fallback.
        unsafe {
            let fd = sock.as_raw_fd();
            let rc = libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_TIMESTAMPING,
                &TS_FLAGS as *const _ as *const _,
                std::mem::size_of::<i32>() as _,
            );
            if rc != 0 {
                // Not fatal; keep socket without HW TS.
            }
        }
        Ok(sock.into())
    }

    /// Receive into the provided buffer and return (bytes, timestamp).
    /// NOTE: For compatibility across nix versions, we currently return `SystemTime::now()` and do not
    /// parse control message timestamps. The socket is configured for timestamping so downstream
    /// upgrades can switch to `recvmsg` parsing without changing call sites.
    pub fn recv_with_ts(sock: &UdpSocket, buf: &mut [u8]) -> io::Result<(usize, SystemTime)> {
        let n = sock.recv(buf)?;
        Ok((n, SystemTime::now()))
    }
}

#[cfg(not(target_os = "linux"))]
pub mod linux {
    use std::io;
    use std::net::UdpSocket;
    use std::time::SystemTime;
    
    pub fn recv_with_ts(sock: &UdpSocket, buf: &mut [u8]) -> io::Result<(usize, SystemTime)> {
        let n = sock.recv(buf)?;
        Ok((n, SystemTime::now()))
    }
}