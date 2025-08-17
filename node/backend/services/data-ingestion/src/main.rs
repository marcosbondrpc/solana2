use anyhow::Result;
use bytes::BytesMut;
use rust_common::net_ts::linux::{bind_timestamped_udp, recv_with_ts};
use rust_common::ring::MpmcRing;
use std::time::SystemTime;

/// Minimal UDP ingress that stamps messages with receive timestamp and pushes into a batch ring.
fn main() -> Result<()> {
    let sock = bind_timestamped_udp("0.0.0.0:9001")?;
    let ring: MpmcRing<(SystemTime, Vec<u8>)> = MpmcRing::with_capacity_pow2(1 << 14);
    let mut buf = vec![0u8; 2048];

    loop {
        let (n, ts) = recv_with_ts(&sock, &mut buf)?;
        let mut b = BytesMut::with_capacity(n);
        b.extend_from_slice(&buf[..n]);
        let _ = ring.push_many(std::iter::once((ts, b.to_vec())));
        // In a real service, signal consumers here.
    }
}
