use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use parking_lot::RwLock;
use prometheus::{
    Counter, Gauge, Histogram, HistogramOpts, IntCounter, IntGauge,
    register_counter, register_gauge, register_histogram, register_int_counter,
    register_int_gauge,
};
use nix::sys::socket::{setsockopt, sockopt};
use socket2::{Domain, Protocol, Socket, Type};
use libc::{SO_TIMESTAMP, SO_TIMESTAMPNS, SO_TIMESTAMPING};
use dashmap::DashMap;

/// Hardware timestamp types
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HardwareTimestamp {
    /// Software timestamp (kernel)
    pub software_ns: u64,
    /// Hardware RX timestamp
    pub hardware_rx_ns: Option<u64>,
    /// Hardware TX timestamp  
    pub hardware_tx_ns: Option<u64>,
}

/// Nanosecond-precision latency attribution
#[derive(Debug, Clone)]
pub struct LatencyBreakdown {
    /// Network RTT
    pub network_rtt_ns: u64,
    /// Packet processing time
    pub processing_ns: u64,
    /// Serialization time
    pub serialization_ns: u64,
    /// Queue wait time
    pub queue_wait_ns: u64,
    /// Submission time
    pub submission_ns: u64,
    /// Total end-to-end
    pub total_ns: u64,
}

/// Metrics collector with nanosecond precision
pub struct MetricsCollector {
    // Counters
    pub packets_received: IntCounter,
    pub packets_sent: IntCounter,
    pub transactions_submitted: IntCounter,
    pub transactions_landed: IntCounter,
    pub bundles_submitted: IntCounter,
    
    // Gauges
    pub current_profit_rate: Gauge,
    pub active_connections: IntGauge,
    pub mempool_size: IntGauge,
    pub congestion_window: IntGauge,
    
    // Histograms (nanosecond buckets)
    pub packet_latency: Histogram,
    pub submission_latency: Histogram,
    pub processing_latency: Histogram,
    pub profit_distribution: Histogram,
    
    // Custom telemetry
    pub latency_breakdown: Arc<RwLock<DashMap<String, LatencyBreakdown>>>,
    pub path_metrics: Arc<RwLock<PathMetrics>>,
}

#[derive(Default)]
pub struct PathMetrics {
    pub tpu_submissions: u64,
    pub tpu_lands: u64,
    pub tpu_avg_latency_ns: u64,
    pub jito_submissions: u64,
    pub jito_lands: u64,
    pub jito_avg_latency_ns: u64,
}

impl MetricsCollector {
    pub fn new() -> Self {
        let latency_buckets = vec![
            100.0,    // 100ns
            500.0,    // 500ns
            1000.0,   // 1μs
            5000.0,   // 5μs
            10000.0,  // 10μs
            50000.0,  // 50μs
            100000.0, // 100μs
            500000.0, // 500μs
            1000000.0, // 1ms
            5000000.0, // 5ms
        ];
        
        Self {
            packets_received: register_int_counter!(
                "mev_packets_received_total",
                "Total packets received"
            ).unwrap(),
            packets_sent: register_int_counter!(
                "mev_packets_sent_total",
                "Total packets sent"
            ).unwrap(),
            transactions_submitted: register_int_counter!(
                "mev_transactions_submitted_total",
                "Total transactions submitted"
            ).unwrap(),
            transactions_landed: register_int_counter!(
                "mev_transactions_landed_total",
                "Total transactions landed on-chain"
            ).unwrap(),
            bundles_submitted: register_int_counter!(
                "mev_bundles_submitted_total",
                "Total Jito bundles submitted"
            ).unwrap(),
            
            current_profit_rate: register_gauge!(
                "mev_current_profit_rate_sol_per_sec",
                "Current profit rate in SOL per second"
            ).unwrap(),
            active_connections: register_int_gauge!(
                "mev_active_connections",
                "Number of active QUIC connections"
            ).unwrap(),
            mempool_size: register_int_gauge!(
                "mev_mempool_size",
                "Current mempool size"
            ).unwrap(),
            congestion_window: register_int_gauge!(
                "mev_congestion_window_bytes",
                "Current congestion window in bytes"
            ).unwrap(),
            
            packet_latency: register_histogram!(
                HistogramOpts::new(
                    "mev_packet_latency_ns",
                    "Packet processing latency in nanoseconds"
                ).buckets(latency_buckets.clone())
            ).unwrap(),
            submission_latency: register_histogram!(
                HistogramOpts::new(
                    "mev_submission_latency_ns",
                    "Transaction submission latency in nanoseconds"
                ).buckets(latency_buckets.clone())
            ).unwrap(),
            processing_latency: register_histogram!(
                HistogramOpts::new(
                    "mev_processing_latency_ns",
                    "MEV opportunity processing latency in nanoseconds"
                ).buckets(latency_buckets.clone())
            ).unwrap(),
            profit_distribution: register_histogram!(
                HistogramOpts::new(
                    "mev_profit_lamports",
                    "Profit distribution in lamports"
                ).buckets(vec![
                    1000000.0,    // 0.001 SOL
                    10000000.0,   // 0.01 SOL
                    100000000.0,  // 0.1 SOL
                    1000000000.0, // 1 SOL
                    10000000000.0, // 10 SOL
                ])
            ).unwrap(),
            
            latency_breakdown: Arc::new(RwLock::new(DashMap::new())),
            path_metrics: Arc::new(RwLock::new(PathMetrics::default())),
        }
    }
    
    /// Record latency with breakdown
    pub fn record_latency_breakdown(&self, id: String, breakdown: LatencyBreakdown) {
        self.latency_breakdown.write().insert(id, breakdown.clone());
        self.submission_latency.observe(breakdown.total_ns as f64);
    }
    
    /// Update path metrics
    pub fn update_path_metrics(&self, is_tpu: bool, landed: bool, latency_ns: u64) {
        let mut metrics = self.path_metrics.write();
        
        if is_tpu {
            metrics.tpu_submissions += 1;
            if landed {
                metrics.tpu_lands += 1;
            }
            // Update average latency (EWMA)
            metrics.tpu_avg_latency_ns = (metrics.tpu_avg_latency_ns * 9 + latency_ns) / 10;
        } else {
            metrics.jito_submissions += 1;
            if landed {
                metrics.jito_lands += 1;
            }
            metrics.jito_avg_latency_ns = (metrics.jito_avg_latency_ns * 9 + latency_ns) / 10;
        }
    }
}

/// Enable hardware timestamping on a socket
pub fn enable_hardware_timestamps(socket: &Socket) -> Result<(), std::io::Error> {
    use std::os::unix::io::AsRawFd;
    let fd = socket.as_raw_fd();
    
    // Enable SO_TIMESTAMPNS for nanosecond precision
    unsafe {
        let enable: i32 = 1;
        let ret = libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            SO_TIMESTAMPNS,
            &enable as *const _ as *const libc::c_void,
            std::mem::size_of::<i32>() as libc::socklen_t,
        );
        
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    
    // Try to enable hardware timestamping if available
    #[cfg(target_os = "linux")]
    {
        unsafe {
            let flags = libc::SOF_TIMESTAMPING_RX_HARDWARE 
                | libc::SOF_TIMESTAMPING_TX_HARDWARE
                | libc::SOF_TIMESTAMPING_RAW_HARDWARE;
            
            let ret = libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                SO_TIMESTAMPING,
                &flags as *const _ as *const libc::c_void,
                std::mem::size_of_val(&flags) as libc::socklen_t,
            );
            
            // It's OK if hardware timestamping fails - fall back to software
            if ret < 0 {
                tracing::warn!("Hardware timestamping not available, using software timestamps");
            }
        }
    }
    
    Ok(())
}

/// Extract timestamp from socket control messages
pub fn extract_timestamp(cmsg: &[u8]) -> Option<HardwareTimestamp> {
    // Parse control messages for timestamps
    // This is platform-specific and requires careful handling
    
    // Simplified implementation - would need full cmsg parsing
    let software_ns = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    
    Some(HardwareTimestamp {
        software_ns,
        hardware_rx_ns: None,
        hardware_tx_ns: None,
    })
}

/// High-precision timer for latency measurement
pub struct PrecisionTimer {
    start: Instant,
    checkpoints: Vec<(String, Duration)>,
}

impl PrecisionTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            checkpoints: Vec::with_capacity(10),
        }
    }
    
    pub fn checkpoint(&mut self, name: impl Into<String>) {
        let elapsed = self.start.elapsed();
        self.checkpoints.push((name.into(), elapsed));
    }
    
    pub fn finish(self) -> LatencyBreakdown {
        let total = self.start.elapsed();
        
        // Extract specific timings from checkpoints
        let mut breakdown = LatencyBreakdown {
            network_rtt_ns: 0,
            processing_ns: 0,
            serialization_ns: 0,
            queue_wait_ns: 0,
            submission_ns: 0,
            total_ns: total.as_nanos() as u64,
        };
        
        // Parse checkpoints to fill breakdown
        for i in 0..self.checkpoints.len() {
            let (name, duration) = &self.checkpoints[i];
            let duration_ns = duration.as_nanos() as u64;
            
            let delta_ns = if i > 0 {
                duration_ns - self.checkpoints[i - 1].1.as_nanos() as u64
            } else {
                duration_ns
            };
            
            match name.as_str() {
                "network_recv" => breakdown.network_rtt_ns = delta_ns,
                "processing" => breakdown.processing_ns = delta_ns,
                "serialization" => breakdown.serialization_ns = delta_ns,
                "queue_wait" => breakdown.queue_wait_ns = delta_ns,
                "submission" => breakdown.submission_ns = delta_ns,
                _ => {}
            }
        }
        
        breakdown
    }
}

/// Real-time telemetry exporter
pub struct TelemetryExporter {
    metrics: Arc<MetricsCollector>,
    export_interval: Duration,
}

impl TelemetryExporter {
    pub fn new(metrics: Arc<MetricsCollector>, export_interval: Duration) -> Self {
        Self {
            metrics,
            export_interval,
        }
    }
    
    pub async fn start(&self) {
        let metrics = Arc::clone(&self.metrics);
        let interval = self.export_interval;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            
            loop {
                interval.tick().await;
                
                // Export current metrics state
                let path_metrics = metrics.path_metrics.read();
                
                tracing::info!(
                    "MEV Metrics - TPU: {}/{} ({:.2}% success, {}ns avg), Jito: {}/{} ({:.2}% success, {}ns avg)",
                    path_metrics.tpu_lands,
                    path_metrics.tpu_submissions,
                    (path_metrics.tpu_lands as f64 / path_metrics.tpu_submissions.max(1) as f64) * 100.0,
                    path_metrics.tpu_avg_latency_ns,
                    path_metrics.jito_lands,
                    path_metrics.jito_submissions,
                    (path_metrics.jito_lands as f64 / path_metrics.jito_submissions.max(1) as f64) * 100.0,
                    path_metrics.jito_avg_latency_ns,
                );
            }
        });
    }
}