use quinn_proto::{
    congestion::{Controller, ControllerOps, Cubic, CubicConfig},
    AckFrequency, Instant, TransportConfig,
};
use std::sync::Arc;
use std::time::Duration;
use parking_lot::Mutex;

/// NanoBurst: Ultra-aggressive congestion controller optimized for MEV
/// Biased for time-to-first-land with minimal inflight window
pub struct NanoBurst {
    /// Current congestion window in bytes
    window: u64,
    /// Maximum window size (24 packets)
    max_window: u64,
    /// Minimum window size (2 packets for safety)
    min_window: u64,
    /// Smoothed RTT in microseconds
    srtt_us: u64,
    /// RTT variance
    rttvar_us: u64,
    /// Bytes in flight
    bytes_in_flight: u64,
    /// Loss detection threshold
    loss_threshold: f64,
    /// Pacing rate in bytes per second
    pacing_rate: u64,
    /// Last update timestamp
    last_update: Instant,
    /// MEV-specific: prioritize first packet delivery
    mev_mode: bool,
    /// Telemetry
    metrics: Arc<Mutex<CongestionMetrics>>,
}

#[derive(Default)]
pub struct CongestionMetrics {
    pub total_losses: u64,
    pub total_acks: u64,
    pub min_rtt_us: u64,
    pub avg_rtt_us: u64,
    pub congestion_events: u64,
}

impl NanoBurst {
    pub fn new(initial_rtt_us: u64, max_packets: u16) -> Self {
        let mtu = 1200; // Conservative MTU for QUIC
        let max_window = (max_packets as u64) * mtu;
        let min_window = 2 * mtu;
        let initial_window = 4 * mtu; // Start with 4 packets
        
        Self {
            window: initial_window,
            max_window,
            min_window,
            srtt_us: initial_rtt_us,
            rttvar_us: initial_rtt_us / 2,
            bytes_in_flight: 0,
            loss_threshold: 0.02, // 2% loss threshold
            pacing_rate: (initial_window * 1_000_000) / initial_rtt_us,
            last_update: Instant::now(),
            mev_mode: true,
            metrics: Arc::new(Mutex::new(CongestionMetrics::default())),
        }
    }
    
    /// Update RTT estimates using exponential weighted moving average
    fn update_rtt(&mut self, rtt_us: u64) {
        if self.srtt_us == 0 {
            self.srtt_us = rtt_us;
            self.rttvar_us = rtt_us / 2;
        } else {
            let diff = if rtt_us > self.srtt_us {
                rtt_us - self.srtt_us
            } else {
                self.srtt_us - rtt_us
            };
            
            // EWMA with alpha = 1/8, beta = 1/4
            self.rttvar_us = (3 * self.rttvar_us + diff) / 4;
            self.srtt_us = (7 * self.srtt_us + rtt_us) / 8;
        }
        
        // Update metrics
        let mut metrics = self.metrics.lock();
        if metrics.min_rtt_us == 0 || rtt_us < metrics.min_rtt_us {
            metrics.min_rtt_us = rtt_us;
        }
        metrics.avg_rtt_us = self.srtt_us;
        
        // Update pacing rate based on new RTT
        self.update_pacing_rate();
    }
    
    /// Calculate pacing rate for smooth transmission
    fn update_pacing_rate(&mut self) {
        if self.srtt_us > 0 {
            // Pace at 1.25x the congestion window rate for slight aggressiveness
            self.pacing_rate = (self.window * 1_250_000) / self.srtt_us;
        }
    }
    
    /// MEV-optimized window adjustment
    fn adjust_window(&mut self, is_loss: bool) {
        if is_loss {
            // On loss, multiplicative decrease but not below minimum
            self.window = self.window.max(self.min_window).saturating_mul(7) / 10;
            self.metrics.lock().congestion_events += 1;
        } else if self.mev_mode {
            // In MEV mode, aggressive increase up to max
            let increment = 1200; // One packet per RTT
            self.window = (self.window + increment).min(self.max_window);
        } else {
            // Standard AIMD increase
            let increment = (1200 * 1200) / self.window;
            self.window = (self.window + increment).min(self.max_window);
        }
        
        self.update_pacing_rate();
    }
}

impl Controller for NanoBurst {
    fn on_ack(
        &mut self,
        _now: Instant,
        sent: Instant,
        bytes: u64,
        _app_limited: bool,
        rtt: &quinn_proto::RttEstimator,
    ) {
        let rtt_us = rtt.get().as_micros() as u64;
        self.update_rtt(rtt_us);
        
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes);
        self.metrics.lock().total_acks += 1;
        
        // Adjust window on successful ACK
        self.adjust_window(false);
    }
    
    fn on_congestion_event(
        &mut self,
        _now: Instant,
        sent: Instant,
        is_persistent_congestion: bool,
        _lost_bytes: u64,
    ) {
        self.metrics.lock().total_losses += 1;
        
        if is_persistent_congestion {
            // Severe congestion: reset to minimum
            self.window = self.min_window;
            self.mev_mode = false; // Temporarily disable aggressive mode
        } else {
            self.adjust_window(true);
        }
        
        self.update_pacing_rate();
    }
    
    fn on_mtu_update(&mut self, new_mtu: u16) {
        // Adjust windows based on new MTU
        let mtu = new_mtu as u64;
        self.max_window = 24 * mtu;
        self.min_window = 2 * mtu;
        self.window = self.window.clamp(self.min_window, self.max_window);
    }
    
    fn window(&self) -> u64 {
        self.window
    }
    
    fn clone_box(&self) -> Box<dyn Controller> {
        Box::new(Self {
            window: self.window,
            max_window: self.max_window,
            min_window: self.min_window,
            srtt_us: self.srtt_us,
            rttvar_us: self.rttvar_us,
            bytes_in_flight: 0,
            loss_threshold: self.loss_threshold,
            pacing_rate: self.pacing_rate,
            last_update: Instant::now(),
            mev_mode: self.mev_mode,
            metrics: Arc::clone(&self.metrics),
        })
    }
    
    fn initial_window(&self) -> u64 {
        4 * 1200 // Start with 4 packets
    }
    
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

/// Configure QUIC transport for MEV optimization
pub fn configure_transport(initial_rtt_us: u64) -> TransportConfig {
    let mut config = TransportConfig::default();
    
    // Ultra-low timeouts for MEV
    config.max_idle_timeout(Some(Duration::from_millis(5000))); // 5s idle timeout
    config.initial_rtt(Duration::from_micros(initial_rtt_us));
    
    // Aggressive retransmission
    config.initial_packet_threshold(2);
    config.packet_threshold(2);
    
    // Disable features that add latency
    config.datagram_receive_buffer_size(Some(0)); // No datagrams
    config.keep_alive_interval(None); // No keep-alive
    
    // Enable ECN for congestion notification
    config.congestion_controller_factory(Arc::new(NanoBurstFactory::new(initial_rtt_us)));
    
    config
}

struct NanoBurstFactory {
    initial_rtt_us: u64,
}

impl NanoBurstFactory {
    fn new(initial_rtt_us: u64) -> Self {
        Self { initial_rtt_us }
    }
}

impl quinn_proto::congestion::ControllerFactory for NanoBurstFactory {
    fn build(
        &self,
        _now: Instant,
        _current_mtu: u16,
    ) -> Box<dyn Controller> {
        Box::new(NanoBurst::new(self.initial_rtt_us, 24))
    }
}