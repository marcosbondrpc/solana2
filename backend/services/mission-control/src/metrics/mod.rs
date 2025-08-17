use dashmap::DashMap;
use parking_lot::RwLock;
use prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec,
    CounterVec, GaugeVec, HistogramVec, Encoder, TextEncoder,
};
use std::collections::HashMap;
use std::sync::Arc;

pub struct MetricsRecorder {
    // RPC metrics
    rpc_latency: HistogramVec,
    rpc_errors: CounterVec,
    method_latencies: Arc<DashMap<String, (f64, f64, f64)>>, // p50, p95, p99
    
    // Node metrics
    node_version: Arc<RwLock<String>>,
    cluster_nodes: GaugeVec,
    current_slot: GaugeVec,
    block_height: GaugeVec,
    epoch: GaugeVec,
    slot_index: GaugeVec,
    slots_per_epoch: GaugeVec,
    
    // Performance metrics
    tps: GaugeVec,
    tps_1min: GaugeVec,
    tps_5min: GaugeVec,
    block_production_rate: GaugeVec,
    
    // Consensus metrics
    active_stake: GaugeVec,
    delinquent_stake: GaugeVec,
    total_stake: GaugeVec,
    consensus_participation: GaugeVec,
    active_validators: GaugeVec,
    delinquent_validators: GaugeVec,
    
    // Fee metrics
    priority_fee_median: GaugeVec,
    priority_fee_p75: GaugeVec,
    priority_fee_p95: GaugeVec,
    
    // Jito metrics
    jito_tips_total: CounterVec,
    jito_tip_amount: HistogramVec,
    jito_region_latency: GaugeVec,
    tip_percentiles: Arc<RwLock<(u64, u64, u64, u64, u64, u64)>>, // p25-p99
    avg_tip: GaugeVec,
    max_tip_24h: GaugeVec,
    bundles_accepted: CounterVec,
    bundles_rejected: CounterVec,
    bundles_landed: CounterVec,
    bundle_acceptance_rate: GaugeVec,
    bundle_simulation_time: HistogramVec,
    bundle_rejections: Arc<DashMap<String, u64>>,
    
    // ShredStream metrics
    shredstream_packets: CounterVec,
    shredstream_pps: GaugeVec,
    shredstream_gaps: CounterVec,
    shredstream_reorders: CounterVec,
    shredstream_latency_p50: GaugeVec,
    shredstream_latency_p99: GaugeVec,
    shredstream_latency_current: Arc<RwLock<f64>>,
    
    // QUIC metrics
    quic_concurrent_connections: GaugeVec,
    quic_open_streams: GaugeVec,
    quic_handshake_success_rate: GaugeVec,
    quic_avg_handshake_time: GaugeVec,
    quic_throttling_events: GaugeVec,
    quic_error_code_15: GaugeVec,
    quic_packet_loss_rate: GaugeVec,
    quic_retransmission_rate: GaugeVec,
    quic_packets_sent: CounterVec,
    quic_current_pps: GaugeVec,
    quic_pps_limit: GaugeVec,
    
    // QoS metrics
    qos_our_stake: GaugeVec,
    qos_total_stake: GaugeVec,
    qos_peer_count: GaugeVec,
}

impl MetricsRecorder {
    pub fn new() -> Result<Self, prometheus::Error> {
        Ok(Self {
            // RPC metrics
            rpc_latency: register_histogram_vec!(
                "rpc_latency_ms",
                "RPC method latency in milliseconds",
                &["method"]
            )?,
            rpc_errors: register_counter_vec!(
                "rpc_errors_total",
                "Total RPC errors by method",
                &["method"]
            )?,
            method_latencies: Arc::new(DashMap::new()),
            
            // Node metrics
            node_version: Arc::new(RwLock::new(String::new())),
            cluster_nodes: register_gauge_vec!(
                "cluster_nodes_total",
                "Total cluster nodes",
                &[]
            )?,
            current_slot: register_gauge_vec!(
                "current_slot",
                "Current slot",
                &[]
            )?,
            block_height: register_gauge_vec!(
                "block_height",
                "Current block height",
                &[]
            )?,
            epoch: register_gauge_vec!(
                "epoch",
                "Current epoch",
                &[]
            )?,
            slot_index: register_gauge_vec!(
                "slot_index",
                "Slot index in current epoch",
                &[]
            )?,
            slots_per_epoch: register_gauge_vec!(
                "slots_per_epoch",
                "Slots per epoch",
                &[]
            )?,
            
            // Performance metrics
            tps: register_gauge_vec!(
                "tps_current",
                "Current transactions per second",
                &[]
            )?,
            tps_1min: register_gauge_vec!(
                "tps_1min",
                "1-minute average TPS",
                &[]
            )?,
            tps_5min: register_gauge_vec!(
                "tps_5min",
                "5-minute average TPS",
                &[]
            )?,
            block_production_rate: register_gauge_vec!(
                "block_production_rate",
                "Block production rate",
                &[]
            )?,
            
            // Consensus metrics
            active_stake: register_gauge_vec!(
                "active_stake",
                "Active validator stake",
                &[]
            )?,
            delinquent_stake: register_gauge_vec!(
                "delinquent_stake",
                "Delinquent validator stake",
                &[]
            )?,
            total_stake: register_gauge_vec!(
                "total_stake",
                "Total stake",
                &[]
            )?,
            consensus_participation: register_gauge_vec!(
                "consensus_participation",
                "Consensus participation rate",
                &[]
            )?,
            active_validators: register_gauge_vec!(
                "active_validators",
                "Number of active validators",
                &[]
            )?,
            delinquent_validators: register_gauge_vec!(
                "delinquent_validators",
                "Number of delinquent validators",
                &[]
            )?,
            
            // Fee metrics
            priority_fee_median: register_gauge_vec!(
                "priority_fee_median",
                "Median priority fee",
                &[]
            )?,
            priority_fee_p75: register_gauge_vec!(
                "priority_fee_p75",
                "75th percentile priority fee",
                &[]
            )?,
            priority_fee_p95: register_gauge_vec!(
                "priority_fee_p95",
                "95th percentile priority fee",
                &[]
            )?,
            
            // Jito metrics
            jito_tips_total: register_counter_vec!(
                "jito_tips_total",
                "Total Jito tips",
                &[]
            )?,
            jito_tip_amount: register_histogram_vec!(
                "jito_tip_amount",
                "Jito tip amounts",
                &[]
            )?,
            jito_region_latency: register_gauge_vec!(
                "jito_region_latency_ms",
                "Jito region latency",
                &["region"]
            )?,
            tip_percentiles: Arc::new(RwLock::new((0, 0, 0, 0, 0, 0))),
            avg_tip: register_gauge_vec!(
                "jito_avg_tip",
                "Average Jito tip",
                &[]
            )?,
            max_tip_24h: register_gauge_vec!(
                "jito_max_tip_24h",
                "Maximum Jito tip in 24h",
                &[]
            )?,
            bundles_accepted: register_counter_vec!(
                "bundles_accepted_total",
                "Total accepted bundles",
                &[]
            )?,
            bundles_rejected: register_counter_vec!(
                "bundles_rejected_total",
                "Total rejected bundles",
                &[]
            )?,
            bundles_landed: register_counter_vec!(
                "bundles_landed_total",
                "Total landed bundles",
                &[]
            )?,
            bundle_acceptance_rate: register_gauge_vec!(
                "bundle_acceptance_rate",
                "Bundle acceptance rate",
                &[]
            )?,
            bundle_simulation_time: register_histogram_vec!(
                "bundle_simulation_time_ms",
                "Bundle simulation time",
                &[]
            )?,
            bundle_rejections: Arc::new(DashMap::new()),
            
            // ShredStream metrics
            shredstream_packets: register_counter_vec!(
                "shredstream_packets_total",
                "Total ShredStream packets",
                &[]
            )?,
            shredstream_pps: register_gauge_vec!(
                "shredstream_pps",
                "ShredStream packets per second",
                &[]
            )?,
            shredstream_gaps: register_counter_vec!(
                "shredstream_gaps_total",
                "Total ShredStream gaps",
                &[]
            )?,
            shredstream_reorders: register_counter_vec!(
                "shredstream_reorders_total",
                "Total ShredStream reorders",
                &[]
            )?,
            shredstream_latency_p50: register_gauge_vec!(
                "shredstream_latency_p50_ms",
                "ShredStream p50 latency",
                &[]
            )?,
            shredstream_latency_p99: register_gauge_vec!(
                "shredstream_latency_p99_ms",
                "ShredStream p99 latency",
                &[]
            )?,
            shredstream_latency_current: Arc::new(RwLock::new(0.0)),
            
            // QUIC metrics
            quic_concurrent_connections: register_gauge_vec!(
                "quic_concurrent_connections",
                "QUIC concurrent connections",
                &[]
            )?,
            quic_open_streams: register_gauge_vec!(
                "quic_open_streams",
                "QUIC open streams",
                &[]
            )?,
            quic_handshake_success_rate: register_gauge_vec!(
                "quic_handshake_success_rate",
                "QUIC handshake success rate",
                &[]
            )?,
            quic_avg_handshake_time: register_gauge_vec!(
                "quic_avg_handshake_time_ms",
                "QUIC average handshake time",
                &[]
            )?,
            quic_throttling_events: register_gauge_vec!(
                "quic_throttling_events",
                "QUIC throttling events",
                &[]
            )?,
            quic_error_code_15: register_gauge_vec!(
                "quic_error_code_15",
                "QUIC error code 15 count",
                &[]
            )?,
            quic_packet_loss_rate: register_gauge_vec!(
                "quic_packet_loss_rate",
                "QUIC packet loss rate",
                &[]
            )?,
            quic_retransmission_rate: register_gauge_vec!(
                "quic_retransmission_rate",
                "QUIC retransmission rate",
                &[]
            )?,
            quic_packets_sent: register_counter_vec!(
                "quic_packets_sent_total",
                "Total QUIC packets sent",
                &[]
            )?,
            quic_current_pps: register_gauge_vec!(
                "quic_current_pps",
                "QUIC current packets per second",
                &[]
            )?,
            quic_pps_limit: register_gauge_vec!(
                "quic_pps_limit",
                "QUIC PPS limit",
                &[]
            )?,
            
            // QoS metrics
            qos_our_stake: register_gauge_vec!(
                "qos_our_stake",
                "Our stake for QoS",
                &[]
            )?,
            qos_total_stake: register_gauge_vec!(
                "qos_total_stake",
                "Total stake in QoS",
                &[]
            )?,
            qos_peer_count: register_gauge_vec!(
                "qos_peer_count",
                "QoS peer count",
                &[]
            )?,
        })
    }
    
    // RPC metric setters
    pub fn record_rpc_latency(&self, method: &str, latency_ms: f64) {
        self.rpc_latency.with_label_values(&[method]).observe(latency_ms);
    }
    
    pub fn increment_rpc_errors(&self, method: &str) {
        self.rpc_errors.with_label_values(&[method]).inc();
    }
    
    pub fn set_method_percentiles(&self, method: &str, p50: f64, p95: f64, p99: f64) {
        self.method_latencies.insert(method.to_string(), (p50, p95, p99));
    }
    
    // Node metric setters
    pub fn set_node_version(&self, version: &str) {
        *self.node_version.write() = version.to_string();
    }
    
    pub fn set_cluster_node_count(&self, count: u64) {
        self.cluster_nodes.with_label_values(&[]).set(count as f64);
    }
    
    pub fn set_current_slot(&self, slot: u64) {
        self.current_slot.with_label_values(&[]).set(slot as f64);
    }
    
    pub fn set_block_height(&self, height: u64) {
        self.block_height.with_label_values(&[]).set(height as f64);
    }
    
    pub fn set_epoch(&self, epoch: u64) {
        self.epoch.with_label_values(&[]).set(epoch as f64);
    }
    
    pub fn set_slot_index(&self, index: u64) {
        self.slot_index.with_label_values(&[]).set(index as f64);
    }
    
    pub fn set_slots_per_epoch(&self, slots: u64) {
        self.slots_per_epoch.with_label_values(&[]).set(slots as f64);
    }
    
    // Performance metric setters
    pub fn set_tps(&self, tps: f64) {
        self.tps.with_label_values(&[]).set(tps);
    }
    
    pub fn set_tps_1min(&self, tps: f64) {
        self.tps_1min.with_label_values(&[]).set(tps);
    }
    
    pub fn set_tps_5min(&self, tps: f64) {
        self.tps_5min.with_label_values(&[]).set(tps);
    }
    
    pub fn set_block_production_rate(&self, rate: f64) {
        self.block_production_rate.with_label_values(&[]).set(rate);
    }
    
    // Consensus metric setters
    pub fn set_active_stake(&self, stake: u64) {
        self.active_stake.with_label_values(&[]).set(stake as f64);
    }
    
    pub fn set_delinquent_stake(&self, stake: u64) {
        self.delinquent_stake.with_label_values(&[]).set(stake as f64);
    }
    
    pub fn set_total_stake(&self, stake: u64) {
        self.total_stake.with_label_values(&[]).set(stake as f64);
    }
    
    pub fn set_consensus_participation(&self, rate: f64) {
        self.consensus_participation.with_label_values(&[]).set(rate);
    }
    
    pub fn set_active_validators(&self, count: u64) {
        self.active_validators.with_label_values(&[]).set(count as f64);
    }
    
    pub fn set_delinquent_validators(&self, count: u64) {
        self.delinquent_validators.with_label_values(&[]).set(count as f64);
    }
    
    // Fee metric setters
    pub fn set_priority_fee_median(&self, fee: u64) {
        self.priority_fee_median.with_label_values(&[]).set(fee as f64);
    }
    
    pub fn set_priority_fee_p75(&self, fee: u64) {
        self.priority_fee_p75.with_label_values(&[]).set(fee as f64);
    }
    
    pub fn set_priority_fee_p95(&self, fee: u64) {
        self.priority_fee_p95.with_label_values(&[]).set(fee as f64);
    }
    
    // Jito metric setters
    pub fn record_jito_tip(&self, amount: u64) {
        self.jito_tip_amount.with_label_values(&[]).observe(amount as f64);
    }
    
    pub fn increment_jito_tips_total(&self) {
        self.jito_tips_total.with_label_values(&[]).inc();
    }
    
    pub fn record_jito_region_latency(&self, region: &str, latency: f64) {
        self.jito_region_latency.with_label_values(&[region]).set(latency);
    }
    
    pub fn set_tip_percentiles(&self, p25: u64, p50: u64, p75: u64, p90: u64, p95: u64, p99: u64) {
        *self.tip_percentiles.write() = (p25, p50, p75, p90, p95, p99);
    }
    
    pub fn set_avg_tip(&self, avg: u64) {
        self.avg_tip.with_label_values(&[]).set(avg as f64);
    }
    
    pub fn set_max_tip_24h(&self, max: u64) {
        self.max_tip_24h.with_label_values(&[]).set(max as f64);
    }
    
    pub fn increment_bundles_accepted(&self) {
        self.bundles_accepted.with_label_values(&[]).inc();
    }
    
    pub fn increment_bundles_rejected(&self) {
        self.bundles_rejected.with_label_values(&[]).inc();
    }
    
    pub fn increment_bundles_landed(&self) {
        self.bundles_landed.with_label_values(&[]).inc();
    }
    
    pub fn set_bundle_acceptance_rate(&self, rate: f64) {
        self.bundle_acceptance_rate.with_label_values(&[]).set(rate);
    }
    
    pub fn record_bundle_simulation_time(&self, time_ms: f64) {
        self.bundle_simulation_time.with_label_values(&[]).observe(time_ms);
    }
    
    pub fn record_bundle_rejection(&self, reason: &str) {
        self.bundle_rejections.entry(reason.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }
    
    // ShredStream metric setters
    pub fn get_shredstream_packets(&self) -> u64 {
        // In production, would get actual value
        0
    }
    
    pub fn set_shredstream_pps(&self, pps: u64) {
        self.shredstream_pps.with_label_values(&[]).set(pps as f64);
    }
    
    pub fn increment_shredstream_gaps(&self, gaps: u64) {
        for _ in 0..gaps {
            self.shredstream_gaps.with_label_values(&[]).inc();
        }
    }
    
    pub fn increment_shredstream_reorders(&self, reorders: u64) {
        for _ in 0..reorders {
            self.shredstream_reorders.with_label_values(&[]).inc();
        }
    }
    
    pub fn set_shredstream_latency_p50(&self, latency: f64) {
        self.shredstream_latency_p50.with_label_values(&[]).set(latency);
    }
    
    pub fn set_shredstream_latency_p99(&self, latency: f64) {
        self.shredstream_latency_p99.with_label_values(&[]).set(latency);
    }
    
    pub fn get_shredstream_latency(&self) -> f64 {
        *self.shredstream_latency_current.read()
    }
    
    // QUIC metric setters
    pub fn set_quic_concurrent_connections(&self, count: u64) {
        self.quic_concurrent_connections.with_label_values(&[]).set(count as f64);
    }
    
    pub fn set_quic_open_streams(&self, count: u64) {
        self.quic_open_streams.with_label_values(&[]).set(count as f64);
    }
    
    pub fn set_quic_handshake_success_rate(&self, rate: f64) {
        self.quic_handshake_success_rate.with_label_values(&[]).set(rate);
    }
    
    pub fn set_quic_avg_handshake_time(&self, time_ms: f64) {
        self.quic_avg_handshake_time.with_label_values(&[]).set(time_ms);
    }
    
    pub fn set_quic_throttling_events(&self, count: u64) {
        self.quic_throttling_events.with_label_values(&[]).set(count as f64);
    }
    
    pub fn set_quic_error_code_15(&self, count: u64) {
        self.quic_error_code_15.with_label_values(&[]).set(count as f64);
    }
    
    pub fn set_quic_packet_loss_rate(&self, rate: f64) {
        self.quic_packet_loss_rate.with_label_values(&[]).set(rate);
    }
    
    pub fn set_quic_retransmission_rate(&self, rate: f64) {
        self.quic_retransmission_rate.with_label_values(&[]).set(rate);
    }
    
    pub fn increment_quic_packets_sent(&self) {
        self.quic_packets_sent.with_label_values(&[]).inc();
    }
    
    pub fn set_quic_current_pps(&self, pps: u64) {
        self.quic_current_pps.with_label_values(&[]).set(pps as f64);
    }
    
    pub fn set_quic_pps_limit(&self, limit: u64) {
        self.quic_pps_limit.with_label_values(&[]).set(limit as f64);
    }
    
    // QoS metric setters
    pub fn set_qos_our_stake(&self, stake: u64) {
        self.qos_our_stake.with_label_values(&[]).set(stake as f64);
    }
    
    pub fn set_qos_total_stake(&self, stake: u64) {
        self.qos_total_stake.with_label_values(&[]).set(stake as f64);
    }
    
    pub fn set_qos_peer_count(&self, count: u64) {
        self.qos_peer_count.with_label_values(&[]).set(count as f64);
    }
    
    // Export metrics for Prometheus
    pub fn export_metrics(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}