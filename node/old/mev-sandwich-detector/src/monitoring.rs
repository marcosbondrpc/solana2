use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use prometheus::{register_counter, register_histogram, register_gauge, Counter, Histogram, Gauge};
use metrics_exporter_prometheus::PrometheusBuilder;
use tracing::{info, warn};

pub struct MetricsCollector {
    // Latency metrics
    pub decision_time: Histogram,
    pub submission_time: Histogram,
    pub batch_processing_time: Histogram,
    pub ml_inference_time: Histogram,
    
    // Throughput metrics
    pub decisions_per_second: Gauge,
    pub bundles_submitted: Counter,
    pub bundles_landed: Counter,
    
    // Success metrics
    pub tpu_success_rate: Gauge,
    pub jito_success_rate: Gauge,
    pub overall_success_rate: Gauge,
    
    // PnL metrics
    pub total_profit: Gauge,
    pub rolling_pnl: Gauge,
    pub profit_per_bundle: Histogram,
    
    // System metrics
    pub dedupe_hits: Counter,
    pub risk_rejections: Counter,
    pub active_connections: Gauge,
    
    // Internal counters
    decision_count: AtomicU64,
    landed_count: AtomicU64,
    total_profit_lamports: AtomicU64,
    
    // Performance tracking
    last_report: Arc<tokio::sync::Mutex<Instant>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        // Start Prometheus exporter
        PrometheusBuilder::new()
            .with_http_listener("0.0.0.0:9091".parse().unwrap())
            .install()
            .expect("Failed to start metrics server");
        
        info!("Metrics server started on :9091");
        
        Self {
            decision_time: register_histogram!(
                "sandwich_decision_time_microseconds",
                "Time to make sandwich decision in microseconds",
                vec![100.0, 500.0, 1000.0, 2000.0, 5000.0, 8000.0, 10000.0, 20000.0]
            ).unwrap(),
            
            submission_time: register_histogram!(
                "sandwich_submission_time_microseconds",
                "Time to submit bundle in microseconds"
            ).unwrap(),
            
            batch_processing_time: register_histogram!(
                "sandwich_batch_processing_time_microseconds",
                "Time to process packet batch"
            ).unwrap(),
            
            ml_inference_time: register_histogram!(
                "sandwich_ml_inference_microseconds",
                "ML model inference time",
                vec![10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0]
            ).unwrap(),
            
            decisions_per_second: register_gauge!(
                "sandwich_decisions_per_second",
                "Current decisions per second"
            ).unwrap(),
            
            bundles_submitted: register_counter!(
                "sandwich_bundles_submitted_total",
                "Total bundles submitted"
            ).unwrap(),
            
            bundles_landed: register_counter!(
                "sandwich_bundles_landed_total",
                "Total bundles landed on-chain"
            ).unwrap(),
            
            tpu_success_rate: register_gauge!(
                "sandwich_tpu_success_rate",
                "TPU submission success rate"
            ).unwrap(),
            
            jito_success_rate: register_gauge!(
                "sandwich_jito_success_rate",
                "Jito submission success rate"
            ).unwrap(),
            
            overall_success_rate: register_gauge!(
                "sandwich_overall_success_rate",
                "Overall bundle landing rate"
            ).unwrap(),
            
            total_profit: register_gauge!(
                "sandwich_total_profit_sol",
                "Total profit in SOL"
            ).unwrap(),
            
            rolling_pnl: register_gauge!(
                "sandwich_rolling_pnl_sol",
                "Rolling 10k trade PnL in SOL"
            ).unwrap(),
            
            profit_per_bundle: register_histogram!(
                "sandwich_profit_per_bundle_lamports",
                "Profit distribution per bundle"
            ).unwrap(),
            
            dedupe_hits: register_counter!(
                "sandwich_dedupe_hits_total",
                "Deduplication cache hits"
            ).unwrap(),
            
            risk_rejections: register_counter!(
                "sandwich_risk_rejections_total",
                "Trades rejected by risk manager"
            ).unwrap(),
            
            active_connections: register_gauge!(
                "sandwich_active_connections",
                "Number of active network connections"
            ).unwrap(),
            
            decision_count: AtomicU64::new(0),
            landed_count: AtomicU64::new(0),
            total_profit_lamports: AtomicU64::new(0),
            
            last_report: Arc::new(tokio::sync::Mutex::new(Instant::now())),
        }
    }
    
    pub fn record_decision_time(&self, duration: Duration) {
        let micros = duration.as_micros() as f64;
        self.decision_time.observe(micros);
        
        self.decision_count.fetch_add(1, Ordering::Relaxed);
        
        // Check SLO
        if micros > 8000.0 {
            warn!("Decision time {}μs exceeds 8ms SLO", micros);
        }
    }
    
    pub fn record_submission_time(&self, duration: Duration) {
        self.submission_time.observe(duration.as_micros() as f64);
        self.bundles_submitted.inc();
    }
    
    pub fn record_batch_processing_time(&self, duration: Duration) {
        self.batch_processing_time.observe(duration.as_micros() as f64);
    }
    
    pub fn record_ml_inference(&self, duration: Duration) {
        let micros = duration.as_micros() as f64;
        self.ml_inference_time.observe(micros);
        
        if micros > 100.0 {
            warn!("ML inference {}μs exceeds 100μs target", micros);
        }
    }
    
    pub fn record_bundle_landed(&self, profit: u64) {
        self.bundles_landed.inc();
        self.landed_count.fetch_add(1, Ordering::Relaxed);
        self.total_profit_lamports.fetch_add(profit, Ordering::Relaxed);
        self.profit_per_bundle.observe(profit as f64);
        
        // Update profit gauge
        let total_sol = self.total_profit_lamports.load(Ordering::Relaxed) as f64 / 1e9;
        self.total_profit.set(total_sol);
    }
    
    pub fn increment_dedupe_hits(&self) {
        self.dedupe_hits.inc();
    }
    
    pub fn increment_risk_rejections(&self) {
        self.risk_rejections.inc();
    }
    
    pub fn update_success_rates(&self, tpu_rate: f64, jito_rate: f64) {
        self.tpu_success_rate.set(tpu_rate);
        self.jito_success_rate.set(jito_rate);
        
        let overall = (tpu_rate + jito_rate) / 2.0;
        self.overall_success_rate.set(overall);
        
        // Check SLOs
        if overall < 0.65 {
            warn!("Overall success rate {:.2}% below 65% SLO", overall * 100.0);
        }
    }
    
    pub async fn periodic_report(&self) {
        let mut last_report = self.last_report.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(*last_report);
        
        if elapsed < Duration::from_secs(10) {
            return;
        }
        
        let decisions = self.decision_count.load(Ordering::Relaxed);
        let landed = self.landed_count.load(Ordering::Relaxed);
        let profit = self.total_profit_lamports.load(Ordering::Relaxed);
        
        let dps = decisions as f64 / elapsed.as_secs_f64();
        self.decisions_per_second.set(dps);
        
        let success_rate = if decisions > 0 {
            landed as f64 / decisions as f64
        } else {
            0.0
        };
        
        info!(
            "Performance Report - DPS: {:.0}, Success: {:.2}%, Profit: {:.4} SOL",
            dps,
            success_rate * 100.0,
            profit as f64 / 1e9
        );
        
        *last_report = now;
    }
    
    pub fn check_pnl_slo(&self, rolling_window: &[i64]) -> bool {
        let sum: i64 = rolling_window.iter().sum();
        let pnl_sol = sum as f64 / 1e9;
        
        self.rolling_pnl.set(pnl_sol);
        
        if sum < 0 {
            warn!("ALERT: Rolling PnL negative: {:.4} SOL", pnl_sol);
            false
        } else {
            true
        }
    }
}