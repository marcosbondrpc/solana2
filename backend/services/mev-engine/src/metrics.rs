use anyhow::Result;
use prometheus::{Counter, Histogram, IntCounter, Registry};
use std::time::Duration;
use tracing::info;

pub struct Metrics {
    mempool_scans: IntCounter,
    mempool_scan_duration: Histogram,
    opportunities_found: IntCounter,
    opportunity_scan_duration: Histogram,
    bundles_submitted: IntCounter,
    bundles_successful: IntCounter,
    bundles_expired: IntCounter,
    errors: Counter,
}

impl Metrics {
    pub fn new() -> Result<Self> {
        let registry = Registry::new();
        
        Ok(Self {
            mempool_scans: IntCounter::new("mev_mempool_scans_total", "Total mempool scans")?,
            mempool_scan_duration: Histogram::with_opts(prometheus::HistogramOpts::new(
                "mev_mempool_scan_duration_seconds",
                "Mempool scan duration",
            ))?,
            opportunities_found: IntCounter::new("mev_opportunities_found_total", "Total opportunities found")?,
            opportunity_scan_duration: Histogram::with_opts(prometheus::HistogramOpts::new(
                "mev_opportunity_scan_duration_seconds",
                "Opportunity scan duration",
            ))?,
            bundles_submitted: IntCounter::new("mev_bundles_submitted_total", "Total bundles submitted")?,
            bundles_successful: IntCounter::new("mev_bundles_successful_total", "Successful bundles")?,
            bundles_expired: IntCounter::new("mev_bundles_expired_total", "Expired bundles")?,
            errors: Counter::new("mev_errors_total", "Total errors")?,
        })
    }
    
    pub fn record_mempool_scan(&self, tx_count: usize, duration: Duration) {
        self.mempool_scans.inc();
        self.mempool_scan_duration.observe(duration.as_secs_f64());
    }
    
    pub fn record_opportunity_scan(&self, opp_count: usize, duration: Duration) {
        self.opportunities_found.inc_by(opp_count as u64);
        self.opportunity_scan_duration.observe(duration.as_secs_f64());
    }
    
    pub fn increment_bundle_success(&self) {
        self.bundles_successful.inc();
    }
    
    pub fn increment_bundle_expired(&self) {
        self.bundles_expired.inc();
    }
    
    pub fn increment_errors(&self, error_type: &str) {
        self.errors.inc();
    }
    
    pub fn report(&self) {
        info!("Metrics snapshot - Bundles submitted: {}, Successful: {}, Expired: {}",
              self.bundles_submitted.get(),
              self.bundles_successful.get(),
              self.bundles_expired.get());
    }
}