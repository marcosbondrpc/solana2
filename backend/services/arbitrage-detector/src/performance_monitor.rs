use anyhow::Result;
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, Registry};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;
use warp::Filter;

#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    registry: Arc<Registry>,
    opportunities_found: Counter,
    opportunities_executed: Counter,
    total_profit: Gauge,
    execution_latency: Histogram,
    cache_hit_rate: Gauge,
    active_connections: Gauge,
}

impl PerformanceMonitor {
    pub fn new() -> Result<Self> {
        let registry = Registry::new();

        let opportunities_found = Counter::new(
            "arbitrage_opportunities_found_total",
            "Total number of arbitrage opportunities found"
        )?;
        registry.register(Box::new(opportunities_found.clone()))?;

        let opportunities_executed = Counter::new(
            "arbitrage_opportunities_executed_total",
            "Total number of arbitrage opportunities executed"
        )?;
        registry.register(Box::new(opportunities_executed.clone()))?;

        let total_profit = Gauge::new(
            "arbitrage_total_profit_sol",
            "Total profit earned in SOL"
        )?;
        registry.register(Box::new(total_profit.clone()))?;

        let execution_latency = Histogram::with_opts(
            HistogramOpts::new(
                "arbitrage_execution_latency_ms",
                "Execution latency in milliseconds"
            )
        )?;
        registry.register(Box::new(execution_latency.clone()))?;

        let cache_hit_rate = Gauge::new(
            "cache_hit_rate",
            "Cache hit rate percentage"
        )?;
        registry.register(Box::new(cache_hit_rate.clone()))?;

        let active_connections = Gauge::new(
            "active_websocket_connections",
            "Number of active WebSocket connections"
        )?;
        registry.register(Box::new(active_connections.clone()))?;

        Ok(Self {
            registry: Arc::new(registry),
            opportunities_found,
            opportunities_executed,
            total_profit,
            execution_latency,
            cache_hit_rate,
            active_connections,
        })
    }

    pub fn record_opportunity_found(&self) {
        self.opportunities_found.inc();
    }

    pub fn record_opportunity_executed(&self) {
        self.opportunities_executed.inc();
    }

    pub fn add_profit(&self, profit: f64) {
        self.total_profit.add(profit);
    }

    pub fn record_execution_time(&self, duration: Duration) {
        self.execution_latency.observe(duration.as_millis() as f64);
    }

    pub fn update_cache_hit_rate(&self, rate: f64) {
        self.cache_hit_rate.set(rate);
    }

    pub fn update_active_connections(&self, count: f64) {
        self.active_connections.set(count);
    }

    pub fn get_registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }

    pub async fn start_metrics_server(&self, port: u16) -> Result<()> {
        let registry = self.registry.clone();
        
        let metrics = warp::path("metrics")
            .and(warp::get())
            .map(move || {
                use prometheus::Encoder;
                let encoder = prometheus::TextEncoder::new();
                let metric_families = registry.gather();
                let mut buffer = Vec::new();
                encoder.encode(&metric_families, &mut buffer).unwrap();
                warp::reply::with_header(
                    String::from_utf8(buffer).unwrap(),
                    "Content-Type",
                    "text/plain; version=0.0.4"
                )
            });

        info!("Starting metrics server on port {}", port);
        tokio::spawn(async move {
            warp::serve(metrics)
                .run(([0, 0, 0, 0], port))
                .await;
        });

        Ok(())
    }
}

pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn log_elapsed(&self) {
        info!(
            "{} took {:?}",
            self.name,
            self.elapsed()
        );
    }
}