use dashmap::DashMap;
use parking_lot::RwLock;
use prometheus::{register_histogram_vec, register_int_counter_vec, HistogramVec, IntCounterVec};
use shared_types::{BundlePriority, Result, SystemError};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

// Metrics
lazy_static::lazy_static! {
    static ref PATH_SELECTIONS: IntCounterVec = register_int_counter_vec!(
        "submission_path_selections_total",
        "Total path selections by type",
        &["path", "priority"]
    ).unwrap();
    
    static ref PATH_SUCCESS_RATE: HistogramVec = register_histogram_vec!(
        "submission_path_success_rate",
        "Success rate per submission path",
        &["path"]
    ).unwrap();
}

const WIN_RATE_WINDOW: usize = 100;
const MIN_SAMPLES: usize = 10;
const JITO_PREFERENCE_THRESHOLD: f64 = 0.7; // Prefer Jito if win rate > 70%
const TPU_FALLBACK_THRESHOLD: f64 = 0.3;    // Fallback to TPU if Jito < 30%

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionPath {
    Jito,
    TPU,
    Both, // Submit to both paths simultaneously
}

/// Dual-rail submission policy with intelligent path selection
pub struct PathSelector {
    jito_stats: Arc<PathStats>,
    tpu_stats: Arc<PathStats>,
    selection_history: Arc<RwLock<VecDeque<(SubmissionPath, Instant)>>>,
    config: PathSelectorConfig,
}

#[derive(Clone)]
pub struct PathSelectorConfig {
    pub enable_dual_rail: bool,
    pub jito_weight: f64,        // Base weight for Jito path
    pub tpu_weight: f64,         // Base weight for TPU path
    pub high_priority_dual: bool, // Always use both for UltraHigh priority
    pub adaptive_routing: bool,   // Enable adaptive routing based on success rates
}

struct PathStats {
    total_submissions: AtomicU64,
    successful_submissions: AtomicU64,
    recent_wins: RwLock<VecDeque<bool>>,
    avg_latency_ms: AtomicU64,
    last_success: RwLock<Option<Instant>>,
}

impl PathStats {
    fn new() -> Self {
        Self {
            total_submissions: AtomicU64::new(0),
            successful_submissions: AtomicU64::new(0),
            recent_wins: RwLock::new(VecDeque::with_capacity(WIN_RATE_WINDOW)),
            avg_latency_ms: AtomicU64::new(0),
            last_success: RwLock::new(None),
        }
    }

    fn record_submission(&self, success: bool, latency_ms: u64) {
        self.total_submissions.fetch_add(1, Ordering::Relaxed);
        
        if success {
            self.successful_submissions.fetch_add(1, Ordering::Relaxed);
            *self.last_success.write() = Some(Instant::now());
        }

        // Update recent wins window
        let mut wins = self.recent_wins.write();
        if wins.len() >= WIN_RATE_WINDOW {
            wins.pop_front();
        }
        wins.push_back(success);

        // Update average latency (exponential moving average)
        let current_avg = self.avg_latency_ms.load(Ordering::Relaxed);
        let new_avg = (current_avg * 9 + latency_ms) / 10;
        self.avg_latency_ms.store(new_avg, Ordering::Relaxed);
    }

    fn win_rate(&self) -> f64 {
        let wins = self.recent_wins.read();
        if wins.len() < MIN_SAMPLES {
            // Not enough data, return overall win rate
            let total = self.total_submissions.load(Ordering::Relaxed);
            if total == 0 {
                return 0.5; // No data, assume 50%
            }
            let successful = self.successful_submissions.load(Ordering::Relaxed);
            successful as f64 / total as f64
        } else {
            let successful = wins.iter().filter(|&&w| w).count();
            successful as f64 / wins.len() as f64
        }
    }

    fn staleness(&self) -> Duration {
        self.last_success
            .read()
            .map(|t| t.elapsed())
            .unwrap_or(Duration::from_secs(3600)) // 1 hour if never successful
    }
}

impl PathSelector {
    pub fn new(config: PathSelectorConfig) -> Self {
        Self {
            jito_stats: Arc::new(PathStats::new()),
            tpu_stats: Arc::new(PathStats::new()),
            selection_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            config,
        }
    }

    /// Select optimal submission path based on current statistics
    pub fn select_path(&self, priority: BundlePriority) -> SubmissionPath {
        // Always use dual-rail for ultra-high priority if configured
        if self.config.high_priority_dual && matches!(priority, BundlePriority::UltraHigh) {
            info!("Selecting BOTH paths for UltraHigh priority bundle");
            PATH_SELECTIONS
                .with_label_values(&["both", &format!("{:?}", priority)])
                .inc();
            return SubmissionPath::Both;
        }

        if !self.config.adaptive_routing {
            // Simple weighted selection
            return self.weighted_selection(priority);
        }

        // Adaptive routing based on win rates
        let jito_win_rate = self.jito_stats.win_rate();
        let tpu_win_rate = self.tpu_stats.win_rate();
        let jito_staleness = self.jito_stats.staleness();
        let tpu_staleness = self.tpu_stats.staleness();

        debug!(
            "Path selection - Jito: {:.2}% (stale: {:?}), TPU: {:.2}% (stale: {:?})",
            jito_win_rate * 100.0,
            jito_staleness,
            tpu_win_rate * 100.0,
            tpu_staleness
        );

        // Decision logic
        let path = if jito_win_rate > JITO_PREFERENCE_THRESHOLD {
            // Strong preference for Jito
            SubmissionPath::Jito
        } else if jito_win_rate < TPU_FALLBACK_THRESHOLD {
            // Jito performing poorly, fallback to TPU
            if tpu_win_rate > 0.5 {
                SubmissionPath::TPU
            } else {
                // Both performing poorly, use both for redundancy
                SubmissionPath::Both
            }
        } else {
            // Middle ground - consider staleness and priority
            if jito_staleness > Duration::from_secs(30) && tpu_win_rate > 0.4 {
                // Jito hasn't won recently, try TPU
                SubmissionPath::TPU
            } else if matches!(priority, BundlePriority::High) && self.config.enable_dual_rail {
                // High priority and moderate performance - use both
                SubmissionPath::Both
            } else {
                // Default to weighted selection
                self.weighted_selection(priority)
            }
        };

        // Record selection
        PATH_SELECTIONS
            .with_label_values(&[path.as_str(), &format!("{:?}", priority)])
            .inc();

        // Update history
        self.selection_history.write().push_back((path, Instant::now()));

        info!("Selected {:?} path for {:?} priority bundle", path, priority);
        path
    }

    /// Simple weighted selection based on configuration
    fn weighted_selection(&self, priority: BundlePriority) -> SubmissionPath {
        let jito_weight = self.config.jito_weight * match priority {
            BundlePriority::UltraHigh | BundlePriority::High => 1.5,
            BundlePriority::Medium => 1.0,
            BundlePriority::Low => 0.8,
        };

        let tpu_weight = self.config.tpu_weight;
        let total_weight = jito_weight + tpu_weight;

        // Random selection based on weights
        let random = rand::random::<f64>() * total_weight;
        
        if random < jito_weight {
            SubmissionPath::Jito
        } else {
            SubmissionPath::TPU
        }
    }

    /// Record submission result for adaptive routing
    pub fn record_result(&self, path: SubmissionPath, success: bool, latency_ms: u64) {
        match path {
            SubmissionPath::Jito => {
                self.jito_stats.record_submission(success, latency_ms);
                PATH_SUCCESS_RATE
                    .with_label_values(&["jito"])
                    .observe(if success { 1.0 } else { 0.0 });
            }
            SubmissionPath::TPU => {
                self.tpu_stats.record_submission(success, latency_ms);
                PATH_SUCCESS_RATE
                    .with_label_values(&["tpu"])
                    .observe(if success { 1.0 } else { 0.0 });
            }
            SubmissionPath::Both => {
                // Record for both paths (we don't know which succeeded)
                self.jito_stats.record_submission(success, latency_ms);
                self.tpu_stats.record_submission(success, latency_ms);
            }
        }
    }

    /// Get current statistics for monitoring
    pub fn get_stats(&self) -> PathSelectorStats {
        PathSelectorStats {
            jito_win_rate: self.jito_stats.win_rate(),
            tpu_win_rate: self.tpu_stats.win_rate(),
            jito_avg_latency_ms: self.jito_stats.avg_latency_ms.load(Ordering::Relaxed),
            tpu_avg_latency_ms: self.tpu_stats.avg_latency_ms.load(Ordering::Relaxed),
            jito_total: self.jito_stats.total_submissions.load(Ordering::Relaxed),
            tpu_total: self.tpu_stats.total_submissions.load(Ordering::Relaxed),
        }
    }
}

impl SubmissionPath {
    pub fn as_str(&self) -> &str {
        match self {
            SubmissionPath::Jito => "jito",
            SubmissionPath::TPU => "tpu",
            SubmissionPath::Both => "both",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PathSelectorStats {
    pub jito_win_rate: f64,
    pub tpu_win_rate: f64,
    pub jito_avg_latency_ms: u64,
    pub tpu_avg_latency_ms: u64,
    pub jito_total: u64,
    pub tpu_total: u64,
}

impl Default for PathSelectorConfig {
    fn default() -> Self {
        Self {
            enable_dual_rail: true,
            jito_weight: 0.7,          // Prefer Jito by default
            tpu_weight: 0.3,
            high_priority_dual: true,   // Use both for UltraHigh
            adaptive_routing: true,     // Enable smart routing
        }
    }
}