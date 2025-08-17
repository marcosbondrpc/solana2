use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use nalgebra::{Matrix2, Vector2};
use solana_sdk::pubkey::Pubkey;

/// Phase-aware Kalman filter for predicting slot timing per leader
/// Tracks slot production patterns and network conditions
pub struct PhaseKalmanPredictor {
    /// Kalman filters per leader
    leader_filters: Arc<RwLock<HashMap<Pubkey, LeaderKalmanFilter>>>,
    /// Global phase tracker
    global_phase: Arc<RwLock<GlobalPhase>>,
    /// Historical slot data
    slot_history: Arc<RwLock<SlotHistory>>,
}

/// Kalman filter state for individual leader
struct LeaderKalmanFilter {
    /// State vector: [slot_duration, drift_rate]
    state: Vector2<f64>,
    /// Covariance matrix
    covariance: Matrix2<f64>,
    /// Process noise
    process_noise: Matrix2<f64>,
    /// Measurement noise
    measurement_noise: f64,
    /// Last update slot
    last_slot: u64,
    /// Leader-specific phase offset
    phase_offset: f64,
    /// Confidence score (0-1)
    confidence: f64,
}

/// Global network phase information
struct GlobalPhase {
    /// Current epoch
    epoch: u64,
    /// Slots per epoch
    slots_per_epoch: u64,
    /// Average slot time (ms)
    avg_slot_time: f64,
    /// Network congestion factor (0-1)
    congestion: f64,
    /// Fork probability estimate
    fork_probability: f64,
}

/// Historical slot timing data
struct SlotHistory {
    /// Slot -> (leader, duration_ms, timestamp)
    slots: Vec<(u64, Pubkey, f64, u64)>,
    /// Maximum history size
    max_size: usize,
}

/// Prediction result
pub struct SlotPrediction {
    /// Expected slot start time (ms from now)
    pub slot_start_ms: f64,
    /// Confidence interval (±ms)
    pub confidence_interval: f64,
    /// Probability of leader producing slot
    pub production_probability: f64,
    /// Recommended transaction send time (ms from now)
    pub optimal_send_time: f64,
    /// Phase within leader's schedule
    pub leader_phase: LeaderPhase,
}

#[derive(Debug, Clone)]
pub enum LeaderPhase {
    /// First slot in leader's rotation
    RotationStart,
    /// Middle slots with stable timing
    Stable,
    /// Last slot before rotation
    RotationEnd,
    /// Uncertain phase
    Unknown,
}

impl PhaseKalmanPredictor {
    pub fn new() -> Self {
        Self {
            leader_filters: Arc::new(RwLock::new(HashMap::new())),
            global_phase: Arc::new(RwLock::new(GlobalPhase {
                epoch: 0,
                slots_per_epoch: 432000,
                avg_slot_time: 400.0, // 400ms nominal
                congestion: 0.0,
                fork_probability: 0.01,
            })),
            slot_history: Arc::new(RwLock::new(SlotHistory {
                slots: Vec::with_capacity(10000),
                max_size: 10000,
            })),
        }
    }
    
    /// Update with observed slot timing
    pub async fn update_slot(&self, slot: u64, leader: Pubkey, duration_ms: f64, timestamp: u64) {
        // Update history
        {
            let mut history = self.slot_history.write().await;
            history.slots.push((slot, leader, duration_ms, timestamp));
            if history.slots.len() > history.max_size {
                history.slots.remove(0);
            }
        }
        
        // Update or create Kalman filter for leader
        let mut filters = self.leader_filters.write().await;
        let filter = filters.entry(leader).or_insert_with(|| {
            LeaderKalmanFilter::new(duration_ms)
        });
        
        // Kalman filter update
        filter.update(slot, duration_ms);
        
        // Update global phase
        self.update_global_phase(slot, duration_ms).await;
    }
    
    /// Predict slot timing for future slot
    pub async fn predict_slot(&self, slot: u64, leader: Pubkey) -> SlotPrediction {
        let filters = self.leader_filters.read().await;
        let global = self.global_phase.read().await;
        
        let (slot_start_ms, confidence_interval, production_prob) = 
            if let Some(filter) = filters.get(&leader) {
                filter.predict(slot, &global)
            } else {
                // No data for leader, use global average
                let start = (slot as f64 - self.current_slot().await as f64) * global.avg_slot_time;
                (start, global.avg_slot_time * 0.2, 0.95)
            };
            
        // Determine leader phase
        let leader_phase = self.determine_phase(slot, &leader).await;
        
        // Calculate optimal send time based on phase
        let optimal_send_time = match leader_phase {
            LeaderPhase::RotationStart => slot_start_ms - 50.0, // Send early
            LeaderPhase::Stable => slot_start_ms - 20.0,        // Normal timing
            LeaderPhase::RotationEnd => slot_start_ms - 80.0,   // Send very early
            LeaderPhase::Unknown => slot_start_ms - 40.0,       // Conservative
        };
        
        SlotPrediction {
            slot_start_ms,
            confidence_interval,
            production_probability: production_prob * (1.0 - global.fork_probability),
            optimal_send_time: optimal_send_time.max(0.0),
            leader_phase,
        }
    }
    
    /// Get current slot estimate
    async fn current_slot(&self) -> u64 {
        let history = self.slot_history.read().await;
        history.slots.last().map(|(slot, _, _, _)| *slot).unwrap_or(0)
    }
    
    /// Update global phase information
    async fn update_global_phase(&self, slot: u64, duration_ms: f64) {
        let mut global = self.global_phase.write().await;
        
        // Update epoch if needed
        let new_epoch = slot / global.slots_per_epoch;
        if new_epoch > global.epoch {
            global.epoch = new_epoch;
        }
        
        // Update average slot time (exponential moving average)
        let alpha = 0.05;
        global.avg_slot_time = alpha * duration_ms + (1.0 - alpha) * global.avg_slot_time;
        
        // Update congestion estimate based on slot time deviation
        let deviation = (duration_ms - 400.0).abs() / 400.0;
        global.congestion = alpha * deviation + (1.0 - alpha) * global.congestion;
        
        // Update fork probability based on recent history
        let history = self.slot_history.read().await;
        if history.slots.len() > 100 {
            let recent_slots: Vec<u64> = history.slots.iter()
                .rev()
                .take(100)
                .map(|(s, _, _, _)| *s)
                .collect();
                
            let mut gaps = 0;
            for i in 1..recent_slots.len() {
                if recent_slots[i-1] != recent_slots[i] + 1 {
                    gaps += 1;
                }
            }
            
            global.fork_probability = gaps as f64 / 100.0;
        }
    }
    
    /// Determine phase within leader's rotation
    async fn determine_phase(&self, slot: u64, leader: &Pubkey) -> LeaderPhase {
        // Simplified phase detection
        // In production, would use actual leader schedule
        let slot_in_rotation = slot % 4; // Assume 4-slot rotations
        
        match slot_in_rotation {
            0 => LeaderPhase::RotationStart,
            1 | 2 => LeaderPhase::Stable,
            3 => LeaderPhase::RotationEnd,
            _ => LeaderPhase::Unknown,
        }
    }
    
    /// Get performance metrics
    pub async fn get_metrics(&self) -> PredictorMetrics {
        let filters = self.leader_filters.read().await;
        let global = self.global_phase.read().await;
        let history = self.slot_history.read().await;
        
        let avg_confidence = if filters.is_empty() {
            0.0
        } else {
            filters.values().map(|f| f.confidence).sum::<f64>() / filters.len() as f64
        };
        
        PredictorMetrics {
            tracked_leaders: filters.len(),
            avg_confidence,
            global_congestion: global.congestion,
            fork_probability: global.fork_probability,
            history_size: history.slots.len(),
        }
    }
}

impl LeaderKalmanFilter {
    fn new(initial_duration: f64) -> Self {
        // Initialize state with observed duration and zero drift
        let state = Vector2::new(initial_duration, 0.0);
        
        // Initial covariance (high uncertainty)
        let covariance = Matrix2::new(
            100.0, 0.0,
            0.0, 1.0
        );
        
        // Process noise (how much the system can change)
        let process_noise = Matrix2::new(
            0.1, 0.0,
            0.0, 0.01
        );
        
        Self {
            state,
            covariance,
            process_noise,
            measurement_noise: 10.0, // Measurement uncertainty in ms
            last_slot: 0,
            phase_offset: 0.0,
            confidence: 0.5,
        }
    }
    
    fn update(&mut self, slot: u64, duration_ms: f64) {
        // Time update (prediction step)
        let dt = (slot - self.last_slot) as f64;
        
        // State transition matrix
        let f = Matrix2::new(
            1.0, dt,
            0.0, 1.0
        );
        
        // Predict state
        self.state = f * self.state;
        
        // Predict covariance
        self.covariance = f * self.covariance * f.transpose() + self.process_noise;
        
        // Measurement update (correction step)
        let h = Vector2::new(1.0, 0.0); // Observation matrix (we observe duration)
        let y = duration_ms - h.dot(&self.state); // Innovation
        let s = h.dot(&(self.covariance * h)) + self.measurement_noise; // Innovation covariance
        let k = self.covariance * h / s; // Kalman gain
        
        // Update state
        self.state += k * y;
        
        // Update covariance
        let i = Matrix2::identity();
        self.covariance = (i - k * h.transpose()) * self.covariance;
        
        // Update confidence based on innovation
        let normalized_innovation = (y.abs() / s.sqrt()).min(3.0) / 3.0;
        self.confidence = 0.9 * self.confidence + 0.1 * (1.0 - normalized_innovation);
        
        self.last_slot = slot;
    }
    
    fn predict(&self, slot: u64, global: &GlobalPhase) -> (f64, f64, f64) {
        let dt = (slot - self.last_slot) as f64;
        
        // Predict duration
        let predicted_duration = self.state[0] + self.state[1] * dt;
        
        // Adjust for congestion
        let congestion_factor = 1.0 + global.congestion * 0.2;
        let adjusted_duration = predicted_duration * congestion_factor;
        
        // Calculate confidence interval
        let prediction_variance = self.covariance[(0, 0)] + 
                                  2.0 * dt * self.covariance[(0, 1)] + 
                                  dt * dt * self.covariance[(1, 1)];
        let confidence_interval = 2.0 * prediction_variance.sqrt(); // 95% CI
        
        // Production probability based on confidence and global factors
        let production_prob = self.confidence * (1.0 - global.congestion * 0.1);
        
        // Calculate start time from now
        let slots_ahead = dt;
        let start_ms = slots_ahead * adjusted_duration;
        
        (start_ms, confidence_interval, production_prob)
    }
}

pub struct PredictorMetrics {
    pub tracked_leaders: usize,
    pub avg_confidence: f64,
    pub global_congestion: f64,
    pub fork_probability: f64,
    pub history_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_kalman_prediction() {
        let predictor = PhaseKalmanPredictor::new();
        let leader = Pubkey::new_unique();
        
        // Feed some slot data
        for i in 0..10 {
            predictor.update_slot(i, leader, 400.0 + (i as f64 * 2.0), i * 400).await;
        }
        
        // Predict future slot
        let prediction = predictor.predict_slot(15, leader).await;
        
        assert!(prediction.slot_start_ms > 0.0);
        assert!(prediction.confidence_interval > 0.0);
        assert!(prediction.production_probability > 0.0);
        assert!(prediction.production_probability <= 1.0);
    }
    
    #[tokio::test]
    async fn test_phase_detection() {
        let predictor = PhaseKalmanPredictor::new();
        let leader = Pubkey::new_unique();
        
        let phase0 = predictor.determine_phase(0, &leader).await;
        let phase1 = predictor.determine_phase(1, &leader).await;
        let phase3 = predictor.determine_phase(3, &leader).await;
        
        assert!(matches!(phase0, LeaderPhase::RotationStart));
        assert!(matches!(phase1, LeaderPhase::Stable));
        assert!(matches!(phase3, LeaderPhase::RotationEnd));
    }
}