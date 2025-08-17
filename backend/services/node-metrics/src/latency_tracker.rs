use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use parking_lot::RwLock;
use crate::ring_buffer::RingBuffer;
use crate::metrics_collector::LatencyStats;

/// Lock-free latency tracker using ring buffer for high-performance metrics collection
pub struct LatencyTracker {
    buffer: RingBuffer<u64>,
    count: AtomicUsize,
    sum: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
}

impl LatencyTracker {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: RingBuffer::new(capacity),
            count: AtomicUsize::new(0),
            sum: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }
    
    pub fn add_sample(&mut self, latency_us: u64) {
        // Add to ring buffer
        if let Some(old_value) = self.buffer.push(latency_us) {
            // Ring buffer is full, subtract old value from sum
            self.sum.fetch_sub(old_value, Ordering::Relaxed);
        } else {
            // Ring buffer not full yet, increment count
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update sum
        self.sum.fetch_add(latency_us, Ordering::Relaxed);
        
        // Update min and max using CAS loop for thread safety
        self.update_min(latency_us);
        self.update_max(latency_us);
    }
    
    fn update_min(&self, value: u64) {
        let mut current = self.min.load(Ordering::Relaxed);
        loop {
            if value >= current {
                break;
            }
            match self.min.compare_exchange_weak(
                current,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }
    
    fn update_max(&self, value: u64) {
        let mut current = self.max.load(Ordering::Relaxed);
        loop {
            if value <= current {
                break;
            }
            match self.max.compare_exchange_weak(
                current,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }
    
    pub fn get_stats(&self) -> LatencyStats {
        let samples = self.buffer.get_all();
        if samples.is_empty() {
            return LatencyStats {
                min: 0,
                max: 0,
                mean: 0.0,
                median: 0.0,
                p95: 0.0,
                p99: 0.0,
                p999: 0.0,
                std_dev: 0.0,
            };
        }
        
        let mut sorted = samples.clone();
        sorted.sort_unstable();
        
        let count = sorted.len();
        let sum: u64 = sorted.iter().sum();
        let mean = sum as f64 / count as f64;
        
        // Calculate percentiles
        let median = sorted[count / 2] as f64;
        let p95 = sorted[(count as f64 * 0.95) as usize] as f64;
        let p99 = sorted[(count as f64 * 0.99) as usize] as f64;
        let p999 = sorted[(count as f64 * 0.999).min((count - 1) as f64) as usize] as f64;
        
        // Calculate standard deviation
        let variance: f64 = sorted
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / count as f64;
        
        let std_dev = variance.sqrt();
        
        LatencyStats {
            min: self.min.load(Ordering::Relaxed),
            max: self.max.load(Ordering::Relaxed),
            mean,
            median,
            p95,
            p99,
            p999,
            std_dev,
        }
    }
    
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.count.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.min.store(u64::MAX, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
    }
}