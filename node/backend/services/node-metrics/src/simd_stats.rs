/// Optimized statistical calculations for ultra-low latency
/// Using manual loop unrolling and cache-friendly algorithms
pub struct SimdStats;

impl SimdStats {
    /// Calculate mean using loop unrolling for optimization
    pub fn mean_simd(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let len = data.len();
        let chunks = len / 4;
        let remainder = len % 4;
        
        let mut sum = 0.0;
        
        // Process 4 elements at a time (manual loop unrolling)
        let mut i = 0;
        while i < chunks * 4 {
            sum += data[i] + data[i + 1] + data[i + 2] + data[i + 3];
            i += 4;
        }
        
        // Process remaining elements
        for j in (chunks * 4)..len {
            sum += data[j];
        }
        
        sum / len as f64
    }
    
    /// Calculate variance using loop unrolling
    pub fn variance_simd(data: &[f64], mean: f64) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let len = data.len();
        let chunks = len / 4;
        let mut sum_sq = 0.0;
        
        // Process 4 elements at a time
        let mut i = 0;
        while i < chunks * 4 {
            let diff0 = data[i] - mean;
            let diff1 = data[i + 1] - mean;
            let diff2 = data[i + 2] - mean;
            let diff3 = data[i + 3] - mean;
            
            sum_sq += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            i += 4;
        }
        
        // Process remaining elements
        for j in (chunks * 4)..len {
            let diff = data[j] - mean;
            sum_sq += diff * diff;
        }
        
        sum_sq / (len - 1) as f64
    }
    
    /// Calculate percentiles using optimized sorting
    pub fn percentile_simd(data: &mut [f64], percentile: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        // Use pdqsort for faster sorting
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((percentile / 100.0) * (data.len() - 1) as f64) as usize;
        data[index.min(data.len() - 1)]
    }
    
    /// Calculate min and max in a single pass
    pub fn min_max_simd(data: &[f64]) -> (f64, f64) {
        if data.is_empty() {
            return (0.0, 0.0);
        }
        
        let mut min = data[0];
        let mut max = data[0];
        
        // Process pairs for better branch prediction
        let mut i = 1;
        while i < data.len() - 1 {
            if data[i] < data[i + 1] {
                min = min.min(data[i]);
                max = max.max(data[i + 1]);
            } else {
                min = min.min(data[i + 1]);
                max = max.max(data[i]);
            }
            i += 2;
        }
        
        // Handle last element if odd number
        if data.len() % 2 == 0 {
            min = min.min(data[data.len() - 1]);
            max = max.max(data[data.len() - 1]);
        }
        
        (min, max)
    }
}