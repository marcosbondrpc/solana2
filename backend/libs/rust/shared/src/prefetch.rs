/// CPU cache prefetch utilities for x86_64
/// Optimizes memory access patterns in hot paths

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _MM_HINT_NTA};

/// Prefetch data into all cache levels (L1, L2, L3)
#[inline(always)]
pub fn prefetch_read_t0<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}

/// Prefetch data into L2 and L3 cache
#[inline(always)]
pub fn prefetch_read_t1<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T1);
    }
}

/// Prefetch data into L3 cache only
#[inline(always)]
pub fn prefetch_read_t2<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T2);
    }
}

/// Non-temporal prefetch (bypass cache, direct to register)
#[inline(always)]
pub fn prefetch_read_nta<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_NTA);
    }
}

/// Prefetch write hint for exclusive cache line ownership
#[inline(always)]
pub fn prefetch_write<T>(ptr: *mut T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // x86_64 specific: prefetchw for write
        std::arch::asm!(
            "prefetchw [{}]",
            in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }
}

/// Memory fence for ensuring ordering
#[inline(always)]
pub fn memory_fence() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// Compiler fence to prevent reordering
#[inline(always)]
pub fn compiler_fence() {
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Cache line size (typically 64 bytes on x86_64)
pub const CACHE_LINE_SIZE: usize = 64;

/// Align data structure to cache line boundary
#[repr(align(64))]
pub struct CacheAligned<T> {
    pub data: T,
}

impl<T> CacheAligned<T> {
    #[inline(always)]
    pub fn new(data: T) -> Self {
        Self { data }
    }
}

/// Prefetch array elements ahead of current position
#[inline(always)]
pub fn prefetch_array_ahead<T>(array: &[T], current_idx: usize, ahead: usize) {
    let prefetch_idx = current_idx + ahead;
    if prefetch_idx < array.len() {
        prefetch_read_t0(&array[prefetch_idx] as *const T);
    }
}

/// Stride prefetching for regular access patterns
pub struct StridePrefetcher {
    stride: usize,
    ahead: usize,
}

impl StridePrefetcher {
    pub fn new(stride: usize, ahead: usize) -> Self {
        Self { stride, ahead }
    }
    
    #[inline(always)]
    pub fn prefetch<T>(&self, base: *const T, current_offset: usize) {
        let prefetch_offset = current_offset + (self.stride * self.ahead);
        unsafe {
            let ptr = (base as *const u8).add(prefetch_offset) as *const T;
            prefetch_read_t0(ptr);
        }
    }
}

/// False sharing prevention padding
#[repr(C)]
pub struct PaddedValue<T> {
    pub value: T,
    _padding: [u8; CACHE_LINE_SIZE - std::mem::size_of::<T>()],
}

impl<T: Default> Default for PaddedValue<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            _padding: [0; CACHE_LINE_SIZE - std::mem::size_of::<T>()],
        }
    }
}

/// NUMA-aware memory allocation hints
pub fn numa_hint_local(ptr: *mut u8, size: usize) {
    #[cfg(target_os = "linux")]
    {
        use libc::{madvise, MADV_HUGEPAGE, MADV_WILLNEED};
        unsafe {
            madvise(ptr as *mut libc::c_void, size, MADV_WILLNEED);
            madvise(ptr as *mut libc::c_void, size, MADV_HUGEPAGE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_aligned() {
        let aligned = CacheAligned::new(42u64);
        assert_eq!(aligned.data, 42);
        
        // Verify alignment
        let addr = &aligned as *const _ as usize;
        assert_eq!(addr % CACHE_LINE_SIZE, 0);
    }
    
    #[test]
    fn test_stride_prefetcher() {
        let data: Vec<u64> = (0..1000).collect();
        let prefetcher = StridePrefetcher::new(8, 4); // 8 bytes stride, 4 elements ahead
        
        for i in 0..data.len() {
            // Simulate processing
            let _value = data[i];
            
            // Prefetch ahead
            if i < data.len() - 4 {
                prefetcher.prefetch(&data[0] as *const u64, i * 8);
            }
        }
    }
}