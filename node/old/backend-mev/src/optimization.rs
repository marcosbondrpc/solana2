use anyhow::Result;
use libc::{
    cpu_set_t, sched_setaffinity, sched_setscheduler, sched_param,
    SCHED_FIFO, CPU_SET, CPU_ZERO, MAP_HUGETLB, MAP_ANONYMOUS, MAP_PRIVATE,
    mmap, munmap, PROT_READ, PROT_WRITE, MAP_FAILED,
};
use std::os::unix::io::AsRawFd;
use std::ptr;
use std::mem;

/// Set process to real-time priority
pub fn set_realtime_priority() -> Result<()> {
    unsafe {
        let param = sched_param {
            sched_priority: 99, // Maximum real-time priority
        };
        
        let ret = sched_setscheduler(0, SCHED_FIFO, &param);
        if ret < 0 {
            // May fail if not running as root - that's OK
            tracing::warn!("Failed to set real-time priority (need root): {}", std::io::Error::last_os_error());
        } else {
            tracing::info!("Set process to real-time priority");
        }
    }
    
    Ok(())
}

/// Pin threads to specific CPU cores for optimal cache usage
pub fn setup_cpu_affinity() -> Result<()> {
    unsafe {
        let mut cpu_set: cpu_set_t = mem::zeroed();
        CPU_ZERO(&mut cpu_set);
        
        // Pin to first 4 cores (adjust based on system)
        for cpu in 0..4 {
            CPU_SET(cpu, &mut cpu_set);
        }
        
        let ret = sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &cpu_set);
        if ret < 0 {
            tracing::warn!("Failed to set CPU affinity: {}", std::io::Error::last_os_error());
        } else {
            tracing::info!("Set CPU affinity to cores 0-3");
        }
    }
    
    Ok(())
}

/// Enable huge pages for reduced TLB misses
pub fn enable_huge_pages() -> Result<()> {
    // Check if huge pages are available
    let hugepage_size = get_hugepage_size()?;
    
    if hugepage_size > 0 {
        tracing::info!("Huge pages available: {} bytes", hugepage_size);
        
        // Allocate some memory with huge pages as a test
        unsafe {
            let size = hugepage_size;
            let addr = mmap(
                ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                -1,
                0,
            );
            
            if addr == MAP_FAILED {
                tracing::warn!("Failed to allocate huge pages: {}", std::io::Error::last_os_error());
            } else {
                // Successfully allocated, now free it
                munmap(addr, size);
                tracing::info!("Huge pages enabled");
            }
        }
    } else {
        tracing::warn!("Huge pages not available on this system");
    }
    
    Ok(())
}

fn get_hugepage_size() -> Result<usize> {
    // Read from /proc/meminfo
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    
    for line in meminfo.lines() {
        if line.starts_with("Hugepagesize:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(size_kb) = parts[1].parse::<usize>() {
                    return Ok(size_kb * 1024);
                }
            }
        }
    }
    
    Ok(0)
}

/// Disable CPU frequency scaling for consistent performance
pub fn disable_cpu_scaling() -> Result<()> {
    let governors = [
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        "/sys/devices/system/cpu/cpu1/cpufreq/scaling_governor",
        "/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor",
        "/sys/devices/system/cpu/cpu3/cpufreq/scaling_governor",
    ];
    
    for governor_path in &governors {
        if let Ok(_) = std::fs::write(governor_path, b"performance") {
            tracing::info!("Set CPU governor to performance for {}", governor_path);
        }
    }
    
    Ok(())
}

/// Optimize network stack settings
pub fn optimize_network_stack() -> Result<()> {
    // These require root access
    let optimizations = [
        ("/proc/sys/net/core/netdev_max_backlog", "50000"),
        ("/proc/sys/net/core/rmem_max", "134217728"),
        ("/proc/sys/net/core/wmem_max", "134217728"),
        ("/proc/sys/net/ipv4/tcp_rmem", "4096 87380 134217728"),
        ("/proc/sys/net/ipv4/tcp_wmem", "4096 65536 134217728"),
        ("/proc/sys/net/ipv4/tcp_congestion_control", "bbr"),
        ("/proc/sys/net/ipv4/tcp_notsent_lowat", "16384"),
        ("/proc/sys/net/ipv4/tcp_low_latency", "1"),
    ];
    
    for (path, value) in &optimizations {
        if let Ok(_) = std::fs::write(path, value.as_bytes()) {
            tracing::info!("Set {} to {}", path, value);
        }
    }
    
    Ok(())
}

/// Memory pool for zero-allocation operations
pub struct MemoryPool {
    pools: Vec<PoolSegment>,
    free_lists: Vec<Vec<usize>>,
}

struct PoolSegment {
    base: *mut u8,
    size: usize,
    block_size: usize,
}

impl MemoryPool {
    pub fn new() -> Self {
        let sizes = [64, 256, 1024, 4096, 16384, 65536];
        let mut pools = Vec::new();
        let mut free_lists = Vec::new();
        
        for &block_size in &sizes {
            let num_blocks = 1000;
            let total_size = block_size * num_blocks;
            
            unsafe {
                let base = mmap(
                    ptr::null_mut(),
                    total_size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                ) as *mut u8;
                
                if base as *mut libc::c_void != MAP_FAILED {
                    pools.push(PoolSegment {
                        base,
                        size: total_size,
                        block_size,
                    });
                    
                    let mut free_list = Vec::with_capacity(num_blocks);
                    for i in 0..num_blocks {
                        free_list.push(i);
                    }
                    free_lists.push(free_list);
                }
            }
        }
        
        Self { pools, free_lists }
    }
    
    pub fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        // Find appropriate pool
        for (i, pool) in self.pools.iter().enumerate() {
            if pool.block_size >= size {
                if let Some(block_idx) = self.free_lists[i].pop() {
                    unsafe {
                        return Some(pool.base.add(block_idx * pool.block_size));
                    }
                }
            }
        }
        None
    }
    
    pub fn deallocate(&mut self, ptr: *mut u8, size: usize) {
        // Find which pool this belongs to
        for (i, pool) in self.pools.iter().enumerate() {
            if pool.block_size >= size {
                unsafe {
                    let offset = ptr.offset_from(pool.base) as usize;
                    if offset < pool.size {
                        let block_idx = offset / pool.block_size;
                        self.free_lists[i].push(block_idx);
                        return;
                    }
                }
            }
        }
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        for pool in &self.pools {
            unsafe {
                munmap(pool.base as *mut libc::c_void, pool.size);
            }
        }
    }
}

/// NUMA-aware memory allocation
pub fn setup_numa_affinity() -> Result<()> {
    // Check NUMA topology
    if let Ok(nodes) = std::fs::read_dir("/sys/devices/system/node") {
        let numa_nodes: Vec<_> = nodes
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_name()
                    .to_str()
                    .map(|s| s.starts_with("node"))
                    .unwrap_or(false)
            })
            .collect();
        
        if numa_nodes.len() > 1 {
            tracing::info!("System has {} NUMA nodes", numa_nodes.len());
            // Would implement NUMA-aware allocation here
        }
    }
    
    Ok(())
}

/// Prefetch memory for reduced cache misses
#[inline(always)]
pub fn prefetch_memory(addr: *const u8, size: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        
        let mut offset = 0;
        while offset < size {
            _mm_prefetch(addr.add(offset) as *const i8, _MM_HINT_T0);
            offset += 64; // Cache line size
        }
    }
}

/// Lock memory pages to prevent swapping
pub fn lock_memory_pages(addr: *const u8, size: usize) -> Result<()> {
    unsafe {
        let ret = libc::mlock(addr as *const libc::c_void, size);
        if ret < 0 {
            tracing::warn!("Failed to lock memory pages: {}", std::io::Error::last_os_error());
        } else {
            tracing::info!("Locked {} bytes of memory", size);
        }
    }
    Ok(())
}

/// Compiler hints for branch prediction
#[inline(always)]
pub fn likely(b: bool) -> bool {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::intrinsics::likely(b)
    }
    #[cfg(not(target_arch = "x86_64"))]
    b
}

#[inline(always)]
pub fn unlikely(b: bool) -> bool {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::intrinsics::unlikely(b)
    }
    #[cfg(not(target_arch = "x86_64"))]
    b
}

/// Initialize all optimizations
pub fn initialize_all() -> Result<()> {
    set_realtime_priority()?;
    setup_cpu_affinity()?;
    enable_huge_pages()?;
    disable_cpu_scaling()?;
    optimize_network_stack()?;
    setup_numa_affinity()?;
    
    tracing::info!("All system optimizations initialized");
    Ok(())
}