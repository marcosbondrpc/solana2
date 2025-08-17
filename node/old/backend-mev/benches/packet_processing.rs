use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use backend_mev::{
    pipeline::{PacketPool, SimdFeatureExtractor, SpscRing},
    mev_engine::{ArbitrageModule, SandwichModule},
};
use std::sync::Arc;

fn benchmark_packet_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("packet_pool");
    
    let pool = Arc::new(PacketPool::new(10_000));
    
    group.bench_function("acquire_release", |b| {
        b.iter(|| {
            let handle = black_box(pool.acquire().unwrap());
            drop(handle);
        });
    });
    
    group.bench_function("parallel_acquire", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..100)
                .filter_map(|_| pool.acquire())
                .collect();
            black_box(handles);
        });
    });
    
    group.finish();
}

fn benchmark_spsc_ring(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsc_ring");
    
    for capacity in [100, 1000, 10000].iter() {
        let mut ring = SpscRing::new(*capacity);
        
        group.throughput(Throughput::Elements(*capacity as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            capacity,
            |b, &cap| {
                b.iter(|| {
                    // Fill ring
                    for i in 0..cap/2 {
                        ring.try_push(i);
                    }
                    // Drain ring
                    for _ in 0..cap/2 {
                        black_box(ring.try_pop());
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_simd_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_features");
    
    // Test different data sizes
    for size in [64, 256, 1024, 4096].iter() {
        let prices: Vec<f64> = (0..*size).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let volumes: Vec<f64> = (0..*size).map(|i| 1000.0 + (i as f64) * 10.0).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("extract_prices", size),
            &prices,
            |b, p| {
                b.iter(|| {
                    black_box(SimdFeatureExtractor::extract_prices_simd(p));
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("calculate_vwap", size),
            &(&prices, &volumes),
            |b, (p, v)| {
                b.iter(|| {
                    black_box(SimdFeatureExtractor::calculate_vwap_simd(p, v));
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_mev_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("mev_detection");
    
    let arb_module = ArbitrageModule::new();
    let sandwich_module = SandwichModule::new();
    
    // Create test transaction data
    let tx_data = vec![0u8; 1024];
    
    group.bench_function("arbitrage_detection", |b| {
        b.iter(|| {
            black_box(arb_module.find_opportunities(10_000_000));
        });
    });
    
    group.bench_function("sandwich_detection", |b| {
        b.iter(|| {
            black_box(sandwich_module.analyze_transaction(&tx_data));
        });
    });
    
    group.finish();
}

fn benchmark_memory_operations(c: &mut Criterion) {
    use backend_mev::optimization::{prefetch_memory, MemoryPool};
    
    let mut group = c.benchmark_group("memory_ops");
    
    // Test prefetching
    let data = vec![0u8; 65536];
    group.bench_function("prefetch_64kb", |b| {
        b.iter(|| {
            prefetch_memory(black_box(data.as_ptr()), black_box(data.len()));
        });
    });
    
    // Test memory pool
    let mut pool = MemoryPool::new();
    
    for size in [64, 256, 1024, 4096].iter() {
        group.bench_with_input(
            BenchmarkId::new("pool_alloc", size),
            size,
            |b, &s| {
                b.iter(|| {
                    let ptr = black_box(pool.allocate(s).unwrap());
                    pool.deallocate(ptr, s);
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_end_to_end_latency(c: &mut Criterion) {
    use std::time::Instant;
    
    let mut group = c.benchmark_group("end_to_end");
    
    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            let start = Instant::now();
            
            // Simulate packet reception
            let packet = vec![0u8; 512];
            
            // Simulate feature extraction
            let _features = SimdFeatureExtractor::extract_prices_simd(&[100.0; 64]);
            
            // Simulate MEV detection
            // (lightweight version for benchmark)
            
            // Simulate transaction building
            let _tx = vec![0u8; 256];
            
            // Measure total time
            black_box(start.elapsed());
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_packet_pool,
    benchmark_spsc_ring,
    benchmark_simd_features,
    benchmark_mev_detection,
    benchmark_memory_operations,
    benchmark_end_to_end_latency
);
criterion_main!(benches);