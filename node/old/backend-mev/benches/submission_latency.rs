use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use backend_mev::{
    dual_path::{DualPathSubmitter, SubmissionRequest, SubmissionPriority},
    congestion::NanoBurst,
};
use solana_sdk::{
    signature::Keypair,
    transaction::Transaction,
    system_instruction,
    pubkey::Pubkey,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn benchmark_submission_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    // Setup
    let auth_keypair = Arc::new(Keypair::new());
    let submitter = rt.block_on(async {
        DualPathSubmitter::new(
            vec!["127.0.0.1:8003".to_string()],
            "127.0.0.1:8005".to_string(),
            auth_keypair.clone(),
            50,
        ).unwrap()
    });
    
    let mut group = c.benchmark_group("submission_latency");
    
    // Benchmark different transaction sizes
    for size in [1, 5, 10, 20].iter() {
        let tx = create_test_transaction(*size);
        let request = SubmissionRequest {
            transaction: tx,
            estimated_profit: 100_000_000,
            deadline: Instant::now() + Duration::from_secs(1),
            priority: SubmissionPriority::High,
            bundle_id: None,
        };
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_instructions", size)),
            &request,
            |b, req| {
                b.to_async(&rt).iter(|| async {
                    let _ = black_box(submitter.submit(req.clone()).await);
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_congestion_control(c: &mut Criterion) {
    let mut group = c.benchmark_group("congestion_control");
    
    // Test NanoBurst performance
    let mut nanoburst = NanoBurst::new(1600, 24);
    
    group.bench_function("nanoburst_on_ack", |b| {
        b.iter(|| {
            nanoburst.on_ack(
                black_box(quinn_proto::Instant::now()),
                black_box(quinn_proto::Instant::now()),
                black_box(1200),
                black_box(false),
                black_box(&quinn_proto::RttEstimator::default()),
            );
        });
    });
    
    group.bench_function("nanoburst_on_congestion", |b| {
        b.iter(|| {
            nanoburst.on_congestion_event(
                black_box(quinn_proto::Instant::now()),
                black_box(quinn_proto::Instant::now()),
                black_box(false),
                black_box(1200),
            );
        });
    });
    
    group.finish();
}

fn benchmark_path_selection(c: &mut Criterion) {
    use backend_mev::dual_path::PathSelector;
    
    let selector = PathSelector {
        tpu_success_rate: Arc::new(parking_lot::RwLock::new(0.7)),
        jito_success_rate: Arc::new(parking_lot::RwLock::new(0.9)),
        recent_latencies: Arc::new(parking_lot::RwLock::new(Vec::new())),
    };
    
    let mut group = c.benchmark_group("path_selection");
    
    for priority in [
        SubmissionPriority::Low,
        SubmissionPriority::Medium,
        SubmissionPriority::High,
        SubmissionPriority::UltraHigh,
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", priority)),
            &priority,
            |b, &p| {
                b.iter(|| {
                    black_box(selector.select_path(
                        p,
                        100_000_000,
                        Instant::now() + Duration::from_millis(100),
                    ));
                });
            },
        );
    }
    
    group.finish();
}

fn create_test_transaction(num_instructions: usize) -> Transaction {
    let payer = Keypair::new();
    let mut instructions = Vec::with_capacity(num_instructions);
    
    for _ in 0..num_instructions {
        instructions.push(system_instruction::transfer(
            &payer.pubkey(),
            &Pubkey::new_unique(),
            1_000_000,
        ));
    }
    
    Transaction::new_with_payer(&instructions, Some(&payer.pubkey()))
}

criterion_group!(
    benches,
    benchmark_submission_latency,
    benchmark_congestion_control,
    benchmark_path_selection
);
criterion_main!(benches);