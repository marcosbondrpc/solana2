use anyhow::Result;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{info, warn};

#[path = "../src/main.rs"]
mod sandwich_detector;

#[path = "../src/network.rs"]
mod network;

#[path = "../src/ml_inference.rs"]  
mod ml_inference;

#[path = "../src/bundle_builder.rs"]
mod bundle_builder;

#[path = "../src/submission.rs"]
mod submission;

use sandwich_detector::SandwichAgent;
use network::{PacketBatch, Packet};
use ml_inference::{MLEngine, SandwichFeatures};

#[tokio::test]
async fn test_end_to_end_sandwich_detection() -> Result<()> {
    // Initialize the sandwich detector
    let agent = SandwichAgent::new().await?;
    
    // Create mock packet batch
    let mut packets = Vec::new();
    for i in 0..10 {
        packets.push(Packet {
            data: create_mock_swap_data(i),
            received_at: Instant::now(),
            source: format!("test_source_{}", i),
        });
    }
    
    let batch = PacketBatch {
        packets,
        received_at: Instant::now(),
        batch_id: "test_batch_001".to_string(),
    };
    
    // Test packet processing latency
    let start = Instant::now();
    let features = agent.ml_engine.extract_features_simd(&batch)?;
    let extraction_time = start.elapsed();
    
    assert!(extraction_time < Duration::from_micros(100), 
            "Feature extraction took {:?}, expected <100μs", extraction_time);
    
    // Test ML inference latency
    let start = Instant::now();
    let (is_sandwich, confidence) = agent.ml_engine.infer(&features).await?;
    let inference_time = start.elapsed();
    
    assert!(inference_time < Duration::from_micros(200),
            "ML inference took {:?}, expected <200μs", inference_time);
    
    info!("Sandwich detection: {} with confidence {:.2}%", 
          is_sandwich, confidence * 100.0);
    
    // Test bundle building if sandwich detected
    if is_sandwich && confidence > 0.7 {
        let start = Instant::now();
        let bundle = agent.bundle_builder.build_sandwich_bundle(
            &batch.packets[0],
            features.clone(),
            confidence
        ).await?;
        let bundle_time = start.elapsed();
        
        assert!(bundle_time < Duration::from_millis(1),
                "Bundle building took {:?}, expected <1ms", bundle_time);
        
        info!("Bundle created with {} transactions", bundle.transactions.len());
    }
    
    // Test full E2E latency
    let start = Instant::now();
    let decision = process_packet_batch(&agent, batch).await?;
    let total_time = start.elapsed();
    
    assert!(total_time < Duration::from_millis(8),
            "E2E decision took {:?}, expected <8ms", total_time);
    
    info!("✅ All latency requirements met!");
    Ok(())
}

#[tokio::test]
async fn test_simd_feature_extraction() -> Result<()> {
    let ml_engine = MLEngine::new(vec![2, 3, 4, 5]).await?;
    
    // Create large batch to test SIMD performance
    let mut packets = Vec::new();
    for i in 0..1000 {
        packets.push(Packet {
            data: create_mock_swap_data(i),
            received_at: Instant::now(),
            source: format!("source_{}", i),
        });
    }
    
    let batch = PacketBatch {
        packets,
        received_at: Instant::now(),
        batch_id: "perf_test".to_string(),
    };
    
    // Benchmark SIMD extraction
    let start = Instant::now();
    for _ in 0..100 {
        let _ = ml_engine.extract_features_simd(&batch)?;
    }
    let elapsed = start.elapsed();
    let per_extraction = elapsed / 100;
    
    assert!(per_extraction < Duration::from_micros(100),
            "SIMD extraction took {:?} per batch, expected <100μs", per_extraction);
    
    info!("✅ SIMD performance validated: {:?}/extraction", per_extraction);
    Ok(())
}

#[tokio::test]
async fn test_multi_bundle_ladder() -> Result<()> {
    let bundle_builder = bundle_builder::BundleBuilder::new().await?;
    
    // Test ladder strategy generation
    let expected_profit = 1_000_000_000; // 1 SOL
    let confidence = 0.85;
    
    let ladder = bundle_builder.generate_tip_ladder(expected_profit, confidence);
    
    assert_eq!(ladder.len(), 4, "Should generate 4-tier ladder");
    
    // Verify tip percentages
    let tip_percentages: Vec<f64> = ladder.iter()
        .map(|t| *t as f64 / expected_profit as f64)
        .collect();
    
    assert!((tip_percentages[0] - 0.50).abs() < 0.01, "Tier 1 should be ~50%");
    assert!((tip_percentages[1] - 0.70).abs() < 0.01, "Tier 2 should be ~70%");
    assert!((tip_percentages[2] - 0.85).abs() < 0.01, "Tier 3 should be ~85%");
    assert!((tip_percentages[3] - 0.95).abs() < 0.01, "Tier 4 should be ~95%");
    
    info!("✅ Multi-bundle ladder validated");
    Ok(())
}

#[tokio::test]
async fn test_dual_path_submission() -> Result<()> {
    let submitter = submission::DualPathSubmitter::new().await?;
    
    // Create mock bundle
    let bundle = create_mock_bundle();
    
    // Test parallel submission
    let start = Instant::now();
    let (tpu_result, jito_result) = tokio::join!(
        submitter.submit_to_tpu(&bundle),
        submitter.submit_to_jito(&bundle)
    );
    let submission_time = start.elapsed();
    
    assert!(submission_time < Duration::from_millis(1),
            "Dual submission took {:?}, expected <1ms", submission_time);
    
    // At least one path should succeed in test
    assert!(tpu_result.is_ok() || jito_result.is_ok(),
            "At least one submission path should succeed");
    
    info!("✅ Dual-path submission validated");
    Ok(())
}

#[tokio::test]
async fn test_redis_state_management() -> Result<()> {
    // Test Redis bundle tracking
    let redis_client = redis::Client::open("redis://127.0.0.1/")?;
    let mut con = redis_client.get_async_connection().await?;
    
    let bundle_id = "test_bundle_001";
    
    // Test bundle creation
    let script = include_str!("../scripts/redis/bundle_tracking.lua");
    let result: (i32, String, String) = redis::Script::new(script)
        .key(bundle_id)
        .key("active_bundles")
        .key("sandwich_metrics")
        .arg("create")
        .arg(Instant::now().elapsed().as_secs())
        .arg(r#"{"expected_profit": 1000000, "confidence": 0.85}"#)
        .invoke_async(&mut con)
        .await?;
    
    assert_eq!(result.0, 1, "Bundle creation should succeed");
    assert_eq!(result.1, "pending", "New bundle should be pending");
    
    // Test tip escalation
    let escalation_script = include_str!("../scripts/redis/tip_escalation.lua");
    let escalation_result: (i32, f64, String) = redis::Script::new(escalation_script)
        .key(format!("bundle:{}", bundle_id))
        .key("competitor_tips")
        .key("network_load")
        .arg(1000000)  // expected profit
        .arg(50000)    // current tip
        .arg(0.85)     // confidence
        .arg(0.5)      // max tip ratio
        .invoke_async(&mut con)
        .await?;
    
    assert_eq!(escalation_result.0, 1, "Tip escalation should succeed");
    assert!(escalation_result.1 > 50000.0, "Tip should be escalated");
    
    info!("✅ Redis state management validated");
    Ok(())
}

#[tokio::test]
async fn test_clickhouse_throughput() -> Result<()> {
    // Test ClickHouse write performance
    let client = clickhouse::Client::default()
        .with_url("http://localhost:8123")
        .with_database("mev_sandwich");
    
    // Prepare batch insert
    let mut inserts = Vec::new();
    for i in 0..10000 {
        inserts.push(SandwichRecord {
            slot: 200000000 + i,
            timestamp: chrono::Utc::now(),
            bundle_id: format!("bundle_{}", i),
            expected_profit: 1000000,
            actual_profit: Some(950000),
            confidence: 0.85,
            decision_time_us: 6500,
            ml_inference_time_us: 95,
        });
    }
    
    // Benchmark insertion
    let start = Instant::now();
    let mut insert = client.insert("mev_sandwich")?;
    for record in inserts {
        insert.write(&record).await?;
    }
    insert.end().await?;
    let elapsed = start.elapsed();
    
    let throughput = 10000.0 / elapsed.as_secs_f64();
    assert!(throughput > 200000.0, 
            "ClickHouse throughput {:.0} rows/s, expected >200k", throughput);
    
    info!("✅ ClickHouse throughput validated: {:.0} rows/s", throughput);
    Ok(())
}

// Helper functions

fn create_mock_swap_data(index: u32) -> Vec<u8> {
    let mut data = vec![0u8; 256];
    
    // Add mock transaction data
    data[0..4].copy_from_slice(&index.to_le_bytes());
    
    // Add mock amounts
    let input_amount = 1000000000u64 + index as u64 * 100000;
    let output_amount = 990000000u64 + index as u64 * 95000;
    data[64..72].copy_from_slice(&input_amount.to_le_bytes());
    data[72..80].copy_from_slice(&output_amount.to_le_bytes());
    
    // Add mock pool reserves
    let reserve_a = 1000000000000u64;
    let reserve_b = 1000000000000u64;
    data[96..104].copy_from_slice(&reserve_a.to_le_bytes());
    data[104..112].copy_from_slice(&reserve_b.to_le_bytes());
    
    // Add mock gas price
    let gas_price = 50000u64;
    data[112..120].copy_from_slice(&gas_price.to_le_bytes());
    
    data
}

fn create_mock_bundle() -> bundle_builder::SandwichBundle {
    bundle_builder::SandwichBundle {
        id: "mock_bundle_001".to_string(),
        transactions: vec![
            vec![1, 2, 3, 4], // frontrun tx
            vec![5, 6, 7, 8], // victim tx
            vec![9, 10, 11, 12], // backrun tx
        ],
        expected_profit: 1000000000,
        gas_estimate: 50000,
        tip: 50000000,
        confidence: 0.85,
        metadata: Default::default(),
    }
}

async fn process_packet_batch(agent: &SandwichAgent, batch: PacketBatch) -> Result<Decision> {
    // Simulate full E2E processing
    let features = agent.ml_engine.extract_features_simd(&batch)?;
    let (is_sandwich, confidence) = agent.ml_engine.infer(&features).await?;
    
    if is_sandwich && confidence > 0.7 {
        let bundle = agent.bundle_builder.build_sandwich_bundle(
            &batch.packets[0],
            features,
            confidence
        ).await?;
        
        let (tpu_result, jito_result) = tokio::join!(
            agent.submitter.submit_to_tpu(&bundle),
            agent.submitter.submit_to_jito(&bundle)
        );
        
        Ok(Decision::Execute { bundle_id: bundle.id })
    } else {
        Ok(Decision::Skip { reason: "Low confidence".to_string() })
    }
}

#[derive(Debug)]
enum Decision {
    Execute { bundle_id: String },
    Skip { reason: String },
}

#[derive(Debug, clickhouse::Row, serde::Serialize)]
struct SandwichRecord {
    slot: u64,
    timestamp: chrono::DateTime<chrono::Utc>,
    bundle_id: String,
    expected_profit: i64,
    actual_profit: Option<i64>,
    confidence: f64,
    decision_time_us: u32,
    ml_inference_time_us: u32,
}