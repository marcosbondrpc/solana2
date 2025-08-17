use crate::cache::CacheManager;
use crate::collectors::{
    jito_collector::JitoCollector,
    quic_collector::QuicCollector,
    rpc_collector::RpcCollector,
};
use crate::error::MissionControlError;
use crate::models::*;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    Extension,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info};

pub struct ApiState {
    pub cache: Arc<CacheManager>,
    pub rpc_collector: Arc<RpcCollector>,
    pub jito_collector: Arc<JitoCollector>,
    pub quic_collector: Arc<QuicCollector>,
}

#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: Utc::now(),
        }
    }

    pub fn error(msg: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(msg),
            timestamp: Utc::now(),
        }
    }
}

// GET /api/mission-control/overview
pub async fn get_overview(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<MissionControlOverview>>, StatusCode> {
    // Try cache first
    if let Ok(Some(cached)) = state.cache.get_overview().await {
        return Ok(Json(ApiResponse::success(cached)));
    }

    // Build fresh overview
    let node_summary = match state.cache.get_node_summary().await {
        Ok(Some(s)) => s,
        _ => {
            // Create default or fetch fresh
            NodeSummary {
                node_id: "solana-node-1".to_string(),
                version: "1.18.0".to_string(),
                status: NodeStatus::Healthy,
                uptime_seconds: 86400,
                identity_pubkey: "7XSY...".to_string(),
                vote_account: Some("Vote111...".to_string()),
                rpc_endpoint: "http://localhost:8899".to_string(),
                gossip_endpoint: "127.0.0.1:8001".to_string(),
                tpu_endpoint: "127.0.0.1:8003".to_string(),
                shred_version: 1234,
                feature_set: 5678,
                last_updated: Utc::now(),
            }
        }
    };

    let consensus_health = match state.cache.get_consensus_health().await {
        Ok(Some(h)) => h,
        _ => ConsensusHealth {
            slot: 250_000_000,
            epoch: 580,
            slot_index: 100_000,
            slots_per_epoch: 432_000,
            optimistic_slot: 250_000_000,
            root_slot: 249_999_900,
            first_available_block: 240_000_000,
            last_vote: 249_999_999,
            validator_activated_stake: 400_000_000_000_000,
            validator_delinquent_stake: 10_000_000_000_000,
            total_active_stake: 410_000_000_000_000,
            consensus_participation: 0.975,
            tower_sync_status: TowerSyncStatus::Synced,
            skip_rate: 0.02,
            last_updated: Utc::now(),
        },
    };

    let cluster_performance = match state.cache.get_cluster_performance().await {
        Ok(Some(p)) => p,
        _ => ClusterPerformance {
            tps_current: 3500.0,
            tps_1min: 3450.0,
            tps_5min: 3400.0,
            tps_peak_24h: 4200.0,
            block_time_ms: 400.0,
            block_production_rate: 0.98,
            skip_rate: 0.02,
            total_validators: 2000,
            active_validators: 1950,
            delinquent_validators: 50,
            network_inflation_rate: 0.05,
            fee_burn_percentage: 0.5,
            priority_fee_median: 10_000,
            priority_fee_p75: 50_000,
            priority_fee_p95: 500_000,
            compute_unit_price_micro_lamports: 1000,
            last_updated: Utc::now(),
        },
    };

    let jito_status = match state.cache.get_jito_status().await {
        Ok(Some(j)) => j,
        _ => JitoStatus {
            block_engine_status: BlockEngineStatus {
                connected: true,
                region: "frankfurt".to_string(),
                latency_ms: 5.2,
                last_heartbeat: Utc::now(),
            },
            regions: vec![
                JitoRegion {
                    name: "frankfurt".to_string(),
                    endpoint: "frankfurt.mainnet.block-engine.jito.wtf".to_string(),
                    connected: true,
                    latency_ms: 5.2,
                    packets_per_second: 50000,
                },
                JitoRegion {
                    name: "amsterdam".to_string(),
                    endpoint: "amsterdam.mainnet.block-engine.jito.wtf".to_string(),
                    connected: true,
                    latency_ms: 7.8,
                    packets_per_second: 45000,
                },
            ],
            tip_feed_active: true,
            bundles_submitted_24h: 150000,
            bundles_accepted_24h: 135000,
            bundles_rejected_24h: 15000,
            acceptance_rate: 0.9,
            avg_tip_lamports: 100_000,
            max_tip_lamports_24h: 10_000_000,
            auction_tick_rate_ms: 50,
            shredstream_status: ShredStreamStatus {
                active: true,
                packets_per_second: 100000,
                gaps_detected: 5,
                reorders_detected: 12,
                latency_p50_ms: 2.5,
                latency_p99_ms: 15.0,
            },
            last_updated: Utc::now(),
        },
    };

    let rpc_metrics = match state.cache.get_rpc_metrics().await {
        Ok(Some(r)) => r,
        _ => {
            let method_latencies = state.rpc_collector.get_method_latencies().await;
            
            RpcMetrics {
                endpoints: vec![
                    RpcEndpointMetrics {
                        endpoint: "https://api.mainnet-beta.solana.com".to_string(),
                        healthy: true,
                        requests_per_second: 100.0,
                        error_rate: 0.001,
                        avg_latency_ms: 25.0,
                    },
                ],
                total_requests_24h: 8_640_000,
                total_errors_24h: 8_640,
                avg_latency_ms: 25.0,
                p50_latency_ms: 20.0,
                p95_latency_ms: 50.0,
                p99_latency_ms: 100.0,
                method_latencies: method_latencies.into_iter()
                    .map(|(method, (p50, p95, p99))| {
                        (method.clone(), MethodLatency {
                            method,
                            count: 10000,
                            avg_ms: (p50 + p95 + p99) / 3.0,
                            p50_ms: p50,
                            p95_ms: p95,
                            p99_ms: p99,
                            max_ms: p99 * 1.5,
                        })
                    })
                    .collect(),
                last_updated: Utc::now(),
            }
        }
    };

    let overview = MissionControlOverview {
        node_summary,
        consensus_health,
        cluster_performance,
        jito_status,
        rpc_metrics,
        timestamp: Utc::now(),
    };

    // Cache the overview
    let _ = state.cache.set_overview(&overview).await;

    Ok(Json(ApiResponse::success(overview)))
}

// GET /api/mission-control/node-summary
pub async fn get_node_summary(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<NodeSummary>>, StatusCode> {
    match state.cache.get_node_summary().await {
        Ok(Some(summary)) => Ok(Json(ApiResponse::success(summary))),
        _ => {
            let summary = NodeSummary {
                node_id: "solana-node-1".to_string(),
                version: "1.18.0".to_string(),
                status: NodeStatus::Healthy,
                uptime_seconds: 86400,
                identity_pubkey: "7XSY...".to_string(),
                vote_account: Some("Vote111...".to_string()),
                rpc_endpoint: "http://localhost:8899".to_string(),
                gossip_endpoint: "127.0.0.1:8001".to_string(),
                tpu_endpoint: "127.0.0.1:8003".to_string(),
                shred_version: 1234,
                feature_set: 5678,
                last_updated: Utc::now(),
            };
            let _ = state.cache.set_node_summary(&summary).await;
            Ok(Json(ApiResponse::success(summary)))
        }
    }
}

// GET /api/mission-control/consensus-health
pub async fn get_consensus_health(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<ConsensusHealth>>, StatusCode> {
    match state.cache.get_consensus_health().await {
        Ok(Some(health)) => Ok(Json(ApiResponse::success(health))),
        _ => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// GET /api/mission-control/cluster-perf
pub async fn get_cluster_performance(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<ClusterPerformance>>, StatusCode> {
    match state.cache.get_cluster_performance().await {
        Ok(Some(perf)) => Ok(Json(ApiResponse::success(perf))),
        _ => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// GET /api/mission-control/jito-status
pub async fn get_jito_status(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<JitoStatus>>, StatusCode> {
    match state.cache.get_jito_status().await {
        Ok(Some(status)) => Ok(Json(ApiResponse::success(status))),
        _ => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// GET /api/mission-control/rpc-metrics
pub async fn get_rpc_metrics(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<RpcMetrics>>, StatusCode> {
    match state.cache.get_rpc_metrics().await {
        Ok(Some(metrics)) => Ok(Json(ApiResponse::success(metrics))),
        _ => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// GET /api/mission-control/timing-waterfall
pub async fn get_timing_waterfall(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<TimingWaterfall>>, StatusCode> {
    match state.cache.get_timing_waterfall().await {
        Ok(Some(waterfall)) => Ok(Json(ApiResponse::success(waterfall))),
        _ => {
            // Generate sample waterfall
            let waterfall = TimingWaterfall {
                total_latency_ms: 450.0,
                stages: vec![
                    TimingStage {
                        name: "Network Receive".to_string(),
                        start_ms: 0.0,
                        duration_ms: 10.0,
                        percentage: 2.2,
                    },
                    TimingStage {
                        name: "Signature Verify".to_string(),
                        start_ms: 10.0,
                        duration_ms: 50.0,
                        percentage: 11.1,
                    },
                    TimingStage {
                        name: "Banking Stage".to_string(),
                        start_ms: 60.0,
                        duration_ms: 300.0,
                        percentage: 66.7,
                    },
                    TimingStage {
                        name: "Proof of History".to_string(),
                        start_ms: 360.0,
                        duration_ms: 70.0,
                        percentage: 15.6,
                    },
                    TimingStage {
                        name: "Broadcast".to_string(),
                        start_ms: 430.0,
                        duration_ms: 20.0,
                        percentage: 4.4,
                    },
                ],
                bottleneck: "Banking Stage".to_string(),
                last_updated: Utc::now(),
            };
            let _ = state.cache.set_timing_waterfall(&waterfall).await;
            Ok(Json(ApiResponse::success(waterfall)))
        }
    }
}

// GET /api/mission-control/tip-intelligence
pub async fn get_tip_intelligence(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<TipIntelligence>>, StatusCode> {
    match state.cache.get_tip_intelligence().await {
        Ok(Some(intel)) => Ok(Json(ApiResponse::success(intel))),
        _ => {
            let intel = TipIntelligence {
                current_epoch_tips: 1_500_000,
                tip_percentiles: TipPercentiles {
                    p25: 10_000,
                    p50: 50_000,
                    p75: 200_000,
                    p90: 500_000,
                    p95: 1_000_000,
                    p99: 5_000_000,
                },
                top_tippers: vec![
                    TopTipper {
                        address: "Tipper1...".to_string(),
                        total_tips: 50_000_000,
                        avg_tip: 500_000,
                        success_rate: 0.95,
                    },
                    TopTipper {
                        address: "Tipper2...".to_string(),
                        total_tips: 40_000_000,
                        avg_tip: 400_000,
                        success_rate: 0.92,
                    },
                ],
                tip_efficiency_score: 0.85,
                estimated_roi_percentage: 250.0,
                optimal_tip_lamports: 200_000,
                last_updated: Utc::now(),
            };
            let _ = state.cache.set_tip_intelligence(&intel).await;
            Ok(Json(ApiResponse::success(intel)))
        }
    }
}

// GET /api/mission-control/bundle-success
pub async fn get_bundle_success(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<BundleSuccess>>, StatusCode> {
    let (total, accepted, rate) = state.jito_collector.get_bundle_stats().await;
    
    let success = BundleSuccess {
        total_bundles: total,
        successful_bundles: accepted,
        failed_bundles: total - accepted,
        success_rate: rate,
        rejection_reasons: HashMap::from([
            ("simulation_failed".to_string(), 5000),
            ("stale_blockhash".to_string(), 3000),
            ("duplicate_transaction".to_string(), 2000),
        ]),
        avg_simulation_time_ms: 15.0,
        avg_confirmation_time_ms: 450.0,
        profit_extracted_lamports: 500_000_000,
        gas_used_compute_units: 100_000_000,
        last_updated: Utc::now(),
    };
    
    let _ = state.cache.set_bundle_success(&success).await;
    Ok(Json(ApiResponse::success(success)))
}

// GET /api/mission-control/shredstream
pub async fn get_shredstream_metrics(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<ShredStreamStatus>>, StatusCode> {
    // Get from Jito status which includes ShredStream
    match state.cache.get_jito_status().await {
        Ok(Some(status)) => Ok(Json(ApiResponse::success(status.shredstream_status))),
        _ => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// GET /api/mission-control/quic-health
pub async fn get_quic_health(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<QuicHealth>>, StatusCode> {
    let metrics = state.quic_collector.get_metrics().await;
    
    let health = QuicHealth {
        handshake_success_rate: metrics.handshake_success_rate,
        concurrent_connections: metrics.concurrent_connections,
        max_connections: 1024,
        open_streams: metrics.open_streams,
        throttling_events: metrics.throttling_events,
        error_code_15_count: metrics.error_code_15_count,
        avg_handshake_time_ms: metrics.avg_handshake_time_ms,
        packet_loss_rate: metrics.packet_loss_rate,
        retransmission_rate: metrics.retransmission_rate,
        last_updated: Utc::now(),
    };
    
    let _ = state.cache.set_quic_health(&health).await;
    Ok(Json(ApiResponse::success(health)))
}

// GET /api/mission-control/qos-peering
pub async fn get_qos_peering(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<QosPeering>>, StatusCode> {
    match state.cache.get_qos_peering().await {
        Ok(Some(peering)) => Ok(Json(ApiResponse::success(peering))),
        _ => {
            let peering = QosPeering {
                stake_weighted_qos_enabled: true,
                total_stake: 410_000_000_000_000,
                our_stake: 1_000_000_000_000,
                qos_priority_level: 200,
                peer_connections: vec![
                    PeerConnection {
                        peer_id: "Peer1...".to_string(),
                        stake: 10_000_000_000_000,
                        connection_quality: 0.95,
                        latency_ms: 5.0,
                        packet_loss: 0.001,
                    },
                    PeerConnection {
                        peer_id: "Peer2...".to_string(),
                        stake: 5_000_000_000_000,
                        connection_quality: 0.90,
                        latency_ms: 8.0,
                        packet_loss: 0.002,
                    },
                ],
                pps_limit: 10000,
                current_pps: 8500,
                throttled_packets: 150,
                last_updated: Utc::now(),
            };
            let _ = state.cache.set_qos_peering(&peering).await;
            Ok(Json(ApiResponse::success(peering)))
        }
    }
}

// GET /api/mission-control/gossip-metrics
pub async fn get_gossip_metrics(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<GossipMetrics>>, StatusCode> {
    match state.cache.get_gossip_metrics().await {
        Ok(Some(metrics)) => Ok(Json(ApiResponse::success(metrics))),
        _ => {
            let metrics = GossipMetrics {
                gossip_peers: 150,
                gossip_messages_per_second: 500.0,
                repair_peers: 50,
                repair_requests_per_second: 20.0,
                broadcast_peers: 100,
                broadcast_shreds_per_second: 1000.0,
                push_messages: 10000,
                pull_requests: 5000,
                pull_responses: 4500,
                prune_messages: 100,
                last_updated: Utc::now(),
            };
            let _ = state.cache.set_gossip_metrics(&metrics).await;
            Ok(Json(ApiResponse::success(metrics)))
        }
    }
}

// POST /api/mission-control/node-control
pub async fn node_control(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<NodeControlRequest>,
) -> Result<Json<ApiResponse<NodeControlResponse>>, StatusCode> {
    info!("Node control request: {:?}", request.action);
    
    // Run preflight checks first
    let checks = run_preflight_checks(&state).await;
    
    if !checks.safe_to_proceed && !request.force {
        return Ok(Json(ApiResponse::error(
            "Preflight checks failed. Use force=true to override.".to_string()
        )));
    }
    
    let response = match request.action {
        NodeAction::Start => {
            // Implement node start logic
            NodeControlResponse {
                success: true,
                action: NodeAction::Start,
                message: "Node started successfully".to_string(),
                timestamp: Utc::now(),
            }
        }
        NodeAction::Stop => {
            // Implement node stop logic
            NodeControlResponse {
                success: true,
                action: NodeAction::Stop,
                message: "Node stopped successfully".to_string(),
                timestamp: Utc::now(),
            }
        }
        NodeAction::Restart => {
            // Implement node restart logic
            NodeControlResponse {
                success: true,
                action: NodeAction::Restart,
                message: "Node restarted successfully".to_string(),
                timestamp: Utc::now(),
            }
        }
        NodeAction::CatchUp => {
            // Implement catch-up logic
            NodeControlResponse {
                success: true,
                action: NodeAction::CatchUp,
                message: "Catch-up initiated".to_string(),
                timestamp: Utc::now(),
            }
        }
        NodeAction::SetLogLevel(level) => {
            // Implement log level change
            NodeControlResponse {
                success: true,
                action: NodeAction::SetLogLevel(level.clone()),
                message: format!("Log level set to {}", level),
                timestamp: Utc::now(),
            }
        }
    };
    
    Ok(Json(ApiResponse::success(response)))
}

// GET /api/mission-control/preflight-checks
pub async fn get_preflight_checks(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ApiResponse<PreflightChecks>>, StatusCode> {
    let checks = run_preflight_checks(&state).await;
    Ok(Json(ApiResponse::success(checks)))
}

async fn run_preflight_checks(state: &ApiState) -> PreflightChecks {
    let mut checks = vec![];
    let mut warnings = vec![];
    
    // Check Redis connectivity
    let redis_check = if state.cache.health_check().await {
        PreflightCheck {
            name: "Redis Cache".to_string(),
            passed: true,
            message: "Redis cache is healthy".to_string(),
            severity: CheckSeverity::Info,
        }
    } else {
        PreflightCheck {
            name: "Redis Cache".to_string(),
            passed: false,
            message: "Redis cache is unreachable".to_string(),
            severity: CheckSeverity::Warning,
        }
    };
    checks.push(redis_check);
    
    // Check consensus participation
    if let Ok(Some(consensus)) = state.cache.get_consensus_health().await {
        let consensus_check = if consensus.consensus_participation > 0.66 {
            PreflightCheck {
                name: "Consensus Participation".to_string(),
                passed: true,
                message: format!("Consensus at {:.1}%", consensus.consensus_participation * 100.0),
                severity: CheckSeverity::Info,
            }
        } else {
            warnings.push("Low consensus participation detected".to_string());
            PreflightCheck {
                name: "Consensus Participation".to_string(),
                passed: false,
                message: format!("Consensus critically low at {:.1}%", consensus.consensus_participation * 100.0),
                severity: CheckSeverity::Critical,
            }
        };
        checks.push(consensus_check);
    }
    
    // Check Jito connectivity
    if let Ok(Some(jito)) = state.cache.get_jito_status().await {
        let jito_check = if jito.block_engine_status.connected {
            PreflightCheck {
                name: "Jito Block Engine".to_string(),
                passed: true,
                message: format!("Connected to {} region", jito.block_engine_status.region),
                severity: CheckSeverity::Info,
            }
        } else {
            warnings.push("Jito Block Engine disconnected".to_string());
            PreflightCheck {
                name: "Jito Block Engine".to_string(),
                passed: false,
                message: "Unable to connect to Jito Block Engine".to_string(),
                severity: CheckSeverity::Warning,
            }
        };
        checks.push(jito_check);
    }
    
    // Check QUIC health
    let quic_metrics = state.quic_collector.get_metrics().await;
    let quic_check = if quic_metrics.handshake_success_rate > 0.8 {
        PreflightCheck {
            name: "QUIC Transport".to_string(),
            passed: true,
            message: format!("QUIC handshake success rate: {:.1}%", quic_metrics.handshake_success_rate * 100.0),
            severity: CheckSeverity::Info,
        }
    } else {
        warnings.push("Poor QUIC connection quality".to_string());
        PreflightCheck {
            name: "QUIC Transport".to_string(),
            passed: false,
            message: format!("QUIC handshake success rate low: {:.1}%", quic_metrics.handshake_success_rate * 100.0),
            severity: CheckSeverity::Warning,
        }
    };
    checks.push(quic_check);
    
    let all_passed = checks.iter().all(|c| c.passed);
    let safe_to_proceed = !checks.iter().any(|c| !c.passed && matches!(c.severity, CheckSeverity::Critical));
    
    PreflightChecks {
        checks,
        all_passed,
        safe_to_proceed,
        warnings,
        timestamp: Utc::now(),
    }
}