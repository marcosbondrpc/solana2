use axum::{
    extract::{Extension, Path, Query},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use prometheus::{Encoder, TextEncoder};

use crate::scrapper::DataScrapper;
use crate::dataset_manager::{DatasetManager, CreateDatasetRequest};
use crate::ml_pipeline::{MLPipeline, TrainRequest};
use crate::storage::ClickHouseStorage;

// Scrapper endpoints
pub async fn start_scrapping(
    Extension(scrapper): Extension<Arc<DataScrapper>>,
    Json(params): Json<ScrapperParams>,
) -> impl IntoResponse {
    tokio::spawn(async move {
        if let Err(e) = scrapper.start_scrapping().await {
            tracing::error!("Scrapper error: {}", e);
        }
    });
    
    Json(serde_json::json!({
        "status": "started",
        "message": "Data scrapping started in background"
    }))
}

pub async fn stop_scrapping(
    Extension(scrapper): Extension<Arc<DataScrapper>>,
) -> impl IntoResponse {
    scrapper.stop();
    
    Json(serde_json::json!({
        "status": "stopped",
        "message": "Data scrapping stopped"
    }))
}

pub async fn get_scrapper_status(
    Extension(scrapper): Extension<Arc<DataScrapper>>,
) -> impl IntoResponse {
    let status = scrapper.get_status();
    Json(status)
}

pub async fn get_progress(
    Extension(scrapper): Extension<Arc<DataScrapper>>,
) -> impl IntoResponse {
    let status = scrapper.get_status();
    
    Json(serde_json::json!({
        "current_slot": status.current_slot,
        "target_slot": status.target_slot,
        "blocks_processed": status.blocks_processed,
        "transactions_processed": status.transactions_processed,
        "percentage": if status.target_slot > 0 {
            (status.current_slot as f64 / status.target_slot as f64 * 100.0)
        } else {
            0.0
        },
        "is_running": status.is_running,
        "errors": status.errors,
    }))
}

// Dataset endpoints
pub async fn list_datasets(
    Extension(manager): Extension<Arc<DatasetManager>>,
) -> impl IntoResponse {
    match manager.list_datasets().await {
        Ok(datasets) => Json(datasets).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            }))
        ).into_response()
    }
}

pub async fn create_dataset(
    Extension(manager): Extension<Arc<DatasetManager>>,
    Json(request): Json<CreateDatasetRequest>,
) -> impl IntoResponse {
    match manager.create_dataset(request).await {
        Ok(dataset) => Json(dataset).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            }))
        ).into_response()
    }
}

pub async fn get_dataset(
    Extension(manager): Extension<Arc<DatasetManager>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match manager.get_dataset(&id).await {
        Ok(Some(dataset)) => Json(dataset).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "Dataset not found"
            }))
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            }))
        ).into_response()
    }
}

pub async fn delete_dataset(
    Extension(manager): Extension<Arc<DatasetManager>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match manager.delete_dataset(&id).await {
        Ok(_) => Json(serde_json::json!({
            "status": "deleted"
        })).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            }))
        ).into_response()
    }
}

pub async fn export_dataset(
    Extension(manager): Extension<Arc<DatasetManager>>,
    Path(id): Path<String>,
    Json(request): Json<ExportRequest>,
) -> impl IntoResponse {
    match manager.export_dataset(&id, request.format).await {
        Ok(data) => Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/octet-stream")
            .header("Content-Disposition", format!("attachment; filename=\"dataset_{}.bin\"", id))
            .body(data.into())
            .unwrap(),
        Err(e) => Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(e.to_string().into())
            .unwrap()
    }
}

// ML endpoints
pub async fn start_training(
    Extension(ml): Extension<Arc<MLPipeline>>,
    Json(request): Json<TrainRequest>,
) -> impl IntoResponse {
    match ml.start_training(request).await {
        Ok(model) => Json(model).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            }))
        ).into_response()
    }
}

pub async fn list_models(
    Extension(ml): Extension<Arc<MLPipeline>>,
) -> impl IntoResponse {
    match ml.list_models().await {
        Ok(models) => Json(models).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            }))
        ).into_response()
    }
}

pub async fn get_model(
    Extension(ml): Extension<Arc<MLPipeline>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match ml.get_model(&id).await {
        Ok(Some(model)) => Json(model).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "Model not found"
            }))
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            }))
        ).into_response()
    }
}

pub async fn evaluate_model(
    Extension(ml): Extension<Arc<MLPipeline>>,
    Path(id): Path<String>,
    body: bytes::Bytes,
) -> impl IntoResponse {
    match ml.evaluate_model(&id, body.to_vec()).await {
        Ok(metrics) => Json(metrics).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            }))
        ).into_response()
    }
}

// Data endpoints
pub async fn get_blocks(
    Extension(storage): Extension<Arc<ClickHouseStorage>>,
    Query(params): Query<BlockQuery>,
) -> impl IntoResponse {
    let start = params.start_slot.unwrap_or(0);
    let end = params.end_slot.unwrap_or(u64::MAX);
    let limit = params.limit.unwrap_or(100).min(1000);
    
    match storage.get_blocks(start, end, limit).await {
        Ok(blocks) => Json(blocks).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            }))
        ).into_response()
    }
}

pub async fn get_transactions(
    Extension(storage): Extension<Arc<ClickHouseStorage>>,
    Query(params): Query<TransactionQuery>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(100).min(1000);
    
    match storage.get_transactions(params.slot, limit).await {
        Ok(transactions) => Json(transactions).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string()
            }))
        ).into_response()
    }
}

pub async fn get_accounts(
    Extension(_storage): Extension<Arc<ClickHouseStorage>>,
) -> impl IntoResponse {
    // TODO: Implement account queries
    Json(serde_json::json!({
        "accounts": []
    }))
}

pub async fn get_programs(
    Extension(_storage): Extension<Arc<ClickHouseStorage>>,
) -> impl IntoResponse {
    // TODO: Implement program queries
    Json(serde_json::json!({
        "programs": []
    }))
}

// Metrics endpoint
pub async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    
    encoder.encode(&metric_families, &mut buffer).unwrap();
    
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", encoder.format_type())
        .body(buffer.into())
        .unwrap()
}

// Request/Response types
#[derive(Debug, Deserialize)]
pub struct ScrapperParams {
    pub start_slot: Option<u64>,
    pub end_slot: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ExportRequest {
    pub format: crate::dataset_manager::ExportFormat,
}

#[derive(Debug, Deserialize)]
pub struct BlockQuery {
    pub start_slot: Option<u64>,
    pub end_slot: Option<u64>,
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct TransactionQuery {
    pub slot: Option<u64>,
    pub limit: Option<u32>,
}