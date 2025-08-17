use std::sync::Arc;
use tokio::signal;
use tracing::{error, info};
use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
    extract::Extension,
};
use tower_http::cors::CorsLayer;
use tower_http::compression::CompressionLayer;

mod config;
mod scrapper;
mod storage;
mod exporter;
mod dataset_manager;
mod ml_pipeline;
mod kafka_producer;
mod api;

use config::Config;
use scrapper::DataScrapper;
use storage::ClickHouseStorage;
use dataset_manager::DatasetManager;
use ml_pipeline::MLPipeline;
use kafka_producer::KafkaProducer;

#[tokio::main(flavor = "multi_thread", worker_threads = 32)]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("data_scrapper=debug,tower_http=debug")
        .json()
        .init();

    info!("Starting Data Scrapper Service - High Performance Mode");

    // Load configuration
    let config = Config::from_env()?;
    
    // Initialize ClickHouse storage
    let storage = Arc::new(ClickHouseStorage::new(&config).await?);
    
    // Initialize Kafka producer for streaming updates
    let kafka_producer = Arc::new(KafkaProducer::new(&config).await?);
    
    // Initialize data scrapper
    let scrapper = Arc::new(DataScrapper::new(
        config.clone(),
        storage.clone(),
        kafka_producer.clone(),
    ).await?);
    
    // Initialize dataset manager
    let dataset_manager = Arc::new(DatasetManager::new(
        storage.clone(),
        config.clone(),
    ));
    
    // Initialize ML pipeline
    let ml_pipeline = Arc::new(MLPipeline::new(
        dataset_manager.clone(),
        config.clone(),
    )?);
    
    // Start scrapper background task
    let scrapper_handle = tokio::spawn({
        let scrapper = scrapper.clone();
        async move {
            if let Err(e) = scrapper.start_scrapping().await {
                error!("Scrapper error: {}", e);
            }
        }
    });
    
    // Build API router
    let app = Router::new()
        // Scrapper control endpoints
        .route("/api/scrapper/start", post(api::start_scrapping))
        .route("/api/scrapper/stop", post(api::stop_scrapping))
        .route("/api/scrapper/status", get(api::get_scrapper_status))
        .route("/api/scrapper/progress", get(api::get_progress))
        
        // Dataset management endpoints
        .route("/api/datasets", get(api::list_datasets))
        .route("/api/datasets", post(api::create_dataset))
        .route("/api/datasets/:id", get(api::get_dataset))
        .route("/api/datasets/:id", delete(api::delete_dataset))
        .route("/api/datasets/:id/export", post(api::export_dataset))
        
        // ML training endpoints
        .route("/api/ml/train", post(api::start_training))
        .route("/api/ml/models", get(api::list_models))
        .route("/api/ml/models/:id", get(api::get_model))
        .route("/api/ml/models/:id/evaluate", post(api::evaluate_model))
        
        // Historical data endpoints
        .route("/api/data/blocks", get(api::get_blocks))
        .route("/api/data/transactions", get(api::get_transactions))
        .route("/api/data/accounts", get(api::get_accounts))
        .route("/api/data/programs", get(api::get_programs))
        
        // Metrics endpoint
        .route("/metrics", get(api::metrics_handler))
        
        // Add shared state
        .layer(Extension(scrapper))
        .layer(Extension(dataset_manager))
        .layer(Extension(ml_pipeline))
        .layer(Extension(storage))
        
        // Add middleware
        .layer(CorsLayer::permissive())
        .layer(CompressionLayer::new());
    
    // Start HTTP server
    let addr = format!("0.0.0.0:{}", config.api_port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    
    info!("Data Scrapper API listening on http://{}", addr);
    
    // Run server
    let server = axum::serve(listener, app);
    
    tokio::select! {
        _ = server => {},
        _ = shutdown_signal() => {
            info!("Shutting down Data Scrapper Service...");
        }
    }
    
    // Cleanup
    scrapper_handle.abort();
    
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}