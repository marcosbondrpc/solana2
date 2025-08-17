use anyhow::Result;
use axum::{
	routing::get,
	response::{IntoResponse, Response},
	Json, Router,
};
use clap::Parser;
use std::net::SocketAddr;
use tokio::{net::TcpListener, signal};
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
	#[arg(short, long, default_value = "config.toml")]
	config: String,

	#[arg(short, long, default_value = "8081")]
	port: u16,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
	tracing_subscriber::fmt().with_env_filter("info").init();

	let args = Args::parse();

	let app = Router::new()
		.route("/health", get(health))
		.route("/metrics", get(prometheus_metrics));

	let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
	let listener = TcpListener::bind(addr).await?;

	info!("Dashboard API listening on 0.0.0.0:{}", args.port);

	axum::serve(listener, app)
		.with_graceful_shutdown(shutdown_signal())
		.await?;

	Ok(())
}

async fn health() -> impl IntoResponse {
	Json(serde_json::json!({
		"status": "healthy",
		"version": env!("CARGO_PKG_VERSION")
	}))
}

async fn prometheus_metrics() -> impl IntoResponse {
	use prometheus::{Encoder, TextEncoder};

	let encoder = TextEncoder::new();
	let metric_families = prometheus::gather();
	let mut buffer = Vec::new();
	encoder.encode(&metric_families, &mut buffer).unwrap();

	Response::builder()
		.header("Content-Type", encoder.format_type())
		.body(String::from_utf8(buffer).unwrap())
		.unwrap()
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