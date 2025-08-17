# Architecture Overview
- Data ingress: Kafka → FastAPI stream endpoints.
- Client transport: WebSockets/WebTransport with binary frames.
- Processing: Web Workers for protobuf decode and zstd WASM; coalescing to reduce UI load.
- Visualization: uPlot + virtualized tables.
- Control surface: protobuf commands with signed requests; confirmations in UI.
- Observability: Grafana via `NEXT_PUBLIC_GRAFANA_URL`.