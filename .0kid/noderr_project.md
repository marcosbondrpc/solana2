# Project Overview
- Workspace: /home/kidgordones/0solana/solana2
- Frontend app: /home/kidgordones/0solana/solana2/frontend2

## PrimaryGoal
Create a production-ready Next.js 14 + TypeScript dashboard in `/home/kidgordones/0solana/solana2/frontend2` that connects to FastAPI + Kafka/ClickHouse/Redis using protobuf-first binary real-time streams (WS/WebTransport), worker-based decoding, zstd (WASM) decompression, and provides operator controls for arbitrage and MEV sandwich modules, datasets, model training/hot-reload, routing/bandit tuning, and SLO guardrails. Target 60 FPS rendering with minimal main-thread CPU, virtualized tables, uPlot charts, and role-guarded actions.

## Success Criteria
- Realtime: >50k events/min without stutter using binary frames + worker coalescing.
- Controls: All actions protobuf-backed, backend signs Ed25519, UX confirmations.
- Datasets/Training: Jobs trigger/monitor from UI; model hot-reload with version display and zero downtime.
- SLO: Guard toggles (throttle/kill) with 2-step confirm and safe defaults.
- Performance: Main-thread CPU <= 20% during sustained load.

## Non-goals
- Implementing backend APIs or infra (FastAPI/Kafka/ClickHouse/Redis).
- Production deployment pipelines.
- Full MEV/arbitrage strategy logic.
- Data migrations or schema design for ClickHouse.
- Non-dashboard services outside `frontend2`.

## Tech Stack
- Next.js 14 (app router), TypeScript
- Tailwind CSS, Radix UI, shadcn/ui
- Zustand (client state)
- uPlot (charts), virtualized tables
- Web Workers + OffscreenCanvas
- protobufjs / ts-proto for protobuf
- zstddec-wasm for zstd decompression
- WebSockets + WebTransport

## External Dependencies
- FastAPI (API/control plane)
- Kafka (stream ingress)
- ClickHouse (analytics store)
- Redis (cache/coordination)

## Required Environment Variables
- NEXT_PUBLIC_API_BASE
- NEXT_PUBLIC_WS_URL
- NEXT_PUBLIC_WT_URL
- NEXT_PUBLIC_GRAFANA_URL
- NEXT_PUBLIC_JWT_AUDIENCE