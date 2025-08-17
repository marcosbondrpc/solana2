# Architecture (Initial)
- Frontend: Next.js 14 app router in `frontend2` with ISR/SSR as needed.
- Realtime path: WS/WebTransport → Service Worker (optional) → Web Worker(s) → protobuf decode → zstd (WASM) decompress → coalesced frames → UI stores (Zustand).
- Rendering: uPlot charts, virtualized tables, 60 FPS goal, minimal main-thread work.
- Controls: protobuf-backed commands; backend Ed25519 signatures; 2-step confirmations.
- Ops modules: Arbitrage, MEV sandwich, datasets, training/hot-reload, routing/bandit tuning, SLO guardrails.
- AuthZ: role-guarded actions in UI.