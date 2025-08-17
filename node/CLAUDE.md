# CLAUDE.md - AI Development Guide

## 🚀 Legendary Solana MEV Infrastructure

Welcome! You're working on a **state-of-the-art MEV system** capable of handling **billions in volume** with **sub-10ms latency**.

### System Overview

This is a **protobuf-first**, **ultra-low-latency** Solana MEV infrastructure with:
- **End-to-end decision latency**: ≤8ms P50, ≤20ms P99
- **Bundle land rates**: ≥65% contested, ≥85% off-peak  
- **Model inference**: ≤100μs P99
- **ClickHouse ingestion**: ≥235k rows/s sustained

### Architecture Components

1. **Ingestion Layer**
   - WebSocket/WebTransport with protobuf
   - Zero-copy message processing
   - QUIC with custom congestion control

2. **Decision Engine**
   - Treelite-compiled models with PGO
   - Thompson Sampling for route selection
   - Decision DNA fingerprinting

3. **Execution Layer**
   - Direct RPC, Jito bundles, hedged sending
   - DSCP marking and SO_TXTIME
   - Leader-phase timing gates

4. **Storage Layer**
   - ClickHouse with Protobuf Kafka engine
   - Typed tables (no JSON in hot path)
   - S3 cold storage with TTL

5. **Control Plane**
   - Ed25519 signing with 2-of-3 multisig
   - ACK hash-chain for audit trail
   - SLO-based kill-switches

### Key Files & Locations

```
/home/kidgordones/0solana/node/
├── arbitrage-data-capture/
│   ├── protocol/           # Protobuf schemas
│   ├── clickhouse/         # DDL and migrations
│   ├── rust-services/      # Core MEV agents
│   └── tools/              # Treelite, DNA anchor
├── frontend/
│   └── apps/dashboard/     # React dashboard
├── backend/
│   └── services/control-plane/  # FastAPI backend
├── api/
│   └── control.py          # Signed command handler
└── Makefile                # All operations
```

### Critical Performance Targets

| Metric | Target | Why Critical |
|--------|--------|--------------|
| Decision Latency P50 | ≤8ms | Beat competitors to opportunities |
| Decision Latency P99 | ≤20ms | Consistent performance under load |
| Bundle Land Rate | ≥65% | Profitability threshold |
| Model Inference | ≤100μs | Leaves time for network/execution |
| Ingestion Rate | ≥200k/s | Handle peak Solana activity |

### Development Commands

```bash
# Essential operations
make legendary          # Bootstrap everything
make lab-smoke-test     # Run comprehensive tests
make tmux              # Start the cockpit
make health-check      # Check system status

# Model operations
make train-model MODULE=mev DATE_RANGE=7d
make models-super      # Build Treelite
make pgo-mev          # Profile-guided optimization
make swap-model       # Hot-reload without restart

# Emergency controls
make emergency-stop    # Kill all trading
make throttle PERCENT=10  # Reduce to 10%
make audit-trail      # View command history
```

### Testing Checklist

Before ANY deployment:
1. ✅ Run `make lab-smoke-test`
2. ✅ Verify P50 ≤ 8ms, P99 ≤ 20ms
3. ✅ Check bundle land rate ≥ 65%
4. ✅ Confirm ClickHouse ingestion ≥ 200k/s
5. ✅ Test kill-switches trigger correctly
6. ✅ Verify Decision DNA on all events
7. ✅ Check ACK chain has no gaps

### Network Access

- **Frontend**: http://45.157.234.184:3001
- **Backend API**: http://45.157.234.184:8000
- **API Docs**: http://45.157.234.184:8000/docs

### GitHub Auto-Sync

The system auto-syncs with GitHub:
- **Pull**: Every minute (automatic restart on changes)
- **Push**: Every 5 minutes (automatic commit)
- **Repository**: https://github.com/marcosbondrpc/solana2

### Safety Mechanisms

1. **SLO Kill-Switches**
   - Auto-throttle if P99 > 20ms
   - Stop if land rate < 55%
   - Kill if negative EV > 1%

2. **Cryptographic Controls**
   - All commands Ed25519 signed
   - 2-of-3 multisig for critical ops
   - Immutable ACK chain

3. **Decision Audit**
   - Every decision has DNA fingerprint
   - Daily Merkle anchor to Solana
   - Full forensic traceability

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Port 3000 conflict | Frontend uses 3001, Grafana disabled |
| Missing protobuf | Run `make proto` |
| Model not loading | Check `make models-super` output |
| Low land rate | Adjust Thompson Sampling priors |
| High latency | Check network DSCP marking |
| ClickHouse lag | Verify Kafka consumer groups |

### Performance Optimization

1. **CPU Optimization**
   ```bash
   sudo cpupower frequency-set -g performance
   taskset -c 0-7 ./mev-agent  # Pin to cores
   ```

2. **Network Tuning**
   ```bash
   sudo sysctl -w net.core.rmem_max=134217728
   sudo sysctl -w net.core.wmem_max=134217728
   ```

3. **Model Optimization**
   ```bash
   make pgo-collect  # Collect profile
   make pgo-build    # Build with profile
   ```

### Important Notes

- **NEVER** disable Decision DNA tracking
- **NEVER** bypass cryptographic signing
- **ALWAYS** test in staging before prod
- **ALWAYS** monitor SLOs in real-time
- **ALWAYS** have kill-switches enabled

### Feature Flags

```bash
export TRANSPORT_MODE=proto      # Use protobuf (not json)
export HEDGED_SEND=on           # Enable hedged sending
export CANARIES=on              # Enable canary transactions
export KILL_SWITCH=on           # Enable auto-throttle
export DNA_TRACKING=on          # Track decision lineage
```

### Monitoring Dashboards

Access Grafana dashboards for:
- **MEV Opportunities**: Real-time feed with DNA
- **Bandit Performance**: Thompson Sampling metrics
- **System Health**: Latency, land rates, ingestion
- **Control Plane**: Commands, ACKs, multisig status

### Contact & Support

- **GitHub**: https://github.com/marcosbondrpc/solana2
- **Lead Developer**: @marcosbondrpc
- **System Status**: Check SYSTEM_STATUS.md

---

## Mission Critical

This system is designed to handle **billions in MEV volume**. Every microsecond matters. Every decision is tracked. Every command is signed. 

The architecture is built for:
- **Speed**: Sub-10ms decisions
- **Scale**: 200k+ events/second
- **Safety**: Cryptographic verification
- **Profit**: Adaptive optimization

Remember: **Built for billions. Optimized for microseconds.**

---

*Last updated: August 16, 2025*
*System version: LEGENDARY v1.0*