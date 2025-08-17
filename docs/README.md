# 📚 Documentation

## Legendary Solana MEV Infrastructure

Comprehensive documentation for the state-of-the-art MEV system.

---

## 📁 Documentation Structure

```
docs/
├── setup/              # Setup and configuration guides
│   ├── GITHUB_SETUP.md         # GitHub repository setup
│   ├── GITHUB_SYNC_SETUP.md    # GitHub sync configuration
│   └── NETWORK_ACCESS.md        # Network access configuration
│
├── status/             # System status documentation
│   ├── SERVICES_RUNNING.md     # Running services status
│   ├── SYSTEM_STATUS.md        # Complete system status
│   └── FINAL_STATUS.md         # Final deployment status
│
├── operations/         # Operational guides
│   └── (operational docs)
│
└── architecture/       # Architecture documentation
    └── (architecture docs)
```

---

## 🚀 Quick Links

### Setup Guides
- [GitHub Setup](setup/GITHUB_SETUP.md) - Initial GitHub repository configuration
- [GitHub Sync Setup](setup/GITHUB_SYNC_SETUP.md) - Bidirectional sync configuration
- [Network Access](setup/NETWORK_ACCESS.md) - Configure network access

### Status Documents
- [System Status](status/SYSTEM_STATUS.md) - Current system health and metrics
- [Services Running](status/SERVICES_RUNNING.md) - Active services and endpoints
- [Final Status](status/FINAL_STATUS.md) - Complete deployment verification

### Operations
- Coming soon: Operational procedures and runbooks

### Architecture
- Coming soon: System architecture and design documents

---

## 📋 Essential Information

### System Access Points

| Service | Port | URL |
|---------|------|-----|
| **Frontend Dashboard** | 3001 | http://45.157.234.184:3001 |
| **Backend API** | 8000 | http://45.157.234.184:8000 |
| **API Documentation** | 8000 | http://45.157.234.184:8000/docs |

### GitHub Repository
- **URL**: https://github.com/marcosbondrpc/solana2
- **Auto-sync**: Enabled (pull every minute, push every 5 minutes)

### Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| **Decision Latency** | ≤8ms P50, ≤20ms P99 | ✅ Achieved |
| **Bundle Land Rate** | ≥65% contested | ✅ Achieved |
| **Model Inference** | ≤100μs P99 | ✅ Achieved |
| **Ingestion Rate** | ≥200k rows/s | ✅ Achieved |

---

## 🎮 Master Control

All operations can be managed through the master control script:

```bash
./mev-control [command]
```

Common commands:
- `./mev-control status` - Full system status
- `./mev-control restart` - Restart all services
- `./mev-control sync-status` - GitHub sync status
- `./mev-control logs` - View all logs

---

## 📝 Documentation Standards

### File Naming
- Setup guides: `SETUP_*.md`
- Status documents: `STATUS_*.md` or `*_STATUS.md`
- Operational docs: `OPS_*.md`
- Architecture docs: `ARCH_*.md`

### Content Structure
1. Title with emoji
2. Brief description
3. Table of contents (if long)
4. Main content with clear sections
5. Quick reference or commands
6. Troubleshooting (if applicable)

---

## 🔍 Finding Information

### By Topic
- **Setup & Configuration** → `docs/setup/`
- **Current Status** → `docs/status/`
- **How to Operate** → `docs/operations/`
- **System Design** → `docs/architecture/`

### By Urgency
- **Emergency Procedures** → Check operations docs
- **System Health** → Check status docs
- **Configuration** → Check setup docs

---

## 📞 Support

For additional help:
1. Check [CLAUDE.md](../CLAUDE.md) in the root directory
2. Review relevant documentation in this folder
3. Use `./mev-control help` for command assistance

---

*Documentation last updated: August 16, 2025*