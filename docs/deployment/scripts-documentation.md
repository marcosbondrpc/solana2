# 📁 Scripts Directory Structure

## Organization

All scripts are organized into logical categories for better maintainability:

```
scripts/
├── sync/                 # GitHub synchronization scripts
│   ├── auto-commit.sh       # Auto-commit and push to GitHub
│   ├── github-pull-sync.sh  # Pull changes from GitHub
│   ├── ensure-sync.sh       # Ensure sync is working
│   └── cron-sync.sh         # Cron job sync wrapper
│
├── services/             # Service management scripts
│   ├── services-manager.sh  # Main service manager
│   ├── start-monitoring-stack.sh  # Start monitoring
│   └── solana-node.sh      # Solana node operations
│
├── monitoring/           # System monitoring scripts
│   ├── monitor-sync.sh     # Monitor GitHub sync
│   ├── test-system.sh      # Test system components
│   ├── test-full-stack.sh  # Full stack testing
│   └── verify-setup.sh     # Verify installation
│
└── setup/               # Setup and configuration scripts
    ├── setup-cron-sync.sh        # Setup cron jobs
    ├── setup-github-sync-service.sh  # Setup systemd
    ├── setup-auto-sync.sh        # Auto-sync setup
    └── fix-docker-permissions.sh # Fix Docker perms
```

## 🎮 Master Control Script

Use the main control script for all operations:

```bash
./mev-control [command]
```

### Available Commands

#### Sync Commands
- `sync-status` - Show GitHub sync status
- `sync-push` - Push changes to GitHub immediately
- `sync-pull` - Pull changes from GitHub immediately
- `sync-ensure` - Ensure sync is working properly

#### Service Commands
- `services-status` - Show all service statuses
- `services-restart` - Restart all services
- `frontend-restart` - Restart frontend only
- `backend-restart` - Restart backend only

#### Monitoring Commands
- `monitor` - Show real-time system monitoring
- `test-system` - Run comprehensive system test
- `verify-setup` - Verify complete setup

#### Docker Commands
- `docker-status` - Show Docker container status
- `docker-restart` - Restart all Docker containers
- `docker-logs` - Show Docker logs

#### Setup Commands
- `setup-sync` - Setup GitHub sync from scratch
- `setup-services` - Setup systemd services

#### Shortcuts
- `status` - Show complete system status
- `restart` - Restart everything
- `logs` - Show all logs

## 📝 Individual Script Usage

### Sync Scripts

```bash
# Auto-commit and push to GitHub
./scripts/sync/auto-commit.sh

# Pull latest from GitHub
./scripts/sync/github-pull-sync.sh

# Ensure sync is working
./scripts/sync/ensure-sync.sh
```

### Service Scripts

```bash
# Manage services (used by systemd)
./scripts/services/services-manager.sh

# Start monitoring stack
./scripts/services/start-monitoring-stack.sh
```

### Monitoring Scripts

```bash
# Monitor sync status
./scripts/monitoring/monitor-sync.sh

# Test system components
./scripts/monitoring/test-system.sh

# Verify setup
./scripts/monitoring/verify-setup.sh
```

### Setup Scripts

```bash
# Setup cron jobs
./scripts/setup/setup-cron-sync.sh

# Setup systemd service
./scripts/setup/setup-github-sync-service.sh

# Fix Docker permissions
./scripts/setup/fix-docker-permissions.sh
```

## 🔄 Automatic Execution

### Cron Jobs
The following scripts run automatically via cron:
- `scripts/sync/auto-commit.sh` - Every 5 minutes
- `scripts/sync/github-pull-sync.sh` - Every minute

### Systemd Service
The following script is managed by systemd:
- `scripts/services/services-manager.sh` - Always running

## 📊 Logs

All scripts write logs to:
- `/home/kidgordones/0solana/node/sync.log` - Sync operations
- `/tmp/frontend.log` - Frontend service
- `/tmp/backend.log` - Backend service

## 🚀 Quick Actions

```bash
# Check everything
./mev-control status

# Restart everything
./mev-control restart

# Force sync now
./mev-control sync-push
./mev-control sync-pull

# View logs
./mev-control logs
```

## 🔧 Troubleshooting

If scripts aren't working:

1. **Check permissions**:
```bash
chmod +x scripts/**/*.sh
chmod +x mev-control
```

2. **Check paths in cron**:
```bash
crontab -l
```

3. **Check systemd service**:
```bash
sudo systemctl status mev-services
```

4. **Run ensure sync**:
```bash
./scripts/sync/ensure-sync.sh
```

---

**Note**: All scripts use absolute paths and are designed to work from any directory.