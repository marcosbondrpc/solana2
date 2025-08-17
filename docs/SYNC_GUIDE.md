# 🔄 GitHub Sync Guide

## Quick Commands

### 1. **Push Changes to GitHub Immediately**
```bash
# Option A: Using Makefile (Recommended)
make sync

# Option B: Using sync script
./scripts/sync-now.sh

# Option C: Manual git commands
git add -A
git commit -m "Your commit message"
git push origin master
```

### 2. **Check Auto-Sync Status**
```bash
# View timer status
systemctl status github-sync.timer

# View recent sync logs
tail -20 /tmp/github-sync.log

# Check if auto-sync is running
ps aux | grep github-sync
```

### 3. **Configure Auto-Sync**

The auto-sync runs automatically every 5 minutes. It:
- Pulls latest changes from GitHub
- Commits any local changes
- Pushes to GitHub

**Files involved:**
- `/home/kidgordones/0solana/solana2/scripts/github-sync.sh` - Main sync script
- `/home/kidgordones/0solana/solana2/scripts/sync-now.sh` - Manual sync trigger
- `/etc/systemd/system/github-sync.timer` - Systemd timer (5-minute interval)
- `/etc/systemd/system/github-sync.service` - Systemd service

### 4. **Stop/Start Auto-Sync**
```bash
# Stop auto-sync
sudo systemctl stop github-sync.timer

# Start auto-sync
sudo systemctl start github-sync.timer

# Disable auto-sync permanently
sudo systemctl disable github-sync.timer

# Enable auto-sync
sudo systemctl enable github-sync.timer
```

### 5. **Troubleshooting**

If sync fails, check:

```bash
# View error logs
tail -50 /tmp/github-sync.log | grep -E "error|fail|Error|Failed"

# Check git status
git status

# Check remote configuration
git remote -v

# Test connection to GitHub
git ls-remote origin
```

### 6. **First-Time Setup**

Before auto-sync works, you need to configure authentication:

```bash
# Set up your GitHub credentials (choose one):

# Option 1: Using token (replace with your token)
git remote set-url origin https://USERNAME:TOKEN@github.com/marcosbondrpc/solana2.git

# Option 2: Using GitHub CLI
gh auth login

# Option 3: Using SSH
ssh-keygen -t ed25519
# Add public key to GitHub
git remote set-url origin git@github.com:marcosbondrpc/solana2.git
```

## 📊 Auto-Sync Schedule

- **Pull from GitHub**: Every 1 minute (cron job)
- **Push to GitHub**: Every 5 minutes (systemd timer)
- **Log rotation**: Daily at midnight
- **Backup**: Before each push

## 🔧 Manual Operations

### Force Push All Changes
```bash
cd /home/kidgordones/0solana/solana2
git add -A
git commit -m "Force sync: $(date)"
git push origin master --force
```

### Pull Latest Changes
```bash
cd /home/kidgordones/0solana/solana2
git pull origin master
```

### View Commit History
```bash
git log --oneline -10
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `scripts/github-sync.sh` | Main sync logic |
| `scripts/sync-now.sh` | Manual trigger |
| `scripts/setup-github-sync.sh` | Initial setup |
| `/tmp/github-sync.log` | Sync logs |
| `.git/config` | Git configuration |
| `.gitignore` | Files to exclude |

## ⚠️ Security Notes

1. **NEVER** commit sensitive files (keys, tokens, passwords)
2. **ALWAYS** use `.gitignore` for private data
3. **REVIEW** changes before pushing: `git diff HEAD`
4. **BACKUP** important data before force pushing

## 🚀 Quick Start

To sync your changes right now:

```bash
# 1. Check what will be synced
git status

# 2. Sync everything
make sync

# 3. Verify it worked
git log -1
```

That's it! Your code is now on GitHub.