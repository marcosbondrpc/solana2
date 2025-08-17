# GitHub Sync Setup Guide

## 🔄 Automatic Bidirectional Sync

This system has automatic bidirectional sync between your local server and GitHub.

### Current Configuration

- **Server IP**: 45.157.234.184
- **Sync Interval**: 
  - Pull from GitHub: Every 1 minute
  - Push to GitHub: Every 5 minutes
- **Repository**: https://github.com/marcosbondrpc/solana2

### SSH Key Setup for GitHub Actions

To enable GitHub Actions to deploy to your server, you need to:

1. **Generate SSH key on server** (if not already done):
```bash
ssh-keygen -t ed25519 -C "github-actions" -f ~/.ssh/github_actions -N ""
```

2. **Add public key to authorized_keys**:
```bash
cat ~/.ssh/github_actions.pub >> ~/.ssh/authorized_keys
```

3. **Copy the PRIVATE key**:
```bash
cat ~/.ssh/github_actions
```

4. **Add to GitHub Secrets**:
   - Go to: https://github.com/marcosbondrpc/solana2/settings/secrets/actions
   - Click "New repository secret"
   - Name: `SERVER_SSH_KEY`
   - Value: Paste the private key content
   - Click "Add secret"

### Local Server Sync (Already Configured)

The server has cron jobs that automatically:
- Pull changes from GitHub every minute
- Push changes to GitHub every 5 minutes

Check status:
```bash
./monitor-sync.sh
```

### GitHub Actions Workflow

The `.github/workflows/sync.yml` file is configured to:
- Trigger on push to main branch
- Trigger on pull requests
- Run every 5 minutes (scheduled)
- Allow manual trigger

### Monitoring

**Check sync status**:
```bash
./monitor-sync.sh
```

**View sync logs**:
```bash
tail -f sync.log
```

**Force sync**:
```bash
./ensure-sync.sh
```

### Troubleshooting

If sync is not working:

1. **Check cron jobs**:
```bash
crontab -l
```

2. **Test GitHub connection**:
```bash
git ls-remote origin HEAD
```

3. **Check SSH key**:
```bash
ssh -T git@github.com
```

4. **Reinstall sync**:
```bash
./ensure-sync.sh
```

### Service Auto-Restart

When files change:
- **Frontend files** (`frontend/`) → Frontend service restarts
- **Backend files** (`backend/`) → Backend service restarts

Services run on:
- Frontend: http://45.157.234.184:3001
- Backend: http://45.157.234.184:8000

---

## Summary

Your system has **complete bidirectional sync**:
- ✅ Local → GitHub (automatic push)
- ✅ GitHub → Local (automatic pull)
- ✅ Service auto-restart on changes
- ✅ GitHub Actions workflow ready

The sync ensures your code is always backed up and synchronized across all environments.