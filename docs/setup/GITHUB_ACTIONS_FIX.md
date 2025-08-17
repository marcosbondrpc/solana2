# 🔧 GitHub Actions SSH Deployment Fix

## Issue Resolution

### Problem
GitHub Actions was failing with error:
```
could not parse as int value for flag port,p: strconv.ParseInt: parsing "": invalid syntax
```

### Root Cause
The `appleboy/ssh-action@v0.1.5` version had issues parsing the port parameter correctly.

### Solution Applied
1. **Upgraded SSH Action Version**
   - Changed from `v0.1.5` to `v1.0.3`
   - This version has better parameter parsing

2. **Fixed Port Parameter**
   - Changed port from `port: 22` to `port: "22"` (quoted string)
   - Ensures proper type handling

### Updated Workflow Configuration

```yaml
- name: Deploy to server via SSH
  uses: appleboy/ssh-action@v1.0.3
  with:
    host: 45.157.234.184
    username: kidgordones
    key: ${{ secrets.SERVER_SSH_KEY }}
    port: "22"
    script: |
      cd /home/kidgordones/0solana/node
      # deployment script continues...
```

## Verification Steps

1. **Check GitHub Actions Status**
   - Go to: https://github.com/marcosbondrpc/solana2/actions
   - Look for green checkmarks on recent commits

2. **Monitor Server Logs**
   ```bash
   tail -f /home/kidgordones/0solana/node/sync.log
   ```
   - Look for "New changes pulled from GitHub" messages
   - Check for service restart messages

3. **Test Manual Trigger**
   - Push a test commit to trigger workflow
   - Monitor Actions tab for execution

## Troubleshooting

### If Actions Still Fail

1. **Verify SSH Key**
   - Ensure `SERVER_SSH_KEY` secret is correctly set in GitHub
   - Key should be the private key content (not path)

2. **Check Network Access**
   - Verify server allows SSH from GitHub Actions IPs
   - Port 22 must be open

3. **Test SSH Locally**
   ```bash
   ssh -i ~/.ssh/github_actions kidgordones@45.157.234.184 -p 22 "echo 'SSH OK'"
   ```

### Alternative Solutions

If the current fix doesn't work, consider:

1. **Use Latest Action Version**
   ```yaml
   uses: appleboy/ssh-action@master
   ```

2. **Try Different SSH Action**
   ```yaml
   uses: cross-the-world/ssh-pipeline@master
   ```

3. **Use Script Mode**
   ```yaml
   - name: Deploy via SSH
     env:
       SSH_KEY: ${{ secrets.SERVER_SSH_KEY }}
     run: |
       echo "$SSH_KEY" > key.pem
       chmod 600 key.pem
       ssh -i key.pem -o StrictHostKeyChecking=no kidgordones@45.157.234.184 'cd /home/kidgordones/0solana/node && git pull'
   ```

## Current Status

✅ **Fixed**: Workflow updated to v1.0.3 with quoted port parameter
✅ **Committed**: Changes pushed to GitHub repository
🔄 **Testing**: Monitoring for successful deployments

## Related Files

- `.github/workflows/sync.yml` - Main workflow file
- `scripts/sync/github-pull-sync.sh` - Server-side sync script
- `docs/setup/GITHUB_SYNC_SETUP.md` - Complete sync documentation

---

*Last updated: August 16, 2025*