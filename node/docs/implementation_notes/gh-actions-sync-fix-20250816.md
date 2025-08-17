# GH Actions Sync Fix - 2025-08-16

Summary
- Resolve appleboy port parse error; ensure reliable GitHub->server sync.

Changes
- Update appleboy/ssh-action to v1.0.3.
- Set port to "22" (literal) to avoid unset secret.
- Use env for SERVER_HOST and SERVER_USER; keep key from secrets.
- Keep script: cd /home/kidgordones/0solana/node && git pull origin main.

Affected Files
- .github/workflows/auto-sync.yml

Prerequisites
- GitHub secret SERVER_SSH_KEY contains the server's private key.

References
- docs/setup/GITHUB_ACTIONS_FIX.md
- docs/setup/GITHUB_ACTIONS_SSH_FIX.md

Next
- Implement server->GitHub auto-push via systemd timer (separate change).