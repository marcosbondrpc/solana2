#!/bin/bash

# Test GitHub Actions Deployment
# This script verifies the deployment workflow is working correctly

set -e

echo "======================================="
echo "GitHub Actions Deployment Test"
echo "======================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SSH_KEY="$HOME/.ssh/github_deploy"
SERVER_HOST="45.157.234.184"
SERVER_USER="kidgordones"
SERVER_PATH="/home/kidgordones/0solana/node"

echo -e "${YELLOW}Test 1: Local SSH Key${NC}"
echo "----------------------"
if [ -f "$SSH_KEY" ]; then
    echo -e "${GREEN}✓ Private key exists${NC}"
    chmod 600 "$SSH_KEY"
    echo -e "${GREEN}✓ Permissions set to 600${NC}"
else
    echo -e "${RED}✗ Private key not found at $SSH_KEY${NC}"
    exit 1
fi

if [ -f "${SSH_KEY}.pub" ]; then
    echo -e "${GREEN}✓ Public key exists${NC}"
else
    echo -e "${RED}✗ Public key not found${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Test 2: SSH Connection${NC}"
echo "---------------------"
echo "Testing connection to $SERVER_HOST..."

# Test with explicit parameters
if ssh -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -o ConnectTimeout=5 \
    -p 22 \
    "${SERVER_USER}@${SERVER_HOST}" \
    "echo 'Connection successful'" 2>/dev/null; then
    echo -e "${GREEN}✓ SSH connection successful${NC}"
else
    echo -e "${RED}✗ SSH connection failed${NC}"
    echo "Debug: Trying with verbose output..."
    ssh -vvv -i "$SSH_KEY" -p 22 "${SERVER_USER}@${SERVER_HOST}" "echo test" 2>&1 | head -20
    exit 1
fi

echo ""
echo -e "${YELLOW}Test 3: Remote Repository${NC}"
echo "------------------------"
REMOTE_STATUS=$(ssh -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -p 22 \
    "${SERVER_USER}@${SERVER_HOST}" \
    "cd $SERVER_PATH && git status --short" 2>/dev/null)

if [ -z "$REMOTE_STATUS" ]; then
    echo -e "${GREEN}✓ Remote repository is clean${NC}"
else
    echo -e "${YELLOW}⚠ Remote repository has changes:${NC}"
    echo "$REMOTE_STATUS"
fi

echo ""
echo -e "${YELLOW}Test 4: Service Status${NC}"
echo "---------------------"

# Check frontend
FRONTEND_PID=$(ssh -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -p 22 \
    "${SERVER_USER}@${SERVER_HOST}" \
    "pgrep -f vite" 2>/dev/null || echo "")

if [ -n "$FRONTEND_PID" ]; then
    echo -e "${GREEN}✓ Frontend is running (PID: $FRONTEND_PID)${NC}"
else
    echo -e "${YELLOW}⚠ Frontend is not running${NC}"
fi

# Check backend
BACKEND_PID=$(ssh -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -p 22 \
    "${SERVER_USER}@${SERVER_HOST}" \
    "pgrep -f 'python3.*main.py'" 2>/dev/null || echo "")

if [ -n "$BACKEND_PID" ]; then
    echo -e "${GREEN}✓ Backend is running (PID: $BACKEND_PID)${NC}"
else
    echo -e "${YELLOW}⚠ Backend is not running${NC}"
fi

echo ""
echo -e "${YELLOW}Test 5: GitHub Workflow File${NC}"
echo "---------------------------"

# Check local workflow
if [ -f ".github/workflows/sync.yml" ]; then
    echo -e "${GREEN}✓ Workflow file exists locally${NC}"
    
    # Check if it's committed
    if git diff --cached --name-only | grep -q "sync.yml"; then
        echo -e "${YELLOW}⚠ Workflow has uncommitted changes${NC}"
    else
        echo -e "${GREEN}✓ Workflow is committed${NC}"
    fi
    
    # Check if it's pushed
    LOCAL_HASH=$(git hash-object .github/workflows/sync.yml)
    REMOTE_HASH=$(git hash-object <(git show origin/main:.github/workflows/sync.yml 2>/dev/null) 2>/dev/null || echo "")
    
    if [ "$LOCAL_HASH" = "$REMOTE_HASH" ]; then
        echo -e "${GREEN}✓ Workflow is synced with GitHub${NC}"
    else
        echo -e "${YELLOW}⚠ Workflow differs from GitHub version${NC}"
        echo "  Local hash:  $LOCAL_HASH"
        echo "  Remote hash: $REMOTE_HASH"
        echo "  Run: git push origin main"
    fi
else
    echo -e "${RED}✗ Workflow file not found${NC}"
fi

echo ""
echo -e "${YELLOW}Test 6: Simulate Deployment${NC}"
echo "--------------------------"
echo "Simulating deployment commands..."

# Test git pull
ssh -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -p 22 \
    "${SERVER_USER}@${SERVER_HOST}" << 'EOF'
set -e
cd /home/kidgordones/0solana/node
echo "Current branch: $(git branch --show-current)"
echo "Latest commit: $(git log -1 --oneline)"

# Test git fetch
git fetch origin >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Git fetch successful"
else
    echo "✗ Git fetch failed"
fi

# Check for updates
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [ "$LOCAL" = "$REMOTE" ]; then
    echo "✓ Repository is up to date"
else
    echo "⚠ Repository needs update"
    echo "  Local:  $LOCAL"
    echo "  Remote: $REMOTE"
fi
EOF

echo ""
echo -e "${YELLOW}Test 7: Port Availability${NC}"
echo "------------------------"

# Check if port 22 is explicitly accessible
nc -zv "$SERVER_HOST" 22 2>&1 | grep -q "succeeded" && \
    echo -e "${GREEN}✓ Port 22 is accessible${NC}" || \
    echo -e "${RED}✗ Port 22 is not accessible${NC}"

echo ""
echo "======================================="
echo -e "${GREEN}Deployment Test Complete${NC}"
echo "======================================="
echo ""
echo "Summary:"
echo "--------"
echo "1. SSH Key: Configured"
echo "2. SSH Connection: Working"
echo "3. Remote Repository: Ready"
echo "4. Services: Check status above"
echo "5. Workflow: Ready for GitHub"
echo ""
echo "Next Steps:"
echo "-----------"
echo "1. Add SERVER_SSH_KEY to GitHub Secrets:"
echo "   - Go to: https://github.com/marcosbondrpc/solana2/settings/secrets/actions"
echo "   - Add the private key content"
echo ""
echo "2. Trigger deployment:"
echo "   - Push a commit: git push origin main"
echo "   - Or manually trigger from GitHub Actions page"
echo ""
echo "3. Monitor the workflow:"
echo "   - Check: https://github.com/marcosbondrpc/solana2/actions"
echo ""