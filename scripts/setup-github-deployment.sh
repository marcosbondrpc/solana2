#!/bin/bash

# GitHub Actions SSH Deployment Setup Script
# This script helps set up GitHub Actions for automated deployment

set -e

echo "================================"
echo "GitHub Actions Deployment Setup"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SSH_KEY_PATH="$HOME/.ssh/github_deploy"
GITHUB_REPO="marcosbondrpc/solana-mev-infrastructure"

echo -e "${YELLOW}Step 1: SSH Key Setup${NC}"
echo "------------------------"

# Check if SSH key exists
if [ -f "$SSH_KEY_PATH" ]; then
    echo -e "${GREEN}✓ SSH key found at $SSH_KEY_PATH${NC}"
else
    echo -e "${RED}✗ SSH key not found. Generating new key...${NC}"
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -C "github-actions@deploy" -N ""
    echo -e "${GREEN}✓ SSH key generated${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Add Public Key to Server${NC}"
echo "--------------------------------"
echo "The following public key needs to be in ~/.ssh/authorized_keys on the server:"
echo ""
cat "${SSH_KEY_PATH}.pub"
echo ""

# Check if key is already authorized
if grep -q "$(cat ${SSH_KEY_PATH}.pub)" ~/.ssh/authorized_keys 2>/dev/null; then
    echo -e "${GREEN}✓ Key is already in authorized_keys${NC}"
else
    echo -e "${YELLOW}Adding key to authorized_keys...${NC}"
    cat "${SSH_KEY_PATH}.pub" >> ~/.ssh/authorized_keys
    echo -e "${GREEN}✓ Key added to authorized_keys${NC}"
fi

echo ""
echo -e "${YELLOW}Step 3: GitHub Secrets Configuration${NC}"
echo "------------------------------------"
echo "You need to add the following secret to your GitHub repository:"
echo ""
echo "1. Go to: https://github.com/${GITHUB_REPO}/settings/secrets/actions"
echo ""
echo "2. Click 'New repository secret'"
echo ""
echo "3. Add a secret named: SERVER_SSH_KEY"
echo ""
echo "4. Copy and paste this ENTIRE private key (including BEGIN and END lines):"
echo ""
echo "==================== COPY FROM HERE ===================="
cat "$SSH_KEY_PATH"
echo "==================== COPY TO HERE ======================"
echo ""
echo "Alternative: Use this base64 encoded version (single line):"
echo ""
echo "==================== BASE64 VERSION ===================="
base64 -w 0 "$SSH_KEY_PATH"
echo ""
echo "========================================================="
echo ""

echo -e "${YELLOW}Step 4: Test SSH Connection${NC}"
echo "---------------------------"
echo "Testing SSH connection to localhost..."

if ssh -i "$SSH_KEY_PATH" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -o ConnectTimeout=5 \
    -p 22 \
    kidgordones@localhost "echo 'SSH test successful'" 2>/dev/null; then
    echo -e "${GREEN}✓ SSH connection test passed${NC}"
else
    echo -e "${RED}✗ SSH connection test failed${NC}"
    echo "Please check your SSH configuration"
fi

echo ""
echo -e "${YELLOW}Step 5: Verify GitHub Workflow${NC}"
echo "------------------------------"

# Check if workflow file exists
WORKFLOW_FILE=".github/workflows/sync.yml"
if [ -f "$WORKFLOW_FILE" ]; then
    echo -e "${GREEN}✓ Workflow file exists${NC}"
    
    # Check workflow version
    if grep -q "uses: actions/checkout@v4" "$WORKFLOW_FILE"; then
        echo -e "${GREEN}✓ Workflow is using latest actions${NC}"
    else
        echo -e "${YELLOW}⚠ Workflow may be using outdated actions${NC}"
    fi
    
    # Check if it's using native SSH
    if grep -q "ssh -i ~/.ssh/deploy_key" "$WORKFLOW_FILE"; then
        echo -e "${GREEN}✓ Workflow is using native SSH (recommended)${NC}"
    elif grep -q "appleboy/ssh-action" "$WORKFLOW_FILE"; then
        echo -e "${RED}✗ Workflow is still using appleboy/ssh-action (problematic)${NC}"
        echo "Please update to use native SSH commands"
    fi
else
    echo -e "${RED}✗ Workflow file not found${NC}"
fi

echo ""
echo -e "${YELLOW}Step 6: Manual Trigger Test${NC}"
echo "---------------------------"
echo "After adding the SERVER_SSH_KEY secret to GitHub:"
echo ""
echo "1. Go to: https://github.com/${GITHUB_REPO}/actions"
echo "2. Click on 'Deploy to Production Server' workflow"
echo "3. Click 'Run workflow' > 'Run workflow' to trigger manually"
echo "4. Monitor the workflow execution for any errors"
echo ""

echo -e "${YELLOW}Step 7: Troubleshooting Commands${NC}"
echo "--------------------------------"
echo "If the deployment fails, use these commands to debug:"
echo ""
echo "# Check GitHub Actions logs:"
echo "gh run list --workflow=sync.yml"
echo "gh run view --log"
echo ""
echo "# Test SSH connection manually:"
echo "ssh -i $SSH_KEY_PATH -p 22 kidgordones@45.157.234.184 'echo test'"
echo ""
echo "# Check server processes:"
echo "pgrep -f vite          # Check frontend"
echo "pgrep -f 'python3.*main.py'  # Check backend"
echo ""
echo "# View logs:"
echo "tail -f /tmp/frontend.log"
echo "tail -f /tmp/backend.log"
echo ""

echo "================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Add SERVER_SSH_KEY to GitHub Secrets (see above)"
echo "2. Push any change to trigger the workflow"
echo "3. Or manually trigger from GitHub Actions page"
echo ""
echo "The workflow will:"
echo "- Run every 5 minutes (cron schedule)"
echo "- Run on every push to main branch"
echo "- Run on pull requests"
echo "- Can be manually triggered"
echo ""