#!/bin/bash

echo "🚀 Pushing to GitHub..."
echo ""
echo "📊 Current status:"
echo "=================="

# Show unpushed commits
UNPUSHED=$(git log --oneline origin/master..HEAD 2>/dev/null | wc -l)
if [ "$UNPUSHED" -gt 0 ]; then
    echo "✅ You have $UNPUSHED unpushed commits:"
    git log --oneline origin/master..HEAD
else
    echo "✅ All commits are already pushed"
fi

echo ""
echo "🔄 Attempting to push..."
echo ""

# Try to push
if git push origin master; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 View your repo at: https://github.com/marcosbondrpc/solana2"
else
    echo ""
    echo "❌ Push failed. You need to configure authentication."
    echo ""
    echo "🔐 To fix this, you need to set up authentication:"
    echo ""
    echo "Option 1: Use a Personal Access Token"
    echo "======================================="
    echo "1. Create a token at: https://github.com/settings/tokens"
    echo "2. Run this command (replace YOUR_TOKEN with your actual token):"
    echo ""
    echo "   git remote set-url origin https://marcosbondrpc:YOUR_TOKEN@github.com/marcosbondrpc/solana2.git"
    echo ""
    echo "3. Then run this script again: ./push-now.sh"
    echo ""
    echo "Option 2: Use GitHub CLI (easier)"
    echo "=================================="
    echo "1. Install GitHub CLI:"
    echo "   sudo apt update && sudo apt install gh"
    echo ""
    echo "2. Login:"
    echo "   gh auth login"
    echo ""
    echo "3. Follow the prompts to authenticate"
    echo "4. Then run: gh repo sync"
fi