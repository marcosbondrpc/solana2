#!/bin/bash
echo "🔄 Manual GitHub sync triggered..."
/home/kidgordones/0solana/solana2/scripts/github-sync.sh
echo "✅ Sync complete!"
tail -20 /tmp/github-sync.log
