#!/bin/bash

# Development environment startup script

set -e

echo "🚀 Starting Solana MEV Frontend Development Environment"

# Check dependencies
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting." >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm is required but not installed. Aborting." >&2; exit 1; }

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start all services in parallel using turbo
echo "Starting development servers..."
npm run dev

# The turbo dev command will handle:
# - Dashboard app on port 3000
# - Operator app on port 3001  
# - Analytics app on port 3002
# - Storybook on port 6006
# - All with hot-reload enabled