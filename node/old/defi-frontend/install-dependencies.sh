#!/bin/bash

echo "Installing additional dependencies for Solana Node Monitoring Dashboard..."

# Install additional production dependencies
npm install --save \
  @radix-ui/react-tabs \
  @radix-ui/react-alert \
  recharts \
  framer-motion \
  valtio \
  zustand \
  @solana/web3.js \
  socket.io \
  socket.io-client \
  class-variance-authority \
  immer \
  zustand-middleware-immer

# Install additional dev dependencies  
npm install --save-dev \
  @types/recharts

echo "Dependencies installed successfully!"
echo ""
echo "To start the monitoring dashboard:"
echo "1. Start the backend monitoring service:"
echo "   node server/monitoring-service.js"
echo ""
echo "2. In another terminal, start the frontend:"
echo "   npm run dev"
echo ""
echo "The dashboard will be available at http://localhost:42391"