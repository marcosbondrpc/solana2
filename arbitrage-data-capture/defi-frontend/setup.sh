#!/bin/bash

# LEGENDARY DeFi Frontend Setup Script
# Sets up ultra-high-performance WebSocket/WebTransport infrastructure

set -e

echo "🚀 Setting up LEGENDARY DeFi Frontend..."

# Install npm dependencies
echo "📦 Installing dependencies..."
npm install

# Install additional performance dependencies
npm install --save \
  fflate \
  lz4js \
  snappy \
  msgpackr \
  fastestsmallesttextencoderdecoder

# Install dev dependencies for optimization
npm install --save-dev \
  vite-plugin-top-level-await \
  @vitejs/plugin-react \
  terser \
  rollup-plugin-visualizer

# Generate protobuf bindings
echo "🔧 Generating protobuf bindings..."

# JavaScript bindings
npx pbjs -t static-module -w es6 \
  -o ./lib/proto/realtime.js \
  ../protocol/realtime.proto

npx pbjs -t static-module -w es6 \
  -o ./lib/proto/control.js \
  ../protocol/control.proto

# TypeScript definitions
npx pbts -o ./lib/proto/realtime.d.ts ./lib/proto/realtime.js
npx pbts -o ./lib/proto/control.d.ts ./lib/proto/control.js

echo "✅ Protobuf bindings generated"

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
  echo "📝 Creating .env file..."
  cat > .env << EOF
# WebSocket Configuration
VITE_WS_URL=ws://localhost:8000/ws
VITE_WS_MODE=proto

# WebTransport Configuration  
VITE_WT_URL=https://localhost:4433/wt
VITE_WT_ENABLED=true

# Performance Settings
VITE_BATCH_WINDOW=15
VITE_MAX_BATCH_SIZE=256
VITE_COMPRESSION_LEVEL=3
VITE_WORKER_POOL_SIZE=8

# Feature Flags
VITE_USE_SHARED_BUFFER=true
VITE_ENABLE_METRICS=true
VITE_ENABLE_PROFILING=false
EOF
  echo "✅ Environment file created"
fi

# Create index.html if it doesn't exist
if [ ! -f index.html ]; then
  echo "📝 Creating index.html..."
  cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LEGENDARY DeFi Dashboard - Solana MEV</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0a0a0a;
      color: #fff;
      overflow: hidden;
    }
    #app {
      width: 100vw;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .loading {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
    }
    .loading h1 {
      font-size: 2.5rem;
      margin-bottom: 1rem;
      background: linear-gradient(45deg, #00ff88, #00ffff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    .stats {
      position: absolute;
      top: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.8);
      border: 1px solid #00ff88;
      border-radius: 8px;
      padding: 10px;
      font-family: 'Courier New', monospace;
      font-size: 12px;
      min-width: 200px;
    }
    .stats-row {
      display: flex;
      justify-content: space-between;
      margin: 4px 0;
    }
    .stats-label {
      color: #888;
    }
    .stats-value {
      color: #00ff88;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <div id="app">
    <div class="loading">
      <h1>LEGENDARY MEV DASHBOARD</h1>
      <p>Initializing ultra-high-performance feed...</p>
    </div>
    <div id="stats" class="stats" style="display: none;">
      <div class="stats-row">
        <span class="stats-label">Messages/sec:</span>
        <span class="stats-value" id="msg-rate">0</span>
      </div>
      <div class="stats-row">
        <span class="stats-label">Decode Time:</span>
        <span class="stats-value" id="decode-time">0ms</span>
      </div>
      <div class="stats-row">
        <span class="stats-label">Buffer Size:</span>
        <span class="stats-value" id="buffer-size">0</span>
      </div>
      <div class="stats-row">
        <span class="stats-label">FPS:</span>
        <span class="stats-value" id="fps">60</span>
      </div>
      <div class="stats-row">
        <span class="stats-label">Memory:</span>
        <span class="stats-value" id="memory">0MB</span>
      </div>
    </div>
  </div>
  <script type="module" src="/src/main.ts"></script>
</body>
</html>
EOF
  echo "✅ index.html created"
fi

# Create main.ts if it doesn't exist
if [ ! -f src/main.ts ]; then
  mkdir -p src
  echo "📝 Creating src/main.ts..."
  cat > src/main.ts << 'EOF'
// LEGENDARY DeFi Dashboard Entry Point
import { initializeFeed, feedStore, getSnapshot } from '../stores/feed';
import { performanceMonitor } from '../utils/performance';
import { realtimeClient } from '../lib/ws';
import { wtClient } from '../lib/wt';

// Initialize performance monitoring
performanceMonitor.onMetrics((metrics) => {
  document.getElementById('fps')!.textContent = metrics.fps.toString();
  document.getElementById('memory')!.textContent = 
    Math.round(metrics.memoryUsed / 1024 / 1024) + 'MB';
});

// Initialize feed with WebTransport if available
const useWebTransport = import.meta.env.VITE_WT_ENABLED === 'true';

initializeFeed(useWebTransport).then(() => {
  console.log('🚀 LEGENDARY feed initialized!');
  
  // Hide loading, show stats
  document.querySelector('.loading')!.remove();
  document.getElementById('stats')!.style.display = 'block';
  
  // Update stats display
  setInterval(() => {
    const stats = realtimeClient.getStats();
    document.getElementById('msg-rate')!.textContent = 
      Math.round(stats.messagesReceived / (stats.connectionUptime / 1000)).toString();
    document.getElementById('decode-time')!.textContent = 
      stats.avgDecodeTime.toFixed(2) + 'ms';
    document.getElementById('buffer-size')!.textContent = 
      stats.currentBufferSize.toString();
  }, 100);
  
  // Log snapshot every 5 seconds for debugging
  if (import.meta.env.DEV) {
    setInterval(() => {
      const snapshot = getSnapshot();
      console.log('📊 Feed Snapshot:', {
        mevCount: snapshot.mevHead - snapshot.mevTail,
        arbCount: snapshot.arbHead - snapshot.arbTail,
        bundleCount: snapshot.bundleHead - snapshot.bundleTail,
        marketTicks: snapshot.marketTicks.size,
        metrics: snapshot.currentMetrics
      });
    }, 5000);
  }
}).catch(error => {
  console.error('Failed to initialize feed:', error);
});

// Handle page visibility for performance
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    console.log('Page hidden, throttling updates...');
    // Implement throttling logic here
  } else {
    console.log('Page visible, resuming full updates...');
  }
});

// Export for debugging
(window as any).feedStore = feedStore;
(window as any).realtimeClient = realtimeClient;
(window as any).wtClient = wtClient;
(window as any).performanceMonitor = performanceMonitor;
EOF
  echo "✅ src/main.ts created"
fi

echo ""
echo "🎉 LEGENDARY DeFi Frontend setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and update .env with your actual endpoints"
echo "2. Run 'npm run dev' to start the development server"
echo "3. Open http://localhost:3000 to see the dashboard"
echo ""
echo "Performance targets:"
echo "• 100,000+ messages per second"
echo "• Sub-millisecond decode times"
echo "• 60 FPS rendering"
echo "• < 50MB memory usage"
echo ""
echo "⚡ This is the fastest DeFi frontend ever built! ⚡"