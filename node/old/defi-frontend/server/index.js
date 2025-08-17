const express = require('express');
const { createServer } = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const { exec, spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const { promisify } = require('util');

const execAsync = promisify(exec);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: ['http://localhost:42391', 'http://0.0.0.0:42391'],
    methods: ['GET', 'POST'],
  },
});

app.use(cors());
app.use(express.json());

// Store for metrics
let metricsCache = {
  node: null,
  system: null,
  jito: null,
  rpc: null,
};

// Monitoring intervals
let monitoringIntervals = [];

// API Routes
app.get('/api/status', (req, res) => {
  res.json({
    connected: true,
    metrics: metricsCache,
    timestamp: new Date(),
  });
});

app.post('/api/execute', async (req, res) => {
  const { script, params } = req.body;
  
  try {
    const startTime = Date.now();
    const { stdout, stderr } = await execAsync(`bash ${script}`, {
      maxBuffer: 1024 * 1024 * 10, // 10MB buffer
      timeout: 60000, // 60 second timeout
    });
    
    const duration = Date.now() - startTime;
    
    res.json({
      success: true,
      output: stdout,
      error: stderr,
      duration,
    });
    
    // Emit log to connected clients
    io.emit('log', {
      timestamp: new Date(),
      level: 'info',
      message: `Executed command: ${path.basename(script)}`,
      source: 'command',
    });
  } catch (error) {
    res.json({
      success: false,
      error: error.message,
      output: error.stdout || '',
    });
    
    io.emit('log', {
      timestamp: new Date(),
      level: 'error',
      message: `Command failed: ${error.message}`,
      source: 'command',
    });
  }
});

app.get('/api/logs', async (req, res) => {
  const { limit = 100, level } = req.query;
  
  try {
    // Read validator log file
    const logPath = '/home/solana/validator.log';
    const logContent = await fs.readFile(logPath, 'utf-8').catch(() => '');
    const lines = logContent.split('\n').slice(-limit);
    
    const logs = lines
      .filter(line => line.trim())
      .map(line => {
        // Parse log line (adjust based on actual log format)
        const match = line.match(/\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\]\s+\[(\w+)\]\s+(.*)/);
        if (match) {
          return {
            timestamp: new Date(match[1]),
            level: match[2].toLowerCase(),
            message: match[3],
            source: 'validator',
          };
        }
        return {
          timestamp: new Date(),
          level: 'info',
          message: line,
          source: 'validator',
        };
      })
      .filter(log => !level || log.level === level);
    
    res.json(logs);
  } catch (error) {
    res.json([]);
  }
});

app.post('/api/network', async (req, res) => {
  const { network } = req.body;
  
  // This would trigger a network switch - implementation depends on your setup
  io.emit('log', {
    timestamp: new Date(),
    level: 'info',
    message: `Switching to ${network}`,
    source: 'system',
  });
  
  res.json({ success: true, network });
});

// System metrics collection
async function collectSystemMetrics() {
  try {
    // CPU usage
    const cpuInfo = os.cpus();
    const cpuUsage = 100 - (cpuInfo.reduce((acc, cpu) => acc + cpu.times.idle, 0) / 
                      cpuInfo.reduce((acc, cpu) => acc + Object.values(cpu.times).reduce((a, b) => a + b, 0), 0)) * 100;
    
    // Memory
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = totalMem - freeMem;
    
    // Load average
    const loadAvg = os.loadavg();
    
    // Network (simplified - would need more sophisticated monitoring)
    const { stdout: netStats } = await execAsync("cat /proc/net/dev | grep -E 'eth0|enp' | head -1");
    const netParts = netStats.trim().split(/\s+/);
    const rxBytes = parseInt(netParts[1] || 0);
    const txBytes = parseInt(netParts[9] || 0);
    
    // Disk usage
    const { stdout: diskStats } = await execAsync("df -B1 /mnt/ledger | tail -1");
    const diskParts = diskStats.trim().split(/\s+/);
    const diskTotal = parseInt(diskParts[1] || 0);
    const diskUsed = parseInt(diskParts[2] || 0);
    
    return {
      cpuUsage,
      cpuCores: cpuInfo.length,
      cpuFreq: cpuInfo[0]?.speed / 1000 || 0, // Convert to GHz
      memoryUsed: usedMem / (1024 * 1024), // Convert to MB
      memoryTotal: totalMem / (1024 * 1024),
      memoryPercent: (usedMem / totalMem) * 100,
      diskUsed: diskUsed / (1024 * 1024 * 1024), // Convert to GB
      diskTotal: diskTotal / (1024 * 1024 * 1024),
      diskPercent: (diskUsed / diskTotal) * 100,
      networkRx: rxBytes,
      networkTx: txBytes,
      loadAverage: loadAvg,
      uptime: os.uptime(),
      processes: parseInt((await execAsync('ps aux | wc -l')).stdout.trim()) - 1,
    };
  } catch (error) {
    console.error('Error collecting system metrics:', error);
    return null;
  }
}

// Solana node metrics collection
async function collectNodeMetrics() {
  try {
    // Get slot information
    const { stdout: slotInfo } = await execAsync(`curl -s -X POST -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"getSlot"}' http://localhost:8899`);
    const slotData = JSON.parse(slotInfo);
    
    // Get epoch info
    const { stdout: epochInfo } = await execAsync(`curl -s -X POST -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"getEpochInfo"}' http://localhost:8899`);
    const epochData = JSON.parse(epochInfo);
    
    // Get version
    const { stdout: versionInfo } = await execAsync(`curl -s -X POST -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"getVersion"}' http://localhost:8899`);
    const versionData = JSON.parse(versionInfo);
    
    // Get health
    const { stdout: healthInfo } = await execAsync(`curl -s -X POST -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' http://localhost:8899`);
    const health = healthInfo.includes('ok') ? 'healthy' : 'error';
    
    return {
      slot: slotData.result || 0,
      blockHeight: epochData.result?.blockHeight || 0,
      epoch: epochData.result?.epoch || 0,
      slotIndex: epochData.result?.slotIndex || 0,
      slotsInEpoch: epochData.result?.slotsInEpoch || 0,
      absoluteSlot: epochData.result?.absoluteSlot || 0,
      transactionCount: epochData.result?.transactionCount || 0,
      skipRate: Math.random() * 5, // Mock data - would need actual calculation
      leaderSlots: 0,
      blocksProduced: 0,
      health,
      version: versionData.result?.['solana-core'] || 'unknown',
      identity: 'validator-identity', // Would need actual identity
      voteAccount: 'vote-account', // Would need actual vote account
      commission: 0,
      activatedStake: 0,
      networkInflation: 0,
      totalSupply: 0,
    };
  } catch (error) {
    console.error('Error collecting node metrics:', error);
    return null;
  }
}

// Jito metrics collection (mock data - would need actual Jito integration)
async function collectJitoMetrics() {
  return {
    bundlesReceived: Math.floor(Math.random() * 1000),
    bundlesLanded: Math.floor(Math.random() * 800),
    bundleRate: 75 + Math.random() * 20,
    mevRewards: Math.random() * 10,
    tipDistribution: Array(10).fill(0).map(() => Math.floor(Math.random() * 100)),
    avgBundleLatency: 50 + Math.random() * 100,
    blockEngineConnected: Math.random() > 0.2,
    relayerConnected: Math.random() > 0.2,
  };
}

// RPC metrics collection
async function collectRPCMetrics() {
  return {
    requestsPerSecond: Math.floor(Math.random() * 1000),
    avgResponseTime: 10 + Math.random() * 50,
    p99ResponseTime: 100 + Math.random() * 200,
    p95ResponseTime: 50 + Math.random() * 100,
    errorRate: Math.random() * 5,
    activeConnections: Math.floor(Math.random() * 100),
    wsConnections: Math.floor(Math.random() * 30),
    httpConnections: Math.floor(Math.random() * 70),
    methods: {
      getSlot: { count: Math.floor(Math.random() * 10000), avgTime: Math.random() * 10, errors: Math.floor(Math.random() * 10) },
      getBlock: { count: Math.floor(Math.random() * 5000), avgTime: Math.random() * 50, errors: Math.floor(Math.random() * 20) },
      getTransaction: { count: Math.floor(Math.random() * 8000), avgTime: Math.random() * 30, errors: Math.floor(Math.random() * 15) },
      sendTransaction: { count: Math.floor(Math.random() * 2000), avgTime: Math.random() * 100, errors: Math.floor(Math.random() * 50) },
      getAccountInfo: { count: Math.floor(Math.random() * 15000), avgTime: Math.random() * 20, errors: Math.floor(Math.random() * 5) },
    },
  };
}

// Start monitoring
function startMonitoring() {
  // System metrics - every 2 seconds
  monitoringIntervals.push(
    setInterval(async () => {
      const metrics = await collectSystemMetrics();
      if (metrics) {
        metricsCache.system = metrics;
        io.emit('system-metrics', metrics);
      }
    }, 2000)
  );
  
  // Node metrics - every 5 seconds
  monitoringIntervals.push(
    setInterval(async () => {
      const metrics = await collectNodeMetrics();
      if (metrics) {
        metricsCache.node = metrics;
        io.emit('node-metrics', metrics);
      }
    }, 5000)
  );
  
  // Jito metrics - every 10 seconds
  monitoringIntervals.push(
    setInterval(async () => {
      const metrics = await collectJitoMetrics();
      if (metrics) {
        metricsCache.jito = metrics;
        io.emit('jito-metrics', metrics);
      }
    }, 10000)
  );
  
  // RPC metrics - every 3 seconds
  monitoringIntervals.push(
    setInterval(async () => {
      const metrics = await collectRPCMetrics();
      if (metrics) {
        metricsCache.rpc = metrics;
        io.emit('rpc-metrics', metrics);
      }
    }, 3000)
  );
}

// WebSocket connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  // Send initial metrics
  socket.emit('connected', { timestamp: new Date() });
  if (metricsCache.node) socket.emit('node-metrics', metricsCache.node);
  if (metricsCache.system) socket.emit('system-metrics', metricsCache.system);
  if (metricsCache.jito) socket.emit('jito-metrics', metricsCache.jito);
  if (metricsCache.rpc) socket.emit('rpc-metrics', metricsCache.rpc);
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
  
  socket.on('execute-command', async (data) => {
    const { script, params } = data;
    try {
      const result = await execAsync(`bash ${script}`);
      socket.emit('command-result', {
        success: true,
        output: result.stdout,
        error: result.stderr,
      });
    } catch (error) {
      socket.emit('command-result', {
        success: false,
        error: error.message,
      });
    }
  });
});

// Start server
const PORT = 42392;
httpServer.listen(PORT, '0.0.0.0', () => {
  console.log(`Backend server running on http://0.0.0.0:${PORT}`);
  startMonitoring();
});