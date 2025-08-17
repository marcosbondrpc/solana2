const express = require('express');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const WebSocket = require('ws');
const http = require('http');

const app = express();
const server = http.createServer(app);

app.use(cors());
app.use(express.json());

// In-memory job storage
const jobs = new Map();
const wsClients = new Map();

// Mock dataset statistics
const datasetStats = {
  raw: {
    bytes: 0,
    rows: 0,
    partitions: 0,
    lastUpdate: null
  },
  swaps: {
    total: 0,
    uniquePairs: 0,
    volume: 0
  },
  arbitrage: {
    opportunities: 0,
    totalProfit: 0,
    successRate: 0
  },
  sandwich: {
    attacks: 0,
    extractedValue: 0,
    victims: 0
  }
};

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    uptime: process.uptime(),
    timestamp: new Date().toISOString()
  });
});

// Start capture job
app.post('/capture/start', (req, res) => {
  const jobId = uuidv4();
  const job = {
    id: jobId,
    type: 'capture',
    status: 'pending',
    progress: 0,
    params: req.body,
    createdAt: new Date().toISOString(),
    blocksWritten: 0,
    bytesProcessed: 0,
    currentSlot: 0,
    totalSlots: 0
  };
  
  jobs.set(jobId, job);
  
  // Simulate job processing
  simulateCaptureJob(jobId);
  
  res.json({ job_id: jobId, status: 'started' });
});

// Get job status
app.get('/jobs/:jobId', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  res.json(job);
});

// Cancel job
app.post('/jobs/:jobId/cancel', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  
  job.status = 'cancelled';
  updateJobClients(req.params.jobId, job);
  
  res.json({ status: 'cancelled' });
});

// Start arbitrage conversion
app.post('/convert/arbitrage/start', (req, res) => {
  const jobId = uuidv4();
  const job = {
    id: jobId,
    type: 'arbitrage',
    status: 'pending',
    progress: 0,
    params: req.body,
    createdAt: new Date().toISOString(),
    opportunitiesFound: 0,
    totalProfit: 0
  };
  
  jobs.set(jobId, job);
  simulateConversionJob(jobId, 'arbitrage');
  
  res.json({ job_id: jobId, status: 'started' });
});

// Start sandwich conversion
app.post('/convert/sandwich/start', (req, res) => {
  const jobId = uuidv4();
  const job = {
    id: jobId,
    type: 'sandwich',
    status: 'pending',
    progress: 0,
    params: req.body,
    createdAt: new Date().toISOString(),
    attacksFound: 0,
    extractedValue: 0
  };
  
  jobs.set(jobId, job);
  simulateConversionJob(jobId, 'sandwich');
  
  res.json({ job_id: jobId, status: 'started' });
});

// Get dataset statistics
app.get('/datasets/stats', (req, res) => {
  res.json(datasetStats);
});

// WebSocket setup for real-time updates
const wss = new WebSocket.Server({ server, path: '/ws' });

wss.on('connection', (ws, req) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const jobId = url.pathname.split('/').pop();
  
  if (jobId && jobId !== 'ws') {
    wsClients.set(jobId, ws);
    
    // Send current job status
    const job = jobs.get(jobId);
    if (job) {
      ws.send(JSON.stringify(job));
    }
    
    ws.on('close', () => {
      wsClients.delete(jobId);
    });
  }
});

// Simulate capture job processing
function simulateCaptureJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) return;
  
  job.status = 'running';
  job.totalSlots = 100000; // Simulate total slots to process
  
  const interval = setInterval(() => {
    if (job.status === 'cancelled') {
      clearInterval(interval);
      return;
    }
    
    // Update progress
    job.progress = Math.min(job.progress + Math.random() * 5, 100);
    job.currentSlot = Math.floor((job.progress / 100) * job.totalSlots);
    job.blocksWritten = Math.floor(job.currentSlot / 100);
    job.bytesProcessed = job.blocksWritten * 1024 * 1024; // 1MB per block
    
    // Update stats
    datasetStats.raw.bytes = job.bytesProcessed;
    datasetStats.raw.rows = job.blocksWritten * 1000;
    datasetStats.raw.partitions = Math.floor(job.blocksWritten / 10);
    datasetStats.raw.lastUpdate = new Date().toISOString();
    
    updateJobClients(jobId, job);
    
    if (job.progress >= 100) {
      job.status = 'completed';
      job.completedAt = new Date().toISOString();
      clearInterval(interval);
      updateJobClients(jobId, job);
    }
  }, 1000);
}

// Simulate conversion job processing
function simulateConversionJob(jobId, type) {
  const job = jobs.get(jobId);
  if (!job) return;
  
  job.status = 'running';
  
  const interval = setInterval(() => {
    if (job.status === 'cancelled') {
      clearInterval(interval);
      return;
    }
    
    // Update progress
    job.progress = Math.min(job.progress + Math.random() * 10, 100);
    
    if (type === 'arbitrage') {
      job.opportunitiesFound = Math.floor(job.progress * 5);
      job.totalProfit = job.opportunitiesFound * (1000 + Math.random() * 5000);
      
      datasetStats.arbitrage.opportunities = job.opportunitiesFound;
      datasetStats.arbitrage.totalProfit = job.totalProfit;
      datasetStats.arbitrage.successRate = 0.72 + Math.random() * 0.1;
    } else if (type === 'sandwich') {
      job.attacksFound = Math.floor(job.progress * 3);
      job.extractedValue = job.attacksFound * (500 + Math.random() * 3000);
      
      datasetStats.sandwich.attacks = job.attacksFound;
      datasetStats.sandwich.extractedValue = job.extractedValue;
      datasetStats.sandwich.victims = job.attacksFound * 2;
    }
    
    updateJobClients(jobId, job);
    
    if (job.progress >= 100) {
      job.status = 'completed';
      job.completedAt = new Date().toISOString();
      clearInterval(interval);
      updateJobClients(jobId, job);
    }
  }, 500);
}

// Update WebSocket clients
function updateJobClients(jobId, job) {
  const ws = wsClients.get(jobId);
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(job));
  }
}

const PORT = process.env.PORT || 8055;
server.listen(PORT, () => {
  console.log(`Historical Capture API running on port ${PORT}`);
  console.log(`WebSocket endpoint: ws://localhost:${PORT}/ws/{job_id}`);
});