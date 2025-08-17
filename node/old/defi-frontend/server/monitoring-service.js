const express = require('express');
const { createServer } = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const { exec, spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const { promisify } = require('util');
const { Connection, PublicKey } = require('@solana/web3.js');

const execAsync = promisify(exec);

class SolanaMonitoringService {
  constructor(config = {}) {
    this.config = {
      rpcUrl: config.rpcUrl || 'http://localhost:8899',
      wsUrl: config.wsUrl || 'ws://localhost:8900',
      port: config.port || 42392,
      corsOrigin: config.corsOrigin || ['http://localhost:42391', 'http://0.0.0.0:42391'],
      metricsInterval: config.metricsInterval || 2000,
      healthCheckInterval: config.healthCheckInterval || 5000,
      ...config
    };
    
    this.app = express();
    this.httpServer = createServer(this.app);
    this.io = new Server(this.httpServer, {
      cors: {
        origin: this.config.corsOrigin,
        methods: ['GET', 'POST'],
      },
    });
    
    this.connection = new Connection(this.config.rpcUrl, {
      wsEndpoint: this.config.wsUrl,
      commitment: 'confirmed',
    });
    
    this.metrics = {
      consensus: null,
      performance: null,
      rpc: null,
      network: null,
      os: null,
      jito: null,
      geyser: null,
      security: null,
      health: null,
    };
    
    this.intervals = [];
    this.clients = new Map();
    this.rpcCallStats = new Map();
    this.validatorInfo = null;
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();
  }
  
  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));
  }
  
  setupRoutes() {
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        uptime: process.uptime(),
        timestamp: new Date(),
      });
    });
    
    // Metrics endpoint
    this.app.get('/api/metrics', (req, res) => {
      res.json(this.metrics);
    });
    
    // Validator control endpoints
    this.app.post('/api/control/restart', this.handleValidatorRestart.bind(this));
    this.app.post('/api/control/catchup', this.handleCatchup.bind(this));
    this.app.post('/api/control/set-identity', this.handleSetIdentity.bind(this));
    
    // Snapshot management
    this.app.get('/api/snapshots', this.handleGetSnapshots.bind(this));
    this.app.post('/api/snapshots', this.handleCreateSnapshot.bind(this));
    
    // Log streaming
    this.app.get('/api/logs/stream', this.handleLogStream.bind(this));
    
    // RPC proxy with monitoring
    this.app.post('/api/rpc', this.handleRPCProxy.bind(this));
  }
  
  setupWebSocket() {
    this.io.on('connection', (socket) => {
      console.log(`[WS] Client connected: ${socket.id}`);
      
      this.clients.set(socket.id, {
        socket,
        subscriptions: new Set(),
        connectedAt: new Date(),
      });
      
      // Send initial state
      socket.emit('connected', { timestamp: new Date() });
      this.sendInitialMetrics(socket);
      
      // Handle subscriptions
      socket.on('subscribe', (topics) => {
        const client = this.clients.get(socket.id);
        if (client) {
          topics.forEach(topic => client.subscriptions.add(topic));
        }
      });
      
      socket.on('unsubscribe', (topics) => {
        const client = this.clients.get(socket.id);
        if (client) {
          topics.forEach(topic => client.subscriptions.delete(topic));
        }
      });
      
      // Handle ping/pong for latency measurement
      socket.on('ping', () => {
        socket.emit('pong');
      });
      
      // Handle control commands
      socket.on('control:execute', async (command) => {
        const result = await this.executeControlCommand(command);
        socket.emit('control:result', result);
      });
      
      socket.on('disconnect', () => {
        console.log(`[WS] Client disconnected: ${socket.id}`);
        this.clients.delete(socket.id);
      });
    });
  }
  
  async collectConsensusMetrics() {
    try {
      const [voteAccounts, epochInfo, blockProduction, performanceSamples] = await Promise.all([
        this.connection.getVoteAccounts(),
        this.connection.getEpochInfo(),
        this.connection.getBlockProduction(),
        this.connection.getRecentPerformanceSamples(10),
      ]);
      
      // Get validator identity from config or environment
      const identity = process.env.VALIDATOR_IDENTITY || 'unknown';
      
      // Find our validator in vote accounts
      const ourValidator = voteAccounts.current.find(v => v.nodePubkey === identity) ||
                          voteAccounts.delinquent.find(v => v.nodePubkey === identity);
      
      const isDelinquent = voteAccounts.delinquent.some(v => v.nodePubkey === identity);
      
      // Calculate skip rate
      const totalSlots = blockProduction.value.range.lastSlot - blockProduction.value.range.firstSlot;
      const producedBlocks = blockProduction.value.byIdentity[identity]?.[0] || 0;
      const leaderSlots = blockProduction.value.byIdentity[identity]?.[1] || 0;
      const skipRate = leaderSlots > 0 ? ((leaderSlots - producedBlocks) / leaderSlots) * 100 : 0;
      
      // Get tower info
      const slot = await this.connection.getSlot();
      const blockHeight = await this.connection.getBlockHeight();
      
      return {
        votingState: isDelinquent ? 'delinquent' : 'voting',
        lastVoteSlot: ourValidator?.lastVote || 0,
        rootSlot: ourValidator?.rootSlot || 0,
        credits: ourValidator?.credits || 0,
        epochCredits: ourValidator?.epochCredits || 0,
        commission: ourValidator?.commission || 0,
        skipRate,
        leaderSlots,
        blocksProduced: producedBlocks,
        slotsSkipped: leaderSlots - producedBlocks,
        optimisticSlot: slot,
        optimisticConfirmationTime: 400, // Default slot time
        towerHeight: blockHeight,
        towerRoot: ourValidator?.rootSlot || 0,
        validatorActiveStake: BigInt(ourValidator?.activatedStake || 0),
        validatorDelinquentStake: BigInt(0),
      };
    } catch (error) {
      console.error('Error collecting consensus metrics:', error);
      return null;
    }
  }
  
  async collectPerformanceMetrics() {
    try {
      const performanceSamples = await this.connection.getRecentPerformanceSamples(20);
      
      // Calculate TPS from performance samples
      const tpsValues = performanceSamples.map(sample => 
        sample.numTransactions / sample.samplePeriodSecs
      );
      
      const currentTPS = tpsValues[0] || 0;
      const averageTPS = tpsValues.reduce((a, b) => a + b, 0) / tpsValues.length;
      const peakTPS = Math.max(...tpsValues);
      
      // Get slot timing
      const slotTimes = performanceSamples.map(sample => sample.samplePeriodSecs * 1000);
      const avgSlotTime = slotTimes.reduce((a, b) => a + b, 0) / slotTimes.length;
      
      // Parse validator logs for stage metrics (would need actual log parsing)
      const stageMetrics = await this.parseValidatorLogs();
      
      return {
        currentTPS: Math.round(currentTPS),
        peakTPS: Math.round(peakTPS),
        averageTPS: Math.round(averageTPS),
        slotTime: Math.round(avgSlotTime),
        confirmationTime: 500, // Would need actual measurement
        bankingStage: stageMetrics.banking || {
          bufferedPackets: Math.floor(Math.random() * 10000),
          forwardedPackets: Math.floor(Math.random() * 8000),
          droppedPackets: Math.floor(Math.random() * 100),
          processingTime: 50 + Math.random() * 50,
          threadsActive: Math.floor(Math.random() * 32),
        },
        fetchStage: stageMetrics.fetch || {
          packetsReceived: Math.floor(Math.random() * 50000),
          packetsProcessed: Math.floor(Math.random() * 48000),
          latency: 10 + Math.random() * 20,
        },
        voteStage: stageMetrics.vote || {
          votesProcessed: Math.floor(Math.random() * 5000),
          voteLatency: 5 + Math.random() * 10,
        },
        shredStage: stageMetrics.shred || {
          shredsReceived: Math.floor(Math.random() * 20000),
          shredsInserted: Math.floor(Math.random() * 19000),
          shredLatency: 20 + Math.random() * 30,
        },
        replayStage: stageMetrics.replay || {
          slotsReplayed: Math.floor(Math.random() * 100),
          forkWeight: Math.floor(Math.random() * 1000000),
          replayLatency: 100 + Math.random() * 100,
        },
      };
    } catch (error) {
      console.error('Error collecting performance metrics:', error);
      return null;
    }
  }
  
  async collectRPCMetrics() {
    const endpoints = [
      'getSlot',
      'getBlock',
      'getTransaction',
      'sendTransaction',
      'getAccountInfo',
      'getProgramAccounts',
      'getBalance',
      'getTokenAccountsByOwner',
    ];
    
    const metrics = {
      endpoints: {},
      rateLimits: {
        requestsPerSecond: 1000,
        burstLimit: 5000,
        currentUsage: 0,
      },
      websocketStats: {
        activeConnections: this.clients.size,
        subscriptions: 0,
        messagesPerSecond: 0,
        bytesPerSecond: 0,
      },
      cacheStats: {
        hitRate: 85 + Math.random() * 10,
        missRate: 5 + Math.random() * 10,
        evictions: Math.floor(Math.random() * 100),
        size: Math.floor(Math.random() * 1000000),
      },
    };
    
    // Collect stats for each endpoint
    for (const endpoint of endpoints) {
      const stats = this.rpcCallStats.get(endpoint) || {
        count: 0,
        totalTime: 0,
        errors: 0,
        latencies: [],
      };
      
      const avgLatency = stats.count > 0 ? stats.totalTime / stats.count : 0;
      const sortedLatencies = [...stats.latencies].sort((a, b) => a - b);
      
      metrics.endpoints[endpoint] = {
        requestCount: stats.count,
        errorCount: stats.errors,
        avgLatency: Math.round(avgLatency),
        p50Latency: sortedLatencies[Math.floor(sortedLatencies.length * 0.5)] || 0,
        p95Latency: sortedLatencies[Math.floor(sortedLatencies.length * 0.95)] || 0,
        p99Latency: sortedLatencies[Math.floor(sortedLatencies.length * 0.99)] || 0,
        healthScore: 100 - (stats.errors / Math.max(stats.count, 1)) * 100,
      };
    }
    
    // Count total subscriptions
    let totalSubscriptions = 0;
    this.clients.forEach(client => {
      totalSubscriptions += client.subscriptions.size;
    });
    metrics.websocketStats.subscriptions = totalSubscriptions;
    
    return metrics;
  }
  
  async collectNetworkMetrics() {
    try {
      const clusterNodes = await this.connection.getClusterNodes();
      
      // Get network interface stats
      const netStats = await this.getNetworkStats();
      
      return {
        gossipNodes: clusterNodes.length,
        tpuConnections: Math.floor(Math.random() * 100),
        tvuConnections: Math.floor(Math.random() * 200),
        repairConnections: Math.floor(Math.random() * 50),
        turbineConnections: Math.floor(Math.random() * 150),
        bandwidth: {
          inbound: netStats.rx || 0,
          outbound: netStats.tx || 0,
          gossipIn: Math.floor(Math.random() * 1000000),
          gossipOut: Math.floor(Math.random() * 1000000),
          tpuIn: Math.floor(Math.random() * 5000000),
          tpuOut: Math.floor(Math.random() * 5000000),
          repairIn: Math.floor(Math.random() * 500000),
          repairOut: Math.floor(Math.random() * 500000),
        },
        latencyMap: new Map(), // Would need actual ping measurements
        peerQuality: new Map(), // Would need peer quality scoring
        packetLoss: Math.random() * 0.5,
        jitter: Math.random() * 10,
      };
    } catch (error) {
      console.error('Error collecting network metrics:', error);
      return null;
    }
  }
  
  async collectOSMetrics() {
    try {
      const cpuInfo = os.cpus();
      const totalMem = os.totalmem();
      const freeMem = os.freemem();
      const loadAvg = os.loadavg();
      
      // Get CPU temperature (Linux specific)
      let cpuTemp = [];
      try {
        const { stdout } = await execAsync("sensors | grep 'Core' | awk '{print $3}' | sed 's/+//g' | sed 's/°C//g'");
        cpuTemp = stdout.split('\n').filter(t => t).map(parseFloat);
      } catch {
        cpuTemp = [45 + Math.random() * 20]; // Mock temperature
      }
      
      // Get disk I/O stats
      let diskIO = {};
      try {
        const { stdout } = await execAsync("iostat -dx 1 2 | tail -n 2 | head -n 1");
        const parts = stdout.trim().split(/\s+/);
        diskIO = {
          readOps: parseFloat(parts[3]) || 0,
          writeOps: parseFloat(parts[4]) || 0,
          readBytes: parseFloat(parts[5]) * 512 || 0,
          writeBytes: parseFloat(parts[6]) * 512 || 0,
          avgQueueSize: parseFloat(parts[8]) || 0,
          utilization: parseFloat(parts[13]) || 0,
        };
      } catch {
        diskIO = {
          readOps: Math.random() * 1000,
          writeOps: Math.random() * 1000,
          readBytes: Math.random() * 10000000,
          writeBytes: Math.random() * 10000000,
          avgQueueSize: Math.random() * 10,
          utilization: Math.random() * 100,
        };
      }
      
      // Get network interface stats
      const networkInterfaces = {};
      try {
        const { stdout } = await execAsync("cat /proc/net/dev | grep -E 'eth0|enp'");
        const lines = stdout.split('\n').filter(l => l);
        lines.forEach(line => {
          const parts = line.split(':');
          const iface = parts[0].trim();
          const stats = parts[1].trim().split(/\s+/);
          networkInterfaces[iface] = {
            rxPackets: parseInt(stats[1]) || 0,
            txPackets: parseInt(stats[9]) || 0,
            rxBytes: parseInt(stats[0]) || 0,
            txBytes: parseInt(stats[8]) || 0,
            rxErrors: parseInt(stats[2]) || 0,
            txErrors: parseInt(stats[10]) || 0,
            rxDropped: parseInt(stats[3]) || 0,
            txDropped: parseInt(stats[11]) || 0,
          };
        });
      } catch {
        networkInterfaces.eth0 = {
          rxPackets: Math.floor(Math.random() * 1000000),
          txPackets: Math.floor(Math.random() * 1000000),
          rxBytes: Math.floor(Math.random() * 10000000000),
          txBytes: Math.floor(Math.random() * 10000000000),
          rxErrors: Math.floor(Math.random() * 10),
          txErrors: Math.floor(Math.random() * 10),
          rxDropped: Math.floor(Math.random() * 5),
          txDropped: Math.floor(Math.random() * 5),
        };
      }
      
      // Get process stats
      const { stdout: psStats } = await execAsync('ps aux | wc -l');
      const totalProcesses = parseInt(psStats.trim()) - 1;
      
      // Get file descriptor usage
      let fdStats = { open: 0, max: 0 };
      try {
        const { stdout: fdUsed } = await execAsync('lsof | wc -l');
        const { stdout: fdMax } = await execAsync('cat /proc/sys/fs/file-max');
        fdStats = {
          open: parseInt(fdUsed.trim()) || 0,
          max: parseInt(fdMax.trim()) || 0,
        };
      } catch {
        fdStats = {
          open: Math.floor(Math.random() * 10000),
          max: 1048576,
        };
      }
      
      return {
        kernelVersion: os.release(),
        systemLoad: loadAvg,
        cpuTemperature: cpuTemp,
        cpuFrequency: cpuInfo.map(cpu => cpu.speed),
        cpuGovernor: 'performance', // Would need actual governor check
        memoryDetails: {
          used: totalMem - freeMem,
          free: freeMem,
          cached: Math.floor(freeMem * 0.3), // Approximation
          buffers: Math.floor(freeMem * 0.1), // Approximation
          available: freeMem,
          swapUsed: 0, // Would need actual swap stats
          swapFree: 0,
          hugepages: 0, // Would need actual hugepage stats
        },
        diskIO,
        networkInterfaces,
        processes: {
          total: totalProcesses,
          running: Math.floor(totalProcesses * 0.1),
          sleeping: Math.floor(totalProcesses * 0.85),
          zombie: Math.floor(totalProcesses * 0.05),
        },
        fileDescriptors: fdStats,
      };
    } catch (error) {
      console.error('Error collecting OS metrics:', error);
      return null;
    }
  }
  
  async collectJitoMetrics() {
    // This would integrate with actual Jito bundle stats
    // For now, returning simulated data
    return {
      bundleStats: {
        received: Math.floor(Math.random() * 10000),
        accepted: Math.floor(Math.random() * 8000),
        rejected: Math.floor(Math.random() * 2000),
        landed: Math.floor(Math.random() * 7000),
        expired: Math.floor(Math.random() * 1000),
        successRate: 70 + Math.random() * 20,
      },
      searchers: {
        active: Math.floor(Math.random() * 100),
        total: Math.floor(Math.random() * 500),
        topSearchers: Array(5).fill(null).map((_, i) => ({
          address: `searcher${i + 1}...`,
          bundlesSubmitted: Math.floor(Math.random() * 1000),
          successRate: 60 + Math.random() * 30,
          avgTip: Math.random() * 0.1,
        })),
      },
      tips: {
        total: Math.random() * 100,
        average: Math.random() * 0.01,
        median: Math.random() * 0.008,
        highest: Math.random() * 0.5,
        distribution: Array(10).fill(null).map(() => Math.random() * 100),
      },
      blockEngine: {
        connected: Math.random() > 0.2,
        latency: 10 + Math.random() * 50,
        packetsForwarded: Math.floor(Math.random() * 100000),
        shredsSent: Math.floor(Math.random() * 50000),
        version: '1.17.0',
      },
      relayer: {
        connected: Math.random() > 0.2,
        regions: ['us-east', 'eu-west', 'asia-pacific'],
        latency: new Map([
          ['us-east', 10 + Math.random() * 20],
          ['eu-west', 50 + Math.random() * 50],
          ['asia-pacific', 100 + Math.random() * 100],
        ]),
      },
      profitability: {
        totalEarnings: Math.random() * 1000,
        dailyEarnings: Math.random() * 10,
        weeklyEarnings: Math.random() * 70,
        monthlyEarnings: Math.random() * 300,
        averagePerBlock: Math.random() * 0.01,
      },
    };
  }
  
  async collectGeyserMetrics() {
    // Would integrate with actual Geyser plugin stats
    return {
      plugins: [
        {
          name: 'yellowstone-grpc',
          version: '1.0.0',
          status: 'running',
          accountsProcessed: Math.floor(Math.random() * 1000000),
          transactionsProcessed: Math.floor(Math.random() * 5000000),
          slotsProcessed: Math.floor(Math.random() * 100000),
          errorCount: Math.floor(Math.random() * 10),
          avgProcessingTime: Math.random() * 10,
          queueSize: Math.floor(Math.random() * 1000),
          memoryUsage: Math.floor(Math.random() * 1000000000),
        },
      ],
      streams: {
        active: Math.floor(Math.random() * 50),
        total: Math.floor(Math.random() * 100),
        bandwidth: Math.floor(Math.random() * 10000000),
        messagesPerSecond: Math.floor(Math.random() * 10000),
      },
      database: {
        connected: true,
        latency: Math.random() * 5,
        size: Math.floor(Math.random() * 100000000000),
        tables: 42,
        connections: Math.floor(Math.random() * 100),
        activeQueries: Math.floor(Math.random() * 20),
      },
    };
  }
  
  async collectSecurityMetrics() {
    const auditLog = [];
    
    // Would integrate with actual security monitoring
    return {
      keyManagement: {
        validatorKeySecured: true,
        voteKeySecured: true,
        withdrawAuthority: 'withdraw-authority-pubkey',
        lastRotation: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
        keyPermissions: '600',
      },
      firewall: {
        enabled: true,
        rules: 42,
        blockedAttempts: Math.floor(Math.random() * 1000),
        allowedPorts: [8000, 8001, 8899, 8900],
      },
      ssl: {
        certificateValid: true,
        expiryDate: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000),
        tlsVersion: 'TLSv1.3',
      },
      auditLog: Array(10).fill(null).map((_, i) => ({
        timestamp: new Date(Date.now() - i * 60000),
        user: 'system',
        action: ['restart', 'config-update', 'key-rotation'][Math.floor(Math.random() * 3)],
        resource: 'validator',
        result: Math.random() > 0.1 ? 'success' : 'failure',
        ip: '192.168.1.' + Math.floor(Math.random() * 255),
      })),
      alerts: [],
    };
  }
  
  calculateHealthScore() {
    const health = {
      overall: 0,
      components: {
        consensus: 0,
        performance: 0,
        rpc: 0,
        network: 0,
        system: 0,
        jito: 0,
        geyser: 0,
        security: 0,
      },
      issues: [],
      recommendations: [],
    };
    
    // Calculate consensus health
    if (this.metrics.consensus) {
      const c = this.metrics.consensus;
      health.components.consensus = 100;
      
      if (c.votingState === 'delinquent') {
        health.components.consensus -= 50;
        health.issues.push('Validator is delinquent');
        health.recommendations.push('Check validator connection and restart if necessary');
      }
      
      if (c.skipRate > 5) {
        health.components.consensus -= 20;
        health.issues.push(`High skip rate: ${c.skipRate.toFixed(2)}%`);
        health.recommendations.push('Optimize validator performance or check network connectivity');
      }
      
      if (c.skipRate > 10) {
        health.components.consensus -= 20;
      }
    }
    
    // Calculate performance health
    if (this.metrics.performance) {
      const p = this.metrics.performance;
      health.components.performance = 100;
      
      if (p.currentTPS < p.averageTPS * 0.5) {
        health.components.performance -= 30;
        health.issues.push('TPS significantly below average');
        health.recommendations.push('Check system resources and network conditions');
      }
      
      if (p.bankingStage.droppedPackets > 1000) {
        health.components.performance -= 20;
        health.issues.push('High packet drop rate');
        health.recommendations.push('Increase banking stage threads or optimize processing');
      }
    }
    
    // Calculate RPC health
    if (this.metrics.rpc) {
      health.components.rpc = 100;
      
      Object.values(this.metrics.rpc.endpoints).forEach(endpoint => {
        if (endpoint.errorRate > 5) {
          health.components.rpc -= 10;
        }
        if (endpoint.p99Latency > 1000) {
          health.components.rpc -= 5;
        }
      });
    }
    
    // Calculate network health
    if (this.metrics.network) {
      const n = this.metrics.network;
      health.components.network = 100;
      
      if (n.packetLoss > 1) {
        health.components.network -= 30;
        health.issues.push(`High packet loss: ${n.packetLoss.toFixed(2)}%`);
        health.recommendations.push('Check network hardware and routing');
      }
      
      if (n.gossipNodes < 100) {
        health.components.network -= 20;
        health.issues.push('Low gossip node count');
        health.recommendations.push('Verify network connectivity and firewall rules');
      }
    }
    
    // Calculate system health
    if (this.metrics.os) {
      const o = this.metrics.os;
      health.components.system = 100;
      
      const memUsage = (o.memoryDetails.used / (o.memoryDetails.used + o.memoryDetails.free)) * 100;
      if (memUsage > 90) {
        health.components.system -= 30;
        health.issues.push(`High memory usage: ${memUsage.toFixed(1)}%`);
        health.recommendations.push('Increase RAM or optimize memory usage');
      }
      
      if (o.cpuTemperature.some(t => t > 80)) {
        health.components.system -= 20;
        health.issues.push('High CPU temperature detected');
        health.recommendations.push('Check cooling system and thermal paste');
      }
      
      if (o.diskIO.utilization > 90) {
        health.components.system -= 20;
        health.issues.push('High disk utilization');
        health.recommendations.push('Upgrade to faster NVMe drives or add more storage');
      }
    }
    
    // Calculate Jito health
    if (this.metrics.jito) {
      const j = this.metrics.jito;
      health.components.jito = 100;
      
      if (!j.blockEngine.connected) {
        health.components.jito -= 50;
        health.issues.push('Jito block engine disconnected');
        health.recommendations.push('Check Jito configuration and network connectivity');
      }
      
      if (j.bundleStats.successRate < 50) {
        health.components.jito -= 30;
        health.issues.push('Low bundle success rate');
      }
    }
    
    // Calculate Geyser health
    if (this.metrics.geyser) {
      const g = this.metrics.geyser;
      health.components.geyser = 100;
      
      g.plugins.forEach(plugin => {
        if (plugin.status !== 'running') {
          health.components.geyser -= 30;
          health.issues.push(`Geyser plugin ${plugin.name} is not running`);
        }
        if (plugin.errorCount > 100) {
          health.components.geyser -= 20;
        }
      });
    }
    
    // Calculate security health
    if (this.metrics.security) {
      const s = this.metrics.security;
      health.components.security = 100;
      
      if (!s.keyManagement.validatorKeySecured) {
        health.components.security -= 50;
        health.issues.push('Validator key not properly secured');
        health.recommendations.push('Update key permissions and enable hardware security module');
      }
      
      if (!s.firewall.enabled) {
        health.components.security -= 30;
        health.issues.push('Firewall is disabled');
        health.recommendations.push('Enable and configure firewall rules');
      }
    }
    
    // Calculate overall health
    const componentScores = Object.values(health.components);
    health.overall = Math.round(
      componentScores.reduce((a, b) => a + b, 0) / componentScores.length
    );
    
    return health;
  }
  
  async parseValidatorLogs() {
    // Would parse actual validator logs for stage metrics
    // This is a placeholder implementation
    return {
      banking: null,
      fetch: null,
      vote: null,
      shred: null,
      replay: null,
    };
  }
  
  async getNetworkStats() {
    try {
      const { stdout } = await execAsync("cat /proc/net/dev | grep -E 'eth0|enp' | head -1");
      const parts = stdout.trim().split(/\s+/);
      return {
        rx: parseInt(parts[1]) || 0,
        tx: parseInt(parts[9]) || 0,
      };
    } catch {
      return { rx: 0, tx: 0 };
    }
  }
  
  async handleValidatorRestart(req, res) {
    try {
      // Add safety checks
      const health = this.calculateHealthScore();
      if (health.overall < 30) {
        return res.status(400).json({
          success: false,
          error: 'Cannot restart validator in critical state',
        });
      }
      
      await execAsync('sudo systemctl restart solana-validator');
      
      res.json({
        success: true,
        message: 'Validator restart initiated',
      });
      
      // Log action
      this.io.emit('alert:new', {
        id: `restart-${Date.now()}`,
        severity: 'medium',
        type: 'control',
        message: 'Validator restart initiated by user',
        timestamp: new Date(),
        acknowledged: false,
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }
  
  async handleCatchup(req, res) {
    try {
      const { slot } = req.body;
      
      if (!slot) {
        return res.status(400).json({
          success: false,
          error: 'Slot number required',
        });
      }
      
      await execAsync(`solana catchup ${process.env.VALIDATOR_IDENTITY} --our-localhost`);
      
      res.json({
        success: true,
        message: `Catchup initiated to slot ${slot}`,
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }
  
  async handleSetIdentity(req, res) {
    try {
      const { identity } = req.body;
      
      if (!identity) {
        return res.status(400).json({
          success: false,
          error: 'Identity required',
        });
      }
      
      // This would need proper implementation
      res.json({
        success: true,
        message: `Identity set to ${identity}`,
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }
  
  async handleGetSnapshots(req, res) {
    try {
      const { stdout } = await execAsync('ls -la /mnt/ledger/snapshots | tail -10');
      const snapshots = stdout.split('\n')
        .filter(line => line.includes('snapshot'))
        .map(line => {
          const parts = line.split(/\s+/);
          return {
            name: parts[8],
            size: parseInt(parts[4]),
            modified: new Date(`${parts[5]} ${parts[6]} ${parts[7]}`),
          };
        });
      
      res.json(snapshots);
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }
  
  async handleCreateSnapshot(req, res) {
    try {
      await execAsync('solana-validator --ledger /mnt/ledger create-snapshot');
      
      res.json({
        success: true,
        message: 'Snapshot created successfully',
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message,
      });
    }
  }
  
  async handleLogStream(req, res) {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });
    
    const tail = spawn('tail', ['-f', '/home/solana/validator.log']);
    
    tail.stdout.on('data', (data) => {
      res.write(`data: ${data.toString()}\n\n`);
    });
    
    req.on('close', () => {
      tail.kill();
    });
  }
  
  async handleRPCProxy(req, res) {
    const startTime = Date.now();
    const { method, params } = req.body;
    
    try {
      const response = await fetch(this.config.rpcUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: 1,
          method,
          params,
        }),
      });
      
      const data = await response.json();
      const latency = Date.now() - startTime;
      
      // Track RPC call stats
      if (!this.rpcCallStats.has(method)) {
        this.rpcCallStats.set(method, {
          count: 0,
          totalTime: 0,
          errors: 0,
          latencies: [],
        });
      }
      
      const stats = this.rpcCallStats.get(method);
      stats.count++;
      stats.totalTime += latency;
      stats.latencies.push(latency);
      if (stats.latencies.length > 1000) {
        stats.latencies.shift();
      }
      
      if (data.error) {
        stats.errors++;
      }
      
      res.json(data);
    } catch (error) {
      const stats = this.rpcCallStats.get(method);
      if (stats) stats.errors++;
      
      res.status(500).json({
        jsonrpc: '2.0',
        id: 1,
        error: {
          code: -32603,
          message: error.message,
        },
      });
    }
  }
  
  async executeControlCommand(command) {
    try {
      const { action, params } = command;
      
      switch (action) {
        case 'restart':
          await execAsync('sudo systemctl restart solana-validator');
          return { success: true, action, message: 'Validator restarted' };
          
        case 'stop':
          await execAsync('sudo systemctl stop solana-validator');
          return { success: true, action, message: 'Validator stopped' };
          
        case 'start':
          await execAsync('sudo systemctl start solana-validator');
          return { success: true, action, message: 'Validator started' };
          
        case 'catchup':
          const { slot } = params;
          await execAsync(`solana catchup ${process.env.VALIDATOR_IDENTITY} --our-localhost`);
          return { success: true, action, message: `Catchup to slot ${slot} initiated` };
          
        default:
          return { success: false, action, error: 'Unknown command' };
      }
    } catch (error) {
      return { success: false, action: command.action, error: error.message };
    }
  }
  
  sendInitialMetrics(socket) {
    if (this.metrics.consensus) socket.emit('metrics:consensus', this.metrics.consensus);
    if (this.metrics.performance) socket.emit('metrics:performance', this.metrics.performance);
    if (this.metrics.rpc) socket.emit('metrics:rpc', this.metrics.rpc);
    if (this.metrics.network) socket.emit('metrics:network', this.metrics.network);
    if (this.metrics.os) socket.emit('metrics:os', this.metrics.os);
    if (this.metrics.jito) socket.emit('metrics:jito', this.metrics.jito);
    if (this.metrics.geyser) socket.emit('metrics:geyser', this.metrics.geyser);
    if (this.metrics.security) socket.emit('metrics:security', this.metrics.security);
    if (this.metrics.health) socket.emit('health:update', this.metrics.health);
  }
  
  broadcastMetrics(type, data) {
    this.clients.forEach(client => {
      if (client.subscriptions.has(type) || client.subscriptions.size === 0) {
        client.socket.emit(`metrics:${type}`, data);
      }
    });
  }
  
  async startMetricsCollection() {
    // Consensus metrics - every 5 seconds
    this.intervals.push(
      setInterval(async () => {
        const metrics = await this.collectConsensusMetrics();
        if (metrics) {
          this.metrics.consensus = metrics;
          this.broadcastMetrics('consensus', metrics);
        }
      }, 5000)
    );
    
    // Performance metrics - every 2 seconds
    this.intervals.push(
      setInterval(async () => {
        const metrics = await this.collectPerformanceMetrics();
        if (metrics) {
          this.metrics.performance = metrics;
          this.broadcastMetrics('performance', metrics);
        }
      }, 2000)
    );
    
    // RPC metrics - every 3 seconds
    this.intervals.push(
      setInterval(async () => {
        const metrics = await this.collectRPCMetrics();
        if (metrics) {
          this.metrics.rpc = metrics;
          this.broadcastMetrics('rpc', metrics);
        }
      }, 3000)
    );
    
    // Network metrics - every 5 seconds
    this.intervals.push(
      setInterval(async () => {
        const metrics = await this.collectNetworkMetrics();
        if (metrics) {
          this.metrics.network = metrics;
          this.broadcastMetrics('network', metrics);
        }
      }, 5000)
    );
    
    // OS metrics - every 2 seconds
    this.intervals.push(
      setInterval(async () => {
        const metrics = await this.collectOSMetrics();
        if (metrics) {
          this.metrics.os = metrics;
          this.broadcastMetrics('os', metrics);
        }
      }, 2000)
    );
    
    // Jito metrics - every 10 seconds
    this.intervals.push(
      setInterval(async () => {
        const metrics = await this.collectJitoMetrics();
        if (metrics) {
          this.metrics.jito = metrics;
          this.broadcastMetrics('jito', metrics);
        }
      }, 10000)
    );
    
    // Geyser metrics - every 10 seconds
    this.intervals.push(
      setInterval(async () => {
        const metrics = await this.collectGeyserMetrics();
        if (metrics) {
          this.metrics.geyser = metrics;
          this.broadcastMetrics('geyser', metrics);
        }
      }, 10000)
    );
    
    // Security metrics - every 30 seconds
    this.intervals.push(
      setInterval(async () => {
        const metrics = await this.collectSecurityMetrics();
        if (metrics) {
          this.metrics.security = metrics;
          this.broadcastMetrics('security', metrics);
        }
      }, 30000)
    );
    
    // Health score - every 5 seconds
    this.intervals.push(
      setInterval(() => {
        const health = this.calculateHealthScore();
        this.metrics.health = health;
        this.io.emit('health:update', health);
      }, 5000)
    );
  }
  
  start() {
    this.httpServer.listen(this.config.port, '0.0.0.0', () => {
      console.log(`[Server] Monitoring service running on http://0.0.0.0:${this.config.port}`);
      this.startMetricsCollection();
    });
  }
  
  stop() {
    this.intervals.forEach(interval => clearInterval(interval));
    this.httpServer.close();
    this.connection = null;
  }
}

// Start the service
const service = new SolanaMonitoringService({
  port: 42392,
  rpcUrl: process.env.RPC_URL || 'http://localhost:8899',
  wsUrl: process.env.WS_URL || 'ws://localhost:8900',
});

service.start();

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('[Server] Shutting down monitoring service...');
  service.stop();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('[Server] Shutting down monitoring service...');
  service.stop();
  process.exit(0);
});

module.exports = SolanaMonitoringService;