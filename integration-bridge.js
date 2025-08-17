/**
 * Elite MEV Detection Integration Bridge
 * Connects Solana transaction stream to detection models and ClickHouse
 * Target: 200k+ events/sec processing with sub-slot detection
 */

const { Connection, PublicKey } = require('@solana/web3.js');
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const WebSocket = require('ws');
const http = require('http');
const crypto = require('crypto');
const axios = require('axios');

// Configuration
const CONFIG = {
    SOLANA_RPC: process.env.SOLANA_RPC || 'https://api.mainnet-beta.solana.com',
    SOLANA_WS: process.env.SOLANA_WS || 'wss://api.mainnet-beta.solana.com',
    CLICKHOUSE_URL: process.env.CLICKHOUSE_URL || 'http://localhost:8123',
    DETECTOR_API: process.env.DETECTOR_API || 'http://localhost:8000',
    
    // Performance settings
    BATCH_SIZE: 1000,
    FLUSH_INTERVAL_MS: 100,
    MAX_QUEUE_SIZE: 10000,
    
    // Detection settings
    ENABLE_REALTIME_DETECTION: true,
    DETECTION_CONFIDENCE_THRESHOLD: 0.7,
    
    // Priority addresses for monitoring
    PRIORITY_ADDRESSES: new Set([
        'B91piBSfCBRs5rUxCMRdJEGv7tNEnFxweWcdQJHJoFpi',
        '6gAnjderE13TGGFeqdPVQ438jp2FPVeyXAszxKu9y338',
        'CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C',
        'E6YoRP3adE5XYneSseLee15wJshDxCsmyD2WtLvAmfLi',
        'pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA'
    ]),
    
    // Known DEX programs
    DEX_PROGRAMS: new Set([
        '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',  // Raydium V4
        'CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK',  // Raydium CPMM
        '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP',  // Orca Whirlpool
        'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc',  // Whirlpool
        '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P',  // Pump.fun
    ])
};

// Express app setup
const app = express();
const server = http.createServer(app);

// CORS headers
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Proxy to MEV Backend (Rust service)
app.use('/api/mev', createProxyMiddleware({
  target: 'http://localhost:8080',
  changeOrigin: true,
  pathRewrite: { '^/api/mev': '' }
}));

// Proxy to API Gateway
app.use('/api/gateway', createProxyMiddleware({
  target: 'https://localhost:3000',
  changeOrigin: true,
  secure: false,
  pathRewrite: { '^/api/gateway': '/api' }
}));

// Proxy to Metrics
app.use('/api/metrics', createProxyMiddleware({
  target: 'http://localhost:9090',
  changeOrigin: true,
  pathRewrite: { '^/api/metrics': '/metrics' }
}));

// WebSocket Bridge
const wss = new WebSocket.Server({ server, path: '/ws' });

// Connect to backend WebSocket services
let backendWS = null;
function connectBackend() {
  backendWS = new WebSocket('ws://localhost:8080/ws');
  
  backendWS.on('open', () => {
    console.log('Connected to MEV backend WebSocket');
  });
  
  backendWS.on('message', (data) => {
    // Broadcast to all frontend clients
    wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(data);
      }
    });
  });
  
  backendWS.on('close', () => {
    console.log('Backend WebSocket disconnected, reconnecting...');
    setTimeout(connectBackend, 1000);
  });
  
  backendWS.on('error', (err) => {
    console.error('Backend WebSocket error:', err);
  });
}

// MEV Detection Statistics
const detectionStats = {
    totalProcessed: 0,
    mevDetected: 0,
    sandwichesFound: 0,
    arbitragesFound: 0,
    priorityAddressHits: 0,
    latencyP50: 0,
    latencyP99: 0
};

// Transaction queue for batch processing
const transactionQueue = [];
let isProcessing = false;

/**
 * Encode instruction sequence into compact numeric tokens
 * Format: (program_id_index << 8) | opcode
 */
async function encodeInstructionSeq(tx) {
    if (!tx.message || !tx.message.instructions) {
        return [];
    }
    
    const encoded = [];
    const { accountKeys, instructions } = tx.message;
    
    for (const instruction of instructions) {
        const programIdIndex = instruction.programIdIndex;
        const opcode = instruction.data ? instruction.data[0] : 0;
        
        // Compact encoding: program index in upper bits, opcode in lower
        const token = (programIdIndex << 8) | (opcode & 0xFF);
        encoded.push(token);
        
        // Add DEX program detection
        if (accountKeys[programIdIndex]) {
            const programId = accountKeys[programIdIndex];
            if (CONFIG.DEX_PROGRAMS.has(programId)) {
                // Mark as DEX interaction with specific program
                encoded.push(0x1000 | Array.from(CONFIG.DEX_PROGRAMS).indexOf(programId));
            }
        }
    }
    
    return encoded;
}

/**
 * Upsert raw transaction to ClickHouse
 */
async function upsertRawTx(parsed) {
    try {
        const { transaction, slot, blockTime, meta } = parsed;
        
        // Extract key fields
        const signature = transaction.signatures[0];
        const accountKeys = transaction.message.accountKeys;
        const instructions = await encodeInstructionSeq(transaction);
        
        // Detect entity (simplified - would use more sophisticated matching in production)
        let entity = 'UNKNOWN';
        for (const key of accountKeys) {
            if (CONFIG.PRIORITY_ADDRESSES.has(key)) {
                entity = key.substring(0, 3); // Use first 3 chars as entity ID
                break;
            }
        }
        
        // Calculate decision DNA
        const dnaInput = `${signature}:${slot}:${instructions.join(',')}`;
        const decisionDna = crypto.createHash('sha256')
            .update(dnaInput)
            .digest('hex')
            .substring(0, 16);
        
        // Prepare ClickHouse insert
        const insertData = {
            timestamp: new Date(blockTime * 1000),
            slot: slot,
            signature: signature,
            entity_id: entity,
            account_keys: accountKeys,
            instruction_seq: instructions,
            fee: meta?.fee || 0,
            compute_units: meta?.computeUnitsConsumed || 0,
            status: meta?.err ? 'failed' : 'success',
            decision_dna: decisionDna
        };
        
        // Add to batch queue
        transactionQueue.push(insertData);
        
        // Process queue if batch size reached
        if (transactionQueue.length >= CONFIG.BATCH_SIZE) {
            await flushTransactionQueue();
        }
        
        return insertData;
    } catch (error) {
        console.error('Error upserting transaction:', error);
        throw error;
    }
}

/**
 * Flush transaction queue to ClickHouse
 */
async function flushTransactionQueue() {
    if (transactionQueue.length === 0 || isProcessing) {
        return;
    }
    
    isProcessing = true;
    const batch = transactionQueue.splice(0, CONFIG.BATCH_SIZE);
    
    try {
        // Format for ClickHouse insert
        const values = batch.map(tx => [
            tx.timestamp.toISOString(),
            tx.slot,
            tx.signature,
            tx.entity_id,
            JSON.stringify(tx.account_keys),
            JSON.stringify(tx.instruction_seq),
            tx.fee,
            tx.compute_units,
            tx.status,
            tx.decision_dna
        ]);
        
        const query = `
            INSERT INTO raw_transactions 
            (timestamp, slot, signature, entity_id, account_keys, 
             instruction_seq, fee, compute_units, status, decision_dna)
            FORMAT JSONEachRow
        `;
        
        await axios.post(CONFIG.CLICKHOUSE_URL, {
            query: query,
            data: values
        });
        
        console.log(`Flushed ${batch.length} transactions to ClickHouse`);
    } catch (error) {
        console.error('Error flushing to ClickHouse:', error);
        // Re-add failed batch to queue
        transactionQueue.unshift(...batch);
    } finally {
        isProcessing = false;
    }
}

// Periodic flush
setInterval(flushTransactionQueue, CONFIG.FLUSH_INTERVAL_MS);

// Handle frontend connections
wss.on('connection', (ws) => {
  console.log('Frontend client connected');
  
  // Send initial data
  ws.send(JSON.stringify({
    type: 'connected',
    timestamp: Date.now(),
    services: {
      mev: 'http://localhost:8080',
      gateway: 'https://localhost:3000',
      metrics: 'http://localhost:9090',
      detector: CONFIG.DETECTOR_API
    },
    stats: detectionStats,
    priorityAddresses: Array.from(CONFIG.PRIORITY_ADDRESSES)
  }));
  
  ws.on('message', async (message) => {
    try {
        const data = JSON.parse(message);
        
        // Handle detection requests
        if (data.type === 'detect') {
            const result = await runDetection(data.transaction);
            ws.send(JSON.stringify({
                type: 'detection_result',
                result: result
            }));
        }
        
        // Forward to backend if connected
        if (backendWS && backendWS.readyState === WebSocket.OPEN) {
            backendWS.send(message);
        }
    } catch (error) {
        console.error('Message handling error:', error);
    }
  });
  
  ws.on('close', () => {
    console.log('Frontend client disconnected');
  });
});

// Detection API endpoint
app.post('/api/detect', express.json(), async (req, res) => {
    try {
        const { transaction, features } = req.body;
        
        // Check if priority address
        const isPriority = features?.accountKeys?.some(key => 
            CONFIG.PRIORITY_ADDRESSES.has(key)
        );
        
        if (isPriority) {
            detectionStats.priorityAddressHits++;
        }
        
        // Run detection
        const response = await axios.post(`${CONFIG.DETECTOR_API}/detect`, {
            transaction,
            features
        }, {
            timeout: 1000
        });
        
        // Update stats
        detectionStats.totalProcessed++;
        if (response.data.is_mev) {
            detectionStats.mevDetected++;
            if (response.data.mev_type === 'sandwich') {
                detectionStats.sandwichesFound++;
            } else if (response.data.mev_type === 'arbitrage') {
                detectionStats.arbitragesFound++;
            }
        }
        
        res.json(response.data);
    } catch (error) {
        console.error('Detection error:', error);
        res.status(500).json({ error: 'Detection failed' });
    }
});

// Entity profile endpoint
app.get('/api/entity/:address', async (req, res) => {
    try {
        const { address } = req.params;
        
        // Fetch entity profile from detector API
        const response = await axios.get(
            `${CONFIG.DETECTOR_API}/entity/${address}`
        );
        
        res.json(response.data);
    } catch (error) {
        console.error('Entity profile error:', error);
        res.status(500).json({ error: 'Failed to fetch entity profile' });
    }
});

// Statistics endpoint
app.get('/api/stats', (req, res) => {
    res.json({
        detection: detectionStats,
        timestamp: Date.now(),
        uptime: process.uptime(),
        connections: {
            frontend: wss.clients.size,
            backend: backendWS && backendWS.readyState === WebSocket.OPEN
        }
    });
});

// Landing rate metrics endpoint
app.get('/api/v1/metrics/landing-rate', async (req, res) => {
    try {
        const { entity, hours = 24 } = req.query;
        
        const query = `
            SELECT 
                entity_id,
                COUNT(*) as total_bundles,
                countIf(landed = 1) as landed_bundles,
                landed_bundles / total_bundles as landing_rate,
                quantile(0.5)(landing_rate) OVER () as median_rate,
                quantile(0.95)(landing_rate) OVER () as p95_rate
            FROM mev_bundles
            WHERE timestamp >= now() - INTERVAL ${parseInt(hours)} HOUR
                ${entity ? `AND entity_id = '${entity}'` : ''}
            GROUP BY entity_id
            ORDER BY landing_rate DESC
            LIMIT 100
        `;
        
        const response = await axios.post(`${CONFIG.CLICKHOUSE_URL}`, {
            query: query
        });
        
        res.json({
            data: response.data,
            timestamp: Date.now(),
            query_hours: hours
        });
    } catch (error) {
        console.error('Landing rate metrics error:', error);
        res.status(500).json({ error: 'Failed to fetch landing rate metrics' });
    }
});

// GET /api/v1/metrics/archetype - Archetype distribution metrics (DEFENSIVE-ONLY)
app.get('/api/v1/metrics/archetype', async (req, res) => {
    try {
        const { days = 7 } = req.query;
        
        const query = `
            WITH entity_metrics AS (
                SELECT 
                    attacker_addr,
                    count() as attack_count,
                    sum(attacker_profit_sol) as total_profit,
                    avg(d_ms) as avg_response_ms,
                    uniqExact(pool) as unique_pools,
                    avg(ensemble_score) as avg_confidence
                FROM ch.candidates
                WHERE detection_ts >= now() - INTERVAL ${parseInt(days)} DAY
                GROUP BY attacker_addr
            )
            SELECT
                CASE
                    WHEN attack_count > 1000 AND avg_response_ms < 20 THEN 'Empire'
                    WHEN unique_pools < 5 AND avg_confidence > 0.8 THEN 'Warlord'
                    WHEN attack_count < 100 THEN 'Guerrilla'
                    ELSE 'Unknown'
                END as archetype,
                count() as entity_count,
                sum(total_profit) as archetype_profit,
                avg(avg_response_ms) as avg_latency,
                avg(avg_confidence) as avg_confidence_score
            FROM entity_metrics
            GROUP BY archetype
        `;
        
        const response = await axios.post(`${CONFIG.CLICKHOUSE_URL}`, {
            query: query
        });
        
        res.json({
            timestamp: new Date().toISOString(),
            period_days: parseInt(days),
            archetypes: response.data,
            metadata: {
                confidence_interval: 0.95,
                detection_only: true
            }
        });
    } catch (error) {
        console.error('Archetype metrics error:', error);
        res.status(500).json({ error: 'Failed to fetch archetype metrics' });
    }
});

// GET /api/v1/metrics/adjacency - Bundle adjacency metrics (DEFENSIVE-ONLY)
app.get('/api/v1/metrics/adjacency', async (req, res) => {
    try {
        const { days = 7 } = req.query;
        
        const query = `
            SELECT
                toDate(detection_ts) as date,
                avg(d_ms) as avg_adjacency_ms,
                quantile(0.5)(d_ms) as p50_adjacency_ms,
                quantile(0.95)(d_ms) as p95_adjacency_ms,
                quantile(0.99)(d_ms) as p99_adjacency_ms,
                countIf(d_ms < 50) / count() as tight_adjacency_rate,
                countIf(d_slots = 0) / count() as same_slot_rate,
                avg(ensemble_score) as avg_detection_confidence
            FROM ch.candidates
            WHERE detection_ts >= now() - INTERVAL ${parseInt(days)} DAY
            GROUP BY date
            ORDER BY date DESC
        `;
        
        const response = await axios.post(`${CONFIG.CLICKHOUSE_URL}`, {
            query: query
        });
        
        res.json({
            timestamp: new Date().toISOString(),
            period_days: parseInt(days),
            daily_metrics: response.data,
            detection_only: true
        });
    } catch (error) {
        console.error('Adjacency metrics error:', error);
        res.status(500).json({ error: 'Failed to fetch adjacency metrics' });
    }
});

// SOL drain/extraction metrics endpoint
app.get('/api/v1/metrics/sol-drain', async (req, res) => {
    try {
        const { entity, days = 30 } = req.query;
        
        const query = `
            SELECT 
                entity_id,
                SUM(profit_sol) as total_extraction,
                SUM(profit_sol) / ${parseInt(days)} as daily_average,
                COUNT(DISTINCT victim_address) as unique_victims,
                SUM(victim_loss_sol) as total_victim_losses,
                SUM(tip_lamports) / 1e9 as total_tips_sol,
                (SUM(profit_sol) - SUM(tip_lamports) / 1e9) as net_profit
            FROM mev_transactions
            WHERE timestamp >= now() - INTERVAL ${parseInt(days)} DAY
                ${entity ? `AND entity_id = '${entity}'` : ''}
            GROUP BY entity_id
            ORDER BY total_extraction DESC
            LIMIT 100
        `;
        
        const response = await axios.post(`${CONFIG.CLICKHOUSE_URL}`, {
            query: query
        });
        
        res.json({
            data: response.data,
            timestamp: Date.now(),
            query_days: days,
            reference: {
                b91_monthly: 7800,
                note: 'B91 reference: ~7,800 SOL/month'
            }
        });
    } catch (error) {
        console.error('SOL drain metrics error:', error);
        res.status(500).json({ error: 'Failed to fetch SOL drain metrics' });
    }
});

// Latency histogram endpoint
app.get('/api/v1/metrics/latency-hist', async (req, res) => {
    try {
        const { entity, hours = 24 } = req.query;
        
        const query = `
            SELECT 
                entity_id,
                quantile(0.25)(decision_latency_ms) as p25,
                quantile(0.50)(decision_latency_ms) as p50,
                quantile(0.75)(decision_latency_ms) as p75,
                quantile(0.90)(decision_latency_ms) as p90,
                quantile(0.95)(decision_latency_ms) as p95,
                quantile(0.99)(decision_latency_ms) as p99,
                quantile(0.999)(decision_latency_ms) as p999,
                avg(decision_latency_ms) as mean,
                stddevPop(decision_latency_ms) as stddev,
                p95 - p50 as p95_p50_spread,
                COUNT(*) as sample_count
            FROM mev_decisions
            WHERE timestamp >= now() - INTERVAL ${parseInt(hours)} HOUR
                AND decision_latency_ms > 0
                AND decision_latency_ms < 10000
                ${entity ? `AND entity_id = '${entity}'` : ''}
            GROUP BY entity_id
            ORDER BY p50 ASC
            LIMIT 100
        `;
        
        const response = await axios.post(`${CONFIG.CLICKHOUSE_URL}`, {
            query: query
        });
        
        res.json({
            data: response.data,
            timestamp: Date.now(),
            query_hours: hours,
            thresholds: {
                ultra_optimized: { p99: 20, spread: 10 },
                optimized: { p99: 100, spread: 50 },
                normal: { p99: 500, spread: 200 }
            }
        });
    } catch (error) {
        console.error('Latency histogram error:', error);
        res.status(500).json({ error: 'Failed to fetch latency histogram' });
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: Date.now(),
    connections: {
      frontend: wss.clients.size,
      backend: backendWS && backendWS.readyState === WebSocket.OPEN
    },
    stats: detectionStats
  });
});

// Start backend connection
connectBackend();

// Periodic stats broadcast
setInterval(() => {
    const statsMessage = JSON.stringify({
        type: 'stats_update',
        stats: detectionStats,
        timestamp: Date.now()
    });
    
    wss.clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(statsMessage);
        }
    });
}, 5000);

const PORT = process.env.PORT || 4000;
server.listen(PORT, () => {
  console.log(`MEV Detection Bridge running on port ${PORT}`);
  console.log(`WebSocket available at ws://localhost:${PORT}/ws`);
  console.log(`Health check at http://localhost:${PORT}/health`);
  console.log(`Detection stats at http://localhost:${PORT}/api/stats`);
});