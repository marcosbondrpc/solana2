// MEV Data Processing Worker for parallel computation
// Handles heavy calculations off the main thread

// Calculate arbitrage profitability with slippage
function calculateArbitrageProfitability(opportunity) {
  const { amountIn, priceA, priceB, slippage, gasEstimate, path } = opportunity;
  
  // Calculate output amounts through the path
  let currentAmount = amountIn;
  let totalGas = gasEstimate || 0;
  
  for (let i = 0; i < path.length - 1; i++) {
    const step = path[i];
    const nextStep = path[i + 1];
    
    // Apply DEX fee (typically 0.3%)
    const feeMultiplier = 0.997;
    currentAmount *= feeMultiplier;
    
    // Apply price impact based on liquidity depth
    const priceImpact = calculatePriceImpact(currentAmount, step.liquidity);
    currentAmount *= (1 - priceImpact);
    
    // Apply slippage
    currentAmount *= (1 - slippage);
  }
  
  const profit = currentAmount - amountIn - totalGas;
  const roi = (profit / amountIn) * 100;
  
  return {
    expectedProfit: profit,
    roi,
    finalAmount: currentAmount,
    viability: profit > 0
  };
}

// Calculate price impact based on trade size and liquidity
function calculatePriceImpact(tradeSize, liquidity) {
  if (!liquidity || liquidity === 0) return 0.01; // 1% default impact
  
  const impactRatio = tradeSize / liquidity;
  // Quadratic impact model
  return Math.min(impactRatio * impactRatio * 0.1, 0.5); // Cap at 50% impact
}

// Process batch of arbitrage opportunities
function processArbitrageOpportunities(opportunities) {
  const processed = opportunities.map(opp => {
    const profitability = calculateArbitrageProfitability(opp);
    
    // Calculate confidence score
    const confidence = calculateConfidence(opp, profitability);
    
    // Determine optimal gas price
    const optimalGas = calculateOptimalGas(profitability.expectedProfit);
    
    return {
      ...opp,
      ...profitability,
      confidence,
      optimalGas,
      score: profitability.expectedProfit * confidence,
      timestamp: Date.now()
    };
  });
  
  // Sort by score
  processed.sort((a, b) => b.score - a.score);
  
  return processed;
}

// Calculate confidence score for opportunity
function calculateConfidence(opportunity, profitability) {
  let confidence = 0.5; // Base confidence
  
  // Factor in profitability
  if (profitability.expectedProfit > 1) confidence += 0.2;
  else if (profitability.expectedProfit > 0.5) confidence += 0.1;
  
  // Factor in slippage tolerance
  if (opportunity.slippage < 0.01) confidence += 0.15;
  else if (opportunity.slippage < 0.02) confidence += 0.1;
  
  // Factor in path complexity
  if (opportunity.path.length === 2) confidence += 0.1;
  else if (opportunity.path.length > 4) confidence -= 0.1;
  
  // Factor in DEX reliability
  const reliableDexes = ['raydium', 'orca', 'phoenix'];
  if (reliableDexes.includes(opportunity.dexA) && reliableDexes.includes(opportunity.dexB)) {
    confidence += 0.1;
  }
  
  return Math.min(Math.max(confidence, 0), 1);
}

// Calculate optimal gas price based on expected profit
function calculateOptimalGas(expectedProfit) {
  if (expectedProfit > 10) return 0.1; // High priority
  if (expectedProfit > 5) return 0.05;
  if (expectedProfit > 1) return 0.02;
  return 0.01; // Low priority
}

// Aggregate statistics from multiple data points
function aggregateStatistics(dataPoints) {
  if (!dataPoints || dataPoints.length === 0) return null;
  
  const stats = {
    count: dataPoints.length,
    sum: 0,
    mean: 0,
    median: 0,
    min: Infinity,
    max: -Infinity,
    variance: 0,
    stdDev: 0,
    p25: 0,
    p75: 0,
    p95: 0,
    p99: 0
  };
  
  // Calculate basic stats
  dataPoints.forEach(value => {
    stats.sum += value;
    stats.min = Math.min(stats.min, value);
    stats.max = Math.max(stats.max, value);
  });
  
  stats.mean = stats.sum / stats.count;
  
  // Sort for percentiles
  const sorted = [...dataPoints].sort((a, b) => a - b);
  
  // Calculate percentiles
  stats.median = sorted[Math.floor(sorted.length / 2)];
  stats.p25 = sorted[Math.floor(sorted.length * 0.25)];
  stats.p75 = sorted[Math.floor(sorted.length * 0.75)];
  stats.p95 = sorted[Math.floor(sorted.length * 0.95)];
  stats.p99 = sorted[Math.floor(sorted.length * 0.99)];
  
  // Calculate variance and standard deviation
  const squaredDiffs = dataPoints.map(value => Math.pow(value - stats.mean, 2));
  stats.variance = squaredDiffs.reduce((a, b) => a + b, 0) / stats.count;
  stats.stdDev = Math.sqrt(stats.variance);
  
  return stats;
}

// Process latency metrics
function processLatencyMetrics(metrics) {
  const processed = {
    timestamp: Date.now(),
    rpc: aggregateStatistics(metrics.map(m => m.rpcLatency)),
    quic: aggregateStatistics(metrics.map(m => m.quicLatency)),
    tcp: aggregateStatistics(metrics.map(m => m.tcpLatency)),
    websocket: aggregateStatistics(metrics.map(m => m.websocketLatency)),
    parsing: aggregateStatistics(metrics.map(m => m.parsingLatency)),
    execution: aggregateStatistics(metrics.map(m => m.executionLatency))
  };
  
  // Calculate comparative metrics
  processed.quicAdvantage = processed.rpc.mean > 0 
    ? ((processed.rpc.mean - processed.quic.mean) / processed.rpc.mean) * 100
    : 0;
  
  processed.overallHealth = calculateNetworkHealth(processed);
  
  return processed;
}

// Calculate network health score
function calculateNetworkHealth(metrics) {
  let score = 100;
  
  // Penalize high latency
  if (metrics.quic.mean > 100) score -= 30;
  else if (metrics.quic.mean > 50) score -= 15;
  else if (metrics.quic.mean > 20) score -= 5;
  
  // Penalize high variance
  if (metrics.quic.stdDev > 50) score -= 20;
  else if (metrics.quic.stdDev > 20) score -= 10;
  
  // Penalize poor p99
  if (metrics.quic.p99 > 200) score -= 20;
  else if (metrics.quic.p99 > 100) score -= 10;
  
  return Math.max(score, 0);
}

// Calculate DEX flow patterns
function analyzeDEXFlows(transactions) {
  const flows = new Map();
  const patterns = [];
  
  transactions.forEach(tx => {
    const key = `${tx.from}_${tx.to}_${tx.token}`;
    
    if (!flows.has(key)) {
      flows.set(key, {
        from: tx.from,
        to: tx.to,
        token: tx.token,
        volume: 0,
        count: 0,
        avgAmount: 0,
        timestamps: []
      });
    }
    
    const flow = flows.get(key);
    flow.volume += tx.amount;
    flow.count++;
    flow.avgAmount = flow.volume / flow.count;
    flow.timestamps.push(tx.timestamp);
    
    // Detect patterns
    if (flow.count > 10) {
      const pattern = detectTradingPattern(flow.timestamps);
      if (pattern) {
        patterns.push({
          ...pattern,
          flow: key
        });
      }
    }
  });
  
  return {
    flows: Array.from(flows.values()),
    patterns,
    totalVolume: Array.from(flows.values()).reduce((sum, f) => sum + f.volume, 0),
    uniquePairs: flows.size
  };
}

// Detect trading patterns in timestamps
function detectTradingPattern(timestamps) {
  if (timestamps.length < 10) return null;
  
  // Sort timestamps
  timestamps.sort((a, b) => a - b);
  
  // Calculate intervals
  const intervals = [];
  for (let i = 1; i < timestamps.length; i++) {
    intervals.push(timestamps[i] - timestamps[i - 1]);
  }
  
  // Check for regular intervals (bot trading)
  const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  const variance = intervals.reduce((sum, i) => sum + Math.pow(i - avgInterval, 2), 0) / intervals.length;
  const stdDev = Math.sqrt(variance);
  
  if (stdDev < avgInterval * 0.1) {
    return {
      type: 'regular',
      interval: avgInterval,
      confidence: 1 - (stdDev / avgInterval)
    };
  }
  
  // Check for burst patterns
  const shortIntervals = intervals.filter(i => i < avgInterval * 0.5).length;
  if (shortIntervals > intervals.length * 0.3) {
    return {
      type: 'burst',
      burstRatio: shortIntervals / intervals.length,
      avgBurstInterval: intervals.filter(i => i < avgInterval * 0.5)
        .reduce((a, b) => a + b, 0) / shortIntervals
    };
  }
  
  return null;
}

// Calculate profit metrics
function calculateProfitMetrics(trades) {
  const metrics = {
    totalProfit: 0,
    totalLoss: 0,
    netProfit: 0,
    winRate: 0,
    avgWin: 0,
    avgLoss: 0,
    maxWin: 0,
    maxLoss: 0,
    profitFactor: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    profitByHour: new Array(24).fill(0),
    profitByDex: {},
    profitByToken: {}
  };
  
  let wins = 0;
  let losses = 0;
  let cumulative = 0;
  let peak = 0;
  let drawdowns = [];
  
  trades.forEach(trade => {
    const profit = trade.profit || 0;
    cumulative += profit;
    
    if (profit > 0) {
      metrics.totalProfit += profit;
      metrics.maxWin = Math.max(metrics.maxWin, profit);
      wins++;
    } else {
      metrics.totalLoss += Math.abs(profit);
      metrics.maxLoss = Math.max(metrics.maxLoss, Math.abs(profit));
      losses++;
    }
    
    // Track by hour
    const hour = new Date(trade.timestamp).getHours();
    metrics.profitByHour[hour] += profit;
    
    // Track by DEX
    if (!metrics.profitByDex[trade.dex]) {
      metrics.profitByDex[trade.dex] = 0;
    }
    metrics.profitByDex[trade.dex] += profit;
    
    // Track by token
    if (!metrics.profitByToken[trade.token]) {
      metrics.profitByToken[trade.token] = 0;
    }
    metrics.profitByToken[trade.token] += profit;
    
    // Calculate drawdown
    if (cumulative > peak) {
      peak = cumulative;
    } else {
      const drawdown = (peak - cumulative) / peak;
      drawdowns.push(drawdown);
      metrics.maxDrawdown = Math.max(metrics.maxDrawdown, drawdown);
    }
  });
  
  metrics.netProfit = metrics.totalProfit - metrics.totalLoss;
  metrics.winRate = trades.length > 0 ? (wins / trades.length) * 100 : 0;
  metrics.avgWin = wins > 0 ? metrics.totalProfit / wins : 0;
  metrics.avgLoss = losses > 0 ? metrics.totalLoss / losses : 0;
  metrics.profitFactor = metrics.totalLoss > 0 ? metrics.totalProfit / metrics.totalLoss : 0;
  
  // Calculate Sharpe ratio (simplified)
  if (trades.length > 1) {
    const returns = trades.map(t => t.profit || 0);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    metrics.sharpeRatio = stdDev > 0 ? avgReturn / stdDev : 0;
  }
  
  return metrics;
}

// Message handler
self.onmessage = function(e) {
  const { type, data, id } = e.data;
  
  try {
    let result;
    
    switch (type) {
      case 'processArbitrage':
        result = processArbitrageOpportunities(data);
        break;
        
      case 'processLatency':
        result = processLatencyMetrics(data);
        break;
        
      case 'analyzeDEXFlows':
        result = analyzeDEXFlows(data);
        break;
        
      case 'calculateProfit':
        result = calculateProfitMetrics(data);
        break;
        
      case 'aggregateStats':
        result = aggregateStatistics(data);
        break;
        
      default:
        throw new Error(`Unknown message type: ${type}`);
    }
    
    self.postMessage({
      id,
      type: 'result',
      data: result
    });
    
  } catch (error) {
    self.postMessage({
      id,
      type: 'error',
      error: error.message
    });
  }
};