/**
 * High-Performance Web Worker for MEV Data Processing
 * Handles arbitrage calculations, profit computations, and pattern detection
 * Optimized for SharedArrayBuffer zero-copy transfers
 */

// MEV calculation types
interface ArbitrageCalculation {
  type: 'arbitrage';
  dexA: {
    name: string;
    reserve0: bigint;
    reserve1: bigint;
    fee: number;
  };
  dexB: {
    name: string;
    reserve0: bigint;
    reserve1: bigint;
    fee: number;
  };
  amountIn: bigint;
  gasPrice: bigint;
  priorityFee: bigint;
}

interface ProfitCalculation {
  type: 'profit';
  transactions: Array<{
    revenue: bigint;
    cost: bigint;
    gasUsed: bigint;
  }>;
  epoch: number;
}

interface PatternDetection {
  type: 'pattern';
  priceHistory: Float64Array;
  volumeHistory: Float64Array;
  windowSize: number;
}

// Shared memory structures for zero-copy communication
let sharedBuffer: SharedArrayBuffer | null = null;
let sharedView: DataView | null = null;
let resultBuffer: SharedArrayBuffer | null = null;
let resultView: DataView | null = null;

// Constants for DEX calculations
const BASIS_POINTS = 10000n;
const LAMPORTS_PER_SOL = 1_000_000_000n;

// Ring buffer for streaming calculations
class RingBuffer {
  private buffer: Float64Array;
  private head = 0;
  private tail = 0;
  private size: number;

  constructor(size: number) {
    this.size = size;
    this.buffer = new Float64Array(size);
  }

  push(value: number): void {
    this.buffer[this.tail] = value;
    this.tail = (this.tail + 1) % this.size;
    if (this.tail === this.head) {
      this.head = (this.head + 1) % this.size;
    }
  }

  getArray(): Float64Array {
    const result = new Float64Array(this.size);
    let idx = 0;
    let current = this.head;
    
    while (current !== this.tail) {
      result[idx++] = this.buffer[current];
      current = (current + 1) % this.size;
    }
    
    return result.slice(0, idx);
  }

  clear(): void {
    this.head = 0;
    this.tail = 0;
  }
}

// Statistical calculations
class Statistics {
  static mean(data: Float64Array): number {
    if (data.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      sum += data[i];
    }
    return sum / data.length;
  }

  static stdDev(data: Float64Array): number {
    const avg = this.mean(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      sum += Math.pow(data[i] - avg, 2);
    }
    return Math.sqrt(sum / data.length);
  }

  static percentile(data: Float64Array, p: number): number {
    const sorted = Array.from(data).sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  static movingAverage(data: Float64Array, window: number): Float64Array {
    const result = new Float64Array(data.length);
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - window + 1);
      let sum = 0;
      for (let j = start; j <= i; j++) {
        sum += data[j];
      }
      result[i] = sum / (i - start + 1);
    }
    return result;
  }

  static exponentialMovingAverage(data: Float64Array, alpha: number): Float64Array {
    const result = new Float64Array(data.length);
    result[0] = data[0];
    for (let i = 1; i < data.length; i++) {
      result[i] = alpha * data[i] + (1 - alpha) * result[i - 1];
    }
    return result;
  }
}

// Arbitrage detection algorithms
class ArbitrageDetector {
  // Calculate output amount for constant product AMM (Uniswap V2 style)
  static getAmountOut(
    amountIn: bigint,
    reserveIn: bigint,
    reserveOut: bigint,
    fee: number // in basis points
  ): bigint {
    const amountInWithFee = amountIn * (BASIS_POINTS - BigInt(fee));
    const numerator = amountInWithFee * reserveOut;
    const denominator = reserveIn * BASIS_POINTS + amountInWithFee;
    return numerator / denominator;
  }

  // Calculate required input for desired output
  static getAmountIn(
    amountOut: bigint,
    reserveIn: bigint,
    reserveOut: bigint,
    fee: number
  ): bigint {
    const numerator = reserveIn * amountOut * BASIS_POINTS;
    const denominator = (reserveOut - amountOut) * (BASIS_POINTS - BigInt(fee));
    return numerator / denominator + 1n;
  }

  // Find optimal arbitrage amount using binary search
  static findOptimalAmount(
    reserveA0: bigint,
    reserveA1: bigint,
    reserveB0: bigint,
    reserveB1: bigint,
    feeA: number,
    feeB: number,
    maxAmount: bigint
  ): { amount: bigint; profit: bigint } {
    let left = 0n;
    let right = maxAmount;
    let bestAmount = 0n;
    let bestProfit = 0n;

    while (right - left > 1n) {
      const mid = (left + right) / 2n;
      
      // Path A -> B
      const amountOut1 = this.getAmountOut(mid, reserveA0, reserveA1, feeA);
      const amountOut2 = this.getAmountOut(amountOut1, reserveB1, reserveB0, feeB);
      const profit = amountOut2 > mid ? amountOut2 - mid : 0n;

      if (profit > bestProfit) {
        bestProfit = profit;
        bestAmount = mid;
      }

      // Check derivative to determine search direction
      const testAmount = mid + 1n;
      const testOut1 = this.getAmountOut(testAmount, reserveA0, reserveA1, feeA);
      const testOut2 = this.getAmountOut(testOut1, reserveB1, reserveB0, feeB);
      const testProfit = testOut2 > testAmount ? testOut2 - testAmount : 0n;

      if (testProfit > profit) {
        left = mid;
      } else {
        right = mid;
      }
    }

    return { amount: bestAmount, profit: bestProfit };
  }

  // Calculate multi-hop arbitrage
  static calculateMultiHop(
    path: Array<{
      reserveIn: bigint;
      reserveOut: bigint;
      fee: number;
    }>,
    amountIn: bigint
  ): bigint {
    let currentAmount = amountIn;
    
    for (const hop of path) {
      currentAmount = this.getAmountOut(
        currentAmount,
        hop.reserveIn,
        hop.reserveOut,
        hop.fee
      );
    }
    
    return currentAmount;
  }
}

// Pattern detection algorithms
class PatternDetector {
  // Detect triangle patterns
  static detectTriangle(prices: Float64Array, window: number): boolean {
    if (prices.length < window) return false;
    
    const recent = prices.slice(-window);
    const high = Math.max(...recent);
    const low = Math.min(...recent);
    const range = high - low;
    
    // Check for converging triangle
    let isConverging = true;
    for (let i = 1; i < recent.length - 1; i++) {
      const localRange = Math.abs(recent[i + 1] - recent[i - 1]);
      if (localRange > range * 0.8) {
        isConverging = false;
        break;
      }
    }
    
    return isConverging;
  }

  // Detect support and resistance levels
  static detectLevels(prices: Float64Array, sensitivity: number = 0.02): number[] {
    const levels: number[] = [];
    const sorted = Array.from(prices).sort((a, b) => a - b);
    
    let lastLevel = sorted[0];
    let count = 1;
    
    for (let i = 1; i < sorted.length; i++) {
      if (Math.abs(sorted[i] - lastLevel) / lastLevel < sensitivity) {
        count++;
      } else {
        if (count >= 3) {
          levels.push(lastLevel);
        }
        lastLevel = sorted[i];
        count = 1;
      }
    }
    
    if (count >= 3) {
      levels.push(lastLevel);
    }
    
    return levels;
  }

  // Detect momentum shifts
  static detectMomentumShift(
    prices: Float64Array,
    volumes: Float64Array,
    window: number = 20
  ): { bullish: boolean; strength: number } {
    const priceMA = Statistics.movingAverage(prices, window);
    const volumeMA = Statistics.movingAverage(volumes, window);
    
    const recentPriceSlope = this.calculateSlope(priceMA.slice(-10));
    const recentVolumeSlope = this.calculateSlope(volumeMA.slice(-10));
    
    const bullish = recentPriceSlope > 0 && recentVolumeSlope > 0;
    const strength = Math.abs(recentPriceSlope) * (1 + Math.abs(recentVolumeSlope));
    
    return { bullish, strength };
  }

  private static calculateSlope(data: Float64Array): number {
    const n = data.length;
    if (n < 2) return 0;
    
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += data[i];
      sumXY += i * data[i];
      sumX2 += i * i;
    }
    
    return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  }
}

// Main message handler
self.onmessage = async (event: MessageEvent) => {
  const { type, data, sharedBuffer: buffer } = event.data;

  try {
    switch (type) {
      case 'init':
        if (buffer instanceof SharedArrayBuffer) {
          sharedBuffer = buffer;
          sharedView = new DataView(sharedBuffer);
          
          // Allocate result buffer
          resultBuffer = new SharedArrayBuffer(1024 * 1024); // 1MB for results
          resultView = new DataView(resultBuffer);
          
          self.postMessage({
            type: 'initialized',
            resultBuffer
          });
        }
        break;

      case 'calculate':
        const result = await processCalculation(data);
        self.postMessage({
          type: 'result',
          data: result
        });
        break;

      case 'batch':
        const results = await processBatch(data);
        self.postMessage({
          type: 'batchResult',
          data: results
        });
        break;

      case 'stream':
        await processStream(data);
        break;

      default:
        self.postMessage({
          type: 'error',
          error: `Unknown message type: ${type}`
        });
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: (error as Error).message
    });
  }
};

// Process single calculation
async function processCalculation(data: any): Promise<any> {
  const startTime = performance.now();
  
  switch (data.type) {
    case 'arbitrage':
      return calculateArbitrage(data as ArbitrageCalculation);
    
    case 'profit':
      return calculateProfit(data as ProfitCalculation);
    
    case 'pattern':
      return detectPatterns(data as PatternDetection);
    
    default:
      throw new Error(`Unknown calculation type: ${data.type}`);
  }
}

// Process batch of calculations
async function processBatch(calculations: any[]): Promise<any[]> {
  const results: any[] = [];
  
  for (const calc of calculations) {
    results.push(await processCalculation(calc));
  }
  
  return results;
}

// Process streaming data
async function processStream(config: any): Promise<void> {
  // Implementation for streaming calculations
  // Would use SharedArrayBuffer for real-time updates
}

// Calculate arbitrage opportunity
function calculateArbitrage(data: ArbitrageCalculation): any {
  const { dexA, dexB, amountIn, gasPrice, priorityFee } = data;
  
  // Find optimal arbitrage amount
  const { amount, profit } = ArbitrageDetector.findOptimalAmount(
    dexA.reserve0,
    dexA.reserve1,
    dexB.reserve0,
    dexB.reserve1,
    dexA.fee,
    dexB.fee,
    amountIn
  );
  
  // Calculate gas cost
  const estimatedGas = 200000n; // Typical arbitrage gas usage
  const totalGasPrice = gasPrice + priorityFee;
  const gasCost = estimatedGas * totalGasPrice;
  
  // Calculate net profit
  const netProfit = profit > gasCost ? profit - gasCost : 0n;
  
  // Calculate ROI
  const roi = amount > 0n ? Number(netProfit * 10000n / amount) / 100 : 0;
  
  return {
    optimalAmount: amount.toString(),
    grossProfit: profit.toString(),
    gasCost: gasCost.toString(),
    netProfit: netProfit.toString(),
    roi,
    profitable: netProfit > 0n,
    confidence: calculateConfidence(profit, gasCost)
  };
}

// Calculate profit metrics
function calculateProfit(data: ProfitCalculation): any {
  const { transactions, epoch } = data;
  
  let totalRevenue = 0n;
  let totalCost = 0n;
  let totalGas = 0n;
  
  for (const tx of transactions) {
    totalRevenue += tx.revenue;
    totalCost += tx.cost;
    totalGas += tx.gasUsed;
  }
  
  const netProfit = totalRevenue - totalCost;
  const avgProfit = transactions.length > 0 ? netProfit / BigInt(transactions.length) : 0n;
  
  // Calculate percentiles
  const profits = transactions.map(tx => Number(tx.revenue - tx.cost));
  const profitArray = new Float64Array(profits);
  
  return {
    epoch,
    totalRevenue: totalRevenue.toString(),
    totalCost: totalCost.toString(),
    netProfit: netProfit.toString(),
    avgProfit: avgProfit.toString(),
    totalGas: totalGas.toString(),
    transactionCount: transactions.length,
    metrics: {
      mean: Statistics.mean(profitArray),
      stdDev: Statistics.stdDev(profitArray),
      p50: Statistics.percentile(profitArray, 50),
      p95: Statistics.percentile(profitArray, 95),
      p99: Statistics.percentile(profitArray, 99)
    }
  };
}

// Detect patterns in price/volume data
function detectPatterns(data: PatternDetection): any {
  const { priceHistory, volumeHistory, windowSize } = data;
  
  // Calculate moving averages
  const priceMA = Statistics.movingAverage(priceHistory, windowSize);
  const priceEMA = Statistics.exponentialMovingAverage(priceHistory, 0.2);
  const volumeMA = Statistics.movingAverage(volumeHistory, windowSize);
  
  // Detect patterns
  const triangle = PatternDetector.detectTriangle(priceHistory, windowSize);
  const levels = PatternDetector.detectLevels(priceHistory);
  const momentum = PatternDetector.detectMomentumShift(priceHistory, volumeHistory, windowSize);
  
  // Calculate volatility
  const returns = new Float64Array(priceHistory.length - 1);
  for (let i = 1; i < priceHistory.length; i++) {
    returns[i - 1] = (priceHistory[i] - priceHistory[i - 1]) / priceHistory[i - 1];
  }
  const volatility = Statistics.stdDev(returns);
  
  return {
    patterns: {
      triangle,
      supportLevels: levels.filter(l => l < priceHistory[priceHistory.length - 1]),
      resistanceLevels: levels.filter(l => l > priceHistory[priceHistory.length - 1]),
      momentum
    },
    indicators: {
      priceMA: Array.from(priceMA.slice(-20)),
      priceEMA: Array.from(priceEMA.slice(-20)),
      volumeMA: Array.from(volumeMA.slice(-20)),
      volatility,
      trend: priceMA[priceMA.length - 1] > priceMA[priceMA.length - 20] ? 'up' : 'down'
    }
  };
}

// Calculate confidence score
function calculateConfidence(profit: bigint, gasCost: bigint): number {
  if (profit <= gasCost) return 0;
  
  const ratio = Number(profit * 100n / gasCost) / 100;
  
  if (ratio > 3) return 0.95;
  if (ratio > 2) return 0.85;
  if (ratio > 1.5) return 0.75;
  if (ratio > 1.2) return 0.65;
  return 0.5;
}

// Export for TypeScript
export {};