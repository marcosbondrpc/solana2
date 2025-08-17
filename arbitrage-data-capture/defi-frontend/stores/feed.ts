/**
 * Ultra-optimized real-time data store
 * Uses Valtio for fine-grained reactivity and minimal re-renders
 * Implements ring buffers, time-based windowing, and intelligent batching
 */

import { proxy, snapshot, subscribe, ref } from 'valtio';
import { subscribeKey } from 'valtio/utils';
import { realtimeClient } from '../lib/ws';
import { wtClient } from '../lib/wt';
import { MessageTypes } from '../lib/ws-proto';

// Store configuration
const CONFIG = {
  maxMevOpportunities: 1000,
  maxArbitrageOpportunities: 1000,
  maxBundleOutcomes: 500,
  maxMarketTicks: 10000,
  maxMetricsHistory: 100,
  aggregationWindow: 1000, // 1 second windows for aggregation
  tickDebounce: 50,        // Debounce market ticks
  gcInterval: 30000,       // Garbage collection every 30s
};

// Type definitions
export interface MevOpportunity {
  tx_hash: string;
  block_hash: string;
  slot: bigint;
  profit_lamports: number;
  probability: number;
  opportunity_type: string;
  target_accounts: string[];
  gas_estimate: bigint;
  priority_fee: bigint;
  raw_transaction: Uint8Array;
  metrics: Map<string, number>;
  timestamp: number;
  _id: string;
}

export interface ArbitrageOpportunity {
  id: string;
  slot: bigint;
  dex_markets: string[];
  profit_estimate: number;
  execution_probability: number;
  gas_cost: bigint;
  route: RouteStep[];
  risk_metrics: Map<string, number>;
  deadline_ns: bigint;
  timestamp: number;
}

export interface RouteStep {
  dex: string;
  pool_address: string;
  token_in: string;
  token_out: string;
  amount_in: bigint;
  amount_out: bigint;
  slippage: number;
}

export interface BundleOutcome {
  bundle_id: string;
  slot: bigint;
  landed: boolean;
  profit_actual: number;
  gas_used: bigint;
  error?: string;
  latency_ms: bigint;
  metadata: Map<string, string>;
  timestamp: number;
}

export interface MarketTick {
  market_id: string;
  timestamp_ns: bigint;
  bid_price: number;
  ask_price: number;
  bid_size: number;
  ask_size: number;
  last_price: number;
  volume_24h: bigint;
  additional_data: Map<string, number>;
}

export interface AggregatedMetrics {
  timestamp: number;
  mevCount: number;
  arbCount: number;
  bundleCount: number;
  successRate: number;
  avgProfit: number;
  totalVolume: number;
  gasUsed: number;
  latencyP50: number;
  latencyP95: number;
  latencyP99: number;
}

// Create reactive store
export const feedStore = proxy({
  // Connection state
  connection: {
    status: 'disconnected' as 'connected' | 'connecting' | 'disconnected',
    transport: 'websocket' as 'websocket' | 'webtransport',
    stats: {
      messagesReceived: 0,
      messagesDropped: 0,
      bytesReceived: 0,
      avgDecodeTime: 0,
      avgBatchSize: 0,
      connectionUptime: 0,
      reconnectCount: 0,
    }
  },
  
  // MEV opportunities (ring buffer)
  mevOpportunities: [] as MevOpportunity[],
  mevHead: 0,
  mevTail: 0,
  
  // Arbitrage opportunities (ring buffer)
  arbOpportunities: [] as ArbitrageOpportunity[],
  arbHead: 0,
  arbTail: 0,
  
  // Bundle outcomes (ring buffer)
  bundleOutcomes: [] as BundleOutcome[],
  bundleHead: 0,
  bundleTail: 0,
  
  // Market ticks (Map for O(1) lookups)
  marketTicks: new Map() as Map<string, MarketTick>,
  
  // Aggregated metrics (time series)
  aggregatedMetrics: [] as AggregatedMetrics[],
  
  // Current metrics
  currentMetrics: {
    mevRate: 0,         // MEV opportunities per second
    arbRate: 0,         // Arb opportunities per second
    bundleRate: 0,      // Bundles per second
    successRate: 0,     // Bundle success rate
    avgLatency: 0,      // Average bundle latency
    totalProfit: 0,     // Total profit in current window
    gasPrice: 0,        // Current gas price estimate
  },
  
  // Filters and settings
  filters: {
    minProfit: 0,
    minProbability: 0.5,
    opportunityTypes: new Set<string>(),
    dexMarkets: new Set<string>(),
    showOnlySuccessful: false,
  },
  
  // Performance tracking
  performance: {
    lastUpdate: 0,
    updateCount: 0,
    droppedUpdates: 0,
    gcRuns: 0,
  }
});

// Use refs for non-reactive data that changes frequently
const tickBuffer = ref(new Map<string, { tick: MarketTick, timer: number }>());
const metricsBuffer = ref<any[]>([]);
const updateQueue = ref<any[]>([]);

/**
 * Initialize the store and connect to real-time feeds
 */
export async function initializeFeed(useWebTransport: boolean = false): Promise<void> {
  // Initialize ring buffers
  feedStore.mevOpportunities = new Array(CONFIG.maxMevOpportunities);
  feedStore.arbOpportunities = new Array(CONFIG.maxArbitrageOpportunities);
  feedStore.bundleOutcomes = new Array(CONFIG.maxBundleOutcomes);
  
  // Choose transport
  const client = useWebTransport ? wtClient : realtimeClient;
  
  // Update connection status
  feedStore.connection.status = 'connecting';
  feedStore.connection.transport = useWebTransport ? 'webtransport' : 'websocket';
  
  // Subscribe to connection events
  client.on('connected', () => {
    feedStore.connection.status = 'connected';
    console.log('[Feed] Connected to real-time feed');
  });
  
  client.on('disconnected', () => {
    feedStore.connection.status = 'disconnected';
  });
  
  client.on('stats', (stats) => {
    Object.assign(feedStore.connection.stats, stats);
  });
  
  // Subscribe to data events
  client.on('mev_opportunity', handleMevOpportunity);
  client.on('arbitrage_opportunity', handleArbitrageOpportunity);
  client.on('bundle_outcome', handleBundleOutcome);
  client.on('market_tick', handleMarketTick);
  client.on('metrics_update', handleMetricsUpdate);
  
  // Connect
  await client.connect();
  
  // Start aggregation timer
  startAggregation();
  
  // Start garbage collection
  startGarbageCollection();
}

/**
 * Handle MEV opportunity
 */
function handleMevOpportunity(data: any): void {
  // Apply filters
  if (data.profit_lamports < feedStore.filters.minProfit) return;
  if (data.probability < feedStore.filters.minProbability) return;
  if (feedStore.filters.opportunityTypes.size > 0 && 
      !feedStore.filters.opportunityTypes.has(data.opportunity_type)) return;
  
  // Create opportunity object
  const opportunity: MevOpportunity = {
    ...data,
    timestamp: Date.now(),
    _id: `mev-${data.tx_hash}-${Date.now()}`
  };
  
  // Add to ring buffer
  const nextIndex = (feedStore.mevHead + 1) % CONFIG.maxMevOpportunities;
  
  // Check for overflow
  if (nextIndex === feedStore.mevTail) {
    // Move tail forward (drop oldest)
    feedStore.mevTail = (feedStore.mevTail + 1) % CONFIG.maxMevOpportunities;
    feedStore.performance.droppedUpdates++;
  }
  
  // Add to buffer
  feedStore.mevOpportunities[feedStore.mevHead] = opportunity;
  feedStore.mevHead = nextIndex;
  
  // Update metrics
  feedStore.currentMetrics.mevRate++;
  feedStore.performance.updateCount++;
  feedStore.performance.lastUpdate = Date.now();
}

/**
 * Handle arbitrage opportunity
 */
function handleArbitrageOpportunity(data: any): void {
  // Apply filters
  if (data.profit_estimate < feedStore.filters.minProfit) return;
  if (data.execution_probability < feedStore.filters.minProbability) return;
  if (feedStore.filters.dexMarkets.size > 0) {
    const hasMatchingDex = data.dex_markets.some((dex: string) => 
      feedStore.filters.dexMarkets.has(dex)
    );
    if (!hasMatchingDex) return;
  }
  
  // Create opportunity object
  const opportunity: ArbitrageOpportunity = {
    ...data,
    timestamp: Date.now()
  };
  
  // Add to ring buffer
  const nextIndex = (feedStore.arbHead + 1) % CONFIG.maxArbitrageOpportunities;
  
  if (nextIndex === feedStore.arbTail) {
    feedStore.arbTail = (feedStore.arbTail + 1) % CONFIG.maxArbitrageOpportunities;
    feedStore.performance.droppedUpdates++;
  }
  
  feedStore.arbOpportunities[feedStore.arbHead] = opportunity;
  feedStore.arbHead = nextIndex;
  
  // Update metrics
  feedStore.currentMetrics.arbRate++;
  feedStore.performance.updateCount++;
  feedStore.performance.lastUpdate = Date.now();
}

/**
 * Handle bundle outcome
 */
function handleBundleOutcome(data: any): void {
  // Apply filters
  if (feedStore.filters.showOnlySuccessful && !data.landed) return;
  
  // Create outcome object
  const outcome: BundleOutcome = {
    ...data,
    timestamp: Date.now()
  };
  
  // Add to ring buffer
  const nextIndex = (feedStore.bundleHead + 1) % CONFIG.maxBundleOutcomes;
  
  if (nextIndex === feedStore.bundleTail) {
    feedStore.bundleTail = (feedStore.bundleTail + 1) % CONFIG.maxBundleOutcomes;
    feedStore.performance.droppedUpdates++;
  }
  
  feedStore.bundleOutcomes[feedStore.bundleHead] = outcome;
  feedStore.bundleHead = nextIndex;
  
  // Update metrics
  feedStore.currentMetrics.bundleRate++;
  if (data.landed) {
    feedStore.currentMetrics.totalProfit += data.profit_actual;
  }
  feedStore.currentMetrics.avgLatency = 
    (feedStore.currentMetrics.avgLatency * 0.9) + (Number(data.latency_ms) * 0.1);
  
  feedStore.performance.updateCount++;
  feedStore.performance.lastUpdate = Date.now();
}

/**
 * Handle market tick with debouncing
 */
function handleMarketTick(data: any): void {
  const marketId = data.market_id;
  
  // Check if we have a pending update for this market
  const pending = tickBuffer.value.get(marketId);
  if (pending) {
    // Clear existing timer
    clearTimeout(pending.timer);
  }
  
  // Set new timer
  const timer = window.setTimeout(() => {
    // Apply update
    feedStore.marketTicks.set(marketId, {
      ...data,
      timestamp_ns: BigInt(data.timestamp_ns)
    });
    
    // Remove from buffer
    tickBuffer.value.delete(marketId);
    
    feedStore.performance.updateCount++;
    feedStore.performance.lastUpdate = Date.now();
  }, CONFIG.tickDebounce);
  
  // Store in buffer
  tickBuffer.value.set(marketId, { tick: data, timer });
}

/**
 * Handle metrics update
 */
function handleMetricsUpdate(data: any): void {
  // Buffer metrics for aggregation
  metricsBuffer.value.push(data);
  
  feedStore.performance.updateCount++;
  feedStore.performance.lastUpdate = Date.now();
}

/**
 * Start aggregation timer
 */
function startAggregation(): void {
  setInterval(() => {
    // Calculate aggregated metrics
    const now = Date.now();
    const successfulBundles = getBundlesInWindow(CONFIG.aggregationWindow)
      .filter(b => b?.landed).length;
    const totalBundles = getBundlesInWindow(CONFIG.aggregationWindow).length;
    
    const metrics: AggregatedMetrics = {
      timestamp: now,
      mevCount: feedStore.currentMetrics.mevRate,
      arbCount: feedStore.currentMetrics.arbRate,
      bundleCount: feedStore.currentMetrics.bundleRate,
      successRate: totalBundles > 0 ? successfulBundles / totalBundles : 0,
      avgProfit: feedStore.currentMetrics.totalProfit / Math.max(1, successfulBundles),
      totalVolume: calculateTotalVolume(),
      gasUsed: calculateGasUsed(),
      latencyP50: calculateLatencyPercentile(0.5),
      latencyP95: calculateLatencyPercentile(0.95),
      latencyP99: calculateLatencyPercentile(0.99),
    };
    
    // Add to history
    feedStore.aggregatedMetrics.push(metrics);
    if (feedStore.aggregatedMetrics.length > CONFIG.maxMetricsHistory) {
      feedStore.aggregatedMetrics.shift();
    }
    
    // Reset counters
    feedStore.currentMetrics.mevRate = 0;
    feedStore.currentMetrics.arbRate = 0;
    feedStore.currentMetrics.bundleRate = 0;
    feedStore.currentMetrics.totalProfit = 0;
    
    // Process buffered metrics
    metricsBuffer.value = [];
    
  }, CONFIG.aggregationWindow);
}

/**
 * Start garbage collection
 */
function startGarbageCollection(): void {
  setInterval(() => {
    // Clean up old market ticks
    const cutoff = Date.now() - 60000; // Keep last minute
    for (const [marketId, tick] of feedStore.marketTicks.entries()) {
      if (Number(tick.timestamp_ns) / 1000000 < cutoff) {
        feedStore.marketTicks.delete(marketId);
      }
    }
    
    feedStore.performance.gcRuns++;
  }, CONFIG.gcInterval);
}

/**
 * Get MEV opportunities as array
 */
export function getMevOpportunities(): MevOpportunity[] {
  const result: MevOpportunity[] = [];
  let current = feedStore.mevTail;
  
  while (current !== feedStore.mevHead) {
    const opp = feedStore.mevOpportunities[current];
    if (opp) result.push(opp);
    current = (current + 1) % CONFIG.maxMevOpportunities;
  }
  
  return result;
}

/**
 * Get arbitrage opportunities as array
 */
export function getArbOpportunities(): ArbitrageOpportunity[] {
  const result: ArbitrageOpportunity[] = [];
  let current = feedStore.arbTail;
  
  while (current !== feedStore.arbHead) {
    const opp = feedStore.arbOpportunities[current];
    if (opp) result.push(opp);
    current = (current + 1) % CONFIG.maxArbitrageOpportunities;
  }
  
  return result;
}

/**
 * Get bundle outcomes as array
 */
export function getBundleOutcomes(): BundleOutcome[] {
  const result: BundleOutcome[] = [];
  let current = feedStore.bundleTail;
  
  while (current !== feedStore.bundleHead) {
    const outcome = feedStore.bundleOutcomes[current];
    if (outcome) result.push(outcome);
    current = (current + 1) % CONFIG.maxBundleOutcomes;
  }
  
  return result;
}

/**
 * Get bundles in time window
 */
function getBundlesInWindow(windowMs: number): BundleOutcome[] {
  const cutoff = Date.now() - windowMs;
  return getBundleOutcomes().filter(b => b.timestamp > cutoff);
}

/**
 * Calculate total volume
 */
function calculateTotalVolume(): number {
  let volume = 0;
  for (const tick of feedStore.marketTicks.values()) {
    volume += Number(tick.volume_24h);
  }
  return volume;
}

/**
 * Calculate gas used
 */
function calculateGasUsed(): number {
  return getBundlesInWindow(CONFIG.aggregationWindow)
    .reduce((sum, b) => sum + Number(b?.gas_used || 0), 0);
}

/**
 * Calculate latency percentile
 */
function calculateLatencyPercentile(percentile: number): number {
  const latencies = getBundlesInWindow(CONFIG.aggregationWindow)
    .map(b => Number(b?.latency_ms || 0))
    .filter(l => l > 0)
    .sort((a, b) => a - b);
  
  if (latencies.length === 0) return 0;
  
  const index = Math.floor(latencies.length * percentile);
  return latencies[index];
}

/**
 * Subscribe to specific opportunity type
 */
export function subscribeToMev(callback: (opportunities: MevOpportunity[]) => void): () => void {
  return subscribe(feedStore, () => {
    callback(getMevOpportunities());
  });
}

/**
 * Subscribe to specific metric
 */
export function subscribeToMetric(
  key: keyof typeof feedStore.currentMetrics,
  callback: (value: number) => void
): () => void {
  return subscribeKey(feedStore.currentMetrics, key, callback);
}

/**
 * Update filters
 */
export function updateFilters(filters: Partial<typeof feedStore.filters>): void {
  Object.assign(feedStore.filters, filters);
}

/**
 * Get snapshot of current state
 */
export function getSnapshot() {
  return snapshot(feedStore);
}

/**
 * Clear all data
 */
export function clearFeed(): void {
  feedStore.mevHead = 0;
  feedStore.mevTail = 0;
  feedStore.arbHead = 0;
  feedStore.arbTail = 0;
  feedStore.bundleHead = 0;
  feedStore.bundleTail = 0;
  feedStore.marketTicks.clear();
  feedStore.aggregatedMetrics = [];
  
  // Reset metrics
  Object.keys(feedStore.currentMetrics).forEach(key => {
    (feedStore.currentMetrics as any)[key] = 0;
  });
}