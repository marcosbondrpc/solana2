import { createClient } from '@clickhouse/client-web';

// ClickHouse connection configuration
const CLICKHOUSE_URL = process.env.NEXT_PUBLIC_CLICKHOUSE_URL || 'http://45.157.234.184:8123';
const CLICKHOUSE_USER = process.env.NEXT_PUBLIC_CLICKHOUSE_USER || 'default';
const CLICKHOUSE_PASSWORD = process.env.NEXT_PUBLIC_CLICKHOUSE_PASSWORD || '';
const CLICKHOUSE_DATABASE = process.env.NEXT_PUBLIC_CLICKHOUSE_DATABASE || 'mev';

// Create ClickHouse client
export const clickhouse = createClient({
  url: CLICKHOUSE_URL,
  username: CLICKHOUSE_USER,
  password: CLICKHOUSE_PASSWORD,
  database: CLICKHOUSE_DATABASE,
  application: 'mev-dashboard',
  clickhouse_settings: {
    max_result_rows: '10000',
    max_result_bytes: '100000000',
    result_overflow_mode: 'break',
    enable_http_compression: 1,
    max_execution_time: 30,
    send_progress_in_http_headers: 1,
  },
});

// Query interfaces
export interface MEVTransaction {
  timestamp: Date;
  block_slot: number;
  transaction_signature: string;
  transaction_type: 'arbitrage' | 'liquidation' | 'sandwich' | 'jit';
  program_id: string;
  input_amount: number;
  output_amount: number;
  profit_amount: number;
  profit_percentage: number;
  gas_used: number;
  latency_ms: number;
  bundle_landed: boolean;
  decision_dna: string;
  route_path: string[];
  slippage_bps: number;
  priority_fee: number;
}

export interface SystemMetrics {
  timestamp: Date;
  metric_name: string;
  metric_value: number;
  p50: number;
  p95: number;
  p99: number;
  count: number;
}

export interface BundleStats {
  timestamp: Date;
  total_bundles: number;
  landed_bundles: number;
  failed_bundles: number;
  land_rate: number;
  avg_latency_ms: number;
  total_profit: number;
}

// Query builders
export class ClickHouseQueries {
  // Get recent MEV transactions
  static async getRecentTransactions(limit = 100): Promise<MEVTransaction[]> {
    const query = `
      SELECT 
        timestamp,
        block_slot,
        transaction_signature,
        transaction_type,
        program_id,
        input_amount,
        output_amount,
        profit_amount,
        profit_percentage,
        gas_used,
        latency_ms,
        bundle_landed,
        decision_dna,
        route_path,
        slippage_bps,
        priority_fee
      FROM mev_transactions
      WHERE timestamp > now() - INTERVAL 1 HOUR
      ORDER BY timestamp DESC
      LIMIT ${limit}
    `;
    
    const result = await clickhouse.query({
      query,
      format: 'JSONEachRow',
    });
    
    return result.json<MEVTransaction>();
  }
  
  // Get system metrics
  static async getSystemMetrics(period = '5m'): Promise<SystemMetrics[]> {
    const query = `
      SELECT 
        toStartOfInterval(timestamp, INTERVAL ${period}) as timestamp,
        metric_name,
        avg(metric_value) as metric_value,
        quantile(0.5)(metric_value) as p50,
        quantile(0.95)(metric_value) as p95,
        quantile(0.99)(metric_value) as p99,
        count() as count
      FROM system_metrics
      WHERE timestamp > now() - INTERVAL 1 HOUR
      GROUP BY timestamp, metric_name
      ORDER BY timestamp DESC
    `;
    
    const result = await clickhouse.query({
      query,
      format: 'JSONEachRow',
    });
    
    return result.json<SystemMetrics>();
  }
  
  // Get bundle statistics
  static async getBundleStats(interval = '1m'): Promise<BundleStats[]> {
    const query = `
      SELECT 
        toStartOfInterval(timestamp, INTERVAL ${interval}) as timestamp,
        count() as total_bundles,
        countIf(bundle_landed = 1) as landed_bundles,
        countIf(bundle_landed = 0) as failed_bundles,
        avg(toUInt8(bundle_landed)) * 100 as land_rate,
        avg(latency_ms) as avg_latency_ms,
        sum(profit_amount) as total_profit
      FROM mev_transactions
      WHERE timestamp > now() - INTERVAL 1 HOUR
      GROUP BY timestamp
      ORDER BY timestamp DESC
    `;
    
    const result = await clickhouse.query({
      query,
      format: 'JSONEachRow',
    });
    
    return result.json<BundleStats>();
  }
  
  // Get top profitable routes
  static async getTopRoutes(limit = 10): Promise<any[]> {
    const query = `
      SELECT 
        arrayStringConcat(route_path, ' -> ') as route,
        count() as transaction_count,
        sum(profit_amount) as total_profit,
        avg(profit_percentage) as avg_profit_percentage,
        avg(latency_ms) as avg_latency,
        avg(toUInt8(bundle_landed)) * 100 as success_rate
      FROM mev_transactions
      WHERE timestamp > now() - INTERVAL 1 HOUR
        AND profit_amount > 0
      GROUP BY route
      ORDER BY total_profit DESC
      LIMIT ${limit}
    `;
    
    const result = await clickhouse.query({
      query,
      format: 'JSONEachRow',
    });
    
    return result.json();
  }
  
  // Stream real-time transactions using HTTP streaming
  static streamTransactions(onData: (tx: MEVTransaction) => void, onError?: (err: Error) => void) {
    const query = `
      SELECT 
        timestamp,
        block_slot,
        transaction_signature,
        transaction_type,
        program_id,
        input_amount,
        output_amount,
        profit_amount,
        profit_percentage,
        gas_used,
        latency_ms,
        bundle_landed,
        decision_dna,
        route_path,
        slippage_bps,
        priority_fee
      FROM mev_transactions
      WHERE timestamp > now() - INTERVAL 1 SECOND
      ORDER BY timestamp DESC
      SETTINGS stream_like_engine_insert_timeout = 1
    `;
    
    // This would need WebSocket or Server-Sent Events for true streaming
    // For now, we'll poll
    const interval = setInterval(async () => {
      try {
        const transactions = await this.getRecentTransactions(10);
        transactions.forEach(onData);
      } catch (err) {
        if (onError) onError(err as Error);
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }
  
  // Get arbitrage opportunities by DEX
  static async getArbitrageByDEX(): Promise<any[]> {
    const query = `
      SELECT 
        program_id as dex,
        count() as opportunity_count,
        sum(profit_amount) as total_profit,
        avg(profit_amount) as avg_profit,
        max(profit_amount) as max_profit,
        avg(latency_ms) as avg_latency,
        avg(toUInt8(bundle_landed)) * 100 as success_rate
      FROM mev_transactions
      WHERE timestamp > now() - INTERVAL 1 HOUR
        AND transaction_type = 'arbitrage'
        AND profit_amount > 0
      GROUP BY dex
      ORDER BY total_profit DESC
    `;
    
    const result = await clickhouse.query({
      query,
      format: 'JSONEachRow',
    });
    
    return result.json();
  }
  
  // Get latency distribution
  static async getLatencyDistribution(): Promise<any[]> {
    const query = `
      SELECT 
        floor(latency_ms / 5) * 5 as latency_bucket,
        count() as count,
        avg(toUInt8(bundle_landed)) * 100 as success_rate
      FROM mev_transactions
      WHERE timestamp > now() - INTERVAL 1 HOUR
      GROUP BY latency_bucket
      ORDER BY latency_bucket
    `;
    
    const result = await clickhouse.query({
      query,
      format: 'JSONEachRow',
    });
    
    return result.json();
  }
}