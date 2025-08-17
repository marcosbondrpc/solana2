/// <reference lib="webworker" />

import { ExportConfig } from '../components/node/DatasetExporter';

interface WorkerMessage {
  type: 'startExport' | 'processChunk' | 'generateFeatures' | 'cancel';
  config?: ExportConfig;
  exportId?: string;
  data?: any[];
  chunkIndex?: number;
}

interface WorkerResponse {
  type: 'progress' | 'preview' | 'features' | 'complete' | 'error';
  data: any;
}

class ExportProcessor {
  private config: ExportConfig | null = null;
  private exportId: string | null = null;
  private processedRows = 0;
  private totalRows = 0;
  private startTime = 0;
  private cancelled = false;

  async processExport(config: ExportConfig, exportId: string) {
    this.config = config;
    this.exportId = exportId;
    this.processedRows = 0;
    this.startTime = Date.now();
    this.cancelled = false;

    try {
      // Simulate initial processing
      await this.delay(500);
      
      // Generate feature analysis
      if (config.features.enabled) {
        const features = await this.generateFeatureAnalysis(config);
        self.postMessage({
          type: 'features',
          data: features,
        });
      }

      // Process data in chunks
      const chunkSize = config.chunking.chunkSize || 100000;
      const totalChunks = Math.ceil(this.totalRows / chunkSize);

      for (let i = 0; i < totalChunks; i++) {
        if (this.cancelled) break;

        await this.processChunk(i, chunkSize);
        
        // Send progress update
        const progress = (this.processedRows / this.totalRows) * 100;
        const elapsed = Date.now() - this.startTime;
        const eta = (elapsed / this.processedRows) * (this.totalRows - this.processedRows);
        
        self.postMessage({
          type: 'progress',
          data: {
            progress,
            processedRows: this.processedRows,
            totalRows: this.totalRows,
            eta: Math.round(eta / 1000),
          },
        });

        // Send preview for first chunk
        if (i === 0) {
          const preview = await this.generatePreview();
          self.postMessage({
            type: 'preview',
            data: preview,
          });
        }
      }

      // Complete
      if (!this.cancelled) {
        self.postMessage({
          type: 'complete',
          data: {
            downloadUrl: `/api/mission-control/export/download/${exportId}`,
            totalRows: this.processedRows,
            processingTime: Date.now() - this.startTime,
          },
        });
      }
    } catch (error) {
      self.postMessage({
        type: 'error',
        data: {
          message: error instanceof Error ? error.message : 'Export processing failed',
        },
      });
    }
  }

  private async processChunk(chunkIndex: number, chunkSize: number) {
    // Simulate chunk processing
    await this.delay(1000);
    
    const rowsInChunk = Math.min(chunkSize, this.totalRows - this.processedRows);
    
    // Apply feature engineering
    if (this.config?.features.enabled) {
      await this.applyFeatureEngineering(chunkIndex, rowsInChunk);
    }
    
    // Apply labels
    if (this.config?.labels) {
      await this.applyLabels(chunkIndex, rowsInChunk);
    }
    
    this.processedRows += rowsInChunk;
  }

  private async generateFeatureAnalysis(config: ExportConfig) {
    // Simulate feature analysis
    await this.delay(1500);

    // Generate mock feature importance
    const features = [
      'jito_tip_mean_5m',
      'bundle_success_rate',
      'landing_delay_p95',
      'rpc_latency_p99',
      'block_production_rate',
      'quic_handshake_success',
      'tpu_throttle_events',
      'slot_distance_to_tip',
      'auction_tick_variance',
      'mev_timing_efficiency',
      'sandwich_detection_score',
      'arbitrage_window_size',
      'gas_price_volatility',
      'network_congestion_index',
      'validator_stake_weight',
    ];

    const categories = ['Jito', 'RPC', 'Node Health', 'MEV Features', 'Network'];
    
    const importance = features.map((feature, idx) => ({
      feature,
      importance: Math.random() * 0.8 + 0.2,
      category: categories[idx % categories.length],
    })).sort((a, b) => b.importance - a.importance);

    // Generate correlation matrix
    const correlation = {
      features: features.slice(0, 10),
      matrix: this.generateCorrelationMatrix(10),
    };

    return { importance, correlation };
  }

  private generateCorrelationMatrix(size: number): number[][] {
    const matrix: number[][] = [];
    
    for (let i = 0; i < size; i++) {
      matrix[i] = [];
      for (let j = 0; j < size; j++) {
        if (i === j) {
          matrix[i][j] = 1;
        } else if (j < i) {
          matrix[i][j] = matrix[j][i];
        } else {
          // Generate realistic correlations
          const baseCorr = (Math.random() - 0.5) * 2;
          const dampened = baseCorr * 0.7; // Dampen extreme correlations
          matrix[i][j] = Math.max(-0.9, Math.min(0.9, dampened));
        }
      }
    }
    
    return matrix;
  }

  private async generatePreview() {
    // Generate mock preview data
    const preview = [];
    const metrics = this.getSelectedMetrics();
    
    for (let i = 0; i < 10; i++) {
      const row: any = {
        timestamp: Date.now() - i * 60000,
        slot: 250000000 - i * 100,
      };
      
      // Add selected metrics
      if (metrics.includes('jito_tips')) {
        row.jito_tip = Math.random() * 1000000;
        row.jito_tip_mean_5m = Math.random() * 1000000;
      }
      
      if (metrics.includes('rpc_latencies')) {
        row.rpc_latency_p50 = Math.random() * 100;
        row.rpc_latency_p99 = Math.random() * 500;
      }
      
      if (metrics.includes('bundle_success')) {
        row.bundle_success = Math.random() > 0.7 ? 1 : 0;
        row.bundle_success_rate_5m = Math.random();
      }
      
      if (metrics.includes('mev_success')) {
        row.mev_opportunity = Math.random() > 0.8 ? 1 : 0;
        row.mev_profit = Math.random() * 10000;
      }
      
      preview.push(row);
    }
    
    return preview;
  }

  private getSelectedMetrics(): string[] {
    if (!this.config) return [];
    
    const metrics: string[] = [];
    const { rpc, jito, quicTpu, nodeHealth, mevFeatures } = this.config.metrics;
    
    if (rpc.latencies) metrics.push('rpc_latencies');
    if (rpc.methodCalls) metrics.push('rpc_methods');
    if (jito.tips) metrics.push('jito_tips');
    if (jito.bundleSuccess) metrics.push('bundle_success');
    if (mevFeatures.tipEfficiency) metrics.push('mev_success');
    
    return metrics;
  }

  private async applyFeatureEngineering(chunkIndex: number, rowCount: number) {
    if (!this.config?.features.enabled) return;
    
    // Simulate feature engineering
    await this.delay(rowCount / 1000); // 1ms per row
    
    // Rolling windows
    if (this.config.features.rollingWindows['1m']) {
      // Apply 1-minute rolling window calculations
    }
    
    if (this.config.features.rollingWindows['5m']) {
      // Apply 5-minute rolling window calculations
    }
    
    // Lag features
    if (this.config.features.lagFeatures.enabled) {
      // Apply lag features
    }
    
    // Advanced features
    if (this.config.features.advancedFeatures.fourierTransform) {
      // Apply FFT
    }
    
    if (this.config.features.advancedFeatures.autoregressive) {
      // Apply AR features
    }
  }

  private async applyLabels(chunkIndex: number, rowCount: number) {
    if (!this.config?.labels) return;
    
    // Simulate label application
    await this.delay(rowCount / 2000); // 0.5ms per row
    
    if (this.config.labels.mevSuccess) {
      // Apply MEV success detection
    }
    
    if (this.config.labels.sandwichDetection) {
      // Apply sandwich attack detection
    }
    
    if (this.config.labels.arbitrageWindows) {
      // Apply arbitrage window detection
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  cancel() {
    this.cancelled = true;
  }

  setTotalRows(total: number) {
    this.totalRows = total;
  }
}

// Worker message handler
const processor = new ExportProcessor();

self.addEventListener('message', async (event: MessageEvent<WorkerMessage>) => {
  const { type, config, exportId, data } = event.data;
  
  switch (type) {
    case 'startExport':
      if (config && exportId) {
        // Set estimated total rows based on time range
        const hours = getHoursFromTimeRange(config.timeRange);
        const estimatedRows = hours * 60 * 60 * 10; // ~10 rows per second
        processor.setTotalRows(estimatedRows);
        
        await processor.processExport(config, exportId);
      }
      break;
      
    case 'cancel':
      processor.cancel();
      break;
  }
});

function getHoursFromTimeRange(timeRange: ExportConfig['timeRange']): number {
  switch (timeRange.preset) {
    case 'last_1h': return 1;
    case 'last_6h': return 6;
    case 'last_24h': return 24;
    case 'last_7d': return 168;
    default: return 24;
  }
}

// Prevent TypeScript error for worker context
export {};