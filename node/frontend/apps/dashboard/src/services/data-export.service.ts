import ApiBase from './api.base';
import { ExportConfig, ExportProgress } from '../components/node/DatasetExporter';

export interface ExportStartResponse {
  exportId: string;
  estimatedRows: number;
  estimatedSize: number;
}

export interface ExportProgressResponse extends ExportProgress {
  features?: {
    importance: Array<{
      feature: string;
      importance: number;
      category: string;
    }>;
    correlation: {
      features: string[];
      matrix: number[][];
    };
  };
}

export class DataExportService extends ApiBase {
  /**
   * Start a new data export
   */
  async startExport(config: ExportConfig) {
    return this.post<ExportStartResponse>('/api/mission-control/export/start', {
      timeRange: config.timeRange,
      metrics: config.metrics,
      format: config.format,
      compression: config.compression,
      features: config.features,
      labels: config.labels,
      chunking: config.chunking,
    });
  }

  /**
   * Get export progress
   */
  async getProgress(exportId: string) {
    return this.get<ExportProgressResponse>(`/api/mission-control/export/progress/${exportId}`);
  }

  /**
   * Cancel an export
   */
  async cancelExport(exportId: string) {
    return this.delete(`/api/mission-control/export/cancel/${exportId}`);
  }

  /**
   * Get export preview
   */
  async getPreview(exportId: string, limit: number = 10) {
    return this.get<any[]>(`/api/mission-control/export/preview/${exportId}?limit=${limit}`);
  }

  /**
   * Download exported file
   */
  async downloadExport(exportId: string) {
    return this.get(`/api/mission-control/export/download/${exportId}`, {
      // Return blob for file download
      headers: {
        'Accept': 'application/octet-stream',
      },
    });
  }

  /**
   * Get feature analysis for the dataset
   */
  async getFeatureAnalysis(config: ExportConfig) {
    return this.post<{
      importance: Array<{
        feature: string;
        importance: number;
        category: string;
      }>;
      correlation: {
        features: string[];
        matrix: number[][];
      };
      statistics: {
        mean: Record<string, number>;
        std: Record<string, number>;
        min: Record<string, number>;
        max: Record<string, number>;
      };
    }>('/api/mission-control/export/analyze', config);
  }

  /**
   * Get available export templates
   */
  async getTemplates() {
    return this.get<Array<{
      id: string;
      name: string;
      description: string;
      config: ExportConfig;
    }>>('/api/mission-control/export/templates');
  }

  /**
   * Save export configuration as template
   */
  async saveTemplate(name: string, description: string, config: ExportConfig) {
    return this.post('/api/mission-control/export/templates', {
      name,
      description,
      config,
    });
  }

  /**
   * Get export history
   */
  async getExportHistory(limit: number = 10) {
    return this.get<Array<{
      id: string;
      timestamp: number;
      status: string;
      config: ExportConfig;
      fileSize: number;
      rowCount: number;
      downloadUrl?: string;
    }>>(`/api/mission-control/export/history?limit=${limit}`);
  }

  /**
   * Stream export data using Server-Sent Events
   */
  streamExport(exportId: string, onMessage: (data: any) => void, onError?: (error: any) => void) {
    const eventSource = new EventSource(`${this.baseUrl}/api/mission-control/export/stream/${exportId}`);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse stream data:', error);
        onError?.(error);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('Stream error:', error);
      onError?.(error);
      eventSource.close();
    };
    
    return () => {
      eventSource.close();
    };
  }

  /**
   * Validate export configuration
   */
  async validateConfig(config: ExportConfig) {
    return this.post<{
      valid: boolean;
      errors?: string[];
      warnings?: string[];
      estimatedRows: number;
      estimatedSize: number;
      estimatedTime: number;
    }>('/api/mission-control/export/validate', config);
  }
}