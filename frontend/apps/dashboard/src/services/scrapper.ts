/**
 * Scrapper Service
 * Handles all data collection, ML training, and dataset management API calls
 */

import ApiBase from './api.base';
import WebSocketBase from './websocket.base';
import { API_CONFIG } from '../config/api.config';
import {
  ApiResponse,
  Dataset,
  DatasetFilters,
  CollectRequest,
  CollectOptions,
  MLModel,
  TrainRequest,
  ExportRequest,
  ExportFormat,
  CompressionType,
  PaginatedResponse,
  PaginationParams,
  ScrapperProgressUpdate,
  ModelTrainingUpdate,
} from '../types/api.types';

export interface ImportRequest {
  file: File;
  name: string;
  type: string;
  overwrite?: boolean;
}

export interface DatasetStats {
  totalDatasets: number;
  totalRecords: number;
  totalSize: number;
  activeCollections: number;
  completedToday: number;
  averageCollectionTime: number;
}

export interface ModelStats {
  totalModels: number;
  trainedModels: number;
  deployedModels: number;
  averageAccuracy: number;
  bestModel: {
    id: string;
    name: string;
    accuracy: number;
  };
}

export interface ScrapperConfig {
  maxConcurrentCollections: number;
  defaultBatchSize: number;
  retentionDays: number;
  autoExportEnabled: boolean;
  compressionEnabled: boolean;
  defaultCompression: CompressionType;
}

class ScrapperService extends ApiBase {
  private progressWS: ScrapperProgressWebSocket | null = null;
  private datasetsCache: Map<string, Dataset> = new Map();
  private modelsCache: Map<string, MLModel> = new Map();

  /**
   * Get all datasets with pagination
   */
  async getDatasets(params?: PaginationParams): Promise<ApiResponse<PaginatedResponse<Dataset>>> {
    const queryParams = new URLSearchParams();
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          queryParams.append(key, String(value));
        }
      });
    }
    
    const endpoint = `${API_CONFIG.endpoints.scrapper.datasets}${
      queryParams.toString() ? `?${queryParams.toString()}` : ''
    }`;
    
    const response = await this.get<PaginatedResponse<Dataset>>(endpoint);
    
    // Update cache
    if (response.success && response.data) {
      response.data.items.forEach(dataset => {
        this.datasetsCache.set(dataset.id, dataset);
      });
    }
    
    return response;
  }

  /**
   * Get a specific dataset by ID
   */
  async getDataset(id: string): Promise<ApiResponse<Dataset>> {
    // Check cache first
    if (this.datasetsCache.has(id)) {
      return {
        success: true,
        data: this.datasetsCache.get(id)!,
        timestamp: Date.now(),
      };
    }
    
    const response = await this.get<Dataset>(`${API_CONFIG.endpoints.scrapper.datasets}/${id}`);
    
    if (response.success && response.data) {
      this.datasetsCache.set(id, response.data);
    }
    
    return response;
  }

  /**
   * Create a new dataset
   */
  async createDataset(name: string, type: string, filters?: DatasetFilters): Promise<ApiResponse<Dataset>> {
    const response = await this.post<Dataset>(API_CONFIG.endpoints.scrapper.datasets, {
      name,
      type,
      filters,
    });
    
    if (response.success && response.data) {
      this.datasetsCache.set(response.data.id, response.data);
    }
    
    return response;
  }

  /**
   * Delete a dataset
   */
  async deleteDataset(id: string): Promise<ApiResponse<void>> {
    const response = await this.delete(`${API_CONFIG.endpoints.scrapper.datasets}/${id}`);
    
    if (response.success) {
      this.datasetsCache.delete(id);
    }
    
    return response;
  }

  /**
   * Start data collection
   */
  async startCollection(request: CollectRequest): Promise<ApiResponse<Dataset>> {
    return this.post<Dataset>(API_CONFIG.endpoints.scrapper.collect, request);
  }

  /**
   * Stop data collection
   */
  async stopCollection(datasetId: string): Promise<ApiResponse<void>> {
    return this.post(`${API_CONFIG.endpoints.scrapper.collect}/${datasetId}/stop`);
  }

  /**
   * Pause data collection
   */
  async pauseCollection(datasetId: string): Promise<ApiResponse<void>> {
    return this.post(`${API_CONFIG.endpoints.scrapper.collect}/${datasetId}/pause`);
  }

  /**
   * Resume data collection
   */
  async resumeCollection(datasetId: string): Promise<ApiResponse<void>> {
    return this.post(`${API_CONFIG.endpoints.scrapper.collect}/${datasetId}/resume`);
  }

  /**
   * Get all ML models
   */
  async getModels(params?: PaginationParams): Promise<ApiResponse<PaginatedResponse<MLModel>>> {
    const queryParams = new URLSearchParams();
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          queryParams.append(key, String(value));
        }
      });
    }
    
    const endpoint = `${API_CONFIG.endpoints.scrapper.models}${
      queryParams.toString() ? `?${queryParams.toString()}` : ''
    }`;
    
    const response = await this.get<PaginatedResponse<MLModel>>(endpoint);
    
    // Update cache
    if (response.success && response.data) {
      response.data.items.forEach(model => {
        this.modelsCache.set(model.id, model);
      });
    }
    
    return response;
  }

  /**
   * Get a specific model by ID
   */
  async getModel(id: string): Promise<ApiResponse<MLModel>> {
    // Check cache first
    if (this.modelsCache.has(id)) {
      return {
        success: true,
        data: this.modelsCache.get(id)!,
        timestamp: Date.now(),
      };
    }
    
    const response = await this.get<MLModel>(`${API_CONFIG.endpoints.scrapper.models}/${id}`);
    
    if (response.success && response.data) {
      this.modelsCache.set(id, response.data);
    }
    
    return response;
  }

  /**
   * Start model training
   */
  async trainModel(request: TrainRequest): Promise<ApiResponse<MLModel>> {
    const response = await this.post<MLModel>(API_CONFIG.endpoints.scrapper.train, request);
    
    if (response.success && response.data) {
      this.modelsCache.set(response.data.id, response.data);
    }
    
    return response;
  }

  /**
   * Stop model training
   */
  async stopTraining(modelId: string): Promise<ApiResponse<void>> {
    return this.post(`${API_CONFIG.endpoints.scrapper.train}/${modelId}/stop`);
  }

  /**
   * Deploy a model
   */
  async deployModel(modelId: string): Promise<ApiResponse<MLModel>> {
    const response = await this.post<MLModel>(`${API_CONFIG.endpoints.scrapper.models}/${modelId}/deploy`);
    
    if (response.success && response.data) {
      this.modelsCache.set(modelId, response.data);
    }
    
    return response;
  }

  /**
   * Delete a model
   */
  async deleteModel(id: string): Promise<ApiResponse<void>> {
    const response = await this.delete(`${API_CONFIG.endpoints.scrapper.models}/${id}`);
    
    if (response.success) {
      this.modelsCache.delete(id);
    }
    
    return response;
  }

  /**
   * Export dataset
   */
  async exportDataset(request: ExportRequest): Promise<ApiResponse<Blob>> {
    const response = await this.post(
      API_CONFIG.endpoints.scrapper.export,
      request,
      {
        headers: {
          Accept: 'application/octet-stream',
        },
      }
    );
    
    return response as ApiResponse<Blob>;
  }

  /**
   * Import dataset
   */
  async importDataset(request: ImportRequest): Promise<ApiResponse<Dataset>> {
    const formData = new FormData();
    formData.append('file', request.file);
    formData.append('name', request.name);
    formData.append('type', request.type);
    
    if (request.overwrite !== undefined) {
      formData.append('overwrite', String(request.overwrite));
    }
    
    const response = await this.request<Dataset>(
      API_CONFIG.endpoints.scrapper.import,
      {
        method: 'POST',
        body: formData,
        headers: {
          // Let browser set Content-Type with boundary
        },
      }
    );
    
    if (response.success && response.data) {
      this.datasetsCache.set(response.data.id, response.data);
    }
    
    return response;
  }

  /**
   * Get scrapper configuration
   */
  async getConfig(): Promise<ApiResponse<ScrapperConfig>> {
    return this.get<ScrapperConfig>(API_CONFIG.endpoints.scrapper.config);
  }

  /**
   * Update scrapper configuration
   */
  async updateConfig(config: Partial<ScrapperConfig>): Promise<ApiResponse<ScrapperConfig>> {
    return this.put<ScrapperConfig>(API_CONFIG.endpoints.scrapper.config, config);
  }

  /**
   * Get dataset statistics
   */
  async getDatasetStats(): Promise<ApiResponse<DatasetStats>> {
    return this.get<DatasetStats>(`${API_CONFIG.endpoints.scrapper.datasets}/stats`);
  }

  /**
   * Get model statistics
   */
  async getModelStats(): Promise<ApiResponse<ModelStats>> {
    return this.get<ModelStats>(`${API_CONFIG.endpoints.scrapper.models}/stats`);
  }

  /**
   * Connect to progress WebSocket stream
   */
  connectProgressWebSocket(
    onProgress?: (update: ScrapperProgressUpdate['payload']) => void,
    onTraining?: (update: ModelTrainingUpdate['payload']) => void
  ): ScrapperProgressWebSocket {
    // Disconnect existing WebSocket if any
    if (this.progressWS) {
      this.progressWS.disconnect();
    }
    
    this.progressWS = new ScrapperProgressWebSocket();
    
    // Set up event handlers
    if (onProgress) {
      this.progressWS.onProgress(onProgress);
    }
    
    if (onTraining) {
      this.progressWS.onTraining(onTraining);
    }
    
    // Auto-connect
    this.progressWS.connect().catch(error => {
      console.error('WebSocket connection failed:', error);
    });
    
    return this.progressWS;
  }

  /**
   * Disconnect progress WebSocket
   */
  disconnectProgressWebSocket(): void {
    if (this.progressWS) {
      this.progressWS.disconnect();
      this.progressWS = null;
    }
  }

  /**
   * Clear caches
   */
  clearCache(): void {
    this.datasetsCache.clear();
    this.modelsCache.clear();
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    this.disconnectProgressWebSocket();
    this.cancelAllRequests();
    this.clearCache();
  }
}

/**
 * WebSocket client for scrapper progress updates
 */
export class ScrapperProgressWebSocket extends WebSocketBase {
  constructor() {
    super(API_CONFIG.ws.scrapperProgress);
  }

  /**
   * Subscribe to collection progress updates
   */
  onProgress(handler: (update: ScrapperProgressUpdate['payload']) => void): () => void {
    return this.on('scrapper-progress', (data: ScrapperProgressUpdate) => {
      handler(data.payload);
    });
  }

  /**
   * Subscribe to model training updates
   */
  onTraining(handler: (update: ModelTrainingUpdate['payload']) => void): () => void {
    return this.on('model-training', (data: ModelTrainingUpdate) => {
      handler(data.payload);
    });
  }

  /**
   * Subscribe to dataset updates
   */
  onDatasetUpdate(handler: (dataset: Dataset) => void): () => void {
    return this.on('dataset-update', handler);
  }

  /**
   * Subscribe to model updates
   */
  onModelUpdate(handler: (model: MLModel) => void): () => void {
    return this.on('model-update', handler);
  }

  /**
   * Subscribe to specific dataset progress
   */
  subscribeToDataset(datasetId: string): void {
    this.send('subscribe-dataset', { datasetId });
  }

  /**
   * Unsubscribe from dataset progress
   */
  unsubscribeFromDataset(datasetId: string): void {
    this.send('unsubscribe-dataset', { datasetId });
  }

  /**
   * Subscribe to specific model training
   */
  subscribeToModel(modelId: string): void {
    this.send('subscribe-model', { modelId });
  }

  /**
   * Unsubscribe from model training
   */
  unsubscribeFromModel(modelId: string): void {
    this.send('unsubscribe-model', { modelId });
  }
}

// Export singleton instance
const scrapperService = new ScrapperService();
export default scrapperService;