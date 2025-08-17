import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as Dialog from '@radix-ui/react-dialog';
import * as Checkbox from '@radix-ui/react-checkbox';
import * as Select from '@radix-ui/react-select';
import * as Tabs from '@radix-ui/react-tabs';
import * as Progress from '@radix-ui/react-progress';
import {
  Download,
  X,
  Check,
  AlertCircle,
  Clock,
  Database,
  Cpu,
  Activity,
  FileText,
  Zap,
  TrendingUp,
  Layers,
  Filter,
  Settings,
  ChevronDown,
  Archive,
  Eye,
  Sparkles,
  Brain,
  Target,
  BarChart3,
  GitBranch,
  Hash,
  Binary,
  Gauge
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { useDataExportStore } from '../../stores/data-export-store';
import { DataExportService } from '../../services/data-export.service';
import { FeatureCorrelationMatrix } from './export/FeatureCorrelationMatrix';
import { FeatureImportanceChart } from './export/FeatureImportanceChart';
import { ExportPreviewTable } from './export/ExportPreviewTable';

// Types
export interface ExportConfig {
  timeRange: TimeRange;
  metrics: MetricSelection;
  format: ExportFormat;
  compression: CompressionType;
  features: FeatureEngineering;
  labels: LabelConfig;
  chunking: ChunkConfig;
}

export interface TimeRange {
  preset: 'last_1h' | 'last_6h' | 'last_24h' | 'last_7d' | 'custom';
  from?: Date;
  to?: Date;
}

export interface MetricSelection {
  rpc: {
    latencies: boolean;
    methodCalls: boolean;
    errorRates: boolean;
    subscriptions: boolean;
  };
  jito: {
    tips: boolean;
    bundleSuccess: boolean;
    auctionData: boolean;
    shredstream: boolean;
    landingDelays: boolean;
  };
  quicTpu: {
    handshakes: boolean;
    streams: boolean;
    throttling: boolean;
    qosMetrics: boolean;
  };
  nodeHealth: {
    slots: boolean;
    blockProduction: boolean;
    votePerformance: boolean;
    leaderSchedule: boolean;
  };
  mevFeatures: {
    timingWaterfalls: boolean;
    tipEfficiency: boolean;
    landingDelays: boolean;
    arbitrageWindows: boolean;
    sandwichPatterns: boolean;
  };
}

export type ExportFormat = 'csv' | 'json' | 'parquet' | 'arrow';
export type CompressionType = 'none' | 'gzip' | 'zstd' | 'lz4';

export interface FeatureEngineering {
  enabled: boolean;
  rollingWindows: {
    '1m': boolean;
    '5m': boolean;
    '15m': boolean;
  };
  lagFeatures: {
    enabled: boolean;
    lags: number[];
  };
  crossFeatures: {
    tipGasRatio: boolean;
    successLatencyCorr: boolean;
    bundleEfficiency: boolean;
  };
  advancedFeatures: {
    fourierTransform: boolean;
    wavelets: boolean;
    autoregressive: boolean;
    polynomialFeatures: boolean;
  };
}

export interface LabelConfig {
  mevSuccess: boolean;
  sandwichDetection: boolean;
  arbitrageWindows: boolean;
  customLabels: Array<{
    name: string;
    condition: string;
  }>;
}

export interface ChunkConfig {
  enabled: boolean;
  chunkSize: number;
  parallel: boolean;
}

export interface ExportProgress {
  id: string;
  status: 'preparing' | 'processing' | 'compressing' | 'complete' | 'error';
  progress: number;
  processedRows: number;
  totalRows: number;
  fileSize: number;
  eta: number;
  downloadUrl?: string;
  error?: string;
}

interface DatasetExporterProps {
  isOpen: boolean;
  onClose: () => void;
}

const DEFAULT_CONFIG: ExportConfig = {
  timeRange: {
    preset: 'last_24h',
  },
  metrics: {
    rpc: {
      latencies: true,
      methodCalls: true,
      errorRates: true,
      subscriptions: false,
    },
    jito: {
      tips: true,
      bundleSuccess: true,
      auctionData: true,
      shredstream: false,
      landingDelays: true,
    },
    quicTpu: {
      handshakes: true,
      streams: true,
      throttling: true,
      qosMetrics: false,
    },
    nodeHealth: {
      slots: true,
      blockProduction: true,
      votePerformance: false,
      leaderSchedule: false,
    },
    mevFeatures: {
      timingWaterfalls: true,
      tipEfficiency: true,
      landingDelays: true,
      arbitrageWindows: true,
      sandwichPatterns: true,
    },
  },
  format: 'parquet',
  compression: 'zstd',
  features: {
    enabled: true,
    rollingWindows: {
      '1m': true,
      '5m': true,
      '15m': false,
    },
    lagFeatures: {
      enabled: true,
      lags: [1, 5, 10, 30],
    },
    crossFeatures: {
      tipGasRatio: true,
      successLatencyCorr: true,
      bundleEfficiency: true,
    },
    advancedFeatures: {
      fourierTransform: false,
      wavelets: false,
      autoregressive: true,
      polynomialFeatures: false,
    },
  },
  labels: {
    mevSuccess: true,
    sandwichDetection: true,
    arbitrageWindows: true,
    customLabels: [],
  },
  chunking: {
    enabled: true,
    chunkSize: 100000,
    parallel: true,
  },
};

export const DatasetExporter: React.FC<DatasetExporterProps> = ({
  isOpen,
  onClose,
}) => {
  const [config, setConfig] = useState<ExportConfig>(DEFAULT_CONFIG);
  const [activeTab, setActiveTab] = useState('metrics');
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);
  const [previewData, setPreviewData] = useState<any[] | null>(null);
  const [featureImportance, setFeatureImportance] = useState<any[] | null>(null);
  const [correlationMatrix, setCorrelationMatrix] = useState<any | null>(null);
  
  const exportService = useRef(new DataExportService());
  const exportWorker = useRef<Worker | null>(null);
  const progressInterval = useRef<NodeJS.Timeout | null>(null);

  // Initialize WebWorker for heavy processing
  useEffect(() => {
    exportWorker.current = new Worker(
      new URL('../../workers/export-processor.worker.ts', import.meta.url),
      { type: 'module' }
    );

    exportWorker.current.onmessage = (event) => {
      const { type, data } = event.data;
      
      switch (type) {
        case 'progress':
          setExportProgress(prev => prev ? { ...prev, ...data } : null);
          break;
        case 'preview':
          setPreviewData(data);
          break;
        case 'features':
          setFeatureImportance(data.importance);
          setCorrelationMatrix(data.correlation);
          break;
        case 'complete':
          handleExportComplete(data);
          break;
        case 'error':
          handleExportError(data);
          break;
      }
    };

    return () => {
      exportWorker.current?.terminate();
      if (progressInterval.current) {
        clearInterval(progressInterval.current);
      }
    };
  }, []);

  // Handle export initiation
  const handleStartExport = useCallback(async () => {
    try {
      // Initialize export
      const exportId = `export-${Date.now()}`;
      setExportProgress({
        id: exportId,
        status: 'preparing',
        progress: 0,
        processedRows: 0,
        totalRows: 0,
        fileSize: 0,
        eta: 0,
      });

      // Start export on backend
      const response = await exportService.current.startExport(config);
      
      if (!response.success) {
        throw new Error(response.error?.message || 'Failed to start export');
      }

      // Update progress with export ID
      setExportProgress(prev => prev ? {
        ...prev,
        id: response.data.exportId,
        totalRows: response.data.estimatedRows,
      } : null);

      // Start progress polling
      progressInterval.current = setInterval(async () => {
        const progressResponse = await exportService.current.getProgress(response.data.exportId);
        
        if (progressResponse.success) {
          setExportProgress(prev => prev ? {
            ...prev,
            ...progressResponse.data,
          } : null);

          if (progressResponse.data.status === 'complete' || progressResponse.data.status === 'error') {
            if (progressInterval.current) {
              clearInterval(progressInterval.current);
            }
          }
        }
      }, 1000);

      // Send to worker for processing
      exportWorker.current?.postMessage({
        type: 'startExport',
        config,
        exportId: response.data.exportId,
      });
    } catch (error) {
      console.error('Export failed:', error);
      setExportProgress(prev => prev ? {
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      } : null);
    }
  }, [config]);

  // Handle export completion
  const handleExportComplete = useCallback((data: any) => {
    setExportProgress(prev => prev ? {
      ...prev,
      status: 'complete',
      progress: 100,
      downloadUrl: data.downloadUrl,
    } : null);

    // Cache export config for future use
    localStorage.setItem('lastExportConfig', JSON.stringify(config));
  }, [config]);

  // Handle export error
  const handleExportError = useCallback((error: any) => {
    setExportProgress(prev => prev ? {
      ...prev,
      status: 'error',
      error: error.message || 'Export failed',
    } : null);
  }, []);

  // Handle download
  const handleDownload = useCallback(async () => {
    if (!exportProgress?.downloadUrl) return;

    try {
      const response = await fetch(exportProgress.downloadUrl);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `solana-dataset-${exportProgress.id}.${config.format}${config.compression !== 'none' ? `.${config.compression}` : ''}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
    }
  }, [exportProgress, config]);

  // Calculate estimated file size
  const estimatedFileSize = useMemo(() => {
    if (!exportProgress) return 0;
    
    const baseSize = exportProgress.totalRows * 0.001; // ~1KB per row estimate
    const compressionRatio = {
      none: 1,
      gzip: 0.3,
      zstd: 0.25,
      lz4: 0.4,
    }[config.compression];
    
    return baseSize * compressionRatio;
  }, [exportProgress, config.compression]);

  // Format ETA
  const formatEta = useCallback((seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  }, []);

  return (
    <Dialog.Root open={isOpen} onOpenChange={onClose}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50" />
        <Dialog.Content className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[90vw] max-w-6xl h-[85vh] bg-gray-900/95 backdrop-blur-xl border border-cyan-500/20 rounded-2xl shadow-2xl z-50 overflow-hidden">
          {/* Glassmorphism background effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 via-transparent to-cyan-500/5 pointer-events-none" />
          
          {/* Header */}
          <div className="relative px-8 py-6 border-b border-gray-800/50 bg-gradient-to-r from-gray-900/50 to-gray-800/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-500 blur-xl opacity-50" />
                  <Database className="relative w-8 h-8 text-cyan-400" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                    Dataset Export
                  </h2>
                  <p className="text-sm text-gray-400 mt-1">
                    Extract high-quality training data for MEV models
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </Button>
            </div>
          </div>

          {/* Content */}
          <div className="flex flex-col h-[calc(100%-5rem)]">
            {!exportProgress ? (
              /* Configuration Tabs */
              <Tabs.Root value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
                <Tabs.List className="flex gap-2 px-8 py-4 bg-gray-900/30 border-b border-gray-800/50">
                  <Tabs.Trigger
                    value="metrics"
                    className={cn(
                      "px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2",
                      activeTab === 'metrics'
                        ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 border border-cyan-500/30"
                        : "text-gray-400 hover:text-white hover:bg-gray-800/50"
                    )}
                  >
                    <Activity className="w-4 h-4" />
                    Metrics Selection
                  </Tabs.Trigger>
                  <Tabs.Trigger
                    value="features"
                    className={cn(
                      "px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2",
                      activeTab === 'features'
                        ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 border border-cyan-500/30"
                        : "text-gray-400 hover:text-white hover:bg-gray-800/50"
                    )}
                  >
                    <Brain className="w-4 h-4" />
                    Feature Engineering
                  </Tabs.Trigger>
                  <Tabs.Trigger
                    value="labels"
                    className={cn(
                      "px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2",
                      activeTab === 'labels'
                        ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 border border-cyan-500/30"
                        : "text-gray-400 hover:text-white hover:bg-gray-800/50"
                    )}
                  >
                    <Target className="w-4 h-4" />
                    Training Labels
                  </Tabs.Trigger>
                  <Tabs.Trigger
                    value="analysis"
                    className={cn(
                      "px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2",
                      activeTab === 'analysis'
                        ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 border border-cyan-500/30"
                        : "text-gray-400 hover:text-white hover:bg-gray-800/50"
                    )}
                  >
                    <BarChart3 className="w-4 h-4" />
                    Feature Analysis
                  </Tabs.Trigger>
                </Tabs.List>

                <div className="flex-1 overflow-y-auto px-8 py-6">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={activeTab}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.2 }}
                    >
                      {/* Metrics Selection Tab */}
                      <Tabs.Content value="metrics" className="space-y-6">
                        <MetricsSelectionPanel config={config} onChange={setConfig} />
                      </Tabs.Content>

                      {/* Feature Engineering Tab */}
                      <Tabs.Content value="features" className="space-y-6">
                        <FeatureEngineeringPanel config={config} onChange={setConfig} />
                      </Tabs.Content>

                      {/* Training Labels Tab */}
                      <Tabs.Content value="labels" className="space-y-6">
                        <TrainingLabelsPanel config={config} onChange={setConfig} />
                      </Tabs.Content>

                      {/* Feature Analysis Tab */}
                      <Tabs.Content value="analysis" className="space-y-6">
                        <FeatureAnalysisPanel 
                          featureImportance={featureImportance}
                          correlationMatrix={correlationMatrix}
                        />
                      </Tabs.Content>
                    </motion.div>
                  </AnimatePresence>
                </div>

                {/* Footer with export controls */}
                <div className="px-8 py-6 border-t border-gray-800/50 bg-gray-900/30">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-6">
                      {/* Time Range Selector */}
                      <TimeRangeSelector config={config} onChange={setConfig} />
                      
                      {/* Format Selector */}
                      <FormatSelector config={config} onChange={setConfig} />
                      
                      {/* Compression Selector */}
                      <CompressionSelector config={config} onChange={setConfig} />
                    </div>

                    <div className="flex items-center gap-4">
                      <div className="text-sm text-gray-400">
                        Estimated size: <span className="text-cyan-400 font-medium">~{estimatedFileSize.toFixed(1)} MB</span>
                      </div>
                      <Button
                        onClick={handleStartExport}
                        className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white px-6"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Start Export
                      </Button>
                    </div>
                  </div>
                </div>
              </Tabs.Root>
            ) : (
              /* Export Progress View */
              <ExportProgressView
                progress={exportProgress}
                config={config}
                previewData={previewData}
                onDownload={handleDownload}
                onCancel={() => {
                  exportService.current.cancelExport(exportProgress.id);
                  setExportProgress(null);
                }}
              />
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};

// Sub-components
const MetricsSelectionPanel: React.FC<{
  config: ExportConfig;
  onChange: (config: ExportConfig) => void;
}> = ({ config, onChange }) => {
  const updateMetric = (category: keyof MetricSelection, metric: string, value: boolean) => {
    onChange({
      ...config,
      metrics: {
        ...config.metrics,
        [category]: {
          ...config.metrics[category],
          [metric]: value,
        },
      },
    });
  };

  return (
    <div className="grid grid-cols-2 gap-6">
      {/* RPC Metrics */}
      <Card className="p-6 bg-gray-800/30 border-gray-700/50">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-purple-500/10">
            <Activity className="w-5 h-5 text-purple-400" />
          </div>
          <h3 className="text-lg font-medium text-white">RPC Metrics</h3>
        </div>
        <div className="space-y-3">
          {Object.entries(config.metrics.rpc).map(([key, value]) => (
            <label key={key} className="flex items-center gap-3 cursor-pointer group">
              <Checkbox.Root
                checked={value}
                onCheckedChange={(checked) => updateMetric('rpc', key, !!checked)}
                className="w-5 h-5 rounded border border-gray-600 bg-gray-800 data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-cyan-500 data-[state=checked]:to-blue-500"
              >
                <Checkbox.Indicator>
                  <Check className="w-3 h-3 text-white" />
                </Checkbox.Indicator>
              </Checkbox.Root>
              <span className="text-gray-300 group-hover:text-white transition-colors">
                {key.replace(/([A-Z])/g, ' $1').trim()}
              </span>
            </label>
          ))}
        </div>
      </Card>

      {/* Jito Metrics */}
      <Card className="p-6 bg-gray-800/30 border-gray-700/50">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-cyan-500/10">
            <Zap className="w-5 h-5 text-cyan-400" />
          </div>
          <h3 className="text-lg font-medium text-white">Jito Metrics</h3>
        </div>
        <div className="space-y-3">
          {Object.entries(config.metrics.jito).map(([key, value]) => (
            <label key={key} className="flex items-center gap-3 cursor-pointer group">
              <Checkbox.Root
                checked={value}
                onCheckedChange={(checked) => updateMetric('jito', key, !!checked)}
                className="w-5 h-5 rounded border border-gray-600 bg-gray-800 data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-cyan-500 data-[state=checked]:to-blue-500"
              >
                <Checkbox.Indicator>
                  <Check className="w-3 h-3 text-white" />
                </Checkbox.Indicator>
              </Checkbox.Root>
              <span className="text-gray-300 group-hover:text-white transition-colors">
                {key.replace(/([A-Z])/g, ' $1').trim()}
              </span>
            </label>
          ))}
        </div>
      </Card>

      {/* Add more metric categories... */}
    </div>
  );
};

const FeatureEngineeringPanel: React.FC<{
  config: ExportConfig;
  onChange: (config: ExportConfig) => void;
}> = ({ config, onChange }) => {
  return (
    <div className="space-y-6">
      {/* Rolling Windows */}
      <Card className="p-6 bg-gray-800/30 border-gray-700/50">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-blue-500/10">
            <TrendingUp className="w-5 h-5 text-blue-400" />
          </div>
          <h3 className="text-lg font-medium text-white">Rolling Windows</h3>
        </div>
        <div className="flex gap-4">
          {Object.entries(config.features.rollingWindows).map(([window, enabled]) => (
            <label key={window} className="flex items-center gap-2 cursor-pointer">
              <Checkbox.Root
                checked={enabled}
                onCheckedChange={(checked) => onChange({
                  ...config,
                  features: {
                    ...config.features,
                    rollingWindows: {
                      ...config.features.rollingWindows,
                      [window]: !!checked,
                    },
                  },
                })}
                className="w-5 h-5 rounded border border-gray-600 bg-gray-800 data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-cyan-500 data-[state=checked]:to-blue-500"
              >
                <Checkbox.Indicator>
                  <Check className="w-3 h-3 text-white" />
                </Checkbox.Indicator>
              </Checkbox.Root>
              <span className="text-gray-300">{window}</span>
            </label>
          ))}
        </div>
      </Card>

      {/* Advanced Features */}
      <Card className="p-6 bg-gray-800/30 border-gray-700/50">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-purple-500/10">
            <Sparkles className="w-5 h-5 text-purple-400" />
          </div>
          <h3 className="text-lg font-medium text-white">Advanced Features</h3>
        </div>
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(config.features.advancedFeatures).map(([feature, enabled]) => (
            <label key={feature} className="flex items-center gap-3 cursor-pointer group">
              <Checkbox.Root
                checked={enabled}
                onCheckedChange={(checked) => onChange({
                  ...config,
                  features: {
                    ...config.features,
                    advancedFeatures: {
                      ...config.features.advancedFeatures,
                      [feature]: !!checked,
                    },
                  },
                })}
                className="w-5 h-5 rounded border border-gray-600 bg-gray-800 data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-cyan-500 data-[state=checked]:to-blue-500"
              >
                <Checkbox.Indicator>
                  <Check className="w-3 h-3 text-white" />
                </Checkbox.Indicator>
              </Checkbox.Root>
              <span className="text-gray-300 group-hover:text-white transition-colors">
                {feature.replace(/([A-Z])/g, ' $1').trim()}
              </span>
            </label>
          ))}
        </div>
      </Card>
    </div>
  );
};

const TrainingLabelsPanel: React.FC<{
  config: ExportConfig;
  onChange: (config: ExportConfig) => void;
}> = ({ config, onChange }) => {
  return (
    <div className="space-y-6">
      <Card className="p-6 bg-gray-800/30 border-gray-700/50">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-green-500/10">
            <Target className="w-5 h-5 text-green-400" />
          </div>
          <h3 className="text-lg font-medium text-white">Automatic Labels</h3>
        </div>
        <div className="space-y-3">
          <label className="flex items-center gap-3 cursor-pointer">
            <Checkbox.Root
              checked={config.labels.mevSuccess}
              onCheckedChange={(checked) => onChange({
                ...config,
                labels: { ...config.labels, mevSuccess: !!checked },
              })}
              className="w-5 h-5 rounded border border-gray-600 bg-gray-800 data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-cyan-500 data-[state=checked]:to-blue-500"
            >
              <Checkbox.Indicator>
                <Check className="w-3 h-3 text-white" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            <span className="text-gray-300">MEV Success Detection</span>
          </label>
          <label className="flex items-center gap-3 cursor-pointer">
            <Checkbox.Root
              checked={config.labels.sandwichDetection}
              onCheckedChange={(checked) => onChange({
                ...config,
                labels: { ...config.labels, sandwichDetection: !!checked },
              })}
              className="w-5 h-5 rounded border border-gray-600 bg-gray-800 data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-cyan-500 data-[state=checked]:to-blue-500"
            >
              <Checkbox.Indicator>
                <Check className="w-3 h-3 text-white" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            <span className="text-gray-300">Sandwich Attack Detection</span>
          </label>
          <label className="flex items-center gap-3 cursor-pointer">
            <Checkbox.Root
              checked={config.labels.arbitrageWindows}
              onCheckedChange={(checked) => onChange({
                ...config,
                labels: { ...config.labels, arbitrageWindows: !!checked },
              })}
              className="w-5 h-5 rounded border border-gray-600 bg-gray-800 data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-cyan-500 data-[state=checked]:to-blue-500"
            >
              <Checkbox.Indicator>
                <Check className="w-3 h-3 text-white" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            <span className="text-gray-300">Arbitrage Window Detection</span>
          </label>
        </div>
      </Card>
    </div>
  );
};

const FeatureAnalysisPanel: React.FC<{
  featureImportance: any[] | null;
  correlationMatrix: any | null;
}> = ({ featureImportance, correlationMatrix }) => {
  return (
    <div className="space-y-6">
      {featureImportance && (
        <FeatureImportanceChart data={featureImportance} />
      )}
      {correlationMatrix && (
        <FeatureCorrelationMatrix data={correlationMatrix} />
      )}
    </div>
  );
};

const TimeRangeSelector: React.FC<{
  config: ExportConfig;
  onChange: (config: ExportConfig) => void;
}> = ({ config, onChange }) => {
  return (
    <Select.Root
      value={config.timeRange.preset}
      onValueChange={(value) => onChange({
        ...config,
        timeRange: { ...config.timeRange, preset: value as any },
      })}
    >
      <Select.Trigger className="flex items-center gap-2 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-300 hover:bg-gray-700">
        <Clock className="w-4 h-4" />
        <Select.Value />
        <ChevronDown className="w-4 h-4" />
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="bg-gray-800 border border-gray-700 rounded-lg shadow-xl">
          <Select.Viewport>
            <Select.Item value="last_1h" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>Last 1 hour</Select.ItemText>
            </Select.Item>
            <Select.Item value="last_6h" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>Last 6 hours</Select.ItemText>
            </Select.Item>
            <Select.Item value="last_24h" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>Last 24 hours</Select.ItemText>
            </Select.Item>
            <Select.Item value="last_7d" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>Last 7 days</Select.ItemText>
            </Select.Item>
            <Select.Item value="custom" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>Custom range</Select.ItemText>
            </Select.Item>
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
};

const FormatSelector: React.FC<{
  config: ExportConfig;
  onChange: (config: ExportConfig) => void;
}> = ({ config, onChange }) => {
  return (
    <Select.Root
      value={config.format}
      onValueChange={(value) => onChange({ ...config, format: value as ExportFormat })}
    >
      <Select.Trigger className="flex items-center gap-2 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-300 hover:bg-gray-700">
        <FileText className="w-4 h-4" />
        <Select.Value />
        <ChevronDown className="w-4 h-4" />
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="bg-gray-800 border border-gray-700 rounded-lg shadow-xl">
          <Select.Viewport>
            <Select.Item value="csv" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>CSV</Select.ItemText>
            </Select.Item>
            <Select.Item value="json" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>JSON</Select.ItemText>
            </Select.Item>
            <Select.Item value="parquet" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>Parquet</Select.ItemText>
            </Select.Item>
            <Select.Item value="arrow" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>Arrow</Select.ItemText>
            </Select.Item>
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
};

const CompressionSelector: React.FC<{
  config: ExportConfig;
  onChange: (config: ExportConfig) => void;
}> = ({ config, onChange }) => {
  return (
    <Select.Root
      value={config.compression}
      onValueChange={(value) => onChange({ ...config, compression: value as CompressionType })}
    >
      <Select.Trigger className="flex items-center gap-2 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-300 hover:bg-gray-700">
        <Archive className="w-4 h-4" />
        <Select.Value />
        <ChevronDown className="w-4 h-4" />
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="bg-gray-800 border border-gray-700 rounded-lg shadow-xl">
          <Select.Viewport>
            <Select.Item value="none" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>None</Select.ItemText>
            </Select.Item>
            <Select.Item value="gzip" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>GZIP</Select.ItemText>
            </Select.Item>
            <Select.Item value="zstd" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>ZSTD</Select.ItemText>
            </Select.Item>
            <Select.Item value="lz4" className="px-4 py-2 text-gray-300 hover:bg-gray-700 cursor-pointer">
              <Select.ItemText>LZ4</Select.ItemText>
            </Select.Item>
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
};

const ExportProgressView: React.FC<{
  progress: ExportProgress;
  config: ExportConfig;
  previewData: any[] | null;
  onDownload: () => void;
  onCancel: () => void;
}> = ({ progress, config, previewData, onDownload, onCancel }) => {
  const statusColors = {
    preparing: 'text-yellow-400',
    processing: 'text-blue-400',
    compressing: 'text-purple-400',
    complete: 'text-green-400',
    error: 'text-red-400',
  };

  const statusIcons = {
    preparing: <Gauge className="w-5 h-5" />,
    processing: <Cpu className="w-5 h-5 animate-spin" />,
    compressing: <Archive className="w-5 h-5" />,
    complete: <Check className="w-5 h-5" />,
    error: <AlertCircle className="w-5 h-5" />,
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8">
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="max-w-2xl w-full space-y-8"
      >
        {/* Progress Status */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-3">
            <div className={cn("p-3 rounded-full bg-gray-800", statusColors[progress.status])}>
              {statusIcons[progress.status]}
            </div>
            <h3 className="text-2xl font-bold text-white capitalize">
              {progress.status === 'processing' ? 'Exporting Data' : progress.status}
            </h3>
          </div>
          
          {progress.error && (
            <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2">
              {progress.error}
            </div>
          )}
        </div>

        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm text-gray-400">
            <span>{progress.processedRows.toLocaleString()} / {progress.totalRows.toLocaleString()} rows</span>
            <span>{progress.progress.toFixed(1)}%</span>
          </div>
          <Progress.Root className="h-3 bg-gray-800 rounded-full overflow-hidden">
            <Progress.Indicator
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
              style={{ width: `${progress.progress}%` }}
            />
          </Progress.Root>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4">
          <Card className="p-4 bg-gray-800/30 border-gray-700/50">
            <div className="text-sm text-gray-400">File Size</div>
            <div className="text-xl font-bold text-white">
              {(progress.fileSize / 1024 / 1024).toFixed(2)} MB
            </div>
          </Card>
          <Card className="p-4 bg-gray-800/30 border-gray-700/50">
            <div className="text-sm text-gray-400">Format</div>
            <div className="text-xl font-bold text-white uppercase">
              {config.format}
            </div>
          </Card>
          <Card className="p-4 bg-gray-800/30 border-gray-700/50">
            <div className="text-sm text-gray-400">ETA</div>
            <div className="text-xl font-bold text-white">
              {progress.eta > 0 ? `~${Math.ceil(progress.eta / 60)}m` : '-'}
            </div>
          </Card>
        </div>

        {/* Preview */}
        {previewData && (
          <ExportPreviewTable data={previewData} />
        )}

        {/* Actions */}
        <div className="flex justify-center gap-4">
          {progress.status === 'complete' ? (
            <Button
              onClick={onDownload}
              className="bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white px-8"
            >
              <Download className="w-4 h-4 mr-2" />
              Download Dataset
            </Button>
          ) : (
            <Button
              onClick={onCancel}
              variant="outline"
              className="border-red-500/30 text-red-400 hover:bg-red-500/10"
            >
              Cancel Export
            </Button>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default DatasetExporter;