import { useState, useCallback } from 'react';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Progress } from '../components/ui/Progress';
import { useToast } from '../hooks/use-toast';
import HistoricalCapturePanel from '../components/HistoricalCapturePanel';

interface DatasetConfig {
  id: string;
  name: string;
  type: 'arbitrage' | 'sandwich' | 'liquidation' | 'jito' | 'custom';
  status: 'idle' | 'collecting' | 'processing' | 'completed' | 'error';
  progress: number;
  totalRecords: number;
  startTime?: Date;
  endTime?: Date;
  filters: {
    minProfit?: number;
    maxSlippage?: number;
    tokenPairs?: string[];
    timeRange?: { start: Date; end: Date };
  };
}

interface MLModel {
  id: string;
  name: string;
  type: string;
  status: 'training' | 'ready' | 'failed';
  accuracy: number;
  loss: number;
  epoch: number;
  maxEpochs: number;
  trainingTime: number;
}

export default function ScrapperPage() {
  const { toast } = useToast();
  const [datasets, setDatasets] = useState<DatasetConfig[]>([
    {
      id: '1',
      name: 'Arbitrage Opportunities Q1',
      type: 'arbitrage',
      status: 'completed',
      progress: 100,
      totalRecords: 45230,
      startTime: new Date('2024-01-01'),
      endTime: new Date('2024-03-31'),
      filters: {
        minProfit: 100,
        tokenPairs: ['SOL/USDC', 'RAY/USDC'],
      },
    },
    {
      id: '2',
      name: 'MEV Sandwich Dataset',
      type: 'sandwich',
      status: 'collecting',
      progress: 67,
      totalRecords: 12450,
      filters: {
        maxSlippage: 2.5,
      },
    },
  ]);

  const [models, setModels] = useState<MLModel[]>([
    {
      id: '1',
      name: 'ArbitragePredictor_v2',
      type: 'RandomForest',
      status: 'ready',
      accuracy: 94.3,
      loss: 0.0234,
      epoch: 100,
      maxEpochs: 100,
      trainingTime: 3240,
    },
    {
      id: '2',
      name: 'SandwichDetector_LSTM',
      type: 'LSTM',
      status: 'training',
      accuracy: 87.2,
      loss: 0.0456,
      epoch: 45,
      maxEpochs: 150,
      trainingTime: 1890,
    },
  ]);

  const [activeTab, setActiveTab] = useState<'capture' | 'datasets' | 'models' | 'export'>('capture');
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);

  const handleCreateDataset = useCallback(() => {
    setIsCreatingDataset(true);
    // Implement dataset creation logic
    const newDataset: DatasetConfig = {
      id: Date.now().toString(),
      name: `Dataset_${new Date().toISOString().split('T')[0]}`,
      type: 'arbitrage',
      status: 'collecting',
      progress: 0,
      totalRecords: 0,
      startTime: new Date(),
      filters: {},
    };
    
    setDatasets(prev => [...prev, newDataset]);
    
    // Simulate progress
    const interval = setInterval(() => {
      setDatasets(prev => prev.map(ds => {
        if (ds.id === newDataset.id && ds.progress < 100) {
          const newProgress = Math.min(100, ds.progress + Math.random() * 10);
          return {
            ...ds,
            progress: newProgress,
            totalRecords: Math.floor(newProgress * 500),
            status: newProgress === 100 ? 'completed' : 'collecting',
          };
        }
        return ds;
      }));
    }, 1000);

    setTimeout(() => {
      clearInterval(interval);
      setIsCreatingDataset(false);
      toast({
        title: 'Dataset Created',
        description: 'New dataset has been created and is collecting data.',
      });
    }, 10000);
  }, [toast]);

  const handleExportDataset = useCallback((datasetId: string) => {
    const dataset = datasets.find(d => d.id === datasetId);
    if (dataset) {
      toast({
        title: 'Export Started',
        description: `Exporting ${dataset.name} with ${dataset.totalRecords} records...`,
      });
      
      // Simulate export
      setTimeout(() => {
        toast({
          title: 'Export Complete',
          description: 'Dataset has been exported to CSV format.',
        });
      }, 2000);
    }
  }, [datasets, toast]);

  const handleTrainModel = useCallback((modelId: string) => {
    setModels(prev => prev.map(model => {
      if (model.id === modelId) {
        return { ...model, status: 'training', epoch: 0 };
      }
      return model;
    }));

    // Simulate training progress
    const interval = setInterval(() => {
      setModels(prev => prev.map(model => {
        if (model.id === modelId && model.epoch < model.maxEpochs) {
          const newEpoch = model.epoch + 1;
          return {
            ...model,
            epoch: newEpoch,
            accuracy: Math.min(99, model.accuracy + Math.random() * 0.5),
            loss: Math.max(0.001, model.loss - Math.random() * 0.001),
            status: newEpoch === model.maxEpochs ? 'ready' : 'training',
          };
        }
        return model;
      }));
    }, 500);

    setTimeout(() => {
      clearInterval(interval);
      toast({
        title: 'Training Complete',
        description: 'Model has been successfully trained.',
      });
    }, 10000);
  }, [toast]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
      case 'ready':
        return 'text-green-400';
      case 'collecting':
      case 'processing':
      case 'training':
        return 'text-yellow-400';
      case 'idle':
        return 'text-gray-400';
      case 'error':
      case 'failed':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            Data Scrapper
          </h1>
          <p className="text-gray-400 mt-1">Historical data capture and ML training datasets</p>
        </div>
        <div className="flex items-center gap-3">
          <Button 
            onClick={handleCreateDataset}
            disabled={isCreatingDataset}
            className="bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600"
          >
            {isCreatingDataset ? 'Creating...' : 'New Dataset'}
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-4 border-b border-gray-700">
        {['capture', 'datasets', 'models', 'export'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as any)}
            className={`pb-3 px-1 capitalize transition-colors ${
              activeTab === tab
                ? 'text-white border-b-2 border-purple-500'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            {tab === 'capture' ? 'Historical Capture' : tab}
          </button>
        ))}
      </div>

      {/* Historical Capture Tab */}
      {activeTab === 'capture' && (
        <HistoricalCapturePanel />
      )}

      {/* Datasets Tab */}
      {activeTab === 'datasets' && (
        <div className="space-y-4">
          <Card className="bg-gray-800 border-gray-700 p-6">
            <h2 className="text-xl font-bold text-white mb-4">Active Datasets</h2>
            <div className="space-y-4">
              {datasets.map((dataset) => (
                <div key={dataset.id} className="bg-gray-900 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <h3 className="text-lg font-semibold text-white">{dataset.name}</h3>
                      <Badge variant="outline" className="text-xs">
                        {dataset.type}
                      </Badge>
                      <span className={`text-sm font-medium ${getStatusColor(dataset.status)}`}>
                        {dataset.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleExportDataset(dataset.id)}
                        disabled={dataset.status !== 'completed'}
                      >
                        Export
                      </Button>
                    </div>
                  </div>

                  {dataset.status === 'collecting' || dataset.status === 'processing' ? (
                    <div className="space-y-2">
                      <Progress value={dataset.progress} className="h-2" />
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Progress: {dataset.progress.toFixed(1)}%</span>
                        <span className="text-gray-400">Records: {dataset.totalRecords.toLocaleString()}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Total Records</p>
                        <p className="text-white font-mono">{dataset.totalRecords.toLocaleString()}</p>
                      </div>
                      {dataset.filters.minProfit && (
                        <div>
                          <p className="text-gray-400">Min Profit</p>
                          <p className="text-green-400 font-mono">${dataset.filters.minProfit}</p>
                        </div>
                      )}
                      {dataset.filters.maxSlippage && (
                        <div>
                          <p className="text-gray-400">Max Slippage</p>
                          <p className="text-yellow-400 font-mono">{dataset.filters.maxSlippage}%</p>
                        </div>
                      )}
                      {dataset.filters.tokenPairs && (
                        <div>
                          <p className="text-gray-400">Token Pairs</p>
                          <p className="text-blue-400 font-mono">{dataset.filters.tokenPairs.length}</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>

          {/* Dataset Configuration */}
          <Card className="bg-gray-800 border-gray-700 p-6">
            <h2 className="text-xl font-bold text-white mb-4">Configuration</h2>
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white">Arbitrage Detection</h3>
                <div className="space-y-2">
                  <label className="flex items-center justify-between">
                    <span className="text-gray-400">Min Profit Threshold</span>
                    <input 
                      type="number" 
                      className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-24 text-white"
                      defaultValue="100"
                    />
                  </label>
                  <label className="flex items-center justify-between">
                    <span className="text-gray-400">Max Latency (ms)</span>
                    <input 
                      type="number" 
                      className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-24 text-white"
                      defaultValue="50"
                    />
                  </label>
                  <label className="flex items-center justify-between">
                    <span className="text-gray-400">Include Failed TXs</span>
                    <input type="checkbox" className="rounded" />
                  </label>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white">MEV Sandwich</h3>
                <div className="space-y-2">
                  <label className="flex items-center justify-between">
                    <span className="text-gray-400">Slippage Tolerance (%)</span>
                    <input 
                      type="number" 
                      className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-24 text-white"
                      defaultValue="2.5"
                      step="0.1"
                    />
                  </label>
                  <label className="flex items-center justify-between">
                    <span className="text-gray-400">Min Volume (SOL)</span>
                    <input 
                      type="number" 
                      className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-24 text-white"
                      defaultValue="10"
                    />
                  </label>
                  <label className="flex items-center justify-between">
                    <span className="text-gray-400">Track Jito Bundles</span>
                    <input type="checkbox" className="rounded" defaultChecked />
                  </label>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Models Tab */}
      {activeTab === 'models' && (
        <div className="space-y-4">
          <Card className="bg-gray-800 border-gray-700 p-6">
            <h2 className="text-xl font-bold text-white mb-4">ML Models</h2>
            <div className="grid grid-cols-2 gap-4">
              {models.map((model) => (
                <div key={model.id} className="bg-gray-900 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{model.name}</h3>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline" className="text-xs">
                          {model.type}
                        </Badge>
                        <span className={`text-sm font-medium ${getStatusColor(model.status)}`}>
                          {model.status}
                        </span>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      onClick={() => handleTrainModel(model.id)}
                      disabled={model.status === 'training'}
                    >
                      {model.status === 'training' ? 'Training...' : 'Train'}
                    </Button>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-400">Training Progress</span>
                        <span className="text-white">{model.epoch}/{model.maxEpochs} epochs</span>
                      </div>
                      <Progress value={(model.epoch / model.maxEpochs) * 100} className="h-2" />
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <p className="text-gray-400 text-xs">Accuracy</p>
                        <p className="text-green-400 font-mono text-lg">{model.accuracy.toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-xs">Loss</p>
                        <p className="text-yellow-400 font-mono text-lg">{model.loss.toFixed(4)}</p>
                      </div>
                    </div>

                    <div className="pt-2 border-t border-gray-800">
                      <p className="text-gray-400 text-xs">Training Time</p>
                      <p className="text-white font-mono">{formatTime(model.trainingTime)}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Model Configuration */}
          <Card className="bg-gray-800 border-gray-700 p-6">
            <h2 className="text-xl font-bold text-white mb-4">Training Configuration</h2>
            <div className="grid grid-cols-3 gap-6">
              <div>
                <h3 className="text-sm font-semibold text-gray-400 mb-2">Hyperparameters</h3>
                <div className="space-y-2">
                  <label className="block">
                    <span className="text-gray-400 text-xs">Learning Rate</span>
                    <input 
                      type="number" 
                      className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-full text-white"
                      defaultValue="0.001"
                      step="0.0001"
                    />
                  </label>
                  <label className="block">
                    <span className="text-gray-400 text-xs">Batch Size</span>
                    <input 
                      type="number" 
                      className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-full text-white"
                      defaultValue="32"
                    />
                  </label>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-gray-400 mb-2">Architecture</h3>
                <div className="space-y-2">
                  <label className="block">
                    <span className="text-gray-400 text-xs">Model Type</span>
                    <select className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-full text-white">
                      <option>LSTM</option>
                      <option>GRU</option>
                      <option>Transformer</option>
                      <option>RandomForest</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-gray-400 text-xs">Hidden Layers</span>
                    <input 
                      type="number" 
                      className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-full text-white"
                      defaultValue="3"
                    />
                  </label>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-gray-400 mb-2">Optimization</h3>
                <div className="space-y-2">
                  <label className="block">
                    <span className="text-gray-400 text-xs">Optimizer</span>
                    <select className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-full text-white">
                      <option>Adam</option>
                      <option>SGD</option>
                      <option>RMSprop</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-gray-400 text-xs">Dropout Rate</span>
                    <input 
                      type="number" 
                      className="bg-gray-900 border border-gray-700 rounded px-2 py-1 w-full text-white"
                      defaultValue="0.2"
                      step="0.1"
                      max="1"
                    />
                  </label>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Export Tab */}
      {activeTab === 'export' && (
        <div className="space-y-4">
          <Card className="bg-gray-800 border-gray-700 p-6">
            <h2 className="text-xl font-bold text-white mb-4">Data Export</h2>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Export Options</h3>
                <div className="space-y-3">
                  <label className="block">
                    <span className="text-gray-400">Format</span>
                    <select className="bg-gray-900 border border-gray-700 rounded px-3 py-2 w-full text-white mt-1">
                      <option>CSV</option>
                      <option>JSON</option>
                      <option>Parquet</option>
                      <option>Arrow</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-gray-400">Compression</span>
                    <select className="bg-gray-900 border border-gray-700 rounded px-3 py-2 w-full text-white mt-1">
                      <option>None</option>
                      <option>GZIP</option>
                      <option>ZSTD</option>
                      <option>LZ4</option>
                    </select>
                  </label>
                  <label className="flex items-center gap-2">
                    <input type="checkbox" className="rounded" defaultChecked />
                    <span className="text-gray-400">Include metadata</span>
                  </label>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Import Data</h3>
                <div className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center">
                  <p className="text-gray-400 mb-2">Drop files here or click to browse</p>
                  <Button variant="outline">Choose Files</Button>
                  <p className="text-xs text-gray-500 mt-2">Supports CSV, JSON, Parquet</p>
                </div>
              </div>
            </div>
          </Card>

          {/* Recent Exports */}
          <Card className="bg-gray-800 border-gray-700 p-6">
            <h2 className="text-xl font-bold text-white mb-4">Recent Exports</h2>
            <div className="space-y-2">
              {[
                { name: 'arbitrage_2024_03_15.csv', size: '245 MB', date: '2024-03-15 14:23' },
                { name: 'sandwich_dataset.parquet', size: '1.2 GB', date: '2024-03-14 09:45' },
                { name: 'training_data_v2.json', size: '456 MB', date: '2024-03-13 18:30' },
              ].map((file, idx) => (
                <div key={idx} className="flex items-center justify-between bg-gray-900 rounded p-3">
                  <div>
                    <p className="text-white font-mono text-sm">{file.name}</p>
                    <p className="text-gray-400 text-xs">{file.date}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-gray-400 text-sm">{file.size}</span>
                    <Button size="sm" variant="outline">Download</Button>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}