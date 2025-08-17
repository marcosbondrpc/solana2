'use client';

import { useState } from 'react';
import { useSnapshot } from 'valtio';
import { mevStore, mevActions } from '../../stores/mevStore';

interface ModelConfig {
  name: string;
  version: string;
  status: 'active' | 'standby' | 'training';
  accuracy: number;
  lastTrained: string;
}

export default function ControlPanel() {
  const store = useSnapshot(mevStore);
  const [activeTab, setActiveTab] = useState<'models' | 'policies' | 'throttle' | 'config'>('models');
  const [throttlePercent, setThrottlePercent] = useState(100);
  const [selectedModel, setSelectedModel] = useState('mev-v1.0');
  
  const [models] = useState<ModelConfig[]>([
    { name: 'mev-v1.0', version: '1.0.0', status: 'active', accuracy: 0.923, lastTrained: '2h ago' },
    { name: 'mev-v1.1', version: '1.1.0', status: 'standby', accuracy: 0.941, lastTrained: '30m ago' },
    { name: 'mev-v2.0', version: '2.0.0-beta', status: 'training', accuracy: 0.912, lastTrained: 'in progress' },
  ]);

  const [policies, setPolicies] = useState([
    { name: 'Auto-throttle on high latency', enabled: true },
    { name: 'Kill on negative EV > 1%', enabled: true },
    { name: 'Pause on land rate < 55%', enabled: true },
    { name: 'Alert on adversary density > 10', enabled: true },
    { name: 'Block honeypot tokens', enabled: true },
  ]);

  const handleModelSwap = async () => {
    if (confirm(`Swap to model ${selectedModel}? This will hot-reload without restart.`)) {
      try {
        const response = await fetch('http://localhost:8000/api/control/swap-model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: selectedModel }),
        });
        if (response.ok) {
          console.log(`Model swapped to ${selectedModel}`);
        }
      } catch (error) {
        console.error('Failed to swap model:', error);
      }
    }
  };

  const handleThrottle = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/control/throttle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ percent: throttlePercent }),
      });
      if (response.ok) {
        console.log(`Throttled to ${throttlePercent}%`);
      }
    } catch (error) {
      console.error('Failed to throttle:', error);
    }
  };

  const handlePolicyToggle = (index: number) => {
    setPolicies(prev => prev.map((policy, i) => 
      i === index ? { ...policy, enabled: !policy.enabled } : policy
    ));
  };

  const handleExportData = () => {
    const data = {
      transactions: store.transactions,
      metrics: store.systemMetrics,
      stats: store.bundleStats,
      timestamp: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mev-data-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-4">
      {/* Tab Navigation */}
      <div className="flex items-center gap-4 mb-4 border-b border-zinc-800 pb-2">
        {(['models', 'policies', 'throttle', 'config'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-3 py-1 text-xs font-medium transition-colors ${
              activeTab === tab
                ? 'text-brand-600 border-b-2 border-brand-600'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="min-h-[120px]">
        {activeTab === 'models' && (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-3">
              {models.map(model => (
                <div 
                  key={model.name}
                  className={`p-3 rounded border cursor-pointer transition-all ${
                    selectedModel === model.name
                      ? 'bg-brand-900/20 border-brand-700'
                      : 'bg-zinc-900 border-zinc-800 hover:border-zinc-700'
                  }`}
                  onClick={() => setSelectedModel(model.name)}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-mono">{model.name}</span>
                    <span className={`text-[10px] px-1 py-0.5 rounded ${
                      model.status === 'active' ? 'bg-green-900/30 text-green-400' :
                      model.status === 'training' ? 'bg-yellow-900/30 text-yellow-400' :
                      'bg-zinc-800 text-zinc-400'
                    }`}>
                      {model.status}
                    </span>
                  </div>
                  <div className="text-[10px] text-zinc-600 space-y-1">
                    <div>Version: {model.version}</div>
                    <div>Accuracy: {(model.accuracy * 100).toFixed(1)}%</div>
                    <div>Trained: {model.lastTrained}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleModelSwap}
                className="px-4 py-1 bg-brand-800 hover:bg-brand-700 text-white rounded text-xs font-medium transition-colors"
              >
                Hot Swap Model
              </button>
              <button
                className="px-4 py-1 bg-zinc-800 hover:bg-zinc-700 text-white rounded text-xs font-medium transition-colors"
              >
                Train New Model
              </button>
            </div>
          </div>
        )}

        {activeTab === 'policies' && (
          <div className="space-y-2">
            {policies.map((policy, idx) => (
              <div 
                key={idx}
                className="flex items-center justify-between p-2 bg-zinc-900 rounded"
              >
                <span className="text-sm text-zinc-300">{policy.name}</span>
                <button
                  onClick={() => handlePolicyToggle(idx)}
                  className={`relative w-10 h-5 rounded-full transition-colors ${
                    policy.enabled ? 'bg-green-600' : 'bg-zinc-700'
                  }`}
                >
                  <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                    policy.enabled ? 'translate-x-5' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            ))}
            <div className="pt-2 text-[10px] text-zinc-600">
              * Policy changes take effect immediately without restart
            </div>
          </div>
        )}

        {activeTab === 'throttle' && (
          <div className="space-y-3">
            <div>
              <label className="text-sm text-zinc-400 block mb-2">
                Trading Volume Throttle
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="10"
                  value={throttlePercent}
                  onChange={(e) => setThrottlePercent(Number(e.target.value))}
                  className="flex-1"
                />
                <span className="text-sm font-mono text-brand-500 w-12">
                  {throttlePercent}%
                </span>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <button
                  onClick={handleThrottle}
                  className="px-4 py-1 bg-yellow-800 hover:bg-yellow-700 text-white rounded text-xs font-medium transition-colors"
                >
                  Apply Throttle
                </button>
                <button
                  onClick={() => setThrottlePercent(100)}
                  className="px-4 py-1 bg-zinc-800 hover:bg-zinc-700 text-white rounded text-xs font-medium transition-colors"
                >
                  Reset to 100%
                </button>
              </div>
            </div>
            <div className="grid grid-cols-4 gap-2 pt-2">
              {[10, 25, 50, 75].map(percent => (
                <button
                  key={percent}
                  onClick={() => {
                    setThrottlePercent(percent);
                    handleThrottle();
                  }}
                  className="px-3 py-2 bg-zinc-900 hover:bg-zinc-800 rounded text-xs transition-colors"
                >
                  {percent}%
                </button>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'config' && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[10px] text-zinc-600 block mb-1">
                  Max Transactions
                </label>
                <input
                  type="number"
                  value={store.settings.maxTransactions}
                  onChange={(e) => mevActions.updateSettings({ maxTransactions: Number(e.target.value) })}
                  className="w-full px-2 py-1 bg-zinc-900 border border-zinc-800 rounded text-xs"
                />
              </div>
              <div>
                <label className="text-[10px] text-zinc-600 block mb-1">
                  Update Interval (ms)
                </label>
                <input
                  type="number"
                  value={store.settings.updateInterval}
                  onChange={(e) => mevActions.updateSettings({ updateInterval: Number(e.target.value) })}
                  className="w-full px-2 py-1 bg-zinc-900 border border-zinc-800 rounded text-xs"
                />
              </div>
              <div>
                <label className="text-[10px] text-zinc-600 block mb-1">
                  Theme
                </label>
                <select
                  value={store.settings.theme}
                  onChange={(e) => mevActions.updateSettings({ theme: e.target.value as any })}
                  className="w-full px-2 py-1 bg-zinc-900 border border-zinc-800 rounded text-xs"
                >
                  <option value="dark">Dark</option>
                  <option value="midnight">Midnight</option>
                  <option value="matrix">Matrix</option>
                </select>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="autoScroll"
                  checked={store.settings.autoScroll}
                  onChange={(e) => mevActions.updateSettings({ autoScroll: e.target.checked })}
                  className="rounded"
                />
                <label htmlFor="autoScroll" className="text-xs text-zinc-400">
                  Auto-scroll feed
                </label>
              </div>
            </div>
            <div className="flex items-center gap-2 pt-2">
              <button
                onClick={handleExportData}
                className="px-4 py-1 bg-zinc-800 hover:bg-zinc-700 text-white rounded text-xs font-medium transition-colors"
              >
                Export Data
              </button>
              <button
                onClick={() => mevActions.reset()}
                className="px-4 py-1 bg-red-900 hover:bg-red-800 text-white rounded text-xs font-medium transition-colors"
              >
                Reset Dashboard
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}