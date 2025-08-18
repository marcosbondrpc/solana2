import { useState } from 'react';
import { useTheme } from '../providers/ThemeProvider';

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [settings, setSettings] = useState({
    autoExecute: true,
    maxGasPrice: 50,
    slippageTolerance: 0.5,
    priorityFee: 0.001,
    rpcEndpoint: 'https://api.mainnet-beta.solana.com',
    wsEndpoint: 'wss://api.mainnet-beta.solana.com',
  });

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-3xl font-bold text-white">Settings</h1>
        <p className="text-gray-400 mt-1">Configure your MEV dashboard preferences</p>
      </div>
      
      {/* General Settings */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold mb-4">General Settings</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Theme
            </label>
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value as any)}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
            >
              <option value="dark">Dark</option>
              <option value="light">Light</option>
              <option value="system">System</option>
            </select>
          </div>
          
          <div>
            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={settings.autoExecute}
                onChange={(e) => setSettings({ ...settings, autoExecute: e.target.checked })}
                className="w-4 h-4 text-blue-500 bg-gray-900 border-gray-600 rounded focus:ring-blue-500"
              />
              <span className="text-white">Auto-execute profitable opportunities</span>
            </label>
          </div>
        </div>
      </div>
      
      {/* Network Settings */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold mb-4">Network Configuration</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              RPC Endpoint
            </label>
            <input
              type="text"
              value={settings.rpcEndpoint}
              onChange={(e) => setSettings({ ...settings, rpcEndpoint: e.target.value })}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              WebSocket Endpoint
            </label>
            <input
              type="text"
              value={settings.wsEndpoint}
              onChange={(e) => setSettings({ ...settings, wsEndpoint: e.target.value })}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
      </div>
      
      {/* Trading Settings */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold mb-4">Trading Parameters</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Max Gas Price (lamports)
            </label>
            <input
              type="number"
              value={settings.maxGasPrice}
              onChange={(e) => setSettings({ ...settings, maxGasPrice: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Slippage Tolerance (%)
            </label>
            <input
              type="number"
              step="0.1"
              value={settings.slippageTolerance}
              onChange={(e) => setSettings({ ...settings, slippageTolerance: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Priority Fee (SOL)
            </label>
            <input
              type="number"
              step="0.0001"
              value={settings.priorityFee}
              onChange={(e) => setSettings({ ...settings, priorityFee: Number(e.target.value) })}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
      </div>
      
      {/* Save Button */}
      <div className="flex justify-end">
        <button className="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors">
          Save Settings
        </button>
      </div>
    </div>
  );
}