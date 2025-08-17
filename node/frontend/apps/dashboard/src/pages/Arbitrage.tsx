import { ArbitrageScanner } from '../components/mev/ArbitrageScanner';

export default function ArbitragePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Arbitrage Scanner</h1>
        <p className="text-gray-400 mt-1">Real-time cross-DEX arbitrage opportunities</p>
      </div>
      
      <ArbitrageScanner />
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold mb-4">Active Pairs</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">SOL/USDC</span>
              <span className="text-green-400">+0.23%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">RAY/USDC</span>
              <span className="text-green-400">+0.15%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">BONK/SOL</span>
              <span className="text-red-400">-0.08%</span>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold mb-4">DEX Coverage</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Raydium</span>
              <span className="text-blue-400">Connected</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Orca</span>
              <span className="text-blue-400">Connected</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Phoenix</span>
              <span className="text-yellow-400">Syncing</span>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold mb-4">Performance</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Opportunities/min</span>
              <span className="text-purple-400">127</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Execution Rate</span>
              <span className="text-purple-400">82%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Profit</span>
              <span className="text-purple-400">$45.23</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}