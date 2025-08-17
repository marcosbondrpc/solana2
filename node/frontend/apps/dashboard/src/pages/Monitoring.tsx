export default function MonitoringPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">System Monitoring</h1>
        <p className="text-gray-400 mt-1">Real-time infrastructure and performance monitoring</p>
      </div>
      
      {/* System Health Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">RPC Health</span>
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
          </div>
          <p className="text-2xl font-bold">Healthy</p>
          <p className="text-xs text-gray-500 mt-1">Latency: 23ms</p>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">WebSocket</span>
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
          </div>
          <p className="text-2xl font-bold">Connected</p>
          <p className="text-xs text-gray-500 mt-1">Messages: 1.2k/s</p>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Mempool</span>
            <span className="w-2 h-2 bg-yellow-400 rounded-full" />
          </div>
          <p className="text-2xl font-bold">Moderate</p>
          <p className="text-xs text-gray-500 mt-1">Queue: 342 txs</p>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">CPU Usage</span>
            <span className="w-2 h-2 bg-green-400 rounded-full" />
          </div>
          <p className="text-2xl font-bold">42%</p>
          <p className="text-xs text-gray-500 mt-1">4 cores active</p>
        </div>
      </div>
      
      {/* Network Metrics */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold mb-4">Network Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-gray-400 mb-2">Transaction Rate</p>
            <div className="h-32 bg-gray-900 rounded flex items-end p-2">
              {[...Array(20)].map((_, i) => (
                <div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-blue-500 to-blue-400 mx-0.5 rounded-t"
                  style={{ height: `${Math.random() * 100}%` }}
                />
              ))}
            </div>
          </div>
          
          <div>
            <p className="text-sm text-gray-400 mb-2">Block Production</p>
            <div className="h-32 bg-gray-900 rounded flex items-end p-2">
              {[...Array(20)].map((_, i) => (
                <div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-green-500 to-green-400 mx-0.5 rounded-t"
                  style={{ height: `${Math.random() * 100}%` }}
                />
              ))}
            </div>
          </div>
          
          <div>
            <p className="text-sm text-gray-400 mb-2">MEV Capture Rate</p>
            <div className="h-32 bg-gray-900 rounded flex items-end p-2">
              {[...Array(20)].map((_, i) => (
                <div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-purple-500 to-purple-400 mx-0.5 rounded-t"
                  style={{ height: `${Math.random() * 100}%` }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
      
      {/* Event Log */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold mb-4">Recent Events</h2>
        <div className="space-y-2 font-mono text-sm">
          <div className="flex items-start space-x-3">
            <span className="text-gray-500">[08:00:01]</span>
            <span className="text-green-400">INFO</span>
            <span className="text-gray-300">Successfully connected to Solana mainnet RPC</span>
          </div>
          <div className="flex items-start space-x-3">
            <span className="text-gray-500">[07:59:45]</span>
            <span className="text-blue-400">DEBUG</span>
            <span className="text-gray-300">WebSocket reconnection attempt successful</span>
          </div>
          <div className="flex items-start space-x-3">
            <span className="text-gray-500">[07:59:32]</span>
            <span className="text-yellow-400">WARN</span>
            <span className="text-gray-300">High mempool congestion detected</span>
          </div>
          <div className="flex items-start space-x-3">
            <span className="text-gray-500">[07:58:12]</span>
            <span className="text-green-400">INFO</span>
            <span className="text-gray-300">MEV bundle submitted successfully</span>
          </div>
          <div className="flex items-start space-x-3">
            <span className="text-gray-500">[07:57:55]</span>
            <span className="text-purple-400">METRIC</span>
            <span className="text-gray-300">Arbitrage opportunity captured: +$124.56</span>
          </div>
        </div>
      </div>
    </div>
  );
}