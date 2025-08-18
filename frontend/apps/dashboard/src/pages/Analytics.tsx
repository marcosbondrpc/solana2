export default function AnalyticsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Analytics</h1>
        <p className="text-gray-400 mt-1">Deep dive into MEV performance metrics</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 className="text-xl font-bold mb-4">Profit Distribution</h2>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Arbitrage</span>
                <span className="text-green-400">45%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-green-400 h-2 rounded-full" style={{ width: '45%' }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Liquidations</span>
                <span className="text-blue-400">30%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-blue-400 h-2 rounded-full" style={{ width: '30%' }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">JIT Trading</span>
                <span className="text-purple-400">25%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-purple-400 h-2 rounded-full" style={{ width: '25%' }} />
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 className="text-xl font-bold mb-4">Weekly Performance</h2>
          <div className="grid grid-cols-7 gap-2">
            {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, i) => (
              <div key={day} className="text-center">
                <p className="text-xs text-gray-400 mb-2">{day}</p>
                <div className="bg-gray-700 rounded h-24 relative overflow-hidden">
                  <div 
                    className="absolute bottom-0 w-full bg-gradient-to-t from-purple-500 to-blue-500 rounded"
                    style={{ height: `${Math.random() * 80 + 20}%` }}
                  />
                </div>
                <p className="text-xs mt-1">${Math.floor(Math.random() * 5000)}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold mb-4">Strategy Effectiveness</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-gray-700">
                <th className="pb-3 text-gray-400">Strategy</th>
                <th className="pb-3 text-gray-400">Attempts</th>
                <th className="pb-3 text-gray-400">Success</th>
                <th className="pb-3 text-gray-400">Profit</th>
                <th className="pb-3 text-gray-400">ROI</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              <tr>
                <td className="py-3">Cross-DEX Arbitrage</td>
                <td className="py-3">1,234</td>
                <td className="py-3 text-green-400">89%</td>
                <td className="py-3">$45,678</td>
                <td className="py-3 text-green-400">+234%</td>
              </tr>
              <tr>
                <td className="py-3">Sandwich Attack</td>
                <td className="py-3">567</td>
                <td className="py-3 text-yellow-400">67%</td>
                <td className="py-3">$23,456</td>
                <td className="py-3 text-yellow-400">+156%</td>
              </tr>
              <tr>
                <td className="py-3">Liquidation</td>
                <td className="py-3">89</td>
                <td className="py-3 text-green-400">95%</td>
                <td className="py-3">$78,901</td>
                <td className="py-3 text-green-400">+567%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}