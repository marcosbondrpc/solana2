import { JitoBundleTracker } from '../components/mev/JitoBundleTracker';

export default function JitoPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Jito Bundle Management</h1>
        <p className="text-gray-400 mt-1">Monitor and optimize Jito bundle submissions</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-r from-green-500/10 to-green-600/10 border border-green-500/20 rounded-lg p-4">
          <p className="text-green-400 text-sm">Bundle Success Rate</p>
          <p className="text-3xl font-bold text-white mt-2">87.3%</p>
        </div>
        <div className="bg-gradient-to-r from-blue-500/10 to-blue-600/10 border border-blue-500/20 rounded-lg p-4">
          <p className="text-blue-400 text-sm">Avg Bundle Size</p>
          <p className="text-3xl font-bold text-white mt-2">4.2 TX</p>
        </div>
        <div className="bg-gradient-to-r from-purple-500/10 to-purple-600/10 border border-purple-500/20 rounded-lg p-4">
          <p className="text-purple-400 text-sm">Tips Paid Today</p>
          <p className="text-3xl font-bold text-white mt-2">12.5 SOL</p>
        </div>
      </div>
      
      <JitoBundleTracker />
      
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h2 className="text-xl font-bold mb-4">Bundle History</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-gray-700">
                <th className="pb-3 text-gray-400">Bundle ID</th>
                <th className="pb-3 text-gray-400">Transactions</th>
                <th className="pb-3 text-gray-400">Tip</th>
                <th className="pb-3 text-gray-400">Status</th>
                <th className="pb-3 text-gray-400">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {[...Array(5)].map((_, i) => (
                <tr key={i}>
                  <td className="py-3 font-mono text-sm">
                    {`0x${Math.random().toString(16).slice(2, 10)}...`}
                  </td>
                  <td className="py-3">{Math.floor(Math.random() * 5) + 1}</td>
                  <td className="py-3">{(Math.random() * 0.1).toFixed(4)} SOL</td>
                  <td className="py-3">
                    <span className={`px-2 py-1 rounded text-xs ${
                      i === 0 ? 'bg-green-500/20 text-green-400' : 
                      i === 3 ? 'bg-red-500/20 text-red-400' : 
                      'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {i === 0 ? 'Success' : i === 3 ? 'Failed' : 'Pending'}
                    </span>
                  </td>
                  <td className="py-3 text-gray-400">{i * 2 + 1}m ago</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}