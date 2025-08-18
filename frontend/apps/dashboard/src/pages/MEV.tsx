import { MEVOpportunitiesPanel } from '../components/mev/MEVOpportunitiesPanel';
import { ProfitDashboard } from '../components/mev/ProfitDashboard';
import { LatencyHeatmap } from '../components/mev/LatencyHeatmap';

export default function MEVPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">MEV Opportunities</h1>
        <p className="text-gray-400 mt-1">Monitor and capture maximum extractable value</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MEVOpportunitiesPanel />
        <ProfitDashboard />
      </div>
      
      <LatencyHeatmap />
    </div>
  );
}