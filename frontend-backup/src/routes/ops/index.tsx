import { Card } from '../../components/ui/Card';
import KPI from '../../components/kpi/KPI';
import Sparkline from '../../components/charts/Sparkline';
import TimeSeriesChart from '../../components/charts/TimeSeriesChart';

export default function OpsHome() {
  const spark = Array.from({ length: 40 }, () => 50 + Math.random() * 20 - 10).map((v, i, a) => Math.max(0, (i ? a[i - 1] : v) + (Math.random() - 0.5) * 5));
  const now = Date.now();
  const series = Array.from({ length: 60 }, (_, i) => ({ t: now - (60 - i) * 60000, v: 1000 + Math.sin(i / 6) * 120 + Math.random() * 40 }));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPI label="Throughput (tps)" value={1342} delta={0.042} />
        <KPI label="Opportunities" value={287} delta={-0.018} />
        <KPI label="Alerts (24h)" value={3} />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        <Card title="Throughput Sparkline">
          <Sparkline data={spark} />
        </Card>
        <Card title="Latency Timeseries" className="xl:col-span-2">
          <TimeSeriesChart data={series} height={240} />
        </Card>
        <Card title="System Health">
          <div className="text-sm text-zinc-500">Live status and KPIs</div>
        </Card>
      </div>
    </div>
  );
}