import { useEffect, useMemo, useState } from 'react';
import { Card } from '../../components/ui/Card';
import KPI from '../../components/kpi/KPI';
import TimeSeriesChart from '../../components/charts/TimeSeriesChart';
import Sparkline from '../../components/charts/Sparkline';
import SimpleTable from '../../components/table/SimpleTable';
import { fetchNodeMetrics, fetchNodes, type NodeMetrics, type NodeInfo } from '../../features/node/api';

export default function NodePage() {
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [selected, setSelected] = useState<string>('default');
  const [data, setData] = useState<NodeMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    fetchNodes().then(ns => {
      if (!alive) return;
      setNodes(ns);
      if (ns.length && selected === 'default') setSelected(ns[0].id);
    });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    fetchNodeMetrics(selected).then(m => {
      if (!alive) return;
      setData(m);
      setLoading(false);
    });
    const iv = setInterval(() => {
      fetchNodeMetrics(selected).then(m => alive && setData(m));
    }, 10000);
    return () => { alive = false; clearInterval(iv); };
  }, [selected]);

  const peers = useMemo(() => {
    const n = data?.peers ?? 0;
    return Array.from({ length: Math.min(n, 10) }, (_, i) => ({ peer: `Peer-${i + 1}`, state: 'connected', rtt_ms: Math.round(20 + Math.random() * 10) }));
  }, [data]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <label className="text-sm text-zinc-600 dark:text-zinc-300">Node</label>
        <select
          className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900"
          value={selected}
          onChange={e => setSelected(e.target.value)}
        >
          {nodes.map(n => <option key={n.id} value={n.id}>{n.name}</option>)}
        </select>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <KPI label="TPS" value={data?.tps ?? 0} />
        <KPI label="Latency (ms)" value={data?.latency_ms ?? 0} />
        <KPI label="Peers" value={data?.peers ?? 0} />
        <KPI label="Slot" value={data?.slot ?? 0} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card title="Latency (ms)">
          {loading || !data ? <div className="h-40 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" /> : (
            <TimeSeriesChart data={data.series_latency} height={240} />
          )}
        </Card>
        <Card title="TPS">
          {loading || !data ? <div className="h-40 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" /> : (
            <TimeSeriesChart data={data.series_tps} height={240} />
          )}
        </Card>
        <Card title="Latency Sparkline (recent)">
          {loading || !data ? <div className="h-40 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" /> : (
            <Sparkline data={data.series_latency.map(p => p.v)} />
          )}
        </Card>
      </div>

      <Card title="Peers">
        <SimpleTable
          columns={[
            { key: 'peer', label: 'Peer' },
            { key: 'state', label: 'State' },
            { key: 'rtt_ms', label: 'RTT (ms)' }
          ]}
          rows={peers}
        />
      </Card>
    </div>
  );
}