import { useEffect, useMemo, useState } from 'react';
import { Card } from '../../components/ui/Card';
import KPI from '../../components/kpi/KPI';
import TimeSeriesChart from '../../components/charts/TimeSeriesChart';
import PaginatedTable from '../../components/table/PaginatedTable';
import { getSummary, listBundles, listAlerts, type Bundle, type Alert } from '../../features/mev/api';
import { fmtDate, fmtNumber } from '../../lib/format/number';

type Filters = { strategy: string; status: string; severity: string; topic: string };

export default function MevPage() {
  const [summary, setSummary] = useState<{ bundles_24h: number; profit_24h_usd: number; alerts_24h: number; series_profit_usd: any[]; series_bundles: any[] } | null>(null);
  const [filters, setFilters] = useState<Filters>({ strategy: '', status: '', severity: '', topic: '' });
  const [bundles, setBundles] = useState<Bundle[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loadingBundles, setLoadingBundles] = useState(true);
  const [loadingAlerts, setLoadingAlerts] = useState(true);
  const [pageB, setPageB] = useState(1);
  const [pageA, setPageA] = useState(1);
  const pageSize = 10;

  const fetchBundles = () => {
    setLoadingBundles(true);
    listBundles({ strategy: filters.strategy || undefined, status: filters.status || undefined, limit: pageSize }).then(p => {
      setBundles(p.items);
      setLoadingBundles(false);
    });
  };
  const fetchAlerts = () => {
    setLoadingAlerts(true);
    listAlerts({ severity: filters.severity || undefined, topic: filters.topic || undefined, limit: pageSize }).then(p => {
      setAlerts(p.items);
      setLoadingAlerts(false);
    });
  };

  useEffect(() => { getSummary().then(setSummary); }, []);
  useEffect(() => { setPageB(1); fetchBundles(); /* eslint-disable-next-line */ }, [filters.strategy, filters.status]);
  useEffect(() => { setPageA(1); fetchAlerts(); /* eslint-disable-next-line */ }, [filters.severity, filters.topic]);

  const bundleRows = useMemo(() => bundles.map(b => ({
    ts: fmtDate(b.ts),
    id: b.id,
    strategy: b.strategy,
    txs: b.txs,
    profit_usd: `$${fmtNumber(b.profit_usd)}`,
    status: b.status
  })), [bundles]);

  const alertRows = useMemo(() => alerts.map(a => ({
    ts: fmtDate(a.ts),
    id: a.id,
    severity: a.severity,
    topic: a.topic,
    message: a.message
  })), [alerts]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPI label="Bundles (24h)" value={summary?.bundles_24h ?? 0} />
        <KPI label="Profit (24h, USD)" value={summary?.profit_24h_usd ?? 0} />
        <KPI label="Alerts (24h)" value={summary?.alerts_24h ?? 0} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Profit (USD)">
          <TimeSeriesChart data={summary?.series_profit_usd ?? []} height={220} />
        </Card>
        <Card title="Bundles Count">
          <TimeSeriesChart data={summary?.series_bundles ?? []} height={220} />
        </Card>
      </div>

      <Card title="Filters">
        <div className="flex flex-wrap items-center gap-2">
          <select value={filters.strategy} onChange={e => setFilters({ ...filters, strategy: e.target.value })} className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900">
            <option value="">All Strategies</option>
            <option value="sandwich">Sandwich</option>
            <option value="arb">Arbitrage</option>
            <option value="liquidation">Liquidation</option>
            <option value="backrun">Backrun</option>
            <option value="other">Other</option>
          </select>
          <select value={filters.status} onChange={e => setFilters({ ...filters, status: e.target.value })} className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900">
            <option value="">All Status</option>
            <option value="observed">Observed</option>
            <option value="submitted">Submitted</option>
            <option value="landed">Landed</option>
            <option value="reverted">Reverted</option>
          </select>
          <select value={filters.severity} onChange={e => setFilters({ ...filters, severity: e.target.value })} className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900">
            <option value="">All Severities</option>
            <option value="critical">Critical</option>
            <option value="warning">Warning</option>
            <option value="info">Info</option>
          </select>
          <input placeholder="Topic" value={filters.topic} onChange={e => setFilters({ ...filters, topic: e.target.value })} className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900" />
          <button onClick={() => { fetchBundles(); fetchAlerts(); }} className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700">Apply</button>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Bundles">
          {loadingBundles ? (
            <div className="h-32 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" />
          ) : (
            <PaginatedTable
              columns={[
                { key: 'ts', label: 'Time' },
                { key: 'id', label: 'Bundle ID' },
                { key: 'strategy', label: 'Strategy' },
                { key: 'txs', label: 'Txs' },
                { key: 'profit_usd', label: 'Profit (USD)' },
                { key: 'status', label: 'Status' }
              ]}
              rows={bundleRows as any[]}
              page={pageB}
              pageSize={pageSize}
              total={bundles.length}
              onPageChange={setPageB}
            />
          )}
        </Card>

        <Card title="Alerts">
          {loadingAlerts ? (
            <div className="h-32 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" />
          ) : (
            <PaginatedTable
              columns={[
                { key: 'ts', label: 'Time' },
                { key: 'id', label: 'ID' },
                { key: 'severity', label: 'Severity' },
                { key: 'topic', label: 'Topic' },
                { key: 'message', label: 'Message' }
              ]}
              rows={alertRows as any[]}
              page={pageA}
              pageSize={pageSize}
              total={alerts.length}
              onPageChange={setPageA}
            />
          )}
        </Card>
      </div>
    </div>
  );
}