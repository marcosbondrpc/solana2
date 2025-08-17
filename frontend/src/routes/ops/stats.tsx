import { useEffect, useMemo, useState } from 'react';
import { Card } from '../../components/ui/Card';
import KPI from '../../components/kpi/KPI';
import TimeSeriesChart from '../../components/charts/TimeSeriesChart';
import BarChart from '../../components/charts/BarChart';
import PaginatedTable from '../../components/table/PaginatedTable';
import { getSummary, getTopN, type Summary as SumT } from '../../features/stats/api';
import { fmtNumber } from '../../lib/format/number';

type Range = '1h' | '6h' | '24h' | '7d';
const PRESETS: Range[] = ['1h','6h','24h','7d'];

export default function StatsPage() {
  const [range, setRange] = useState<Range>('24h');
  const [summary, setSummary] = useState<SumT | null>(null);
  const [top, setTop] = useState<{ throughput: any[]; errors: any[]; profit_usd: any[] } | null>(null);
  const [selectedAsset, setSelectedAsset] = useState<string | null>(null);

  const load = () => {
    getSummary(range).then(setSummary);
    getTopN(range).then(setTop);
  };

  useEffect(() => { load(); /* eslint-disable-next-line */ }, [range]);

  const throughputBars = useMemo(() => (top?.throughput || []).map((d: any) => ({ name: d.name, value: d.value })), [top]);
  const errorBars = useMemo(() => (top?.errors || []).map((d: any) => ({ name: d.name, value: d.value })), [top]);
  const profitBars = useMemo(() => (top?.profit_usd || []).map((d: any) => ({ name: d.name, value: d.value })), [top]);

  const tableRows = useMemo(() => {
    const src = top?.throughput || [];
    return src.map((d: any) => ({
      asset: d.name,
      throughput: d.value,
      errors: (top?.errors.find((e: any) => e.name === d.name)?.value) ?? 0,
      profit_usd: (top?.profit_usd.find((p: any) => p.name === d.name)?.value) ?? 0
    }));
  }, [top]);

  return (
    <div className="space-y-4">
      <Card title="Time Range">
        <div className="flex flex-wrap items-center gap-2">
          {PRESETS.map(p => (
            <button
              key={p}
              onClick={() => setRange(p)}
              className={`px-3 py-1.5 rounded border border-zinc-300 dark:border-zinc-700 ${range === p ? 'bg-brand-600 text-white' : ''}`}
              aria-pressed={range === p}
            >
              {p.toUpperCase()}
            </button>
          ))}
          <button onClick={load} className="px-3 py-1.5 rounded border border-zinc-300 dark:border-zinc-700">Refresh</button>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <KPI label="TPS Avg" value={summary?.tps_avg ?? 0} />
        <KPI label="TPS p95" value={summary?.tps_p95 ?? 0} />
        <KPI label="Latency p95 (ms)" value={summary?.latency_p95_ms ?? 0} />
        <KPI label="Errors (24h)" value={summary?.errors_24h ?? 0} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="TPS">
          <TimeSeriesChart data={summary?.series_tps ?? []} height={240} />
        </Card>
        <Card title="Latency (ms)">
          <TimeSeriesChart data={summary?.series_latency_ms ?? []} height={240} />
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card title="Top Throughput">
          <BarChart data={throughputBars} height={260} />
        </Card>
        <Card title="Top Errors">
          <BarChart data={errorBars} height={260} />
        </Card>
        <Card title="Top Profit (USD)">
          <BarChart data={profitBars} height={260} />
        </Card>
      </div>

      <Card title="Assets Overview">
        <PaginatedTable
          columns={[
            { key: 'asset', label: 'Asset' },
            { key: 'throughput', label: 'Throughput' },
            { key: 'errors', label: 'Errors' },
            { key: 'profit_usd', label: 'Profit (USD)' }
          ]}
          rows={(tableRows as any[]).map(r => ({
            ...r,
            asset: (
              <button onClick={() => setSelectedAsset(String((r as any).asset))} className="underline">
                {(r as any).asset}
              </button>
            ),
            profit_usd: `$${fmtNumber((r as any).profit_usd)}`
          }))}
          page={1}
          pageSize={Math.max(10, (tableRows as any[]).length)}
          total={(tableRows as any[]).length}
          onPageChange={() => {}}
        />
      </Card>

      {selectedAsset ? (
        <Card title={`Drilldown: ${selectedAsset}`}>
          <div className="text-sm text-zinc-500 dark:text-zinc-400">Preset drilldown placeholder for {selectedAsset}. Add dedicated series and tables when real endpoints are wired.</div>
          <div className="mt-3">
            <button onClick={() => setSelectedAsset(null)} className="px-3 py-1.5 rounded border border-zinc-300 dark:border-zinc-700">Close</button>
          </div>
        </Card>
      ) : null}
    </div>
  );
}