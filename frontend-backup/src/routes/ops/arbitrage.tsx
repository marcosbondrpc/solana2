import { useEffect, useMemo, useState } from 'react';
import { Card } from '../../components/ui/Card';
import KPI from '../../components/kpi/KPI';
import PaginatedTable from '../../components/table/PaginatedTable';
import TimeSeriesChart from '../../components/charts/TimeSeriesChart';
import BarChart from '../../components/charts/BarChart';
import { listOpportunities, getSummary, type Opportunity } from '../../features/arbitrage/api';
import { fmtDate, fmtNumber } from '../../lib/format/number';

type Filter = { asset: string; status: string; minSpread: number };

export default function ArbitragePage() {
  const [summary, setSummary] = useState<{ count_24h: number; filled_24h: number; pnl_24h_usd: number; series_count: any[]; series_avg_spread: any[] } | null>(null);
  const [filter, setFilter] = useState<Filter>({ asset: '', status: '', minSpread: 5 });
  const [rows, setRows] = useState<Opportunity[]>([]);
  const [cursor, setCursor] = useState<string | undefined>();
  const [total, setTotal] = useState<number | undefined>();
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const pageSize = 10;

  const fetchPage = (cur?: string) => {
    setLoading(true);
    listOpportunities({ asset: filter.asset || undefined, status: filter.status || undefined, min_spread_bps: filter.minSpread, cursor: cur, limit: pageSize }).then(p => {
      setRows(p.items);
      setCursor(p.next_cursor);
      setTotal(p.total);
      setLoading(false);
    });
  };

  useEffect(() => {
    getSummary().then(setSummary);
  }, []);

  useEffect(() => {
    setPage(1);
    fetchPage(undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter.asset, filter.status, filter.minSpread]);

  const tableRows = useMemo(() => {
    return rows.map(o => ({
      ts: fmtDate(o.ts),
      asset: o.asset,
      venues: `${o.leg_buy_venue} → ${o.leg_sell_venue}`,
      spread_bps: o.spread_bps,
      size: o.size,
      est_pnl_usd: `$${fmtNumber(o.est_pnl_usd)}`,
      status: o.status
    }));
  }, [rows]);

  const pageChange = (p: number) => {
    setPage(p);
    fetchPage(cursor);
  };

  const spreadBuckets = useMemo(() => {
    const buckets: Record<string, number> = {};
    rows.forEach(o => {
      const b = `${Math.floor(o.spread_bps / 5) * 5}-${Math.floor(o.spread_bps / 5) * 5 + 4}bp`;
      buckets[b] = (buckets[b] || 0) + 1;
    });
    return Object.entries(buckets).map(([name, value]) => ({ name, value }));
  }, [rows]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPI label="Opportunities (24h)" value={summary?.count_24h ?? 0} />
        <KPI label="Filled (24h)" value={summary?.filled_24h ?? 0} />
        <KPI label="PnL (24h, USD)" value={summary?.pnl_24h_usd ?? 0} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Opportunities per Interval">
          <TimeSeriesChart data={summary?.series_count ?? []} height={220} />
        </Card>
        <Card title="Average Spread (bps)">
          <TimeSeriesChart data={summary?.series_avg_spread ?? []} height={220} />
        </Card>
      </div>

      <Card title="Filters">
        <div className="flex flex-wrap items-center gap-2">
          <input
            placeholder="Asset (e.g., SOL/USDC)"
            value={filter.asset}
            onChange={e => setFilter({ ...filter, asset: e.target.value })}
            className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900"
          />
          <select
            value={filter.status}
            onChange={e => setFilter({ ...filter, status: e.target.value })}
            className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900"
          >
            <option value="">All Status</option>
            <option value="open">Open</option>
            <option value="filled">Filled</option>
            <option value="expired">Expired</option>
            <option value="cancelled">Cancelled</option>
          </select>
          <input
            type="number"
            min={0}
            step={1}
            value={filter.minSpread}
            onChange={e => setFilter({ ...filter, minSpread: Number(e.target.value) })}
            className="w-28 px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900"
          />
          <span className="text-sm text-zinc-500 dark:text-zinc-400">Min Spread (bps)</span>
          <button onClick={() => fetchPage(undefined)} className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700">Apply</button>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card title="Spread Distribution">
          <BarChart data={spreadBuckets} height={240} />
        </Card>
        <div className="md:col-span-2">
          <Card title="Opportunities">
            {loading ? (
              <div className="h-32 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" />
            ) : (
              <PaginatedTable
                columns={[
                  { key: 'ts', label: 'Time' },
                  { key: 'asset', label: 'Asset' },
                  { key: 'venues', label: 'Venues' },
                  { key: 'spread_bps', label: 'Spread (bps)' },
                  { key: 'size', label: 'Size' },
                  { key: 'est_pnl_usd', label: 'Est. PnL (USD)' },
                  { key: 'status', label: 'Status' }
                ]}
                rows={tableRows as any[]}
                page={page}
                pageSize={pageSize}
                total={total}
                onPageChange={pageChange}
              />
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}