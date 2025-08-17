import { useEffect, useMemo, useState } from 'react';
import { Card } from '../../components/ui/Card';
import KPI from '../../components/kpi/KPI';
import PaginatedTable from '../../components/table/PaginatedTable';
import { listJobs, controlJob, exportJobs, type Job } from '../../features/scraper/api';
import { fmtDate } from '../../lib/format/number';

type Filter = { q: string; status: string };

export default function ScraperPage() {
  const [filter, setFilter] = useState<Filter>({ q: '', status: '' });
  const [jobs, setJobs] = useState<Job[]>([]);
  const [cursor, setCursor] = useState<string | undefined>();
  const [total, setTotal] = useState<number | undefined>();
  const [page, setPage] = useState(1);
  const pageSize = 10;
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Record<string, boolean>>({});

  const fetchPage = (cur?: string) => {
    setLoading(true);
    listJobs({ q: filter.q, status: filter.status, cursor: cur, limit: pageSize }).then(p => {
      setJobs(p.items);
      setCursor(p.next_cursor);
      setTotal(p.total);
      setLoading(false);
      setSelected({});
    });
  };

  useEffect(() => {
    setPage(1);
    fetchPage(undefined);
  }, [filter.q, filter.status]);

  const rows = useMemo(() => {
    return jobs.map(j => ({
      select: (
        <input type="checkbox" checked={!!selected[j.id]} onChange={e => setSelected(s => ({ ...s, [j.id]: e.target.checked }))} />
      ),
      id: j.id,
      status: j.status,
      source: j.source,
      started_at: fmtDate(j.started_at),
      finished_at: j.finished_at ? fmtDate(j.finished_at) : '-',
      progress: `${j.progress}%`,
      items: j.items,
      errors: j.errors,
      actions: (
        <div className="flex gap-2">
          {j.status === 'running'
            ? <button onClick={() => controlJob(j.id, 'stop')} className="px-2 py-1 rounded border border-zinc-300 dark:border-zinc-700">Stop</button>
            : <button onClick={() => controlJob(j.id, 'start')} className="px-2 py-1 rounded border border-zinc-300 dark:border-zinc-700">Start</button>
          }
        </div>
      )
    }));
  }, [jobs, selected]);

  const selectedIds = useMemo(() => Object.entries(selected).filter(([, v]) => v).map(([k]) => k), [selected]);

  const exportSelected = async () => {
    const blob = await exportJobs(selectedIds);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `scraper-jobs-export-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const pageChange = (p: number) => {
    setPage(p);
    fetchPage(cursor);
  };

  const running = jobs.filter(j => j.status === 'running').length;
  const success24h = jobs.filter(j => j.status === 'success').length;
  const failed24h = jobs.filter(j => j.status === 'failed').length;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPI label="Running Jobs" value={running} />
        <KPI label="Completed (24h)" value={success24h} />
        <KPI label="Failed (24h)" value={failed24h} />
      </div>

      <Card title="Filters">
        <div className="flex flex-wrap items-center gap-2">
          <input
            placeholder="Search..."
            value={filter.q}
            onChange={e => setFilter({ ...filter, q: e.target.value })}
            className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900"
          />
          <select
            value={filter.status}
            onChange={e => setFilter({ ...filter, status: e.target.value })}
            className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900"
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="success">Success</option>
            <option value="failed">Failed</option>
            <option value="stopped">Stopped</option>
          </select>
          <button onClick={() => fetchPage(undefined)} className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700">Apply</button>
          <button disabled={!selectedIds.length} onClick={exportSelected} className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 disabled:opacity-50">Export Selected</button>
        </div>
      </Card>

      <Card title="Jobs">
        {loading ? (
          <div className="h-32 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" />
        ) : (
          <PaginatedTable
            columns={[
              { key: 'select', label: '' },
              { key: 'id', label: 'ID' },
              { key: 'status', label: 'Status' },
              { key: 'source', label: 'Source' },
              { key: 'started_at', label: 'Started' },
              { key: 'finished_at', label: 'Finished' },
              { key: 'progress', label: 'Progress' },
              { key: 'items', label: 'Items' },
              { key: 'errors', label: 'Errors' },
              { key: 'actions', label: 'Actions' }
            ]}
            rows={rows}
            page={page}
            pageSize={pageSize}
            total={total}
            onPageChange={pageChange}
          />
        )}
      </Card>
    </div>
  );
}