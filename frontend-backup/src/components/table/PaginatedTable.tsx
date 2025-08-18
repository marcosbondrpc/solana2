import { useMemo } from 'react';

export default function PaginatedTable({
  columns, rows, page, pageSize, total, onPageChange
}: {
  columns: { key: string; label: string }[];
  rows: Record<string, any>[];
  page: number;
  pageSize: number;
  total?: number;
  onPageChange: (p: number) => void;
}) {
  const pages = useMemo(() => {
    const t = total ?? rows.length;
    return Math.max(1, Math.ceil(t / pageSize));
  }, [total, rows.length, pageSize]);

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
        <table className="min-w-full text-sm">
          <thead className="bg-zinc-50 dark:bg-zinc-900/40">
            <tr>
              {columns.map(c => (
                <th key={c.key} className="text-left px-3 py-2 font-medium text-zinc-600 dark:text-zinc-300">{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t border-zinc-200 dark:border-zinc-800">
                {columns.map(c => (
                  <td key={c.key} className="px-3 py-2">{String(r[c.key] ?? '')}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between">
        <div className="text-xs text-zinc-500 dark:text-zinc-400">Page {page} of {pages}</div>
        <div className="flex gap-2">
          <button disabled={page <= 1} onClick={() => onPageChange(page - 1)} className="px-3 py-1.5 rounded border border-zinc-300 dark:border-zinc-700 disabled:opacity-50">Prev</button>
          <button disabled={page >= pages} onClick={() => onPageChange(page + 1)} className="px-3 py-1.5 rounded border border-zinc-300 dark:border-zinc-700 disabled:opacity-50">Next</button>
        </div>
      </div>
    </div>
  );
}