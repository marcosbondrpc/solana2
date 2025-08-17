import { ReactNode } from 'react';

export default function SimpleTable({ columns, rows }: { columns: { key: string; label: string }[]; rows: Record<string, ReactNode | string | number | null | undefined>[] }) {
  return (
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
              {columns.map(c => {
                const v = r[c.key] as ReactNode | string | number | null | undefined;
                return <td key={c.key} className="px-3 py-2">{v as ReactNode}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}