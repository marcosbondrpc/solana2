import { fmtNumber, fmtPercent } from '../../lib/format/number';

export default function KPI({ label, value, delta }: { label: string; value: number; delta?: number }) {
  const sign = delta === undefined ? '' : delta >= 0 ? '+' : '';
  const tone = delta === undefined ? '' : delta >= 0 ? 'text-emerald-600' : 'text-rose-600';
  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-4 bg-white/70 dark:bg-zinc-900/50">
      <div className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{fmtNumber(value)}</div>
      {delta !== undefined ? <div className={`mt-1 text-xs ${tone}`}>{sign}{fmtPercent(delta)}</div> : null}
    </div>
  );
}