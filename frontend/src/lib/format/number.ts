export function fmtNumber(n: number): string {
  return Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 2 }).format(n);
}
export function fmtPercent(v: number, digits = 2): string {
  return `${(v * 100).toFixed(digits)}%`;
}
export function fmtDate(ts: number | string): string {
  const d = typeof ts === 'string' ? new Date(ts) : new Date(ts);
  return d.toISOString().replace('T', ' ').slice(0, 19);
}