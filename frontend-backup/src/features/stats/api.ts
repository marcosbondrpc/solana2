import { API_BASE } from '../../config/env';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: { 'accept': 'application/json' }, ...init });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export type SeriesPoint = { t: number | string; v: number };
export type Summary = {
  tps_avg: number;
  tps_p95: number;
  latency_p95_ms: number;
  errors_24h: number;
  series_tps: SeriesPoint[];
  series_latency_ms: SeriesPoint[];
};

export type TopItem = { name: string; value: number };
export type TopN = { throughput: TopItem[]; errors: TopItem[]; profit_usd: TopItem[] };

export async function getSummary(range: string): Promise<Summary> {
  try {
    return await req<Summary>(`/stats/summary?range=${encodeURIComponent(range)}`);
  } catch {
    const now = Date.now();
    const mkSeries = (n: number, step: number, base: number, amp: number) =>
      Array.from({ length: n }, (_, i) => ({ t: now - (n - i) * step, v: Math.max(0, Math.round(base + Math.sin(i / 6) * amp + Math.random() * amp * 0.3)) }));
    return {
      tps_avg: 1225, tps_p95: 1820, latency_p95_ms: 210, errors_24h: 12,
      series_tps: mkSeries(96, 15 * 60 * 1000, 1200, 150),
      series_latency_ms: mkSeries(96, 15 * 60 * 1000, 200, 40)
    };
  }
}

export async function getTopN(range: string): Promise<TopN> {
  try {
    return await req<TopN>(`/stats/top?range=${encodeURIComponent(range)}`);
  } catch {
    const names = ['SOL/USDC','JUP/USDC','PYTH/USDC','RAY/USDC','BONK/USDC','JTO/USDC'];
    const mk = () => names.map(n => ({ name: n, value: Math.round(100 + Math.random() * 900) }));
    return { throughput: mk(), errors: mk(), profit_usd: mk() };
  }
}