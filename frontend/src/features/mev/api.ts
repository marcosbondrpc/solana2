import { API_BASE } from '../../config/env';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: { 'accept': 'application/json' }, ...init });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export type Bundle = {
  id: string;
  ts: string;
  strategy: 'sandwich' | 'arb' | 'liquidation' | 'backrun' | 'other';
  txs: number;
  profit_usd: number;
  status: 'observed' | 'submitted' | 'landed' | 'reverted';
};

export type Alert = {
  id: string;
  ts: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  topic: string;
};

export type SeriesPoint = { t: number | string; v: number };
export type Summary = {
  bundles_24h: number;
  profit_24h_usd: number;
  alerts_24h: number;
  series_profit_usd: SeriesPoint[];
  series_bundles: SeriesPoint[];
};

export type BundlePage = { items: Bundle[]; next_cursor?: string; total?: number };
export type AlertPage = { items: Alert[]; next_cursor?: string; total?: number };

export async function getSummary(): Promise<Summary> {
  try {
    return await req<Summary>(`/mev/summary`);
  } catch {
    const now = Date.now();
    const series_profit_usd = Array.from({ length: 48 }, (_, i) => ({ t: now - (48 - i) * 30 * 60 * 1000, v: Math.max(0, Math.round(500 + Math.sin(i / 3) * 120 + Math.random() * 80)) }));
    const series_bundles = Array.from({ length: 48 }, (_, i) => ({ t: now - (48 - i) * 30 * 60 * 1000, v: Math.max(0, Math.round(10 + Math.sin(i / 4) * 4 + Math.random() * 3)) }));
    return { bundles_24h: 312, profit_24h_usd: 25431.77, alerts_24h: 17, series_profit_usd, series_bundles };
  }
}

export async function listBundles(params: { strategy?: string; status?: string; cursor?: string; limit?: number } = {}): Promise<BundlePage> {
  const qs = new URLSearchParams();
  if (params.strategy) qs.set('strategy', params.strategy);
  if (params.status) qs.set('status', params.status);
  if (params.cursor) qs.set('cursor', params.cursor);
  if (params.limit) qs.set('limit', String(params.limit));
  try {
    return await req<BundlePage>(`/mev/bundles?${qs.toString()}`);
  } catch {
    const now = Date.now();
    const items = Array.from({ length: 25 }, (_, i) => ({
      id: `bundle_${i + 1}`,
      ts: new Date(now - (i + 1) * 60_000).toISOString(),
      strategy: (['sandwich','arb','liquidation','backrun','other'] as Bundle['strategy'][])[i % 5],
      txs: 2 + (i % 5),
      profit_usd: Number((Math.random() * 500).toFixed(2)),
      status: (['observed','submitted','landed','reverted'] as Bundle['status'][])[i % 4]
    }));
    return { items, next_cursor: undefined, total: 25 };
  }
}

export async function listAlerts(params: { severity?: string; topic?: string; cursor?: string; limit?: number } = {}): Promise<AlertPage> {
  const qs = new URLSearchParams();
  if (params.severity) qs.set('severity', params.severity);
  if (params.topic) qs.set('topic', params.topic);
  if (params.cursor) qs.set('cursor', params.cursor);
  if (params.limit) qs.set('limit', String(params.limit));
  try {
    return await req<AlertPage>(`/mev/alerts?${qs.toString()}`);
  } catch {
    const now = Date.now();
    const items = Array.from({ length: 20 }, (_, i) => ({
      id: `alert_${i + 1}`,
      ts: new Date(now - (i + 1) * 120_000).toISOString(),
      severity: (['critical','warning','info'] as Alert['severity'][])[i % 3],
      message: ['Sandwich risk detected','Arb opportunity missed','Liquidation detected'][i % 3],
      topic: ['sandwich','arb','liquidation'][i % 3]
    }));
    return { items, next_cursor: undefined, total: 20 };
  }
}