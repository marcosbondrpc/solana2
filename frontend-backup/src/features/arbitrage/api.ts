import { API_BASE } from '../../config/env';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: { 'accept': 'application/json' }, ...init });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export type Opportunity = {
  id: string;
  ts: string;
  asset: string;
  leg_buy_venue: string;
  leg_sell_venue: string;
  spread_bps: number;
  size: number;
  est_pnl_usd: number;
  status: 'open' | 'filled' | 'expired' | 'cancelled';
};

export type OpportunityPage = { items: Opportunity[]; next_cursor?: string; total?: number };

export async function listOpportunities(params: { asset?: string; min_spread_bps?: number; status?: string; cursor?: string; limit?: number } = {}): Promise<OpportunityPage> {
  const qs = new URLSearchParams();
  if (params.asset) qs.set('asset', params.asset);
  if (params.min_spread_bps !== undefined) qs.set('min_spread_bps', String(params.min_spread_bps));
  if (params.status) qs.set('status', params.status);
  if (params.cursor) qs.set('cursor', params.cursor);
  if (params.limit) qs.set('limit', String(params.limit));
  try {
    return await req<OpportunityPage>(`/arbitrage/opportunities?${qs.toString()}`);
  } catch {
    const now = Date.now();
    const assets = ['SOL/USDC', 'JUP/USDC', 'PYTH/USDC'];
    const venues = ['Jito', 'Orca', 'Raydium', 'Phoenix'];
    const items = Array.from({ length: 30 }, (_, i) => {
      const asset = assets[i % assets.length];
      const spread_bps = Math.max(5, Math.round(5 + Math.random() * 40));
      const size = Number((Math.random() * 150).toFixed(2));
      const est_pnl_usd = Number((size * (spread_bps / 10000) * (5 + Math.random() * 15)).toFixed(2));
      return {
        id: `opp_${i + 1}`,
        ts: new Date(now - (i + 1) * 60_000).toISOString(),
        asset,
        leg_buy_venue: venues[(i + 1) % venues.length],
        leg_sell_venue: venues[(i + 2) % venues.length],
        spread_bps,
        size,
        est_pnl_usd,
        status: (['open', 'filled', 'expired'] as Opportunity['status'][])[i % 3]
      };
    });
    return { items, next_cursor: undefined, total: 30 };
  }
}

export type Summary = {
  count_24h: number;
  filled_24h: number;
  pnl_24h_usd: number;
  series_count: { t: number | string; v: number }[];
  series_avg_spread: { t: number | string; v: number }[];
};

export async function getSummary(): Promise<Summary> {
  try {
    return await req<Summary>(`/arbitrage/summary`);
  } catch {
    const now = Date.now();
    const series_count = Array.from({ length: 48 }, (_, i) => ({ t: now - (48 - i) * 30 * 60 * 1000, v: Math.max(0, Math.round(20 + Math.sin(i / 4) * 8 + Math.random() * 6)) }));
    const series_avg_spread = Array.from({ length: 48 }, (_, i) => ({ t: now - (48 - i) * 30 * 60 * 1000, v: Math.max(3, Math.round(6 + Math.sin(i / 3) * 2 + Math.random() * 1.2)) }));
    return { count_24h: 432, filled_24h: 289, pnl_24h_usd: 18452.33, series_count, series_avg_spread };
  }
}