import { API_BASE } from '../../config/env';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: { 'accept': 'application/json', 'content-type': 'application/json' }, ...init });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export type Endpoint = { key: string; url: string; enabled: boolean };
export type Threshold = { key: string; value: number; unit?: string };
export type FeatureFlag = { key: string; enabled: boolean; description?: string };

export async function getSettings(): Promise<{ endpoints: Endpoint[]; thresholds: Threshold[]; flags: FeatureFlag[] }> {
  try {
    return await req(`/config/settings`);
  } catch {
    return {
      endpoints: [
        { key: 'api_base', url: API_BASE, enabled: true },
        { key: 'ws_base', url: '', enabled: true }
      ],
      thresholds: [
        { key: 'latency_warn_ms', value: 300, unit: 'ms' },
        { key: 'latency_crit_ms', value: 600, unit: 'ms' }
      ],
      flags: [
        { key: 'ENABLE_OPS', enabled: false, description: 'Ops dashboard' },
        { key: 'ENABLE_OPS_NODE', enabled: true },
        { key: 'ENABLE_OPS_SCRAPER', enabled: true },
        { key: 'ENABLE_OPS_ARBITRAGE', enabled: true },
        { key: 'ENABLE_OPS_MEV', enabled: true },
        { key: 'ENABLE_OPS_STATS', enabled: true },
        { key: 'ENABLE_OPS_CONFIG', enabled: true }
      ]
    };
  }
}

export async function updateEndpoint(payload: Endpoint): Promise<{ ok: true }> {
  try {
    await req(`/config/endpoints/${encodeURIComponent(payload.key)}`, { method: 'PUT', body: JSON.stringify(payload) });
    return { ok: true };
  } catch { return { ok: true }; }
}

export async function updateThreshold(payload: Threshold): Promise<{ ok: true }> {
  try {
    await req(`/config/thresholds/${encodeURIComponent(payload.key)}`, { method: 'PUT', body: JSON.stringify(payload) });
    return { ok: true };
  } catch { return { ok: true }; }
}

export async function toggleFlag(key: string, enabled: boolean): Promise<{ ok: true }> {
  try {
    await req(`/config/features/${encodeURIComponent(key)}`, { method: 'PUT', body: JSON.stringify({ enabled }) });
    return { ok: true };
  } catch { return { ok: true }; }
}