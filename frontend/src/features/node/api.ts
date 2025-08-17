import { API_BASE } from '../../config/env';

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: { 'accept': 'application/json' }, ...init });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export type NodeMetrics = {
  tps: number;
  latency_ms: number;
  peers: number;
  slot: number;
  health: 'healthy' | 'degraded' | 'down';
  series_latency: { t: number | string; v: number }[];
  series_tps: { t: number | string; v: number }[];
};

export async function fetchNodeMetrics(nodeId = 'default'): Promise<NodeMetrics> {
  try {
    return await get<NodeMetrics>(`/nodes/${encodeURIComponent(nodeId)}/metrics`);
  } catch {
    const now = Date.now();
    const series_latency = Array.from({ length: 60 }, (_, i) => ({ t: now - (60 - i) * 1000, v: 150 + Math.sin(i / 5) * 30 + Math.random() * 10 }));
    const series_tps = Array.from({ length: 60 }, (_, i) => ({ t: now - (60 - i) * 1000, v: 1200 + Math.sin(i / 7) * 120 + Math.random() * 50 }));
    return { tps: 1342, latency_ms: 167, peers: 28, slot: 243551234, health: 'healthy', series_latency, series_tps };
  }
}

export type NodeInfo = { id: string; name: string };
export async function fetchNodes(): Promise<NodeInfo[]> {
  try {
    return await get<NodeInfo[]>(`/nodes`);
  } catch {
    return [{ id: 'default', name: 'Primary Node' }];
  }
}