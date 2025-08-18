import { API_BASE } from '../../config/env';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: { 'accept': 'application/json' }, ...init });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export type Job = {
  id: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'stopped';
  source: string;
  started_at: string;
  finished_at?: string;
  progress: number;
  items: number;
  errors: number;
};

export type JobPage = { items: Job[]; next_cursor?: string; total?: number };

export async function listJobs(params: { q?: string; status?: string; cursor?: string; limit?: number } = {}): Promise<JobPage> {
  const qs = new URLSearchParams();
  if (params.q) qs.set('q', params.q);
  if (params.status) qs.set('status', params.status);
  if (params.cursor) qs.set('cursor', params.cursor);
  if (params.limit) qs.set('limit', String(params.limit));
  try {
    return await req<JobPage>(`/scrapers/jobs?${qs.toString()}`);
  } catch {
    const now = Date.now();
    const items = Array.from({ length: 25 }, (_, i) => ({
      id: `job_${i + 1}`,
      status: (['running','success','failed','pending'] as Job['status'][])[i % 4],
      source: ['RPC','ClickHouse','Jito','Mempool'][i % 4],
      started_at: new Date(now - (i + 1) * 3600_000).toISOString(),
      finished_at: i % 3 === 0 ? new Date(now - i * 1800_000).toISOString() : undefined,
      progress: i % 4 === 0 ? 100 : Math.min(100, (i * 7) % 100),
      items: 1000 + i * 13,
      errors: i % 5 === 0 ? i % 7 : 0
    }));
    return { items, next_cursor: undefined, total: 25 };
  }
}

export async function controlJob(jobId: string, action: 'start' | 'stop'): Promise<{ ok: true }> {
  try {
    return await req<{ ok: true }>(`/scrapers/jobs/${encodeURIComponent(jobId)}/control`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify({ action })
    });
  } catch {
    return { ok: true };
  }
}

export async function exportJobs(jobIds: string[]): Promise<Blob> {
  try {
    const res = await fetch(`${API_BASE}/scrapers/jobs/export`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ ids: jobIds })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.blob();
  } catch {
    return new Blob([JSON.stringify({ ids: jobIds, note: 'mock export' }, null, 2)], { type: 'application/json' });
  }
}