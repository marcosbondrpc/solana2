import { useEffect, useMemo, useState } from 'react';
import { Card } from '../../components/ui/Card';
import SimpleTable from '../../components/table/SimpleTable';
import { getSettings, updateEndpoint, updateThreshold, toggleFlag, type Endpoint, type Threshold, type FeatureFlag } from '../../features/config/api';

export default function ConfigPage() {
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [thresholds, setThresholds] = useState<Threshold[]>([]);
  const [flags, setFlags] = useState<FeatureFlag[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = () => {
    setLoading(true);
    getSettings().then(s => {
      setEndpoints(s.endpoints);
      setThresholds(s.thresholds);
      setFlags(s.flags);
      setLoading(false);
    });
  };

  useEffect(() => { load(); }, []);

  const endpointRows = useMemo(() => endpoints.map(e => ({
    key: e.key,
    url: (
      <input
        className="w-full px-2 py-1 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900"
        defaultValue={e.url}
        onBlur={ev => setEndpoints(prev => prev.map(p => p.key === e.key ? { ...p, url: (ev.target as HTMLInputElement).value } : p))}
      />
    ),
    enabled: (
      <input
        type="checkbox"
        checked={e.enabled}
        onChange={ev => setEndpoints(prev => prev.map(p => p.key === e.key ? { ...p, enabled: (ev.target as HTMLInputElement).checked } : p))}
      />
    )
  })), [endpoints]);

  const thresholdRows = useMemo(() => thresholds.map(t => ({
    key: t.key,
    value: (
      <input
        type="number"
        className="w-32 px-2 py-1 rounded border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900"
        defaultValue={t.value}
        onBlur={ev => setThresholds(prev => prev.map(p => p.key === t.key ? { ...p, value: Number((ev.target as HTMLInputElement).value) } : p))}
      />
    ),
    unit: t.unit || ''
  })), [thresholds]);

  const flagRows = useMemo(() => flags.map(f => ({
    key: f.key,
    enabled: (
      <input
        type="checkbox"
        checked={f.enabled}
        onChange={ev => setFlags(prev => prev.map(p => p.key === f.key ? { ...p, enabled: (ev.target as HTMLInputElement).checked } : p))}
      />
    ),
    description: f.description || ''
  })), [flags]);

  const saveAll = async () => {
    setSaving(true);
    await Promise.all([
      ...endpoints.map(updateEndpoint),
      ...thresholds.map(updateThreshold),
      ...flags.map(f => toggleFlag(f.key, f.enabled))
    ]);
    setSaving(false);
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button onClick={load} className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700">Reload</button>
        <button onClick={saveAll} disabled={saving} className="px-3 py-2 rounded border border-zinc-300 dark:border-zinc-700 disabled:opacity-50">{saving ? 'Saving…' : 'Save All'}</button>
      </div>

      <Card title="Endpoints">
        {loading ? <div className="h-20 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" /> : (
          <SimpleTable
            columns={[{ key: 'key', label: 'Key' }, { key: 'url', label: 'URL' }, { key: 'enabled', label: 'Enabled' }]}
            rows={endpointRows as any[]}
          />
        )}
      </Card>

      <Card title="Thresholds">
        {loading ? <div className="h-20 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" /> : (
          <SimpleTable
            columns={[{ key: 'key', label: 'Key' }, { key: 'value', label: 'Value' }, { key: 'unit', label: 'Unit' }]}
            rows={thresholdRows as any[]}
          />
        )}
      </Card>

      <Card title="Feature Flags">
        {loading ? <div className="h-20 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" /> : (
          <SimpleTable
            columns={[{ key: 'key', label: 'Key' }, { key: 'enabled', label: 'Enabled' }, { key: 'description', label: 'Description' }]}
            rows={flagRows as any[]}
          />
        )}
      </Card>
    </div>
  );
}