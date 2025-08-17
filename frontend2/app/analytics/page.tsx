"use client";
import { useState } from "react";

export default function AnalyticsPage() {
  const [q, setQ] = useState<string>("SELECT 1");
  const [out, setOut] = useState<string>("");

  const run = async () => {
    try {
      const url = `${process.env.NEXT_PUBLIC_API_BASE}/ch/query?q=${encodeURIComponent(q)}`;
      const res = await fetch(url, { headers: { "Authorization": `Bearer ${localStorage.getItem("jwt")||""}` } });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setOut(await res.text());
    } catch (e: any) { setOut(`Error: ${e.message}`); }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <input className="px-2 py-1 bg-zinc-900 border border-zinc-800 rounded w-full" value={q} onChange={e=>setQ(e.target.value)} />
        <button className="btn" onClick={run}>Run</button>
        <a className="btn" href={process.env.NEXT_PUBLIC_GRAFANA_URL} target="_blank">Grafana</a>
      </div>
      <pre className="text-xs whitespace-pre-wrap bg-zinc-950 border border-zinc-800 rounded p-3 max-h-[420px] overflow-auto">{out}</pre>
    </div>
  );
}