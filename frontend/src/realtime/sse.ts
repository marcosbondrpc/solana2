export type Stats = { t: "stats_update"; ts: number };

export function connectSSE(base = "", path = "/events", onStats?: (s: Stats) => void) {
	const url = `${base.replace(/\/$/, "")}${path}`;
	const es = new EventSource(url);
	es.addEventListener("stats_update", (e: MessageEvent) => {
		try {
			const data = JSON.parse((e as any).data);
			onStats?.(data);
		} catch {}
	});
	return () => es.close();
}