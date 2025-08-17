import React from "react";
import { connectSSE, Stats } from "../realtime/sse";

export function MetricsBar({ base = "" }: { base?: string }) {
	const [ts, setTs] = React.useState<number | null>(null);
	React.useEffect(() => connectSSE(base, "/events", (s: Stats) => setTs(s.ts)), [base]);
	return (
		<div style={{ padding: 8, fontFamily: "monospace", fontSize: 12 }}>
			<span>stats_update ts: {ts ?? "…"}</span>
		</div>
	);
}