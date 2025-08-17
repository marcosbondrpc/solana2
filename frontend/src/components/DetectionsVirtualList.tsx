import React from "react";
import { FixedSizeList as List, ListChildComponentProps } from "react-window";
import { getRows, subscribe } from "../store/detectionsStore";

export function DetectionsVirtualList() {
	const [, setTick] = React.useState(0);
	React.useEffect(() => subscribe(() => setTick(x => x + 1)), []);
	const data = getRows();

	const Row = ({ index, style }: ListChildComponentProps) => {
		const r = data[index];
		return (
			<div style={style}>
				<div style={{ display: "flex", gap: 8, fontFamily: "monospace", fontSize: 12 }}>
					<span>{r.seq}</span>
					<span>{r.ts}</span>
					<span>{r.kind}</span>
					<span title={r.sig}>{r.sig.slice(0, 8)}…</span>
					<span title={r.address}>{r.address.slice(0, 6)}…</span>
					<span>{r.score.toFixed(2)}</span>
				</div>
			</div>
		);
	};

	return (
		<List height={600} width={"100%"} itemCount={data.length} itemSize={28}>
			{Row}
		</List>
	);
}