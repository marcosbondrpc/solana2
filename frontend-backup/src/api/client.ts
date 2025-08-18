export const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? "";

export type Detection = {
	seq: number;
	ts: string;
	slot: number;
	kind: string;
	sig: string;
	address: string;
	score: number;
};

export type SnapshotResponse = {
	as_of_seq: number;
	detections: Detection[];
};

export async function fetchSnapshot(limit = 200): Promise<SnapshotResponse> {
	const res = await fetch(`${API_BASE}/api/snapshot?limit=${limit}`);
	if (!res.ok) throw new Error(`snapshot ${res.status}`);
	return res.json();
}