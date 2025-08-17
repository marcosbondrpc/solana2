import type { Detection } from "../api/client";

type Subscriber = () => void;

const MAX_ROWS = 1000;

let rows: Detection[] = [];
let lastSeq = 0;
const subs = new Set<Subscriber>();

export function subscribe(fn: Subscriber) {
	subs.add(fn);
	return () => subs.delete(fn);
}

function emit() {
	for (const fn of subs) fn();
}

export function bootstrapSnapshot(snapRows: Detection[], as_of_seq: number) {
	rows = snapRows.slice(0, MAX_ROWS);
	lastSeq = Math.max(as_of_seq, lastSeq);
	emit();
}

export function pushDetections(batch: Detection[]) {
	if (!batch || batch.length === 0) return;
	batch.sort((a, b) => a.seq - b.seq);
	for (const d of batch) {
		lastSeq = Math.max(lastSeq, d.seq);
	}
	rows = batch.concat(rows);
	if (rows.length > MAX_ROWS) rows = rows.slice(0, MAX_ROWS);
	emit();
}

export function getRows() {
	return rows;
}

export function getLastSeq() {
	return lastSeq;
}