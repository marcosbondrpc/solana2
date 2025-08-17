/// <reference lib="webworker" />
import { decode } from "@msgpack/msgpack";

self.onmessage = (e: MessageEvent) => {
	const buf = e.data as ArrayBuffer;
	const msg = decode(new Uint8Array(buf)) as any;
	(self as any).postMessage(msg);
};
export {};