import { decode } from "@msgpack/msgpack";
import { pushDetections, getLastSeq } from "../store/detectionsStore";

type Listener = (ev: MessageEvent) => void;

export function connectWS(base = "", path = "/ws") {
	let delay = 250;
	const max = 8000;
	let ws: WebSocket | null = null;
	let hbTimer: any;

	const open = () => {
		ws = new WebSocket(`${base.replace(/\/$/, "")}${path}`);
		ws.binaryType = "arraybuffer";
		ws.onopen = () => {
			delay = 250;
			const last_seq = getLastSeq();
			const sub = { t: "subscribe", last_seq };
			ws!.send((window as any).MessagePack?.encode ? (window as any).MessagePack.encode(sub) : JSON.stringify(sub));
		};
		ws.onmessage = (ev: MessageEvent) => {
			const data = ev.data;
			let msg: any;
			if (data instanceof ArrayBuffer) {
				msg = decode(new Uint8Array(data));
			} else if (typeof data === "string") {
				try { msg = JSON.parse(data); } catch { return; }
			} else {
				return;
			}
			switch (msg.t) {
				case "detection_batch":
					if (Array.isArray(msg.data)) pushDetections(msg.data);
					break;
				case "needs_snapshot":
					// caller should re-bootstrap; here we just ignore and rely on app bootstrap logic
					break;
				default:
					break;
			}
		};
		ws.onclose = () => {
			clearTimeout(hbTimer);
			setTimeout(open, delay);
			delay = Math.min(delay * 2, max);
		};
	};

	open();
	return () => {
		try { ws?.close(); } catch {}
		clearTimeout(hbTimer);
	};
}