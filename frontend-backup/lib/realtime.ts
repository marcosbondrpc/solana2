const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "";
const WT_URL = process.env.NEXT_PUBLIC_WT_URL || "";

export function connectRealtime(token: string, onBatch: (arr: any[]) => void) {
  const dec = new Worker(new URL("../workers/protoDecoder.worker.ts", import.meta.url), { type: "module" });
  const coal = new Worker(new URL("../workers/protoCoalesce.worker.ts", import.meta.url), { type: "module" });

  coal.onmessage = (e) => {
    if (e.data?.type === "batch") onBatch(e.data.items);
  };
  dec.onmessage = (e) => {
    if (e.data?.type === "env") coal.postMessage({ type: "env", env: e.data.env });
  };

  const hasWT = typeof (globalThis as any).WebTransport !== "undefined" && WT_URL;
  if (hasWT) {
    import("./webtransport").then(({ openWT, readUnidi }) => {
      (async () => {
        try {
          const wt = await openWT(`${WT_URL}?token=${encodeURIComponent(token)}`);
          for await (const ab of readUnidi(wt)) dec.postMessage({ type: "frame", buf: ab }, [ab]);
        } catch (e) {
          const ws = new WebSocket(`${WS_URL}?token=${encodeURIComponent(token)}`);
          ws.binaryType = "arraybuffer";
          ws.onmessage = (ev) => dec.postMessage({ type: "frame", buf: ev.data }, [ev.data]);
        }
      })();
    });
    return () => {};
  } else {
    const ws = new WebSocket(`${WS_URL}?token=${encodeURIComponent(token)}`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (ev) => dec.postMessage({ type: "frame", buf: ev.data }, [ev.data]);
    return () => ws.close();
  }
}