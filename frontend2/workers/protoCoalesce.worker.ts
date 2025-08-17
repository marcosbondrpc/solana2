let bucket: any[] = [];
let timer: any = null;
const BATCH_MS = 16;
const MAX_BATCH = 256;

function flush() {
  if (bucket.length) {
    (self as any).postMessage({ type: "batch", items: bucket });
    bucket = [];
  }
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
}

onmessage = (e: MessageEvent) => {
  const d: any = e.data;
  if (d?.type === "env") {
    bucket.push(d.env);
    if (bucket.length >= MAX_BATCH) flush();
    else if (!timer) timer = setTimeout(flush, BATCH_MS);
  } else if (d?.type === "flush") flush();
};