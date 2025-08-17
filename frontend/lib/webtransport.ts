export async function openWT(url: string) {
  // @ts-ignore
  const wt = new WebTransport(url);
  await wt.ready;
  return wt as any;
}

export async function* readUnidi(wt: any): AsyncIterable<ArrayBuffer> {
  const reader = wt.incomingUnidirectionalStreams.getReader();
  while (true) {
    const { value: stream, done } = await reader.read();
    if (done) break;
    const r = (stream as ReadableStream<Uint8Array>).getReader();
    while (true) {
      const { value, done: d2 } = await r.read();
      if (d2) break;
      if (value) yield value.buffer;
    }
  }
}