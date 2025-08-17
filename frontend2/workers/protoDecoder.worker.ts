const COMPRESS_MARK = 0x28;

function decodeEnvelope(u8: Uint8Array): any {
  return { raw: u8.byteLength };
}

async function decodeFrame(buf: ArrayBuffer) {
  const u8 = new Uint8Array(buf);
  const first = u8[0];
  const raw = first === COMPRESS_MARK ? u8 : u8;
  return decodeEnvelope(raw);
}

onmessage = async (e: MessageEvent) => {
  if ((e.data as any)?.type === "frame") {
    try {
      const env = await decodeFrame((e.data as any).buf);
      (self as any).postMessage({ type: "env", env });
    } catch {}
  }
};