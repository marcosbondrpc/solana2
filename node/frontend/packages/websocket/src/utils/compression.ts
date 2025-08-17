import { gzip, ungzip } from 'pako';

export async function compress(data: string | ArrayBuffer): Promise<ArrayBuffer> {
  const input = typeof data === 'string' ? new TextEncoder().encode(data) : new Uint8Array(data);
  return gzip(input).buffer;
}

export async function decompress(data: ArrayBuffer): Promise<string> {
  const decompressed = ungzip(new Uint8Array(data));
  return new TextDecoder().decode(decompressed);
}

export function isCompressed(data: ArrayBuffer): boolean {
  const view = new DataView(data);
  return view.byteLength > 2 && view.getUint8(0) === 0x1f && view.getUint8(1) === 0x8b;
}