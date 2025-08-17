declare class WebTransport {
  constructor(url: string);
  ready: Promise<void>;
  closed: Promise<void>;
  createBidirectionalStream(): Promise<{ readable: ReadableStream<Uint8Array>; writable: WritableStream<Uint8Array> }>;
  createUnidirectionalStream(): Promise<WritableStream<Uint8Array>>;
  incomingUnidirectionalStreams: ReadableStream<ReadableStream<Uint8Array>>;
  close(opts?: { closeCode?: number; reason?: string }): void;
}