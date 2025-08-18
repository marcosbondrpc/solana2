export class MessageParser {
  private textDecoder = new TextDecoder();
  private textEncoder = new TextEncoder();

  parse(data: string | ArrayBuffer): any {
    if (typeof data === 'string') {
      try {
        return JSON.parse(data);
      } catch {
        return { type: 'raw', data };
      }
    }
    
    if (data instanceof ArrayBuffer) {
      const text = this.textDecoder.decode(data);
      try {
        return JSON.parse(text);
      } catch {
        return { type: 'binary', data };
      }
    }
    
    return { type: 'unknown', data };
  }

  stringify(data: any): string {
    if (typeof data === 'string') {
      return data;
    }
    return JSON.stringify(data);
  }

  encode(data: any): ArrayBuffer {
    const str = this.stringify(data);
    return this.textEncoder.encode(str).buffer;
  }
}