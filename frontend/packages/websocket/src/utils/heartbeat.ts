export class HeartbeatManager {
  private interval: NodeJS.Timeout | null = null;
  private lastPong = Date.now();
  private readonly pingInterval: number;
  private readonly pongTimeout: number;
  private onPing?: () => void;
  private onTimeout?: () => void;

  constructor(options: {
    pingInterval?: number;
    pongTimeout?: number;
    onPing?: () => void;
    onTimeout?: () => void;
  } = {}) {
    this.pingInterval = options.pingInterval ?? 30000;
    this.pongTimeout = options.pongTimeout ?? 10000;
    this.onPing = options.onPing;
    this.onTimeout = options.onTimeout;
  }

  start(): void {
    this.stop();
    this.lastPong = Date.now();
    
    this.interval = setInterval(() => {
      const now = Date.now();
      
      if (now - this.lastPong > this.pingInterval + this.pongTimeout) {
        this.onTimeout?.();
        return;
      }
      
      this.onPing?.();
    }, this.pingInterval);
  }

  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  pong(): void {
    this.lastPong = Date.now();
  }

  isAlive(): boolean {
    return Date.now() - this.lastPong <= this.pingInterval + this.pongTimeout;
  }
}