export class ExponentialBackoff {
  private attempt = 0;
  private readonly maxAttempts: number;
  private readonly baseDelay: number;
  private readonly maxDelay: number;
  private readonly jitter: boolean;

  constructor(options: {
    maxAttempts?: number;
    baseDelay?: number;
    maxDelay?: number;
    jitter?: boolean;
  } = {}) {
    this.maxAttempts = options.maxAttempts ?? 10;
    this.baseDelay = options.baseDelay ?? 1000;
    this.maxDelay = options.maxDelay ?? 30000;
    this.jitter = options.jitter ?? true;
  }

  nextDelay(): number {
    if (this.attempt >= this.maxAttempts) {
      throw new Error('Max attempts reached');
    }
    
    const exponentialDelay = Math.min(
      this.baseDelay * Math.pow(2, this.attempt),
      this.maxDelay
    );
    
    const delay = this.jitter
      ? exponentialDelay * (0.5 + Math.random() * 0.5)
      : exponentialDelay;
    
    this.attempt++;
    return Math.round(delay);
  }

  reset(): void {
    this.attempt = 0;
  }

  canRetry(): boolean {
    return this.attempt < this.maxAttempts;
  }

  get attempts(): number {
    return this.attempt;
  }
}