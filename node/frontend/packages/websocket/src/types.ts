export type WebSocketState = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketConfig {
  url: string;
  protocols?: string | string[];
  reconnectDelay?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  compression?: boolean;
  binaryType?: 'arraybuffer' | 'blob';
  maxMessageSize?: number;
  queueSize?: number;
  concurrency?: number;
}

export interface WebSocketMessage<T = any> {
  id?: string;
  type: string;
  data: T;
  timestamp?: number;
  metadata?: Record<string, any>;
}

export type SubscriptionHandler<T = any> = (message: WebSocketMessage<T>) => void;

export interface ConnectionEvents {
  connect: () => void;
  disconnect: (code: number, reason: string) => void;
  error: (error: Error) => void;
  message: (message: WebSocketMessage) => void;
  reconnect: (attempt: number) => void;
  ping: () => void;
  pong: () => void;
}