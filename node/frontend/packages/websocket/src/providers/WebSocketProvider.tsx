import { createContext, useContext, useEffect, useRef, useState, ReactNode } from 'react';

interface WebSocketConfig {
  url: string;
  protocols?: string[];
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  enableCompression?: boolean;
  binaryType?: 'arraybuffer' | 'blob';
}

interface WebSocketContextType {
  ws: WebSocket | null;
  isConnected: boolean;
  isConnecting: boolean;
  error: Error | null;
  send: (data: any) => void;
  subscribe: (event: string, handler: (data: any) => void) => () => void;
  reconnect: () => void;
  disconnect: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
  config: WebSocketConfig;
  children: ReactNode;
}

export function WebSocketProvider({ config, children }: WebSocketProviderProps) {
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<NodeJS.Timeout>();
  const heartbeatInterval = useRef<NodeJS.Timeout>();
  const eventHandlers = useRef<Map<string, Set<(data: any) => void>>>(new Map());
  const isIntentionalDisconnect = useRef(false);

  const connect = () => {
    if (isConnecting || isConnected || isIntentionalDisconnect.current) return;
    
    setIsConnecting(true);
    setError(null);

    try {
      console.log(`[WebSocket] Attempting to connect to ${config.url}`);
      const websocket = new WebSocket(config.url, config.protocols);
      
      if (config.binaryType) {
        websocket.binaryType = config.binaryType;
      }

      websocket.onopen = () => {
        console.log(`[WebSocket] Successfully connected to ${config.url}`);
        setWs(websocket);
        setIsConnected(true);
        setIsConnecting(false);
        setError(null);
        reconnectAttempts.current = 0;
        
        // Start heartbeat
        if (config.heartbeatInterval) {
          heartbeatInterval.current = setInterval(() => {
            if (websocket.readyState === WebSocket.OPEN) {
              try {
                websocket.send(JSON.stringify({ type: 'ping' }));
              } catch (err) {
                console.warn('[WebSocket] Failed to send heartbeat:', err);
              }
            }
          }, config.heartbeatInterval);
        }
      };

      websocket.onmessage = (event) => {
        try {
          const data = event.data instanceof ArrayBuffer 
            ? event.data 
            : JSON.parse(event.data);
          
          // Handle different message types
          if (typeof data === 'object' && data.type) {
            const handlers = eventHandlers.current.get(data.type);
            if (handlers) {
              handlers.forEach(handler => {
                try {
                  handler(data);
                } catch (err) {
                  console.error('[WebSocket] Handler error:', err);
                }
              });
            }
          }
          
          // Broadcast to all message handlers
          const allHandlers = eventHandlers.current.get('message');
          if (allHandlers) {
            allHandlers.forEach(handler => {
              try {
                handler(data);
              } catch (err) {
                console.error('[WebSocket] Message handler error:', err);
              }
            });
          }
        } catch (err) {
          console.warn('[WebSocket] Failed to parse message:', err);
        }
      };

      websocket.onerror = (event) => {
        console.warn('[WebSocket] Connection error occurred');
        const errorMsg = `WebSocket connection failed to ${config.url}`;
        setError(new Error(errorMsg));
        
        // Don't set connecting to false here, let onclose handle it
      };

      websocket.onclose = (event) => {
        console.log(`[WebSocket] Connection closed: code=${event.code}, reason=${event.reason || 'No reason provided'}, clean=${event.wasClean}`);
        setWs(null);
        setIsConnected(false);
        setIsConnecting(false);
        
        // Clear heartbeat
        if (heartbeatInterval.current) {
          clearInterval(heartbeatInterval.current);
          heartbeatInterval.current = undefined;
        }
        
        // Only attempt reconnection if:
        // 1. It wasn't an intentional disconnect
        // 2. We haven't exceeded max attempts
        // 3. The close wasn't clean (unless it's a normal closure)
        if (
          !isIntentionalDisconnect.current &&
          config.maxReconnectAttempts && 
          reconnectAttempts.current < config.maxReconnectAttempts &&
          (!event.wasClean || event.code !== 1000)
        ) {
          reconnectAttempts.current++;
          const delay = Math.min(
            (config.reconnectInterval || 5000) * Math.pow(2, reconnectAttempts.current - 1), // Increased base from 2000ms to 5000ms
            30000
          );
          
          console.log(`[WebSocket] Scheduling reconnection attempt ${reconnectAttempts.current}/${config.maxReconnectAttempts} in ${delay}ms`);
          reconnectTimeout.current = setTimeout(() => {
            console.log(`[WebSocket] Executing reconnection attempt ${reconnectAttempts.current}`);
            connect();
          }, delay);
        } else if (reconnectAttempts.current >= (config.maxReconnectAttempts || 0)) {
          console.log('[WebSocket] Max reconnection attempts reached, giving up');
          setError(new Error('Maximum reconnection attempts exceeded'));
        }
      };
    } catch (err) {
      console.error('[WebSocket] Failed to create WebSocket:', err);
      setError(err as Error);
      setIsConnecting(false);
      
      // Schedule retry for connection creation failures
      if (
        !isIntentionalDisconnect.current &&
        config.maxReconnectAttempts && 
        reconnectAttempts.current < config.maxReconnectAttempts
      ) {
        reconnectAttempts.current++;
        const delay = Math.min(
          (config.reconnectInterval || 5000) * Math.pow(2, reconnectAttempts.current - 1), // Increased base from 2000ms to 5000ms
          30000
        );
        console.log(`[WebSocket] Retrying connection in ${delay}ms (attempt ${reconnectAttempts.current}/${config.maxReconnectAttempts})`);
        reconnectTimeout.current = setTimeout(connect, delay);
      }
    }
  };

  const disconnect = () => {
    console.log('[WebSocket] Disconnecting intentionally');
    isIntentionalDisconnect.current = true;
    
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = undefined;
    }
    if (heartbeatInterval.current) {
      clearInterval(heartbeatInterval.current);
      heartbeatInterval.current = undefined;
    }
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      ws.close(1000, 'Client disconnect');
    }
    setWs(null);
    setIsConnected(false);
    setIsConnecting(false);
    setError(null);
  };

  const send = (data: any) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] Cannot send message - not connected');
      return;
    }
    
    try {
      const message = typeof data === 'string' ? data : JSON.stringify(data);
      ws.send(message);
    } catch (err) {
      console.error('[WebSocket] Failed to send message:', err);
    }
  };

  const subscribe = (event: string, handler: (data: any) => void) => {
    if (!eventHandlers.current.has(event)) {
      eventHandlers.current.set(event, new Set());
    }
    eventHandlers.current.get(event)!.add(handler);
    
    // Return unsubscribe function
    return () => {
      const handlers = eventHandlers.current.get(event);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          eventHandlers.current.delete(event);
        }
      }
    };
  };

  const reconnect = () => {
    console.log('[WebSocket] Manual reconnection requested');
    isIntentionalDisconnect.current = false;
    disconnect();
    reconnectAttempts.current = 0;
    setTimeout(connect, 1000); // Increased from 100ms to 1000ms
  };

  useEffect(() => {
    // Prevent double connection in React Strict Mode
    let isEffectActive = true;
    
    // Increased delay to prevent race conditions and double connections
    const connectionTimer = setTimeout(() => {
      if (isEffectActive && !isConnected && !isConnecting) {
        isIntentionalDisconnect.current = false;
        connect();
      }
    }, 1000); // Increased from 100ms to 1000ms
    
    return () => {
      isEffectActive = false;
      clearTimeout(connectionTimer);
      console.log('[WebSocket] Component unmounting, cleaning up');
      isIntentionalDisconnect.current = true;
      disconnect();
    };
  }, [config.url]);

  const value: WebSocketContextType = {
    ws,
    isConnected,
    isConnecting,
    error,
    send,
    subscribe,
    reconnect,
    disconnect,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}