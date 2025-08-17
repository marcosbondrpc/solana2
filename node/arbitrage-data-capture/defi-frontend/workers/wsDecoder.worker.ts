/**
 * High-performance Web Worker for parallel message decoding
 * Handles protobuf + zstd decompression in background thread
 */

import { decodeBatch, initializeProto } from '../lib/ws-proto';
import * as Comlink from 'comlink';

// Shared buffer for zero-copy operations
let sharedBuffer: SharedArrayBuffer | null = null;
let sharedView: DataView | null = null;

// Performance tracking
const stats = {
  messagesProcessed: 0,
  batchesProcessed: 0,
  totalDecodeTime: 0,
  errors: 0
};

/**
 * Message processor exposed via Comlink
 */
export class MessageProcessor {
  private initialized = false;
  
  async initialize(): Promise<void> {
    if (!this.initialized) {
      await initializeProto();
      this.initialized = true;
      console.log('[Worker] Decoder initialized');
    }
  }
  
  async processBatch(
    buffers: ArrayBuffer[],
    compressed: boolean = true
  ): Promise<{ messages: any[], timing: number }> {
    const startTime = performance.now();
    
    try {
      // Ensure initialized
      if (!this.initialized) {
        await this.initialize();
      }
      
      // Decode batch
      const messages = await decodeBatch(buffers, compressed);
      
      // Update stats
      const timing = performance.now() - startTime;
      stats.messagesProcessed += messages.length;
      stats.batchesProcessed++;
      stats.totalDecodeTime += timing;
      
      return { messages, timing };
      
    } catch (error) {
      stats.errors++;
      console.error('[Worker] Decode error:', error);
      return { messages: [], timing: 0 };
    }
  }
  
  getStats(): typeof stats {
    return { ...stats };
  }
  
  resetStats(): void {
    stats.messagesProcessed = 0;
    stats.batchesProcessed = 0;
    stats.totalDecodeTime = 0;
    stats.errors = 0;
  }
}

// Handle direct messages (non-Comlink)
self.onmessage = async (event: MessageEvent) => {
  const { type, ...data } = event.data;
  
  switch (type) {
    case 'init-shared':
      // Initialize shared buffer
      sharedBuffer = data.buffer;
      sharedView = new DataView(sharedBuffer!);
      console.log('[Worker] Shared buffer initialized');
      break;
      
    case 'decode-batch':
      // Decode batch of messages
      const { batch, compressed, timestamp } = data;
      const startTime = performance.now();
      
      try {
        // Initialize if needed
        await initializeProto();
        
        // Decode messages
        const messages = await decodeBatch(batch, compressed);
        
        // Send back decoded messages
        self.postMessage({
          type: 'decoded',
          payload: messages,
          timing: performance.now() - startTime,
          originalTimestamp: timestamp
        });
        
        // Update stats
        stats.messagesProcessed += messages.length;
        stats.batchesProcessed++;
        stats.totalDecodeTime += (performance.now() - startTime);
        
      } catch (error) {
        stats.errors++;
        console.error('[Worker] Decode error:', error);
        
        self.postMessage({
          type: 'error',
          error: error instanceof Error ? error.message : 'Unknown error',
          originalTimestamp: timestamp
        });
      }
      break;
      
    case 'get-stats':
      self.postMessage({
        type: 'stats',
        stats: { ...stats }
      });
      break;
      
    case 'reset-stats':
      stats.messagesProcessed = 0;
      stats.batchesProcessed = 0;
      stats.totalDecodeTime = 0;
      stats.errors = 0;
      
      self.postMessage({
        type: 'stats-reset',
        success: true
      });
      break;
      
    default:
      console.warn('[Worker] Unknown message type:', type);
  }
};

// Expose via Comlink for structured communication
Comlink.expose(new MessageProcessor());

// Log initialization
console.log('[Worker] WebSocket decoder worker started');