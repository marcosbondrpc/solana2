/**
 * MEV Backend Service - Complete API Integration
 * Connects all frontend components to backend operations
 */

import ky from 'ky';
import { Ed25519Provider } from '@metamask/key-tree';
import nacl from 'tweetnacl';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://45.157.234.184:8000';
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://45.157.234.184:8000';

export interface MEVOpportunity {
  id: string;
  type: 'arbitrage' | 'sandwich' | 'jit' | 'liquidation' | 'flashloan';
  dexes: string[];
  tokens: string[];
  expectedProfit: number;
  confidence: number;
  gasEstimate: number;
  deadline: number;
  route: string[];
  metadata: Record<string, any>;
}

export interface BundleSubmission {
  opportunityId: string;
  transactions: string[];
  signature: string;
  priority: number;
}

export interface ExecutionResult {
  success: boolean;
  bundleId: string;
  landedSlot?: number;
  actualProfit?: number;
  error?: string;
}

export interface ThompsonStats {
  arms: Array<{
    name: string;
    alpha: number;
    beta: number;
    ev: number;
    samples: number;
  }>;
  totalReward: number;
  totalSamples: number;
}

export interface RiskStatus {
  isActive: boolean;
  currentExposure: number;
  maxExposure: number;
  stopLoss: number;
  sloViolations: number;
  killSwitchEnabled: boolean;
}

export interface SystemMetrics {
  latencyP50: number;
  latencyP99: number;
  bundleLandRate: number;
  ingestionRate: number;
  modelInferenceTime: number;
  decisionDnaCount: number;
}

class MEVService {
  private api: typeof ky;
  private keyPair?: nacl.SignKeyPair;
  private wsConnections: Map<string, WebSocket> = new Map();

  constructor() {
    this.api = ky.create({
      prefixUrl: API_BASE,
      timeout: 5000,
      retry: {
        limit: 2,
        methods: ['get'],
      },
      hooks: {
        beforeRequest: [
          request => {
            request.headers.set('X-Client', 'mev-frontend');
          }
        ]
      }
    });
  }

  /**
   * Initialize Ed25519 keypair for signing
   */
  async initializeSigning() {
    // In production, load from secure storage
    this.keyPair = nacl.sign.keyPair();
    console.log('✅ Ed25519 keypair initialized');
  }

  /**
   * Sign a command with Ed25519
   */
  private async signCommand(command: any): Promise<string> {
    if (!this.keyPair) {
      await this.initializeSigning();
    }
    const message = new TextEncoder().encode(JSON.stringify(command));
    const signature = nacl.sign.detached(message, this.keyPair!.secretKey);
    return Buffer.from(signature).toString('hex');
  }

  // ============= MEV Operations =============

  /**
   * Scan for all MEV opportunities
   */
  async scanOpportunities(): Promise<MEVOpportunity[]> {
    return this.api.post('api/mev/scan').json();
  }

  /**
   * Get real-time MEV opportunities
   */
  async getOpportunities(
    limit: number = 100,
    type?: string,
    minProfit?: number
  ): Promise<MEVOpportunity[]> {
    const params = new URLSearchParams();
    params.append('limit', limit.toString());
    if (type) params.append('type', type);
    if (minProfit) params.append('min_profit', minProfit.toString());

    return this.api.get('api/mev/opportunities', { searchParams: params }).json();
  }

  /**
   * Execute a specific MEV opportunity
   */
  async executeOpportunity(
    opportunityId: string,
    priority: number = 1
  ): Promise<ExecutionResult> {
    const command = { opportunityId, priority, timestamp: Date.now() };
    const signature = await this.signCommand(command);

    return this.api.post(`api/mev/execute/${opportunityId}`, {
      json: { priority, signature }
    }).json();
  }

  /**
   * Simulate bundle execution
   */
  async simulateBundle(bundle: BundleSubmission): Promise<{
    expectedProfit: number;
    successProbability: number;
    gasUsed: number;
  }> {
    return this.api.post('api/mev/simulate', { json: bundle }).json();
  }

  /**
   * Submit bundle to Jito
   */
  async submitBundle(bundle: BundleSubmission): Promise<ExecutionResult> {
    const signature = await this.signCommand(bundle);
    return this.api.post('api/mev/bundle/submit', {
      json: { ...bundle, signature }
    }).json();
  }

  // ============= Statistics & Monitoring =============

  /**
   * Get MEV performance statistics
   */
  async getStats(): Promise<{
    totalOpportunities: number;
    executedOpportunities: number;
    totalProfit: number;
    successRate: number;
    avgLatency: number;
  }> {
    return this.api.get('api/mev/stats').json();
  }

  /**
   * Get Thompson Sampling statistics
   */
  async getBanditStats(): Promise<ThompsonStats> {
    return this.api.get('api/mev/bandit/stats').json();
  }

  /**
   * Reset Thompson Sampling
   */
  async resetBandit(): Promise<void> {
    const command = { action: 'reset', timestamp: Date.now() };
    const signature = await this.signCommand(command);
    await this.api.post('api/mev/bandit/reset', { json: { signature } });
  }

  /**
   * Get system metrics
   */
  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.api.get('api/mev/metrics').json();
  }

  // ============= Risk Management =============

  /**
   * Get risk status
   */
  async getRiskStatus(): Promise<RiskStatus> {
    return this.api.get('api/mev/risk/status').json();
  }

  /**
   * Emergency stop
   */
  async emergencyStop(): Promise<void> {
    const command = { action: 'emergency_stop', timestamp: Date.now() };
    const signature = await this.signCommand(command);
    await this.api.post('api/mev/risk/kill-switch', {
      json: { enable: true, signature }
    });
  }

  /**
   * Resume operations
   */
  async resumeOperations(): Promise<void> {
    const command = { action: 'resume', timestamp: Date.now() };
    const signature = await this.signCommand(command);
    await this.api.post('api/mev/risk/kill-switch', {
      json: { enable: false, signature }
    });
  }

  /**
   * Set throttle percentage
   */
  async setThrottle(percent: number): Promise<void> {
    const command = { throttle: percent, timestamp: Date.now() };
    const signature = await this.signCommand(command);
    await this.api.post('api/mev/risk/throttle', {
      json: { percent, signature }
    });
  }

  // ============= Control Plane =============

  /**
   * Sign and submit a control command
   */
  async submitSignedCommand(command: any): Promise<{
    ackHash: string;
    timestamp: number;
  }> {
    const signature = await this.signCommand(command);
    return this.api.post('api/mev/control/sign', {
      json: { command, signature }
    }).json();
  }

  /**
   * Get audit trail
   */
  async getAuditTrail(limit: number = 100): Promise<Array<{
    timestamp: number;
    command: any;
    signature: string;
    ackHash: string;
  }>> {
    return this.api.get('api/mev/control/audit', {
      searchParams: { limit }
    }).json();
  }

  // ============= WebSocket Connections =============

  /**
   * Connect to opportunity stream
   */
  connectOpportunityStream(
    onMessage: (opportunity: MEVOpportunity) => void,
    onError?: (error: Error) => void
  ): () => void {
    const ws = new WebSocket(`${WS_BASE}/ws/opportunities`);
    ws.binaryType = 'arraybuffer';

    ws.onmessage = (event) => {
      try {
        if (event.data instanceof ArrayBuffer) {
          // Handle protobuf message
          const opportunity = this.decodeProtobuf(event.data);
          onMessage(opportunity);
        } else {
          // Handle JSON fallback
          const opportunity = JSON.parse(event.data);
          onMessage(opportunity);
        }
      } catch (error) {
        onError?.(error as Error);
      }
    };

    ws.onerror = (event) => {
      onError?.(new Error('WebSocket error'));
    };

    ws.onclose = () => {
      // Auto-reconnect after 1 second
      setTimeout(() => this.connectOpportunityStream(onMessage, onError), 1000);
    };

    this.wsConnections.set('opportunities', ws);

    // Return cleanup function
    return () => {
      ws.close();
      this.wsConnections.delete('opportunities');
    };
  }

  /**
   * Connect to execution stream
   */
  connectExecutionStream(
    onMessage: (execution: ExecutionResult) => void,
    onError?: (error: Error) => void
  ): () => void {
    const ws = new WebSocket(`${WS_BASE}/ws/executions`);
    
    ws.onmessage = (event) => {
      try {
        const execution = JSON.parse(event.data);
        onMessage(execution);
      } catch (error) {
        onError?.(error as Error);
      }
    };

    ws.onerror = () => onError?.(new Error('WebSocket error'));
    
    this.wsConnections.set('executions', ws);

    return () => {
      ws.close();
      this.wsConnections.delete('executions');
    };
  }

  /**
   * Connect to metrics stream
   */
  connectMetricsStream(
    onMessage: (metrics: SystemMetrics) => void,
    onError?: (error: Error) => void
  ): () => void {
    const ws = new WebSocket(`${WS_BASE}/ws/metrics`);
    
    ws.onmessage = (event) => {
      try {
        const metrics = JSON.parse(event.data);
        onMessage(metrics);
      } catch (error) {
        onError?.(error as Error);
      }
    };

    ws.onerror = () => onError?.(new Error('WebSocket error'));
    
    this.wsConnections.set('metrics', ws);

    return () => {
      ws.close();
      this.wsConnections.delete('metrics');
    };
  }

  /**
   * Decode protobuf message (placeholder - implement with actual proto schemas)
   */
  private decodeProtobuf(data: ArrayBuffer): MEVOpportunity {
    // This would use the actual protobuf decoder
    // For now, return a mock opportunity
    return {
      id: 'mock-' + Date.now(),
      type: 'arbitrage',
      dexes: ['Orca', 'Raydium'],
      tokens: ['SOL', 'USDC'],
      expectedProfit: Math.random() * 1000,
      confidence: Math.random(),
      gasEstimate: 5000,
      deadline: Date.now() + 30000,
      route: ['Orca', 'Raydium'],
      metadata: {}
    };
  }

  /**
   * Disconnect all WebSocket connections
   */
  disconnectAll() {
    this.wsConnections.forEach(ws => ws.close());
    this.wsConnections.clear();
  }
}

// Export singleton instance
export const mevService = new MEVService();

// Initialize on import
if (typeof window !== 'undefined') {
  mevService.initializeSigning().catch(console.error);
}