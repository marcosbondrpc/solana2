export interface Command {
  id: string;
  name: string;
  description: string;
  category: 'deploy' | 'monitor' | 'maintain' | 'optimize' | 'utils';
  script: string;
  dangerous?: boolean;
  requiresConfirmation?: boolean;
  parameters?: CommandParameter[];
}

export interface CommandParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'select';
  label: string;
  required?: boolean;
  default?: any;
  options?: Array<{ label: string; value: string }>;
}

export interface ExecutionResult {
  success: boolean;
  output?: string;
  error?: string;
  exitCode?: number;
  duration?: number;
}

export interface WebSocketMessage {
  type: 'metrics' | 'logs' | 'command' | 'error' | 'connected';
  data: any;
  timestamp: Date;
}

export interface ChartDataPoint {
  time: string;
  value: number;
  label?: string;
}

export interface NetworkStatus {
  network: 'mainnet-beta' | 'testnet' | 'devnet';
  endpoint: string;
  healthy: boolean;
  latency: number;
  blockHeight: number;
  slot: number;
}