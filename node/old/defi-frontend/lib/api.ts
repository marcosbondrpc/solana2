import type { ExecutionResult } from '@/types';

const API_BASE = 'http://localhost:42392/api';

export async function executeCommand(script: string, params?: Record<string, any>): Promise<ExecutionResult> {
  try {
    const response = await fetch(`${API_BASE}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        script,
        params,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Command execution error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

export async function getNodeStatus() {
  try {
    const response = await fetch(`${API_BASE}/status`);
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch node status:', error);
    return null;
  }
}

export async function getSystemMetrics() {
  try {
    const response = await fetch(`${API_BASE}/metrics/system`);
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch system metrics:', error);
    return null;
  }
}

export async function getLogs(limit = 100, level?: string) {
  try {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (level) params.append('level', level);
    
    const response = await fetch(`${API_BASE}/logs?${params}`);
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch logs:', error);
    return [];
  }
}

export async function switchNetwork(network: 'mainnet-beta' | 'testnet' | 'devnet') {
  try {
    const response = await fetch(`${API_BASE}/network`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ network }),
    });

    return await response.json();
  } catch (error) {
    console.error('Failed to switch network:', error);
    return { success: false, error: 'Failed to switch network' };
  }
}