import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatLamports(lamports: number): string {
  const sol = lamports / 1e9;
  if (sol >= 1000) {
    return `${(sol / 1000).toFixed(2)}k`;
  }
  if (sol >= 1) {
    return sol.toFixed(3);
  }
  return sol.toFixed(6);
}

export function formatRelativeTime(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp;
  
  if (diff < 1000) return 'just now';
  if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return `${Math.floor(diff / 86400000)}d ago`;
}

export function formatNumber(num: number): string {
  if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
  return num.toFixed(2);
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function formatLatency(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function getLatencyColor(ms: number): string {
  if (ms <= 8) return 'text-green-500';
  if (ms <= 20) return 'text-yellow-500';
  if (ms <= 50) return 'text-orange-500';
  return 'text-red-500';
}

export function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.9) return 'text-red-500';
  if (confidence >= 0.7) return 'text-orange-500';
  if (confidence >= 0.5) return 'text-yellow-500';
  return 'text-gray-500';
}

export function truncateAddress(address: string, length = 4): string {
  if (!address) return '';
  return `${address.slice(0, length)}...${address.slice(-length)}`;
}