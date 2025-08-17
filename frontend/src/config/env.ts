export const API_BASE = (import.meta as any).env?.VITE_API_BASE || '/api/v1';
export const WS_BASE = (import.meta as any).env?.VITE_WS_BASE || '';
export const ENABLE_OPS = String((import.meta as any).env?.VITE_ENABLE_OPS || 'false').toLowerCase() === 'true';