import { withAuthHeaders } from "./auth";

const BASE = process.env.NEXT_PUBLIC_API_BASE || "";

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, await withAuthHeaders());
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body: any, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, await withAuthHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json", ...(init.headers||{}) },
    body: JSON.stringify(body),
    ...init
  }));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}