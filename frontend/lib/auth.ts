import { decodeJwt } from "jose";

export async function getToken() {
  const t = typeof window !== "undefined" ? localStorage.getItem("jwt") : null;
  if (!t) throw new Error("No JWT");
  return t;
}
export async function withAuthHeaders(init: RequestInit = {}) {
  const token = await getToken();
  const headers = new Headers(init.headers || {});
  headers.set("Authorization", `Bearer ${token}`);
  return { ...init, headers };
}
export function hasAudience(jwt: string, audRequired: string) {
  try {
    const payload: any = decodeJwt(jwt);
    const aud = payload?.aud;
    if (Array.isArray(aud)) return aud.includes(audRequired);
    if (typeof aud === "string") return aud === audRequired;
    return false;
  } catch { return false; }
}
export function isOperator(): boolean {
  if (typeof window === "undefined") return false;
  const t = localStorage.getItem("jwt");
  const aud = process.env.NEXT_PUBLIC_JWT_AUDIENCE || "";
  return !!(t && aud && hasAudience(t, aud));
}