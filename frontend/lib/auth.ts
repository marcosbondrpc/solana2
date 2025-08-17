/**
 * Enterprise-grade authentication with JWT and RBAC
 * Supports viewer, analyst, operator, ml_engineer, admin roles
 */

import { decodeJwt } from "jose";
import { z } from 'zod';
import React from 'react';

// Keep existing basic functions for backward compatibility
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

// Enhanced RBAC system
export enum Role {
  VIEWER = 'viewer',
  ANALYST = 'analyst', 
  OPERATOR = 'operator',
  ML_ENGINEER = 'ml_engineer',
  ADMIN = 'admin'
}

// Permission matrix
export const Permissions = {
  [Role.VIEWER]: [
    'dashboard.view',
    'metrics.view',
    'events.view'
  ],
  [Role.ANALYST]: [
    'dashboard.view',
    'metrics.view',
    'events.view',
    'queries.execute',
    'datasets.export',
    'reports.create'
  ],
  [Role.OPERATOR]: [
    'dashboard.view',
    'metrics.view',
    'events.view',
    'queries.execute',
    'datasets.export',
    'reports.create',
    'models.deploy',
    'system.monitor',
    'alerts.manage'
  ],
  [Role.ML_ENGINEER]: [
    'dashboard.view',
    'metrics.view',
    'events.view',
    'queries.execute',
    'datasets.export',
    'reports.create',
    'models.train',
    'models.deploy',
    'models.shadow',
    'models.canary',
    'experiments.create'
  ],
  [Role.ADMIN]: [
    '*' // All permissions
  ]
};

// JWT token schema
const TokenSchema = z.object({
  sub: z.string().uuid(),
  role: z.nativeEnum(Role),
  permissions: z.array(z.string()),
  exp: z.number(),
  iat: z.number(),
  jti: z.string().uuid(),
  mfa: z.boolean().optional(),
  ip: z.string().ip().optional()
});

export type Token = z.infer<typeof TokenSchema>;

// User session schema
const UserSessionSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  role: z.nativeEnum(Role),
  permissions: z.array(z.string()),
  token: z.string(),
  refreshToken: z.string(),
  expiresAt: z.number(),
  mfaEnabled: z.boolean(),
  mfaVerified: z.boolean(),
  lastActivity: z.number()
});

export type UserSession = z.infer<typeof UserSessionSchema>;

class AuthService {
  private session: UserSession | null = null;
  private refreshTimer: NodeJS.Timeout | null = null;
  private activityTimer: NodeJS.Timeout | null = null;
  private readonly TOKEN_KEY = 'jwt'; // Use existing key for compatibility
  private readonly REFRESH_KEY = 'refresh_token';
  private readonly SESSION_KEY = 'user_session';
  private readonly ACTIVITY_TIMEOUT = 30 * 60 * 1000; // 30 minutes
  private readonly REFRESH_BUFFER = 5 * 60 * 1000; // 5 minutes before expiry
  private listeners = new Map<string, Set<Function>>();

  constructor() {
    this.loadSession();
    this.setupActivityTracking();
    this.setupTokenRefresh();
  }

  private loadSession(): void {
    try {
      const stored = localStorage.getItem(this.SESSION_KEY);
      if (stored) {
        const session = JSON.parse(stored);
        const validated = UserSessionSchema.parse(session);
        
        // Check if session is still valid
        if (validated.expiresAt > Date.now()) {
          this.session = validated;
        } else {
          this.clearSession();
        }
      }
    } catch (error) {
      console.error('Failed to load session:', error);
      this.clearSession();
    }
  }

  private saveSession(): void {
    if (this.session) {
      localStorage.setItem(this.SESSION_KEY, JSON.stringify(this.session));
      localStorage.setItem(this.TOKEN_KEY, this.session.token);
      localStorage.setItem(this.REFRESH_KEY, this.session.refreshToken);
    }
  }

  private clearSession(): void {
    this.session = null;
    localStorage.removeItem(this.SESSION_KEY);
    localStorage.removeItem(this.TOKEN_KEY);
    localStorage.removeItem(this.REFRESH_KEY);
    
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }
    
    this.emit('logout');
  }

  private setupActivityTracking(): void {
    const trackActivity = () => {
      if (this.session) {
        this.session.lastActivity = Date.now();
        this.saveSession();
        
        // Reset activity timer
        if (this.activityTimer) {
          clearTimeout(this.activityTimer);
        }
        
        this.activityTimer = setTimeout(() => {
          console.warn('Session expired due to inactivity');
          this.logout();
        }, this.ACTIVITY_TIMEOUT);
      }
    };

    // Track user activity
    if (typeof window !== 'undefined') {
      ['mousedown', 'keydown', 'scroll', 'touchstart'].forEach(event => {
        document.addEventListener(event, trackActivity, { passive: true });
      });
    }

    trackActivity();
  }

  private setupTokenRefresh(): void {
    if (!this.session) return;

    const timeUntilRefresh = this.session.expiresAt - Date.now() - this.REFRESH_BUFFER;
    
    if (timeUntilRefresh > 0) {
      this.refreshTimer = setTimeout(() => {
        this.refreshToken();
      }, timeUntilRefresh);
    } else {
      // Token needs immediate refresh
      this.refreshToken();
    }
  }

  public async login(email: string, password: string, mfaCode?: string): Promise<UserSession> {
    try {
      const response = await fetch('http://45.157.234.184:8000/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password, mfaCode })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Login failed');
      }

      const data = await response.json();
      
      // Validate and store session
      this.session = UserSessionSchema.parse({
        id: data.user.id,
        email: data.user.email,
        role: data.user.role,
        permissions: this.getPermissions(data.user.role),
        token: data.token,
        refreshToken: data.refreshToken,
        expiresAt: data.expiresAt,
        mfaEnabled: data.user.mfaEnabled,
        mfaVerified: !!mfaCode,
        lastActivity: Date.now()
      });

      this.saveSession();
      this.setupTokenRefresh();
      this.emit('login', this.session);

      return this.session;
    } catch (error) {
      console.error('Login failed:', error);
      throw error;
    }
  }

  public async refreshToken(): Promise<void> {
    if (!this.session) return;

    try {
      const response = await fetch('http://45.157.234.184:8000/api/auth/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.session.refreshToken}`
        }
      });

      if (!response.ok) {
        throw new Error('Token refresh failed');
      }

      const data = await response.json();
      
      // Update session with new tokens
      this.session.token = data.token;
      this.session.refreshToken = data.refreshToken;
      this.session.expiresAt = data.expiresAt;
      
      this.saveSession();
      this.setupTokenRefresh();
      this.emit('refresh', this.session);
    } catch (error) {
      console.error('Token refresh failed:', error);
      this.logout();
    }
  }

  public async verifyMFA(code: string): Promise<boolean> {
    if (!this.session) return false;

    try {
      const response = await fetch('http://45.157.234.184:8000/api/auth/mfa/verify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.session.token}`
        },
        body: JSON.stringify({ code })
      });

      if (response.ok) {
        this.session.mfaVerified = true;
        this.saveSession();
        this.emit('mfa-verified');
        return true;
      }

      return false;
    } catch (error) {
      console.error('MFA verification failed:', error);
      return false;
    }
  }

  public logout(): void {
    if (this.session) {
      // Notify server
      fetch('http://45.157.234.184:8000/api/auth/logout', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.session.token}`
        }
      }).catch(console.error);
    }

    this.clearSession();
  }

  public getPermissions(role: Role): string[] {
    const perms = Permissions[role];
    if (perms.includes('*')) {
      // Admin has all permissions
      return Object.values(Permissions)
        .flat()
        .filter(p => p !== '*');
    }
    return perms;
  }

  public hasPermission(permission: string): boolean {
    if (!this.session) return false;
    return this.session.permissions.includes(permission) || 
           this.session.permissions.includes('*');
  }

  public requirePermission(permission: string): void {
    if (!this.hasPermission(permission)) {
      throw new Error(`Permission denied: ${permission}`);
    }
  }

  public async requireMFA(action: string): Promise<boolean> {
    if (!this.session?.mfaEnabled) return true;
    
    if (!this.session.mfaVerified) {
      // Prompt for MFA
      this.emit('mfa-required', action);
      return false;
    }

    return true;
  }

  public getSession(): UserSession | null {
    return this.session;
  }

  public isAuthenticated(): boolean {
    return !!this.session && this.session.expiresAt > Date.now();
  }

  public getAuthHeaders(): Record<string, string> {
    if (!this.session) return {};
    return {
      'Authorization': `Bearer ${this.session.token}`
    };
  }

  // Event emitter methods
  public on(event: string, handler: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
  }

  public off(event: string, handler: Function): void {
    this.listeners.get(event)?.delete(handler);
  }

  private emit(event: string, ...args: any[]): void {
    this.listeners.get(event)?.forEach(handler => {
      try {
        handler(...args);
      } catch (error) {
        console.error(`Error in auth event handler for ${event}:`, error);
      }
    });
  }

  // Audit logging
  public async logAction(action: string, details: any): Promise<void> {
    if (!this.session) return;

    try {
      await fetch('http://45.157.234.184:8000/api/audit/log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...this.getAuthHeaders()
        },
        body: JSON.stringify({
          action,
          details,
          timestamp: Date.now(),
          userId: this.session.id,
          role: this.session.role
        })
      });
    } catch (error) {
      console.error('Failed to log audit action:', error);
    }
  }
}

// Export singleton instance
export const auth = new AuthService();

// React hook for auth state
export function useAuth() {
  const [session, setSession] = React.useState(auth.getSession());
  
  React.useEffect(() => {
    const handleChange = () => setSession(auth.getSession());
    auth.on('login', handleChange);
    auth.on('logout', handleChange);
    auth.on('refresh', handleChange);
    
    return () => {
      auth.off('login', handleChange);
      auth.off('logout', handleChange);
      auth.off('refresh', handleChange);
    };
  }, []);

  return {
    session,
    isAuthenticated: auth.isAuthenticated(),
    hasPermission: (perm: string) => auth.hasPermission(perm),
    requirePermission: (perm: string) => auth.requirePermission(perm),
    login: auth.login.bind(auth),
    logout: auth.logout.bind(auth),
    verifyMFA: auth.verifyMFA.bind(auth)
  };
}