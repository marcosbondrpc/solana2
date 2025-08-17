import fs from 'fs';
import path from 'path';

export const config = {
  port: parseInt(process.env.API_PORT || '3000'),
  host: process.env.API_HOST || '0.0.0.0',
  
  mtls: {
    enabled: process.env.MTLS_ENABLED === 'true',
    serverKey: process.env.SERVER_KEY_PATH ? 
      fs.readFileSync(process.env.SERVER_KEY_PATH) : undefined,
    serverCert: process.env.SERVER_CERT_PATH ? 
      fs.readFileSync(process.env.SERVER_CERT_PATH) : undefined,
    caCert: process.env.CA_CERT_PATH ? 
      fs.readFileSync(process.env.CA_CERT_PATH) : undefined,
    clientCertHeader: 'X-Client-Cert',
    allowedCNs: process.env.ALLOWED_CNS?.split(',') || []
  },
  
  oidc: {
    issuer: process.env.OIDC_ISSUER || 'https://auth.example.com',
    clientId: process.env.OIDC_CLIENT_ID || 'solana-monitor',
    clientSecret: process.env.OIDC_CLIENT_SECRET || '',
    redirectUri: process.env.OIDC_REDIRECT_URI || 'http://localhost:3000/callback',
    scope: 'openid profile email roles'
  },
  
  jwt: {
    secret: process.env.JWT_SECRET || 'change-me-in-production',
    expiresIn: '1h',
    refreshExpiresIn: '7d'
  },
  
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD,
    db: parseInt(process.env.REDIS_DB || '0'),
    keyPrefix: 'solana-gateway:'
  },
  
  rateLimit: {
    max: parseInt(process.env.RATE_LIMIT_MAX || '100'),
    timeWindow: '1 minute',
    redis: true,
    keyGenerator: (req: any) => {
      return req.headers['x-forwarded-for'] || 
             req.headers['x-real-ip'] || 
             req.connection.remoteAddress;
    }
  },
  
  csrf: {
    enabled: process.env.CSRF_ENABLED !== 'false',
    secret: process.env.CSRF_SECRET || 'csrf-secret-change-me',
    cookieName: '_csrf',
    headerName: 'x-csrf-token'
  },
  
  audit: {
    enabled: true,
    signatureKey: process.env.AUDIT_SIGNATURE_KEY || 'audit-sign-key',
    storageType: process.env.AUDIT_STORAGE || 'redis',
    retention: parseInt(process.env.AUDIT_RETENTION_DAYS || '90')
  },
  
  security: {
    allowedOrigins: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3001'],
    ipAllowlist: process.env.IP_ALLOWLIST?.split(',') || [],
    requireTwoFactor: process.env.REQUIRE_2FA === 'true',
    sessionTimeout: parseInt(process.env.SESSION_TIMEOUT || '3600000'), // 1 hour
    maxFailedAttempts: parseInt(process.env.MAX_FAILED_ATTEMPTS || '5')
  },
  
  services: {
    rpcProbe: process.env.RPC_PROBE_URL || 'http://localhost:3010',
    validatorAgent: process.env.VALIDATOR_AGENT_URL || 'http://localhost:3020',
    jitoProbe: process.env.JITO_PROBE_URL || 'http://localhost:3030',
    geyserProbe: process.env.GEYSER_PROBE_URL || 'http://localhost:3040',
    metrics: process.env.METRICS_URL || 'http://localhost:3050',
    controls: process.env.CONTROLS_URL || 'http://localhost:3060'
  }
};