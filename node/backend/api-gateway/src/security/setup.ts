import { FastifyInstance } from 'fastify';
import helmet from '@fastify/helmet';
import cors from '@fastify/cors';
import rateLimit from '@fastify/rate-limit';
import Redis from 'ioredis';
import crypto from 'crypto';
import { config } from '../config';

const redis = new Redis({
  host: config.redis.host,
  port: config.redis.port,
  password: config.redis.password,
  db: config.redis.db,
  keyPrefix: config.redis.keyPrefix
});

export async function setupSecurity(server: FastifyInstance) {
  // Helmet for security headers
  await server.register(helmet, {
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        scriptSrc: ["'self'"],
        imgSrc: ["'self'", 'data:', 'https:'],
        connectSrc: ["'self'"],
        fontSrc: ["'self'"],
        objectSrc: ["'none'"],
        mediaSrc: ["'self'"],
        frameSrc: ["'none'"],
      },
    },
    crossOriginEmbedderPolicy: true,
    crossOriginOpenerPolicy: true,
    crossOriginResourcePolicy: { policy: "cross-origin" },
    dnsPrefetchControl: true,
    frameguard: { action: 'deny' },
    hidePoweredBy: true,
    hsts: {
      maxAge: 31536000,
      includeSubDomains: true,
      preload: true
    },
    ieNoOpen: true,
    noSniff: true,
    originAgentCluster: true,
    permittedCrossDomainPolicies: false,
    referrerPolicy: { policy: "same-origin" },
    xssFilter: true,
  });
  
  // CORS configuration
  await server.register(cors, {
    origin: (origin, cb) => {
      if (!origin || config.security.allowedOrigins.includes(origin)) {
        cb(null, true);
      } else {
        cb(new Error('Not allowed by CORS'));
      }
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token', 'X-Request-ID'],
    exposedHeaders: ['X-Request-ID', 'X-RateLimit-Limit', 'X-RateLimit-Remaining'],
  });
  
  // Rate limiting with Redis
  await server.register(rateLimit, {
    global: true,
    max: config.rateLimit.max,
    timeWindow: config.rateLimit.timeWindow,
    redis: redis,
    keyGenerator: config.rateLimit.keyGenerator,
    errorResponseBuilder: (request, context) => {
      return {
        statusCode: 429,
        error: 'Too Many Requests',
        message: `Rate limit exceeded, retry in ${context.after}`,
        date: Date.now(),
        expiresIn: context.ttl
      };
    }
  });
  
  // IP Allowlisting
  if (config.security.ipAllowlist.length > 0) {
    server.addHook('onRequest', async (request, reply) => {
      const clientIp = request.headers['x-forwarded-for'] || 
                      request.headers['x-real-ip'] || 
                      request.socket.remoteAddress;
      
      const ip = Array.isArray(clientIp) ? clientIp[0] : clientIp?.split(',')[0];
      
      if (!ip || !config.security.ipAllowlist.includes(ip)) {
        reply.code(403).send({ error: 'Forbidden: IP not allowed' });
      }
    });
  }
  
  // mTLS Client Certificate Validation
  if (config.mtls.enabled) {
    server.addHook('onRequest', async (request, reply) => {
      const cert = (request.raw.socket as any).getPeerCertificate();
      
      if (!cert || !cert.subject) {
        reply.code(401).send({ error: 'Client certificate required' });
        return;
      }
      
      const cn = cert.subject.CN;
      if (!config.mtls.allowedCNs.includes(cn)) {
        reply.code(403).send({ error: 'Client certificate not authorized' });
        return;
      }
      
      // Store cert info for audit
      request.headers['x-client-cn'] = cn;
      request.headers['x-client-cert-serial'] = cert.serialNumber;
    });
  }
  
  // CSRF Protection
  if (config.csrf.enabled) {
    const csrfTokens = new Map<string, { token: string; expires: number }>();
    
    server.decorateRequest('csrfToken', null);
    
    server.addHook('onRequest', async (request, reply) => {
      // Skip CSRF for GET requests and API endpoints
      if (request.method === 'GET' || request.url.startsWith('/api/health')) {
        return;
      }
      
      const sessionId = request.headers['x-session-id'] as string;
      const providedToken = request.headers[config.csrf.headerName] as string;
      
      if (!sessionId || !providedToken) {
        reply.code(403).send({ error: 'CSRF token required' });
        return;
      }
      
      const storedData = csrfTokens.get(sessionId);
      if (!storedData || storedData.expires < Date.now()) {
        reply.code(403).send({ error: 'CSRF token expired' });
        return;
      }
      
      const expectedToken = crypto
        .createHmac('sha256', config.csrf.secret)
        .update(sessionId + storedData.token)
        .digest('hex');
      
      if (providedToken !== expectedToken) {
        reply.code(403).send({ error: 'Invalid CSRF token' });
        return;
      }
      
      (request as any).csrfToken = providedToken;
    });
    
    // CSRF token generation endpoint
    server.get('/api/csrf-token', async (request, reply) => {
      const sessionId = request.headers['x-session-id'] as string || crypto.randomBytes(32).toString('hex');
      const token = crypto.randomBytes(32).toString('hex');
      const expires = Date.now() + 3600000; // 1 hour
      
      csrfTokens.set(sessionId, { token, expires });
      
      // Cleanup old tokens
      for (const [key, value] of csrfTokens.entries()) {
        if (value.expires < Date.now()) {
          csrfTokens.delete(key);
        }
      }
      
      const csrfToken = crypto
        .createHmac('sha256', config.csrf.secret)
        .update(sessionId + token)
        .digest('hex');
      
      reply.send({
        sessionId,
        csrfToken,
        expiresIn: 3600
      });
    });
  }
  
  // Security headers hook
  server.addHook('onSend', async (request, reply) => {
    reply.header('X-Request-ID', request.id);
    reply.header('X-Content-Type-Options', 'nosniff');
    reply.header('X-Frame-Options', 'DENY');
    reply.header('X-XSS-Protection', '1; mode=block');
    reply.header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
  });
}