import { FastifyInstance } from 'fastify';
import { Issuer, generators } from 'openid-client';
import jwt from '@fastify/jwt';
import { config } from '../config';
import crypto from 'crypto';

interface SessionData {
  userId: string;
  email: string;
  roles: string[];
  permissions: string[];
  twoFactorVerified: boolean;
  loginTime: number;
  lastActivity: number;
}

const sessions = new Map<string, SessionData>();

export async function setupAuth(server: FastifyInstance) {
  // Register JWT plugin
  await server.register(jwt, {
    secret: config.jwt.secret,
    sign: {
      expiresIn: config.jwt.expiresIn
    }
  });
  
  // Initialize OIDC
  const issuer = await Issuer.discover(config.oidc.issuer);
  const client = new issuer.Client({
    client_id: config.oidc.clientId,
    client_secret: config.oidc.clientSecret,
    redirect_uris: [config.oidc.redirectUri],
    response_types: ['code'],
    token_endpoint_auth_method: 'client_secret_post'
  });
  
  // Authentication decorator
  server.decorate('authenticate', async function(request: any, reply: any) {
    try {
      const token = request.headers.authorization?.replace('Bearer ', '');
      
      if (!token) {
        reply.code(401).send({ error: 'No token provided' });
        return;
      }
      
      const decoded = await request.jwtVerify();
      const sessionId = decoded.sessionId;
      
      const session = sessions.get(sessionId);
      if (!session) {
        reply.code(401).send({ error: 'Session expired' });
        return;
      }
      
      // Check session timeout
      if (Date.now() - session.lastActivity > config.security.sessionTimeout) {
        sessions.delete(sessionId);
        reply.code(401).send({ error: 'Session timeout' });
        return;
      }
      
      // Check 2FA requirement for admin operations
      if (decoded.requiresTwoFactor && !session.twoFactorVerified) {
        reply.code(403).send({ error: '2FA verification required' });
        return;
      }
      
      // Update last activity
      session.lastActivity = Date.now();
      sessions.set(sessionId, session);
      
      request.user = session;
    } catch (err) {
      reply.code(401).send({ error: 'Invalid token' });
    }
  });
  
  // OIDC login initiation
  server.get('/auth/login', async (request, reply) => {
    const state = generators.state();
    const nonce = generators.nonce();
    const codeVerifier = generators.codeVerifier();
    const codeChallenge = generators.codeChallenge(codeVerifier);
    
    // Store state data
    await server.redis.setex(
      `auth:state:${state}`,
      300, // 5 minutes
      JSON.stringify({ nonce, codeVerifier })
    );
    
    const authUrl = client.authorizationUrl({
      scope: config.oidc.scope,
      state,
      nonce,
      code_challenge: codeChallenge,
      code_challenge_method: 'S256'
    });
    
    reply.redirect(authUrl);
  });
  
  // OIDC callback
  server.get('/auth/callback', async (request: any, reply) => {
    try {
      const { code, state } = request.query;
      
      // Retrieve state data
      const stateData = await server.redis.get(`auth:state:${state}`);
      if (!stateData) {
        reply.code(400).send({ error: 'Invalid state' });
        return;
      }
      
      const { nonce, codeVerifier } = JSON.parse(stateData);
      await server.redis.del(`auth:state:${state}`);
      
      // Exchange code for tokens
      const tokenSet = await client.callback(
        config.oidc.redirectUri,
        { code, state },
        { nonce, code_verifier: codeVerifier, state }
      );
      
      const claims = tokenSet.claims();
      
      // Create session
      const sessionId = crypto.randomBytes(32).toString('hex');
      const sessionData: SessionData = {
        userId: claims.sub,
        email: claims.email as string,
        roles: claims.roles as string[] || ['viewer'],
        permissions: derivePermissions(claims.roles as string[] || ['viewer']),
        twoFactorVerified: false,
        loginTime: Date.now(),
        lastActivity: Date.now()
      };
      
      sessions.set(sessionId, sessionData);
      
      // Generate JWT
      const token = await reply.jwtSign({
        sessionId,
        userId: claims.sub,
        email: claims.email,
        roles: sessionData.roles,
        requiresTwoFactor: sessionData.roles.includes('admin')
      });
      
      // Log authentication event
      await logAuthEvent({
        type: 'login',
        userId: claims.sub,
        email: claims.email,
        roles: sessionData.roles,
        ip: request.ip,
        userAgent: request.headers['user-agent'],
        timestamp: new Date().toISOString()
      });
      
      reply.send({
        token,
        refreshToken: tokenSet.refresh_token,
        expiresIn: config.jwt.expiresIn,
        requiresTwoFactor: sessionData.roles.includes('admin'),
        user: {
          id: claims.sub,
          email: claims.email,
          roles: sessionData.roles
        }
      });
    } catch (err) {
      server.log.error(err);
      reply.code(500).send({ error: 'Authentication failed' });
    }
  });
  
  // Token refresh
  server.post('/auth/refresh', async (request: any, reply) => {
    try {
      const { refreshToken } = request.body;
      
      const tokenSet = await client.refresh(refreshToken);
      const claims = tokenSet.claims();
      
      // Generate new JWT
      const sessionId = crypto.randomBytes(32).toString('hex');
      const sessionData: SessionData = {
        userId: claims.sub,
        email: claims.email as string,
        roles: claims.roles as string[] || ['viewer'],
        permissions: derivePermissions(claims.roles as string[] || ['viewer']),
        twoFactorVerified: false,
        loginTime: Date.now(),
        lastActivity: Date.now()
      };
      
      sessions.set(sessionId, sessionData);
      
      const token = await reply.jwtSign({
        sessionId,
        userId: claims.sub,
        email: claims.email,
        roles: sessionData.roles
      });
      
      reply.send({
        token,
        refreshToken: tokenSet.refresh_token,
        expiresIn: config.jwt.expiresIn
      });
    } catch (err) {
      reply.code(401).send({ error: 'Invalid refresh token' });
    }
  });
  
  // Logout
  server.post('/auth/logout', { preHandler: server.authenticate }, async (request: any, reply) => {
    const token = request.headers.authorization?.replace('Bearer ', '');
    const decoded = await request.jwtVerify();
    
    sessions.delete(decoded.sessionId);
    
    await logAuthEvent({
      type: 'logout',
      userId: request.user.userId,
      email: request.user.email,
      ip: request.ip,
      timestamp: new Date().toISOString()
    });
    
    reply.send({ message: 'Logged out successfully' });
  });
}

function derivePermissions(roles: string[]): string[] {
  const permissions: string[] = [];
  
  if (roles.includes('admin')) {
    permissions.push(
      'node:control',
      'node:restart',
      'node:configure',
      'snapshot:manage',
      'ledger:repair',
      'metrics:write',
      'audit:read'
    );
  }
  
  if (roles.includes('operator')) {
    permissions.push(
      'node:status',
      'node:logs',
      'metrics:read',
      'snapshot:create',
      'alert:manage'
    );
  }
  
  if (roles.includes('viewer')) {
    permissions.push(
      'node:status',
      'metrics:read',
      'logs:read'
    );
  }
  
  return [...new Set(permissions)];
}

async function logAuthEvent(event: any) {
  // Implement audit logging
  console.log('Auth event:', event);
}