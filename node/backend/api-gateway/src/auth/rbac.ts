import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import speakeasy from 'speakeasy';
import qrcode from 'qrcode';
import crypto from 'crypto';

interface RBACOptions {
  roles?: string[];
  permissions?: string[];
  requireTwoFactor?: boolean;
}

const twoFactorSecrets = new Map<string, string>();
const failedAttempts = new Map<string, number>();

export async function setupRBAC(server: FastifyInstance) {
  // RBAC middleware decorator
  server.decorate('authorize', function(options: RBACOptions) {
    return async function(request: any, reply: FastifyReply) {
      if (!request.user) {
        reply.code(401).send({ error: 'Authentication required' });
        return;
      }
      
      // Check roles
      if (options.roles && options.roles.length > 0) {
        const hasRole = options.roles.some(role => request.user.roles.includes(role));
        if (!hasRole) {
          await logAccessDenied(request, 'Insufficient role');
          reply.code(403).send({ error: 'Insufficient privileges' });
          return;
        }
      }
      
      // Check permissions
      if (options.permissions && options.permissions.length > 0) {
        const hasPermission = options.permissions.every(permission => 
          request.user.permissions.includes(permission)
        );
        if (!hasPermission) {
          await logAccessDenied(request, 'Missing permission');
          reply.code(403).send({ error: 'Missing required permissions' });
          return;
        }
      }
      
      // Check 2FA requirement
      if (options.requireTwoFactor && !request.user.twoFactorVerified) {
        reply.code(403).send({ 
          error: '2FA verification required',
          requiresTwoFactor: true 
        });
        return;
      }
    };
  });
  
  // 2FA setup endpoint
  server.post('/auth/2fa/setup', 
    { preHandler: server.authenticate }, 
    async (request: any, reply) => {
      const userId = request.user.userId;
      
      // Generate secret
      const secret = speakeasy.generateSecret({
        name: `Solana Monitor (${request.user.email})`,
        issuer: 'Solana Node Monitor',
        length: 32
      });
      
      // Store secret temporarily
      twoFactorSecrets.set(userId, secret.base32);
      
      // Generate QR code
      const qrCodeUrl = await qrcode.toDataURL(secret.otpauth_url!);
      
      reply.send({
        secret: secret.base32,
        qrCode: qrCodeUrl,
        manualEntry: secret.otpauth_url
      });
    }
  );
  
  // 2FA verification endpoint
  server.post('/auth/2fa/verify',
    { preHandler: server.authenticate },
    async (request: any, reply) => {
      const { token } = request.body;
      const userId = request.user.userId;
      
      // Check failed attempts
      const attempts = failedAttempts.get(userId) || 0;
      if (attempts >= config.security.maxFailedAttempts) {
        reply.code(429).send({ 
          error: 'Too many failed attempts. Account locked temporarily.' 
        });
        return;
      }
      
      const secret = twoFactorSecrets.get(userId);
      if (!secret) {
        reply.code(400).send({ error: '2FA not set up' });
        return;
      }
      
      const verified = speakeasy.totp.verify({
        secret,
        encoding: 'base32',
        token,
        window: 2
      });
      
      if (!verified) {
        failedAttempts.set(userId, attempts + 1);
        
        // Lock account after max attempts
        if (attempts + 1 >= config.security.maxFailedAttempts) {
          setTimeout(() => {
            failedAttempts.delete(userId);
          }, 15 * 60 * 1000); // Reset after 15 minutes
        }
        
        reply.code(401).send({ error: 'Invalid 2FA token' });
        return;
      }
      
      // Clear failed attempts
      failedAttempts.delete(userId);
      
      // Update session
      request.user.twoFactorVerified = true;
      
      // Log successful 2FA
      await logSecurityEvent({
        type: '2fa_verified',
        userId,
        email: request.user.email,
        ip: request.ip,
        timestamp: new Date().toISOString()
      });
      
      reply.send({ 
        success: true,
        message: '2FA verified successfully' 
      });
    }
  );
  
  // Role management endpoints (admin only)
  server.put('/api/users/:userId/roles',
    { 
      preHandler: [
        server.authenticate,
        server.authorize({ roles: ['admin'], requireTwoFactor: true })
      ]
    },
    async (request: any, reply) => {
      const { userId } = request.params;
      const { roles } = request.body;
      
      // Validate roles
      const validRoles = ['admin', 'operator', 'viewer'];
      const invalidRoles = roles.filter((r: string) => !validRoles.includes(r));
      
      if (invalidRoles.length > 0) {
        reply.code(400).send({ 
          error: 'Invalid roles',
          invalidRoles 
        });
        return;
      }
      
      // Update user roles (would interact with identity provider in production)
      await updateUserRoles(userId, roles);
      
      // Audit log
      await logSecurityEvent({
        type: 'role_change',
        targetUserId: userId,
        newRoles: roles,
        changedBy: request.user.userId,
        ip: request.ip,
        timestamp: new Date().toISOString()
      });
      
      reply.send({ 
        success: true,
        userId,
        roles 
      });
    }
  );
  
  // Permission check endpoint
  server.post('/api/permissions/check',
    { preHandler: server.authenticate },
    async (request: any, reply) => {
      const { permissions } = request.body;
      
      const hasPermissions = permissions.every((p: string) => 
        request.user.permissions.includes(p)
      );
      
      reply.send({
        hasPermissions,
        userPermissions: request.user.permissions,
        requestedPermissions: permissions,
        missing: permissions.filter((p: string) => 
          !request.user.permissions.includes(p)
        )
      });
    }
  );
}

async function logAccessDenied(request: any, reason: string) {
  const event = {
    type: 'access_denied',
    userId: request.user?.userId,
    email: request.user?.email,
    path: request.url,
    method: request.method,
    reason,
    ip: request.ip,
    userAgent: request.headers['user-agent'],
    timestamp: new Date().toISOString()
  };
  
  // Log to audit system
  console.log('Access denied:', event);
}

async function logSecurityEvent(event: any) {
  // Implement security event logging
  console.log('Security event:', event);
}

async function updateUserRoles(userId: string, roles: string[]) {
  // In production, this would update the identity provider
  console.log(`Updating roles for user ${userId}:`, roles);
}