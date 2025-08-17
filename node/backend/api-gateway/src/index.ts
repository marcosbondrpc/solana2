import Fastify from 'fastify';
import { setupSecurity } from './security/setup';
import { setupAuth } from './auth/oidc';
import { setupRBAC } from './auth/rbac';
import { setupAudit } from './audit/logger';
import { setupRoutes } from './routes';
import { setupMetrics } from './metrics';
import { config } from './config';

const server = Fastify({
  logger: {
    level: 'info',
    transport: {
      target: 'pino-pretty',
      options: {
        colorize: true,
        translateTime: 'HH:MM:ss Z',
        ignore: 'pid,hostname'
      }
    }
  },
  trustProxy: true,
  requestIdHeader: 'x-request-id',
  requestIdLogLabel: 'reqId',
  genReqId: (req) => req.headers['x-request-id'] || require('uuid').v4(),
  https: config.mtls.enabled ? {
    key: config.mtls.serverKey,
    cert: config.mtls.serverCert,
    ca: config.mtls.caCert,
    requestCert: true,
    rejectUnauthorized: true
  } : undefined
});

async function bootstrap() {
  try {
    // Setup security middleware
    await setupSecurity(server);
    
    // Setup authentication
    await setupAuth(server);
    
    // Setup RBAC
    await setupRBAC(server);
    
    // Setup audit logging
    await setupAudit(server);
    
    // Setup metrics collection
    await setupMetrics(server);
    
    // Setup API routes
    await setupRoutes(server);
    
    // Start server
    await server.listen({ 
      port: config.port, 
      host: config.host 
    });
    
    server.log.info(`API Gateway started on ${config.host}:${config.port}`);
    
    // Graceful shutdown
    process.on('SIGTERM', async () => {
      server.log.info('SIGTERM received, shutting down gracefully');
      await server.close();
      process.exit(0);
    });
    
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
}

bootstrap();