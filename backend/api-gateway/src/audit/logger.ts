import pino from 'pino';

const baseLogger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: process.env.NODE_ENV !== 'production' ? {
    target: 'pino-pretty',
    options: { colorize: true, translateTime: 'SYS:standard' }
  } : undefined
});

export async function setupAudit(server: any) {
  if (!server) return;
  server.addHook('onRequest', async (req: any) => {
    baseLogger.info({ method: req.method, url: req.url }, 'request');
  });
}

export default baseLogger;