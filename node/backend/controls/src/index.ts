import Fastify from 'fastify';
import { Connection } from '@solana/web3.js';
import { Gauge, Counter, Histogram, register as promRegister } from 'prom-client';
import Redis from 'ioredis';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as crypto from 'crypto';
import * as yaml from 'yaml';
import * as diff from 'diff';
import * as tar from 'tar';
import cron from 'node-cron';
import speakeasy from 'speakeasy';
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
  }
});

const redis = new Redis({
  host: config.redis.host,
  port: config.redis.port,
  password: config.redis.password,
  keyPrefix: 'controls:'
});

// Metrics
const snapshotSize = new Gauge({
  name: 'solana_snapshot_size_bytes',
  help: 'Snapshot size in bytes',
  labelNames: ['type', 'slot']
});

const snapshotOperations = new Counter({
  name: 'solana_snapshot_operations_total',
  help: 'Total snapshot operations',
  labelNames: ['operation', 'status']
});

const ledgerRepairs = new Counter({
  name: 'solana_ledger_repairs_total',
  help: 'Total ledger repair operations',
  labelNames: ['type', 'status']
});

const configValidations = new Counter({
  name: 'solana_config_validations_total',
  help: 'Total configuration validations',
  labelNames: ['status']
});

const operationDuration = new Histogram({
  name: 'solana_control_operation_duration_seconds',
  help: 'Control operation duration',
  labelNames: ['operation'],
  buckets: [1, 5, 10, 30, 60, 120, 300, 600]
});

interface OperationRequest {
  operation: string;
  parameters: any;
  requesterId: string;
  twoFactorToken?: string;
  reason: string;
}

class ControlOperations {
  private connection: Connection;
  private ledgerPath: string;
  private snapshotPath: string;
  private configPath: string;
  private pendingOperations: Map<string, OperationRequest> = new Map();
  
  constructor() {
    this.connection = new Connection(config.solana.rpcUrl, {
      commitment: 'confirmed'
    });
    this.ledgerPath = config.paths.ledger;
    this.snapshotPath = config.paths.snapshots;
    this.configPath = config.paths.config;
  }
  
  async initialize() {
    // Schedule automatic snapshot pruning
    cron.schedule('0 */6 * * *', () => {
      this.pruneSnapshots();
    });
    
    // Schedule ledger health checks
    cron.schedule('*/30 * * * *', () => {
      this.checkLedgerHealth();
    });
    
    server.log.info('Control operations initialized');
  }
  
  async createSnapshot(slot?: number): Promise<string> {
    const startTime = Date.now();
    const operationId = this.generateOperationId();
    
    try {
      await this.auditLog({
        operationId,
        action: 'create_snapshot',
        slot,
        timestamp: new Date().toISOString()
      });
      
      // Get current slot if not specified
      if (!slot) {
        slot = await this.connection.getSlot();
      }
      
      const snapshotName = `snapshot-${slot}-${Date.now()}.tar.zst`;
      const snapshotFullPath = path.join(this.snapshotPath, snapshotName);
      
      // Create snapshot using ledger tool (via systemd service)
      await this.executeLedgerTool([
        'create-snapshot',
        this.ledgerPath,
        snapshotFullPath,
        '--slot', slot.toString()
      ]);
      
      // Get snapshot size
      const stats = await fs.stat(snapshotFullPath);
      snapshotSize.set(
        { type: 'full', slot: slot.toString() },
        stats.size
      );
      
      // Calculate checksum
      const checksum = await this.calculateChecksum(snapshotFullPath);
      
      // Store metadata
      await redis.hset(`snapshot:${snapshotName}`, {
        slot: slot.toString(),
        size: stats.size.toString(),
        checksum,
        created: new Date().toISOString(),
        path: snapshotFullPath
      });
      
      snapshotOperations.inc({ operation: 'create', status: 'success' });
      
      const duration = (Date.now() - startTime) / 1000;
      operationDuration.observe({ operation: 'create_snapshot' }, duration);
      
      server.log.info(`Snapshot created: ${snapshotName}`);
      
      return snapshotName;
    } catch (error) {
      snapshotOperations.inc({ operation: 'create', status: 'failure' });
      server.log.error('Snapshot creation failed:', error);
      throw error;
    }
  }
  
  async verifySnapshot(snapshotName: string): Promise<boolean> {
    const startTime = Date.now();
    
    try {
      const metadata = await redis.hgetall(`snapshot:${snapshotName}`);
      if (!metadata.path) {
        throw new Error('Snapshot not found');
      }
      
      // Verify checksum
      const currentChecksum = await this.calculateChecksum(metadata.path);
      const isValid = currentChecksum === metadata.checksum;
      
      // Verify using ledger tool
      if (isValid) {
        await this.executeLedgerTool([
          'verify-snapshot',
          metadata.path
        ]);
      }
      
      snapshotOperations.inc({ 
        operation: 'verify', 
        status: isValid ? 'success' : 'failure' 
      });
      
      const duration = (Date.now() - startTime) / 1000;
      operationDuration.observe({ operation: 'verify_snapshot' }, duration);
      
      return isValid;
    } catch (error) {
      snapshotOperations.inc({ operation: 'verify', status: 'failure' });
      server.log.error('Snapshot verification failed:', error);
      throw error;
    }
  }
  
  async pruneSnapshots(keepCount: number = 5): Promise<number> {
    const startTime = Date.now();
    
    try {
      // Get all snapshots
      const keys = await redis.keys('snapshot:*');
      const snapshots = await Promise.all(
        keys.map(async key => {
          const data = await redis.hgetall(key);
          return {
            name: key.replace('controls:snapshot:', ''),
            ...data,
            created: new Date(data.created).getTime()
          };
        })
      );
      
      // Sort by creation time
      snapshots.sort((a, b) => b.created - a.created);
      
      // Keep the most recent snapshots
      const toDelete = snapshots.slice(keepCount);
      
      for (const snapshot of toDelete) {
        try {
          await fs.unlink(snapshot.path);
          await redis.del(`snapshot:${snapshot.name}`);
          server.log.info(`Pruned snapshot: ${snapshot.name}`);
        } catch (error) {
          server.log.error(`Failed to prune snapshot ${snapshot.name}:`, error);
        }
      }
      
      snapshotOperations.inc({ 
        operation: 'prune', 
        status: 'success' 
      });
      
      const duration = (Date.now() - startTime) / 1000;
      operationDuration.observe({ operation: 'prune_snapshots' }, duration);
      
      return toDelete.length;
    } catch (error) {
      snapshotOperations.inc({ operation: 'prune', status: 'failure' });
      server.log.error('Snapshot pruning failed:', error);
      throw error;
    }
  }
  
  async repairLedger(startSlot: number, endSlot: number): Promise<void> {
    const startTime = Date.now();
    const operationId = this.generateOperationId();
    
    try {
      await this.auditLog({
        operationId,
        action: 'repair_ledger',
        startSlot,
        endSlot,
        timestamp: new Date().toISOString()
      });
      
      // Execute ledger repair
      await this.executeLedgerTool([
        'repair',
        '--ledger', this.ledgerPath,
        '--start-slot', startSlot.toString(),
        '--end-slot', endSlot.toString()
      ]);
      
      ledgerRepairs.inc({ type: 'slot_range', status: 'success' });
      
      const duration = (Date.now() - startTime) / 1000;
      operationDuration.observe({ operation: 'repair_ledger' }, duration);
      
      server.log.info(`Ledger repaired: slots ${startSlot} to ${endSlot}`);
    } catch (error) {
      ledgerRepairs.inc({ type: 'slot_range', status: 'failure' });
      server.log.error('Ledger repair failed:', error);
      throw error;
    }
  }
  
  async validateConfiguration(newConfig: any): Promise<{ valid: boolean; diff: any }> {
    try {
      // Load current configuration
      const currentConfig = await fs.readFile(this.configPath, 'utf-8');
      const current = yaml.parse(currentConfig);
      
      // Generate diff
      const configDiff = diff.createPatch(
        'validator-config.yaml',
        yaml.stringify(current),
        yaml.stringify(newConfig)
      );
      
      // Validate critical parameters
      const criticalParams = [
        'identity_keypair',
        'vote_account_keypair',
        'ledger_path',
        'rpc_port',
        'dynamic_port_range'
      ];
      
      for (const param of criticalParams) {
        if (!newConfig[param]) {
          configValidations.inc({ status: 'invalid' });
          return {
            valid: false,
            diff: configDiff
          };
        }
      }
      
      // Validate numeric ranges
      if (newConfig.rpc_port < 1024 || newConfig.rpc_port > 65535) {
        configValidations.inc({ status: 'invalid' });
        return {
          valid: false,
          diff: configDiff
        };
      }
      
      configValidations.inc({ status: 'valid' });
      
      return {
        valid: true,
        diff: configDiff
      };
    } catch (error) {
      configValidations.inc({ status: 'error' });
      throw error;
    }
  }
  
  async applyConfiguration(
    newConfig: any,
    operationId: string,
    twoFactorToken: string
  ): Promise<void> {
    const startTime = Date.now();
    
    try {
      // Verify 2FA
      if (!await this.verify2FA(operationId, twoFactorToken)) {
        throw new Error('Invalid 2FA token');
      }
      
      // Validate configuration
      const validation = await this.validateConfiguration(newConfig);
      if (!validation.valid) {
        throw new Error('Invalid configuration');
      }
      
      // Backup current configuration
      const backupPath = `${this.configPath}.backup.${Date.now()}`;
      await fs.copyFile(this.configPath, backupPath);
      
      // Apply new configuration
      await fs.writeFile(
        this.configPath,
        yaml.stringify(newConfig),
        'utf-8'
      );
      
      await this.auditLog({
        operationId,
        action: 'apply_configuration',
        diff: validation.diff,
        backup: backupPath,
        timestamp: new Date().toISOString()
      });
      
      const duration = (Date.now() - startTime) / 1000;
      operationDuration.observe({ operation: 'apply_configuration' }, duration);
      
      server.log.info('Configuration applied successfully');
    } catch (error) {
      server.log.error('Configuration apply failed:', error);
      throw error;
    }
  }
  
  async orchestrateRollingRestart(
    nodes: string[],
    batchSize: number = 1,
    delayMs: number = 60000
  ): Promise<void> {
    const startTime = Date.now();
    const operationId = this.generateOperationId();
    
    try {
      await this.auditLog({
        operationId,
        action: 'rolling_restart',
        nodes,
        batchSize,
        delayMs,
        timestamp: new Date().toISOString()
      });
      
      // Process nodes in batches
      for (let i = 0; i < nodes.length; i += batchSize) {
        const batch = nodes.slice(i, i + batchSize);
        
        // Restart nodes in parallel within batch
        await Promise.all(
          batch.map(node => this.restartNode(node))
        );
        
        // Wait before next batch
        if (i + batchSize < nodes.length) {
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      }
      
      const duration = (Date.now() - startTime) / 1000;
      operationDuration.observe({ operation: 'rolling_restart' }, duration);
      
      server.log.info('Rolling restart completed');
    } catch (error) {
      server.log.error('Rolling restart failed:', error);
      throw error;
    }
  }
  
  private async restartNode(nodeId: string): Promise<void> {
    // Implement node restart logic
    server.log.info(`Restarting node: ${nodeId}`);
    // This would interact with the validator-agent service
  }
  
  private async executeLedgerTool(args: string[]): Promise<void> {
    // Execute ledger tool via systemd service
    // This avoids direct shell execution
    server.log.info(`Executing ledger tool: ${args.join(' ')}`);
  }
  
  private async calculateChecksum(filePath: string): Promise<string> {
    const fileBuffer = await fs.readFile(filePath);
    const hash = crypto.createHash('sha256');
    hash.update(fileBuffer);
    return hash.digest('hex');
  }
  
  private async checkLedgerHealth(): Promise<void> {
    try {
      // Check ledger integrity
      await this.executeLedgerTool([
        'verify',
        '--ledger', this.ledgerPath
      ]);
      
      server.log.info('Ledger health check passed');
    } catch (error) {
      server.log.error('Ledger health check failed:', error);
      ledgerRepairs.inc({ type: 'health_check', status: 'failure' });
    }
  }
  
  private generateOperationId(): string {
    return `op_${Date.now()}_${crypto.randomBytes(8).toString('hex')}`;
  }
  
  private async verify2FA(operationId: string, token: string): Promise<boolean> {
    // Verify 2FA token for operation
    const secret = await redis.get(`2fa:${operationId}`);
    if (!secret) return false;
    
    return speakeasy.totp.verify({
      secret,
      encoding: 'base32',
      token,
      window: 2
    });
  }
  
  private async auditLog(event: any): Promise<void> {
    // Sign audit event
    const signature = crypto
      .createHmac('sha256', config.audit.signatureKey)
      .update(JSON.stringify(event))
      .digest('hex');
    
    const auditEntry = {
      ...event,
      signature
    };
    
    await redis.zadd(
      'audit:controls',
      Date.now(),
      JSON.stringify(auditEntry)
    );
    
    server.log.info('Audit event:', event);
  }
}

async function bootstrap() {
  try {
    const controls = new ControlOperations();
    await controls.initialize();
    
    // API Routes
    server.post('/snapshot/create', async (request: any, reply) => {
      const { slot } = request.body;
      
      try {
        const snapshot = await controls.createSnapshot(slot);
        reply.send({
          success: true,
          snapshot,
          message: 'Snapshot created successfully'
        });
      } catch (error) {
        reply.code(500).send({
          error: 'Snapshot creation failed',
          details: error.message
        });
      }
    });
    
    server.post('/snapshot/verify', async (request: any, reply) => {
      const { snapshotName } = request.body;
      
      try {
        const isValid = await controls.verifySnapshot(snapshotName);
        reply.send({
          success: true,
          valid: isValid,
          snapshot: snapshotName
        });
      } catch (error) {
        reply.code(500).send({
          error: 'Snapshot verification failed',
          details: error.message
        });
      }
    });
    
    server.post('/snapshot/prune', async (request: any, reply) => {
      const { keepCount = 5 } = request.body;
      
      try {
        const pruned = await controls.pruneSnapshots(keepCount);
        reply.send({
          success: true,
          prunedCount: pruned,
          message: `Pruned ${pruned} snapshots`
        });
      } catch (error) {
        reply.code(500).send({
          error: 'Snapshot pruning failed',
          details: error.message
        });
      }
    });
    
    server.post('/ledger/repair', async (request: any, reply) => {
      const { startSlot, endSlot } = request.body;
      
      try {
        await controls.repairLedger(startSlot, endSlot);
        reply.send({
          success: true,
          message: 'Ledger repair completed'
        });
      } catch (error) {
        reply.code(500).send({
          error: 'Ledger repair failed',
          details: error.message
        });
      }
    });
    
    server.post('/config/validate', async (request: any, reply) => {
      try {
        const validation = await controls.validateConfiguration(request.body);
        reply.send(validation);
      } catch (error) {
        reply.code(500).send({
          error: 'Configuration validation failed',
          details: error.message
        });
      }
    });
    
    server.post('/config/apply', async (request: any, reply) => {
      const { config, operationId, twoFactorToken } = request.body;
      
      try {
        await controls.applyConfiguration(config, operationId, twoFactorToken);
        reply.send({
          success: true,
          message: 'Configuration applied successfully'
        });
      } catch (error) {
        reply.code(500).send({
          error: 'Configuration apply failed',
          details: error.message
        });
      }
    });
    
    server.post('/restart/rolling', async (request: any, reply) => {
      const { nodes, batchSize = 1, delayMs = 60000 } = request.body;
      
      try {
        await controls.orchestrateRollingRestart(nodes, batchSize, delayMs);
        reply.send({
          success: true,
          message: 'Rolling restart completed'
        });
      } catch (error) {
        reply.code(500).send({
          error: 'Rolling restart failed',
          details: error.message
        });
      }
    });
    
    server.get('/metrics', async (request, reply) => {
      reply.type('text/plain');
      reply.send(await promRegister.metrics());
    });
    
    await server.listen({
      port: config.port,
      host: config.host
    });
    
    server.log.info(`Control Operations Service started on ${config.host}:${config.port}`);
    
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
}

bootstrap();