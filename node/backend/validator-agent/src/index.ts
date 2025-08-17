import dbus from 'dbus-next';
import Fastify from 'fastify';
import { Gauge, Counter, register as promRegister } from 'prom-client';
import Redis from 'ioredis';
import { Connection, PublicKey } from '@solana/web3.js';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as yaml from 'yaml';
import cron from 'node-cron';
import { JournalReader } from './journal-reader';
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
  keyPrefix: 'validator-agent:'
});

// Metrics
const validatorStatus = new Gauge({
  name: 'solana_validator_status',
  help: 'Validator service status (0=stopped, 1=running, 2=restarting)',
  labelNames: ['service']
});

const restartCount = new Counter({
  name: 'solana_validator_restarts_total',
  help: 'Total validator restarts',
  labelNames: ['reason']
});

const voteStatus = new Gauge({
  name: 'solana_validator_vote_status',
  help: 'Vote status (0=disabled, 1=enabled)',
  labelNames: ['identity']
});

const configChanges = new Counter({
  name: 'solana_validator_config_changes_total',
  help: 'Total configuration changes',
  labelNames: ['type']
});

const journalErrors = new Counter({
  name: 'solana_validator_journal_errors_total',
  help: 'Journal errors by type',
  labelNames: ['error_type', 'severity']
});

class ValidatorAgent {
  private systemBus: any;
  private sessionBus: any;
  private systemd: any;
  private journalReader: JournalReader;
  private connection: Connection;
  private validatorIdentity?: PublicKey;
  private configPath: string;
  private currentConfig: any;
  
  constructor() {
    this.configPath = config.validator.configPath;
    this.connection = new Connection(config.solana.rpcUrl, {
      commitment: 'confirmed'
    });
    this.journalReader = new JournalReader();
  }
  
  async initialize() {
    // Connect to D-Bus
    this.systemBus = dbus.systemBus();
    this.sessionBus = dbus.sessionBus();
    
    // Get systemd manager interface
    const systemdObj = await this.systemBus.getProxyObject(
      'org.freedesktop.systemd1',
      '/org/freedesktop/systemd1'
    );
    
    this.systemd = systemdObj.getInterface('org.freedesktop.systemd1.Manager');
    
    // Load validator config
    await this.loadConfig();
    
    // Start journal monitoring
    this.startJournalMonitoring();
    
    // Start health monitoring
    this.startHealthMonitoring();
    
    server.log.info('Validator agent initialized');
  }
  
  private async loadConfig() {
    try {
      const configContent = await fs.readFile(this.configPath, 'utf-8');
      this.currentConfig = yaml.parse(configContent);
      
      if (this.currentConfig.identity_keypair) {
        const keypairPath = this.currentConfig.identity_keypair;
        const keypairData = await fs.readFile(keypairPath, 'utf-8');
        const keypair = JSON.parse(keypairData);
        this.validatorIdentity = new PublicKey(keypair.publicKey);
      }
    } catch (error) {
      server.log.error('Failed to load validator config:', error);
    }
  }
  
  async getServiceStatus(serviceName: string = 'solana-validator.service'): Promise<string> {
    try {
      const unit = await this.systemd.GetUnit(serviceName);
      const unitObj = await this.systemBus.getProxyObject(
        'org.freedesktop.systemd1',
        unit
      );
      const properties = unitObj.getInterface('org.freedesktop.DBus.Properties');
      
      const activeState = await properties.Get(
        'org.freedesktop.systemd1.Unit',
        'ActiveState'
      );
      
      // Update metric
      const statusMap: { [key: string]: number } = {
        'inactive': 0,
        'active': 1,
        'activating': 2,
        'deactivating': 2,
        'failed': -1
      };
      
      validatorStatus.set(
        { service: serviceName },
        statusMap[activeState.value] || 0
      );
      
      return activeState.value;
    } catch (error) {
      server.log.error('Failed to get service status:', error);
      throw error;
    }
  }
  
  async restartService(
    serviceName: string = 'solana-validator.service',
    reason: string = 'manual'
  ): Promise<void> {
    try {
      // Audit log
      await this.auditLog({
        action: 'restart_service',
        service: serviceName,
        reason,
        timestamp: new Date().toISOString()
      });
      
      // Restart via D-Bus
      await this.systemd.RestartUnit(serviceName, 'replace');
      
      restartCount.inc({ reason });
      
      // Wait for service to be active
      await this.waitForService(serviceName, 'active', 60000);
      
      server.log.info(`Service ${serviceName} restarted successfully`);
    } catch (error) {
      server.log.error('Failed to restart service:', error);
      throw error;
    }
  }
  
  async stopService(serviceName: string = 'solana-validator.service'): Promise<void> {
    try {
      await this.auditLog({
        action: 'stop_service',
        service: serviceName,
        timestamp: new Date().toISOString()
      });
      
      await this.systemd.StopUnit(serviceName, 'replace');
      
      await this.waitForService(serviceName, 'inactive', 30000);
      
      server.log.info(`Service ${serviceName} stopped`);
    } catch (error) {
      server.log.error('Failed to stop service:', error);
      throw error;
    }
  }
  
  async startService(serviceName: string = 'solana-validator.service'): Promise<void> {
    try {
      await this.auditLog({
        action: 'start_service',
        service: serviceName,
        timestamp: new Date().toISOString()
      });
      
      await this.systemd.StartUnit(serviceName, 'replace');
      
      await this.waitForService(serviceName, 'active', 60000);
      
      server.log.info(`Service ${serviceName} started`);
    } catch (error) {
      server.log.error('Failed to start service:', error);
      throw error;
    }
  }
  
  private async waitForService(
    serviceName: string,
    targetState: string,
    timeout: number
  ): Promise<void> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const state = await this.getServiceStatus(serviceName);
      if (state === targetState) {
        return;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    throw new Error(`Service ${serviceName} did not reach ${targetState} state within ${timeout}ms`);
  }
  
  async toggleVote(enable: boolean): Promise<void> {
    try {
      const voteAccountPath = this.currentConfig.vote_account_keypair;
      
      if (enable) {
        // Enable voting by adding vote account to config
        this.currentConfig.vote_account = voteAccountPath;
      } else {
        // Disable voting by removing vote account from config
        delete this.currentConfig.vote_account;
      }
      
      await this.saveConfig();
      
      voteStatus.set(
        { identity: this.validatorIdentity?.toString() || 'unknown' },
        enable ? 1 : 0
      );
      
      await this.auditLog({
        action: 'toggle_vote',
        enabled: enable,
        timestamp: new Date().toISOString()
      });
      
      // Restart validator to apply changes
      await this.restartService('solana-validator.service', 'vote_toggle');
      
      server.log.info(`Vote ${enable ? 'enabled' : 'disabled'}`);
    } catch (error) {
      server.log.error('Failed to toggle vote:', error);
      throw error;
    }
  }
  
  async updateFlags(flags: { [key: string]: any }): Promise<void> {
    try {
      const backup = { ...this.currentConfig };
      
      // Apply flag updates
      Object.entries(flags).forEach(([key, value]) => {
        if (value === null) {
          delete this.currentConfig[key];
        } else {
          this.currentConfig[key] = value;
        }
      });
      
      // Validate configuration
      await this.validateConfig();
      
      // Save configuration
      await this.saveConfig();
      
      configChanges.inc({ type: 'flags' });
      
      await this.auditLog({
        action: 'update_flags',
        changes: flags,
        backup,
        timestamp: new Date().toISOString()
      });
      
      server.log.info('Validator flags updated');
    } catch (error) {
      server.log.error('Failed to update flags:', error);
      throw error;
    }
  }
  
  private async validateConfig(): Promise<void> {
    // Validate critical configuration parameters
    const required = ['identity_keypair', 'ledger_path', 'rpc_port'];
    
    for (const param of required) {
      if (!this.currentConfig[param]) {
        throw new Error(`Missing required parameter: ${param}`);
      }
    }
    
    // Validate paths exist
    const paths = [
      this.currentConfig.identity_keypair,
      this.currentConfig.vote_account_keypair,
      this.currentConfig.ledger_path
    ].filter(Boolean);
    
    for (const p of paths) {
      try {
        await fs.access(p);
      } catch {
        throw new Error(`Path does not exist: ${p}`);
      }
    }
  }
  
  private async saveConfig(): Promise<void> {
    // Create backup
    const backupPath = `${this.configPath}.backup.${Date.now()}`;
    await fs.copyFile(this.configPath, backupPath);
    
    // Write new configuration
    const configYaml = yaml.stringify(this.currentConfig);
    await fs.writeFile(this.configPath, configYaml, 'utf-8');
    
    // Keep only last 10 backups
    const dir = path.dirname(this.configPath);
    const files = await fs.readdir(dir);
    const backups = files
      .filter(f => f.includes('.backup.'))
      .sort()
      .slice(0, -10);
    
    for (const backup of backups) {
      await fs.unlink(path.join(dir, backup));
    }
  }
  
  private startJournalMonitoring() {
    this.journalReader.on('error', (entry) => {
      const errorType = this.classifyError(entry.message);
      const severity = entry.priority <= 3 ? 'critical' : 'warning';
      
      journalErrors.inc({ error_type: errorType, severity });
      
      // Store in Redis for analysis
      redis.zadd(
        'journal:errors',
        Date.now(),
        JSON.stringify({
          timestamp: entry.timestamp,
          severity,
          type: errorType,
          message: entry.message
        })
      );
    });
    
    this.journalReader.start('solana-validator.service');
  }
  
  private classifyError(message: string): string {
    if (message.includes('connection')) return 'connection';
    if (message.includes('memory')) return 'memory';
    if (message.includes('disk')) return 'disk';
    if (message.includes('vote')) return 'vote';
    if (message.includes('snapshot')) return 'snapshot';
    if (message.includes('ledger')) return 'ledger';
    return 'other';
  }
  
  private startHealthMonitoring() {
    // Monitor service health every 30 seconds
    cron.schedule('*/30 * * * * *', async () => {
      try {
        const status = await this.getServiceStatus();
        
        // Check if validator is healthy via RPC
        if (status === 'active' && this.validatorIdentity) {
          const voteAccounts = await this.connection.getVoteAccounts();
          const isVoting = voteAccounts.current.some(
            v => v.nodePubkey === this.validatorIdentity!.toString()
          );
          
          voteStatus.set(
            { identity: this.validatorIdentity.toString() },
            isVoting ? 1 : 0
          );
        }
      } catch (error) {
        server.log.error('Health monitoring error:', error);
      }
    });
  }
  
  private async auditLog(event: any) {
    await redis.zadd(
      'audit:validator',
      Date.now(),
      JSON.stringify(event)
    );
    
    server.log.info('Audit event:', event);
  }
}

async function bootstrap() {
  try {
    const agent = new ValidatorAgent();
    await agent.initialize();
    
    // API Routes
    server.get('/status', async (request, reply) => {
      const status = await agent.getServiceStatus();
      
      reply.send({
        status,
        identity: agent.validatorIdentity?.toString(),
        timestamp: new Date().toISOString()
      });
    });
    
    server.post('/restart', async (request: any, reply) => {
      const { reason = 'manual' } = request.body;
      
      try {
        await agent.restartService('solana-validator.service', reason);
        reply.send({ success: true, message: 'Service restarted' });
      } catch (error) {
        reply.code(500).send({ 
          error: 'Restart failed',
          details: error.message 
        });
      }
    });
    
    server.post('/stop', async (request, reply) => {
      try {
        await agent.stopService();
        reply.send({ success: true, message: 'Service stopped' });
      } catch (error) {
        reply.code(500).send({ 
          error: 'Stop failed',
          details: error.message 
        });
      }
    });
    
    server.post('/start', async (request, reply) => {
      try {
        await agent.startService();
        reply.send({ success: true, message: 'Service started' });
      } catch (error) {
        reply.code(500).send({ 
          error: 'Start failed',
          details: error.message 
        });
      }
    });
    
    server.post('/vote/:action', async (request: any, reply) => {
      const { action } = request.params;
      
      if (action !== 'enable' && action !== 'disable') {
        reply.code(400).send({ error: 'Invalid action' });
        return;
      }
      
      try {
        await agent.toggleVote(action === 'enable');
        reply.send({ 
          success: true, 
          message: `Vote ${action}d` 
        });
      } catch (error) {
        reply.code(500).send({ 
          error: 'Vote toggle failed',
          details: error.message 
        });
      }
    });
    
    server.put('/config/flags', async (request: any, reply) => {
      try {
        await agent.updateFlags(request.body);
        reply.send({ 
          success: true, 
          message: 'Flags updated' 
        });
      } catch (error) {
        reply.code(500).send({ 
          error: 'Flag update failed',
          details: error.message 
        });
      }
    });
    
    server.get('/metrics', async (request, reply) => {
      reply.type('text/plain');
      reply.send(await promRegister.metrics());
    });
    
    server.get('/journal/errors', async (request: any, reply) => {
      const { hours = 1 } = request.query;
      const since = Date.now() - (hours * 3600000);
      
      const errors = await redis.zrangebyscore('journal:errors', since, Date.now());
      const parsed = errors.map(e => JSON.parse(e));
      
      reply.send({
        period: `${hours}h`,
        count: parsed.length,
        errors: parsed
      });
    });
    
    await server.listen({ 
      port: config.port, 
      host: config.host 
    });
    
    server.log.info(`Validator Agent started on ${config.host}:${config.port}`);
    
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
}

bootstrap();