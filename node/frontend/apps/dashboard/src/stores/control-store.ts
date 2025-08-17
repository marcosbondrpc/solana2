/**
 * Control Plane Store for MEV system governance
 * Handles commands, multisig verification, and kill-switch operations
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { devtools } from 'zustand/middleware';
import * as ed from '@noble/ed25519';

export interface Command {
  id: string;
  timestamp: number;
  type: 'config' | 'kill_switch' | 'budget' | 'strategy' | 'model' | 'route';
  action: string;
  params: Record<string, any>;
  issuer: string;
  issuerPubkey: string;
  signature: string;
  status: 'pending' | 'acked' | 'executed' | 'rejected' | 'expired';
  ackChain: AckSignature[];
  requiredAcks: number;
  expiresAt: number;
  executedAt?: number;
  rejectedReason?: string;
}

export interface AckSignature {
  signer: string;
  pubkey: string;
  signature: string;
  timestamp: number;
  approved: boolean;
  comment?: string;
}

export interface KillSwitch {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  autoTrigger: boolean;
  triggerConditions: TriggerCondition[];
  lastTriggered?: number;
  triggeredBy?: string;
  cooldownPeriod: number;
  inCooldown: boolean;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

export interface TriggerCondition {
  metric: string;
  operator: '<' | '>' | '<=' | '>=' | '==' | '!=';
  threshold: number;
  duration: number; // How long condition must be true (ms)
  currentValue?: number;
  violated: boolean;
  violatedSince?: number;
}

export interface SLO {
  id: string;
  name: string;
  target: number;
  current: number;
  window: '1m' | '5m' | '15m' | '1h' | '24h';
  metric: string;
  status: 'healthy' | 'warning' | 'breached';
  errorBudgetRemaining: number;
  lastBreached?: number;
  breachCount: number;
}

export interface MultisigConfig {
  requiredSignatures: number;
  signers: Array<{
    id: string;
    name: string;
    pubkey: string;
    role: 'admin' | 'operator' | 'observer';
    active: boolean;
  }>;
  timelock: number; // Delay before execution (ms)
  emergencyOverride: boolean;
  emergencySigners: string[];
}

export interface AuditLog {
  id: string;
  timestamp: number;
  commandId: string;
  action: string;
  actor: string;
  result: 'success' | 'failure';
  details: any;
  ipAddress?: string;
  userAgent?: string;
}

interface ControlState {
  // Core data
  commands: Map<string, Command>;
  killSwitches: Map<string, KillSwitch>;
  slos: Map<string, SLO>;
  multisigConfig: MultisigConfig;
  auditLogs: AuditLog[];
  
  // Current state
  systemLocked: boolean;
  emergencyMode: boolean;
  maintenanceMode: boolean;
  currentSigner: {
    id: string;
    name: string;
    pubkey: string;
    privateKey?: string; // Only stored temporarily in memory
  } | null;
  
  // Statistics
  totalCommands: number;
  executedCommands: number;
  rejectedCommands: number;
  killSwitchTriggers: number;
  sloBreaches: number;
  
  // Actions
  issueCommand: (command: Omit<Command, 'id' | 'timestamp' | 'status' | 'ackChain'>) => Promise<string>;
  signCommand: (commandId: string, approved: boolean, comment?: string) => Promise<void>;
  executeCommand: (commandId: string) => Promise<void>;
  rejectCommand: (commandId: string, reason: string) => void;
  
  toggleKillSwitch: (switchId: string, enabled: boolean) => void;
  triggerKillSwitch: (switchId: string, triggeredBy: string) => void;
  updateKillSwitchConditions: (switchId: string, conditions: TriggerCondition[]) => void;
  
  updateSLO: (sloId: string, update: Partial<SLO>) => void;
  checkSLOBreaches: () => void;
  
  updateMultisigConfig: (config: Partial<MultisigConfig>) => void;
  setSigner: (signer: ControlState['currentSigner']) => void;
  
  addAuditLog: (log: Omit<AuditLog, 'id' | 'timestamp'>) => void;
  
  setSystemLocked: (locked: boolean) => void;
  setEmergencyMode: (emergency: boolean) => void;
  setMaintenanceMode: (maintenance: boolean) => void;
  
  cleanupExpiredCommands: () => void;
  reset: () => void;
}

const initialState = {
  commands: new Map(),
  killSwitches: new Map([
    ['bundle_failure', {
      id: 'bundle_failure',
      name: 'Bundle Failure Rate',
      description: 'Disable MEV when bundle landing rate drops',
      enabled: true,
      autoTrigger: true,
      triggerConditions: [{
        metric: 'bundle_land_rate',
        operator: '<' as const,
        threshold: 0.65,
        duration: 60000,
        violated: false
      }],
      cooldownPeriod: 300000,
      inCooldown: false,
      priority: 'high' as const
    }],
    ['latency_spike', {
      id: 'latency_spike',
      name: 'Latency Spike',
      description: 'Disable trading during extreme latency',
      enabled: true,
      autoTrigger: true,
      triggerConditions: [{
        metric: 'p99_latency',
        operator: '>' as const,
        threshold: 50,
        duration: 30000,
        violated: false
      }],
      cooldownPeriod: 180000,
      inCooldown: false,
      priority: 'critical' as const
    }],
    ['profit_loss', {
      id: 'profit_loss',
      name: 'Profit Loss Protection',
      description: 'Stop trading when losing money',
      enabled: true,
      autoTrigger: true,
      triggerConditions: [{
        metric: 'hourly_profit',
        operator: '<' as const,
        threshold: -100,
        duration: 300000,
        violated: false
      }],
      cooldownPeriod: 600000,
      inCooldown: false,
      priority: 'high' as const
    }]
  ]),
  slos: new Map([
    ['latency_p50', {
      id: 'latency_p50',
      name: 'P50 Latency',
      target: 8,
      current: 0,
      window: '5m' as const,
      metric: 'latency_p50',
      status: 'healthy' as const,
      errorBudgetRemaining: 100,
      breachCount: 0
    }],
    ['latency_p99', {
      id: 'latency_p99',
      name: 'P99 Latency',
      target: 20,
      current: 0,
      window: '5m' as const,
      metric: 'latency_p99',
      status: 'healthy' as const,
      errorBudgetRemaining: 100,
      breachCount: 0
    }],
    ['bundle_land_rate', {
      id: 'bundle_land_rate',
      name: 'Bundle Landing Rate',
      target: 85,
      current: 0,
      window: '15m' as const,
      metric: 'bundle_success_rate',
      status: 'healthy' as const,
      errorBudgetRemaining: 100,
      breachCount: 0
    }],
    ['clickhouse_ingestion', {
      id: 'clickhouse_ingestion',
      name: 'ClickHouse Ingestion',
      target: 200000,
      current: 0,
      window: '1m' as const,
      metric: 'clickhouse_rows_per_second',
      status: 'healthy' as const,
      errorBudgetRemaining: 100,
      breachCount: 0
    }]
  ]),
  multisigConfig: {
    requiredSignatures: 2,
    signers: [],
    timelock: 30000, // 30 seconds
    emergencyOverride: true,
    emergencySigners: []
  },
  auditLogs: [],
  systemLocked: false,
  emergencyMode: false,
  maintenanceMode: false,
  currentSigner: null,
  totalCommands: 0,
  executedCommands: 0,
  rejectedCommands: 0,
  killSwitchTriggers: 0,
  sloBreaches: 0
};

export const useControlStore = create<ControlState>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        ...initialState,
        
        issueCommand: async (command) => {
          const id = `cmd-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
          const timestamp = Date.now();
          const expiresAt = timestamp + 300000; // 5 minute expiry
          
          const fullCommand: Command = {
            ...command,
            id,
            timestamp,
            status: 'pending',
            ackChain: [],
            requiredAcks: get().multisigConfig.requiredSignatures,
            expiresAt
          };
          
          // Sign the command if we have a current signer
          const signer = get().currentSigner;
          if (signer?.privateKey) {
            const message = JSON.stringify({
              type: command.type,
              action: command.action,
              params: command.params,
              timestamp,
              expiresAt
            });
            
            const messageBytes = new TextEncoder().encode(message);
            const signature = await ed.signAsync(messageBytes, signer.privateKey);
            
            fullCommand.signature = Buffer.from(signature).toString('hex');
            fullCommand.issuer = signer.id;
            fullCommand.issuerPubkey = signer.pubkey;
          }
          
          set((state) => {
            state.commands.set(id, fullCommand);
            state.totalCommands++;
          });
          
          get().addAuditLog({
            commandId: id,
            action: 'command_issued',
            actor: signer?.id || 'system',
            result: 'success',
            details: { command: fullCommand }
          });
          
          return id;
        },
        
        signCommand: async (commandId, approved, comment) => {
          const command = get().commands.get(commandId);
          const signer = get().currentSigner;
          
          if (!command || !signer?.privateKey) {
            throw new Error('Invalid command or signer');
          }
          
          const message = JSON.stringify({
            commandId,
            approved,
            comment,
            timestamp: Date.now()
          });
          
          const messageBytes = new TextEncoder().encode(message);
          const signature = await ed.signAsync(messageBytes, signer.privateKey);
          
          const ack: AckSignature = {
            signer: signer.id,
            pubkey: signer.pubkey,
            signature: Buffer.from(signature).toString('hex'),
            timestamp: Date.now(),
            approved,
            comment
          };
          
          set((state) => {
            const cmd = state.commands.get(commandId);
            if (cmd) {
              cmd.ackChain.push(ack);
              
              // Check if we have enough approvals
              const approvals = cmd.ackChain.filter(a => a.approved).length;
              if (approvals >= cmd.requiredAcks) {
                cmd.status = 'acked';
              }
            }
          });
          
          get().addAuditLog({
            commandId,
            action: 'command_signed',
            actor: signer.id,
            result: 'success',
            details: { approved, comment }
          });
        },
        
        executeCommand: async (commandId) => {
          const command = get().commands.get(commandId);
          
          if (!command || command.status !== 'acked') {
            throw new Error('Command not ready for execution');
          }
          
          // Check timelock
          const timeSinceAck = Date.now() - Math.max(...command.ackChain.map(a => a.timestamp));
          if (timeSinceAck < get().multisigConfig.timelock) {
            throw new Error(`Timelock not expired: ${get().multisigConfig.timelock - timeSinceAck}ms remaining`);
          }
          
          set((state) => {
            const cmd = state.commands.get(commandId);
            if (cmd) {
              cmd.status = 'executed';
              cmd.executedAt = Date.now();
              state.executedCommands++;
            }
          });
          
          // Execute the actual command
          // This would trigger the appropriate action in the system
          
          get().addAuditLog({
            commandId,
            action: 'command_executed',
            actor: 'system',
            result: 'success',
            details: { command }
          });
        },
        
        rejectCommand: (commandId, reason) => {
          set((state) => {
            const cmd = state.commands.get(commandId);
            if (cmd) {
              cmd.status = 'rejected';
              cmd.rejectedReason = reason;
              state.rejectedCommands++;
            }
          });
          
          get().addAuditLog({
            commandId,
            action: 'command_rejected',
            actor: get().currentSigner?.id || 'system',
            result: 'success',
            details: { reason }
          });
        },
        
        toggleKillSwitch: (switchId, enabled) => {
          set((state) => {
            const killSwitch = state.killSwitches.get(switchId);
            if (killSwitch) {
              killSwitch.enabled = enabled;
            }
          });
          
          get().addAuditLog({
            commandId: '',
            action: 'kill_switch_toggled',
            actor: get().currentSigner?.id || 'system',
            result: 'success',
            details: { switchId, enabled }
          });
        },
        
        triggerKillSwitch: (switchId, triggeredBy) => {
          set((state) => {
            const killSwitch = state.killSwitches.get(switchId);
            if (killSwitch && !killSwitch.inCooldown) {
              killSwitch.lastTriggered = Date.now();
              killSwitch.triggeredBy = triggeredBy;
              killSwitch.inCooldown = true;
              state.killSwitchTriggers++;
              
              // Set cooldown timer
              setTimeout(() => {
                set((s) => {
                  const ks = s.killSwitches.get(switchId);
                  if (ks) ks.inCooldown = false;
                });
              }, killSwitch.cooldownPeriod);
              
              // Set system to emergency mode for critical switches
              if (killSwitch.priority === 'critical') {
                state.emergencyMode = true;
              }
            }
          });
          
          get().addAuditLog({
            commandId: '',
            action: 'kill_switch_triggered',
            actor: triggeredBy,
            result: 'success',
            details: { switchId }
          });
        },
        
        updateKillSwitchConditions: (switchId, conditions) => {
          set((state) => {
            const killSwitch = state.killSwitches.get(switchId);
            if (killSwitch) {
              killSwitch.triggerConditions = conditions;
            }
          });
        },
        
        updateSLO: (sloId, update) => {
          set((state) => {
            const slo = state.slos.get(sloId);
            if (slo) {
              Object.assign(slo, update);
              
              // Update status based on current vs target
              if (slo.metric.includes('latency')) {
                // For latency, lower is better
                if (slo.current > slo.target * 1.5) {
                  slo.status = 'breached';
                } else if (slo.current > slo.target) {
                  slo.status = 'warning';
                } else {
                  slo.status = 'healthy';
                }
              } else {
                // For other metrics, higher is better
                if (slo.current < slo.target * 0.5) {
                  slo.status = 'breached';
                } else if (slo.current < slo.target) {
                  slo.status = 'warning';
                } else {
                  slo.status = 'healthy';
                }
              }
              
              // Update error budget
              const performance = slo.metric.includes('latency') 
                ? Math.max(0, 1 - (slo.current / slo.target))
                : Math.min(1, slo.current / slo.target);
              slo.errorBudgetRemaining = Math.max(0, performance * 100);
              
              // Track breaches
              if (slo.status === 'breached') {
                slo.lastBreached = Date.now();
                slo.breachCount++;
                state.sloBreaches++;
              }
            }
          });
        },
        
        checkSLOBreaches: () => {
          const state = get();
          
          for (const [id, slo] of state.slos) {
            if (slo.status === 'breached') {
              // Check if we should trigger a kill switch
              for (const [switchId, killSwitch] of state.killSwitches) {
                if (killSwitch.enabled && killSwitch.autoTrigger) {
                  for (const condition of killSwitch.triggerConditions) {
                    if (condition.metric === slo.metric) {
                      // Update condition with current value
                      condition.currentValue = slo.current;
                      
                      // Check if condition is violated
                      const wasViolated = condition.violated;
                      condition.violated = state.evaluateCondition(condition, slo.current);
                      
                      if (condition.violated && !wasViolated) {
                        condition.violatedSince = Date.now();
                      } else if (!condition.violated) {
                        condition.violatedSince = undefined;
                      }
                      
                      // Trigger if violated for long enough
                      if (condition.violated && condition.violatedSince) {
                        const violationDuration = Date.now() - condition.violatedSince;
                        if (violationDuration >= condition.duration) {
                          state.triggerKillSwitch(switchId, 'auto-slo-breach');
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        },
        
        evaluateCondition: (condition: TriggerCondition, value: number): boolean => {
          switch (condition.operator) {
            case '<': return value < condition.threshold;
            case '>': return value > condition.threshold;
            case '<=': return value <= condition.threshold;
            case '>=': return value >= condition.threshold;
            case '==': return value === condition.threshold;
            case '!=': return value !== condition.threshold;
            default: return false;
          }
        },
        
        updateMultisigConfig: (config) => {
          set((state) => {
            Object.assign(state.multisigConfig, config);
          });
        },
        
        setSigner: (signer) => {
          set((state) => {
            state.currentSigner = signer;
          });
        },
        
        addAuditLog: (log) => {
          set((state) => {
            const fullLog: AuditLog = {
              ...log,
              id: `audit-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              timestamp: Date.now()
            };
            
            state.auditLogs.unshift(fullLog);
            
            // Keep only last 10000 logs
            if (state.auditLogs.length > 10000) {
              state.auditLogs.pop();
            }
          });
        },
        
        setSystemLocked: (locked) => {
          set((state) => {
            state.systemLocked = locked;
          });
          
          get().addAuditLog({
            commandId: '',
            action: locked ? 'system_locked' : 'system_unlocked',
            actor: get().currentSigner?.id || 'system',
            result: 'success',
            details: {}
          });
        },
        
        setEmergencyMode: (emergency) => {
          set((state) => {
            state.emergencyMode = emergency;
          });
          
          get().addAuditLog({
            commandId: '',
            action: emergency ? 'emergency_mode_enabled' : 'emergency_mode_disabled',
            actor: get().currentSigner?.id || 'system',
            result: 'success',
            details: {}
          });
        },
        
        setMaintenanceMode: (maintenance) => {
          set((state) => {
            state.maintenanceMode = maintenance;
          });
          
          get().addAuditLog({
            commandId: '',
            action: maintenance ? 'maintenance_mode_enabled' : 'maintenance_mode_disabled',
            actor: get().currentSigner?.id || 'system',
            result: 'success',
            details: {}
          });
        },
        
        cleanupExpiredCommands: () => {
          const now = Date.now();
          
          set((state) => {
            for (const [id, command] of state.commands) {
              if (command.status === 'pending' && command.expiresAt < now) {
                command.status = 'expired';
              }
            }
            
            // Remove very old commands (> 24 hours)
            const cutoff = now - 86400000;
            for (const [id, command] of state.commands) {
              if (command.timestamp < cutoff) {
                state.commands.delete(id);
              }
            }
          });
        },
        
        reset: () => {
          set(initialState);
        }
      }))
    ),
    {
      name: 'control-store'
    }
  )
);

// Selectors
export const selectPendingCommands = (state: ControlState) =>
  Array.from(state.commands.values())
    .filter(c => c.status === 'pending' || c.status === 'acked')
    .sort((a, b) => b.timestamp - a.timestamp);

export const selectActiveKillSwitches = (state: ControlState) =>
  Array.from(state.killSwitches.values())
    .filter(ks => ks.enabled);

export const selectBreachedSLOs = (state: ControlState) =>
  Array.from(state.slos.values())
    .filter(slo => slo.status === 'breached' || slo.status === 'warning');

export const selectRecentAuditLogs = (state: ControlState) =>
  state.auditLogs.slice(0, 100);