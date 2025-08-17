/**
 * Control Store - Ultra-high-performance state management
 * Handles all control plane operations with microsecond precision
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { subscribeWithSelector } from 'zustand/middleware';
import { apiService, type ControlCommand, type PolicyUpdate, type ModelSwap, type KillSwitch } from '../lib/api-service';

interface CommandHistoryEntry {
  id: string;
  command: ControlCommand;
  timestamp: number;
  status: 'pending' | 'executed' | 'failed';
  response?: any;
  error?: string;
  signature?: string;
  ack_hash?: string;
}

interface ACKChainEntry {
  hash: string;
  parent_hash: string;
  command_id: string;
  timestamp: number;
  signature: string;
  verified: boolean;
}

interface ControlPolicy {
  id: string;
  type: string;
  enabled: boolean;
  thresholds: Record<string, number>;
  rules: Record<string, string>;
  last_modified: number;
  effective_from: number;
}

interface ModelDeployment {
  id: string;
  path: string;
  type: string;
  version: string;
  status: 'loading' | 'ready' | 'error';
  performance: {
    inference_p50_us: number;
    inference_p99_us: number;
    throughput_rps: number;
  };
  metadata: Record<string, string>;
}

interface KillSwitchState {
  active: boolean;
  target: string;
  reason: string;
  triggered_at?: number;
  expires_at?: number;
  force: boolean;
}

interface MultisigState {
  required_signatures: number;
  current_signers: string[];
  pending_commands: Array<{
    command: ControlCommand;
    signatures: string[];
    expires_at: number;
  }>;
}

interface ControlState {
  // Command Management
  commandHistory: CommandHistoryEntry[];
  pendingCommands: Map<string, CommandHistoryEntry>;
  
  // ACK Chain
  ackChain: ACKChainEntry[];
  ackChainValid: boolean;
  lastAckHash: string | null;
  
  // Policies
  policies: Map<string, ControlPolicy>;
  activePolicies: string[];
  
  // Models
  deployedModels: Map<string, ModelDeployment>;
  activeModel: string | null;
  modelSwapInProgress: boolean;
  
  // Kill Switches
  killSwitches: Map<string, KillSwitchState>;
  emergencyStopActive: boolean;
  throttleLevel: number; // 0-100
  
  // Multisig
  multisigState: MultisigState;
  
  // System Status
  systemStatus: 'operational' | 'degraded' | 'emergency';
  lastHeartbeat: number;
  controlPlaneLatency: number;
  
  // SLO Tracking
  sloViolations: Array<{
    metric: string;
    threshold: number;
    actual: number;
    timestamp: number;
  }>;
  
  // Actions
  sendCommand: (command: ControlCommand) => Promise<void>;
  updatePolicy: (policy: PolicyUpdate) => Promise<void>;
  swapModel: (swap: ModelSwap) => Promise<void>;
  triggerKillSwitch: (killSwitch: KillSwitch) => Promise<void>;
  emergencyStop: () => Promise<void>;
  setThrottle: (percent: number) => Promise<void>;
  loadCommandHistory: () => Promise<void>;
  verifyACKChain: () => Promise<boolean>;
  signCommand: (command: ControlCommand, privateKey: string) => string;
  addMultisigSignature: (commandId: string, signature: string) => void;
  executeMultisigCommand: (commandId: string) => Promise<void>;
  checkSLOs: () => void;
  updateSystemStatus: () => Promise<void>;
  clearHistory: () => void;
}

export const useControlStore = create<ControlState>()(
  subscribeWithSelector(
    immer((set, get) => ({
      // Initial state
      commandHistory: [],
      pendingCommands: new Map(),
      ackChain: [],
      ackChainValid: true,
      lastAckHash: null,
      policies: new Map(),
      activePolicies: [],
      deployedModels: new Map(),
      activeModel: null,
      modelSwapInProgress: false,
      killSwitches: new Map(),
      emergencyStopActive: false,
      throttleLevel: 100,
      multisigState: {
        required_signatures: 2,
        current_signers: [],
        pending_commands: []
      },
      systemStatus: 'operational',
      lastHeartbeat: Date.now(),
      controlPlaneLatency: 0,
      sloViolations: [],

      // Actions
      sendCommand: async (command: ControlCommand) => {
        const commandId = `cmd-${Date.now()}-${Math.random()}`;
        const entry: CommandHistoryEntry = {
          id: commandId,
          command,
          timestamp: Date.now(),
          status: 'pending'
        };

        set(state => {
          state.pendingCommands.set(commandId, entry);
        });

        try {
          const startTime = performance.now();
          const response = await apiService.sendCommand(command);
          const latency = performance.now() - startTime;

          set(state => {
            const cmd = state.pendingCommands.get(commandId);
            if (cmd) {
              cmd.status = 'executed';
              cmd.response = response;
              state.commandHistory.unshift(cmd);
              state.pendingCommands.delete(commandId);
            }
            state.controlPlaneLatency = latency;
          });

          // Update ACK chain if response includes hash
          if (response.ack_hash) {
            set(state => {
              state.ackChain.push({
                hash: response.ack_hash,
                parent_hash: state.lastAckHash || '',
                command_id: commandId,
                timestamp: Date.now(),
                signature: response.signature || '',
                verified: true
              });
              state.lastAckHash = response.ack_hash;
            });
          }

        } catch (error: any) {
          set(state => {
            const cmd = state.pendingCommands.get(commandId);
            if (cmd) {
              cmd.status = 'failed';
              cmd.error = error.message;
              state.commandHistory.unshift(cmd);
              state.pendingCommands.delete(commandId);
            }
          });
          throw error;
        }
      },

      updatePolicy: async (policy: PolicyUpdate) => {
        const response = await apiService.updatePolicy(policy);
        
        set(state => {
          const policyState: ControlPolicy = {
            id: policy.policy_id,
            type: policy.policy_type,
            enabled: policy.enabled,
            thresholds: policy.thresholds,
            rules: policy.rules,
            last_modified: Date.now(),
            effective_from: policy.effective_from || Date.now()
          };
          
          state.policies.set(policy.policy_id, policyState);
          
          if (policy.enabled) {
            if (!state.activePolicies.includes(policy.policy_id)) {
              state.activePolicies.push(policy.policy_id);
            }
          } else {
            state.activePolicies = state.activePolicies.filter(id => id !== policy.policy_id);
          }
        });

        return response;
      },

      swapModel: async (swap: ModelSwap) => {
        set(state => {
          state.modelSwapInProgress = true;
        });

        try {
          const response = await apiService.swapModel(swap);
          
          set(state => {
            const deployment: ModelDeployment = {
              id: swap.model_id,
              path: swap.model_path,
              type: swap.model_type,
              version: swap.version,
              status: 'loading',
              performance: {
                inference_p50_us: 0,
                inference_p99_us: 0,
                throughput_rps: 0
              },
              metadata: swap.metadata
            };
            
            state.deployedModels.set(swap.model_id, deployment);
            state.activeModel = swap.model_id;
            state.modelSwapInProgress = false;
          });

          // Poll for model ready status
          const checkStatus = setInterval(async () => {
            const status = await apiService.getModelStatus(swap.model_id);
            if (status && status[0]) {
              set(state => {
                const model = state.deployedModels.get(swap.model_id);
                if (model) {
                  model.status = status[0].status;
                  model.performance = {
                    inference_p50_us: status[0].inference_p50_us,
                    inference_p99_us: status[0].inference_p99_us,
                    throughput_rps: status[0].predictions_per_sec
                  };
                }
              });

              if (status[0].status === 'ready' || status[0].status === 'error') {
                clearInterval(checkStatus);
              }
            }
          }, 1000);

          return response;
        } catch (error) {
          set(state => {
            state.modelSwapInProgress = false;
          });
          throw error;
        }
      },

      triggerKillSwitch: async (killSwitch: KillSwitch) => {
        const response = await apiService.triggerKillSwitch(killSwitch);
        
        set(state => {
          const switchState: KillSwitchState = {
            active: true,
            target: killSwitch.target,
            reason: killSwitch.reason,
            triggered_at: Date.now(),
            expires_at: Date.now() + killSwitch.duration_ms,
            force: killSwitch.force || false
          };
          
          state.killSwitches.set(killSwitch.target, switchState);
          
          if (killSwitch.target === 'all' || killSwitch.force) {
            state.emergencyStopActive = true;
            state.systemStatus = 'emergency';
          }
        });

        // Auto-clear after duration
        setTimeout(() => {
          set(state => {
            state.killSwitches.delete(killSwitch.target);
            if (state.killSwitches.size === 0) {
              state.emergencyStopActive = false;
              state.systemStatus = 'operational';
            }
          });
        }, killSwitch.duration_ms);

        return response;
      },

      emergencyStop: async () => {
        return get().triggerKillSwitch({
          target: 'all',
          reason: 'Emergency stop triggered',
          duration_ms: 60000, // 1 minute
          force: true
        });
      },

      setThrottle: async (percent: number) => {
        const clampedPercent = Math.max(0, Math.min(100, percent));
        
        set(state => {
          state.throttleLevel = clampedPercent;
          if (clampedPercent < 50) {
            state.systemStatus = 'degraded';
          }
        });

        return get().sendCommand({
          module: 'throttle',
          action: 'set',
          params: { percent: clampedPercent }
        });
      },

      loadCommandHistory: async () => {
        const history = await apiService.getCommandHistory(100);
        set(state => {
          state.commandHistory = history.map(h => ({
            id: h.id,
            command: h.command,
            timestamp: h.timestamp,
            status: h.status,
            response: h.response,
            error: h.error,
            signature: h.signature,
            ack_hash: h.ack_hash
          }));
        });
      },

      verifyACKChain: async () => {
        const chain = await apiService.getACKChain();
        
        let valid = true;
        let previousHash = null;
        
        for (const entry of chain) {
          if (previousHash && entry.parent_hash !== previousHash) {
            valid = false;
            break;
          }
          previousHash = entry.hash;
        }
        
        set(state => {
          state.ackChain = chain;
          state.ackChainValid = valid;
          state.lastAckHash = chain[chain.length - 1]?.hash || null;
        });
        
        return valid;
      },

      signCommand: (command: ControlCommand, privateKey: string) => {
        // Ed25519 signing would happen here
        // For now, return mock signature
        return `ed25519:${btoa(JSON.stringify(command))}:${privateKey.slice(0, 8)}`;
      },

      addMultisigSignature: (commandId: string, signature: string) => {
        set(state => {
          const pending = state.multisigState.pending_commands.find(c => 
            JSON.stringify(c.command).includes(commandId)
          );
          if (pending && !pending.signatures.includes(signature)) {
            pending.signatures.push(signature);
          }
        });
      },

      executeMultisigCommand: async (commandId: string) => {
        const state = get();
        const pending = state.multisigState.pending_commands.find(c => 
          JSON.stringify(c.command).includes(commandId)
        );
        
        if (!pending || pending.signatures.length < state.multisigState.required_signatures) {
          throw new Error('Insufficient signatures');
        }
        
        const response = await apiService.submitSignedCommand(
          pending.command,
          pending.signatures
        );
        
        set(state => {
          state.multisigState.pending_commands = state.multisigState.pending_commands.filter(
            c => JSON.stringify(c.command) !== JSON.stringify(pending.command)
          );
        });
        
        return response;
      },

      checkSLOs: () => {
        const metrics = get();
        const violations: typeof metrics.sloViolations = [];
        
        // Check latency SLO
        if (metrics.controlPlaneLatency > 20) {
          violations.push({
            metric: 'control_plane_latency',
            threshold: 20,
            actual: metrics.controlPlaneLatency,
            timestamp: Date.now()
          });
        }
        
        set(state => {
          state.sloViolations = violations;
          if (violations.length > 0) {
            state.systemStatus = 'degraded';
          }
        });
      },

      updateSystemStatus: async () => {
        const status = await apiService.getControlStatus();
        
        set(state => {
          state.lastHeartbeat = Date.now();
          state.systemStatus = status.status || 'operational';
          
          if (status.models) {
            for (const model of status.models) {
              if (state.deployedModels.has(model.id)) {
                const existing = state.deployedModels.get(model.id)!;
                existing.status = model.status;
                existing.performance = model.performance;
              }
            }
          }
          
          if (status.policies) {
            for (const policy of status.policies) {
              state.policies.set(policy.id, policy);
            }
          }
        });
      },

      clearHistory: () => {
        set(state => {
          state.commandHistory = [];
          state.pendingCommands.clear();
          state.sloViolations = [];
        });
      }
    }))
  )
);

// Auto-update system status every 5 seconds
setInterval(() => {
  useControlStore.getState().updateSystemStatus();
}, 5000);

// Check SLOs every second
setInterval(() => {
  useControlStore.getState().checkSLOs();
}, 1000);