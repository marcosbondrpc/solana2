/**
 * React Hooks for MEV Operations
 * Provides easy-to-use hooks for all MEV backend functionality
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState, useCallback, useRef } from 'react';
import { toast } from 'sonner';
import { mevService, MEVOpportunity, ExecutionResult, SystemMetrics, ThompsonStats, RiskStatus } from '../lib/api/mev-service';

// ============= Real-time Opportunities Hook =============
export const useMEVOpportunities = (autoRefresh: boolean = true) => {
  const [opportunities, setOpportunities] = useState<MEVOpportunity[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const cleanupRef = useRef<(() => void) | null>(null);

  // Initial fetch
  const { data: initialData, isLoading, error } = useQuery({
    queryKey: ['mev-opportunities'],
    queryFn: () => mevService.getOpportunities(100),
    refetchInterval: autoRefresh ? 5000 : false,
    staleTime: 2000,
  });

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (autoRefresh) {
      cleanupRef.current = mevService.connectOpportunityStream(
        (opportunity) => {
          setOpportunities(prev => {
            // Keep only last 100 opportunities
            const updated = [opportunity, ...prev].slice(0, 100);
            return updated;
          });
          setIsConnected(true);
        },
        (error) => {
          console.error('WebSocket error:', error);
          setIsConnected(false);
          toast.error('Lost connection to opportunity stream');
        }
      );
    }

    return () => {
      cleanupRef.current?.();
    };
  }, [autoRefresh]);

  // Merge initial data with real-time updates
  useEffect(() => {
    if (initialData && opportunities.length === 0) {
      setOpportunities(initialData);
    }
  }, [initialData]);

  return {
    opportunities,
    isLoading,
    error,
    isConnected,
    refetch: () => mevService.scanOpportunities().then(setOpportunities),
  };
};

// ============= Execute Opportunity Hook =============
export const useExecuteOpportunity = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ opportunityId, priority }: { opportunityId: string; priority?: number }) =>
      mevService.executeOpportunity(opportunityId, priority),
    onSuccess: (result) => {
      if (result.success) {
        toast.success(`Bundle landed in slot ${result.landedSlot}! Profit: $${result.actualProfit?.toFixed(2)}`);
      } else {
        toast.error(`Execution failed: ${result.error}`);
      }
      // Invalidate opportunities to refresh the list
      queryClient.invalidateQueries({ queryKey: ['mev-opportunities'] });
    },
    onError: (error) => {
      toast.error(`Execution error: ${error.message}`);
    },
  });
};

// ============= System Metrics Hook =============
export const useSystemMetrics = (refreshInterval: number = 1000) => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const cleanup = mevService.connectMetricsStream(
      (newMetrics) => {
        setMetrics(newMetrics);
        setIsConnected(true);
      },
      (error) => {
        console.error('Metrics stream error:', error);
        setIsConnected(false);
      }
    );

    return cleanup;
  }, []);

  return { metrics, isConnected };
};

// ============= Thompson Sampling Stats Hook =============
export const useBanditStats = () => {
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ['bandit-stats'],
    queryFn: () => mevService.getBanditStats(),
    refetchInterval: 5000,
  });

  const resetMutation = useMutation({
    mutationFn: () => mevService.resetBandit(),
    onSuccess: () => {
      toast.success('Thompson Sampling reset successfully');
      queryClient.invalidateQueries({ queryKey: ['bandit-stats'] });
    },
  });

  return {
    stats: data,
    isLoading,
    error,
    reset: resetMutation.mutate,
    isResetting: resetMutation.isPending,
  };
};

// ============= Risk Management Hook =============
export const useRiskManagement = () => {
  const queryClient = useQueryClient();

  const { data: riskStatus, isLoading } = useQuery({
    queryKey: ['risk-status'],
    queryFn: () => mevService.getRiskStatus(),
    refetchInterval: 2000,
  });

  const emergencyStopMutation = useMutation({
    mutationFn: () => mevService.emergencyStop(),
    onSuccess: () => {
      toast.error('EMERGENCY STOP ACTIVATED', { duration: 10000 });
      queryClient.invalidateQueries({ queryKey: ['risk-status'] });
    },
  });

  const resumeMutation = useMutation({
    mutationFn: () => mevService.resumeOperations(),
    onSuccess: () => {
      toast.success('Operations resumed');
      queryClient.invalidateQueries({ queryKey: ['risk-status'] });
    },
  });

  const throttleMutation = useMutation({
    mutationFn: (percent: number) => mevService.setThrottle(percent),
    onSuccess: (_, percent) => {
      toast.info(`Throttled to ${percent}%`);
      queryClient.invalidateQueries({ queryKey: ['risk-status'] });
    },
  });

  return {
    riskStatus,
    isLoading,
    emergencyStop: emergencyStopMutation.mutate,
    resume: resumeMutation.mutate,
    setThrottle: throttleMutation.mutate,
    isEmergencyStopping: emergencyStopMutation.isPending,
    isResuming: resumeMutation.isPending,
  };
};

// ============= Bundle Simulation Hook =============
export const useBundleSimulation = () => {
  return useMutation({
    mutationFn: mevService.simulateBundle,
    onSuccess: (result) => {
      toast.info(`Simulation: ${(result.successProbability * 100).toFixed(1)}% success, $${result.expectedProfit.toFixed(2)} profit`);
    },
  });
};

// ============= MEV Stats Hook =============
export const useMEVStats = () => {
  return useQuery({
    queryKey: ['mev-stats'],
    queryFn: () => mevService.getStats(),
    refetchInterval: 10000,
  });
};

// ============= Audit Trail Hook =============
export const useAuditTrail = (limit: number = 100) => {
  return useQuery({
    queryKey: ['audit-trail', limit],
    queryFn: () => mevService.getAuditTrail(limit),
    refetchInterval: 30000,
  });
};

// ============= Execution Stream Hook =============
export const useExecutionStream = () => {
  const [executions, setExecutions] = useState<ExecutionResult[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const cleanup = mevService.connectExecutionStream(
      (execution) => {
        setExecutions(prev => [execution, ...prev].slice(0, 50));
        setIsConnected(true);
        
        // Show toast for important executions
        if (execution.success && execution.actualProfit && execution.actualProfit > 100) {
          toast.success(`💰 Big win! $${execution.actualProfit.toFixed(2)} profit`);
        }
      },
      (error) => {
        console.error('Execution stream error:', error);
        setIsConnected(false);
      }
    );

    return cleanup;
  }, []);

  return { executions, isConnected };
};

// ============= Combined MEV Dashboard Hook =============
export const useMEVDashboard = () => {
  const opportunities = useMEVOpportunities(true);
  const execute = useExecuteOpportunity();
  const metrics = useSystemMetrics();
  const bandit = useBanditStats();
  const risk = useRiskManagement();
  const stats = useMEVStats();
  const executions = useExecutionStream();

  return {
    opportunities: opportunities.opportunities,
    isOpportunitiesConnected: opportunities.isConnected,
    executeOpportunity: execute.mutate,
    isExecuting: execute.isPending,
    
    metrics: metrics.metrics,
    isMetricsConnected: metrics.isConnected,
    
    banditStats: bandit.stats,
    resetBandit: bandit.reset,
    
    riskStatus: risk.riskStatus,
    emergencyStop: risk.emergencyStop,
    resumeOperations: risk.resume,
    setThrottle: risk.setThrottle,
    
    mevStats: stats.data,
    
    executions: executions.executions,
    isExecutionsConnected: executions.isConnected,
    
    isLoading: opportunities.isLoading || bandit.isLoading || risk.isLoading || stats.isLoading,
  };
};