'use client';

import { useEffect, useState } from 'react';
import { useSnapshot } from 'valtio';
import { mevStore } from '../../stores/mevStore';

interface RiskAlert {
  id: string;
  type: 'honeypot' | 'adversary' | 'slippage' | 'frontrun' | 'policy';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: number;
  details?: any;
}

interface PolicyGate {
  name: string;
  enabled: boolean;
  status: 'active' | 'triggered' | 'disabled';
  threshold: number;
  current: number;
}

export default function RiskMonitor() {
  const store = useSnapshot(mevStore);
  const [alerts, setAlerts] = useState<RiskAlert[]>([]);
  const [adversaryDensity, setAdversaryDensity] = useState<number[][]>([]);
  const [tokenRisks, setTokenRisks] = useState<Record<string, number>>({});
  const [policyGates, setPolicyGates] = useState<PolicyGate[]>([
    { name: 'Latency P99', enabled: true, status: 'active', threshold: 20, current: 0 },
    { name: 'Land Rate', enabled: true, status: 'active', threshold: 55, current: 0 },
    { name: 'Negative EV', enabled: true, status: 'active', threshold: 1, current: 0 },
    { name: 'Gas Spike', enabled: true, status: 'active', threshold: 500, current: 120 },
    { name: 'Adversary Count', enabled: true, status: 'active', threshold: 10, current: 3 },
  ]);
  const [killSwitchArmed, setKillSwitchArmed] = useState(true);

  // Generate adversary density heatmap
  useEffect(() => {
    const generateDensity = () => {
      const grid = Array(5).fill(null).map(() =>
        Array(10).fill(null).map(() => Math.random() * Math.random()) // Quadratic for clustering
      );
      setAdversaryDensity(grid);
    };

    generateDensity();
    const interval = setInterval(generateDensity, 5000);
    return () => clearInterval(interval);
  }, []);

  // Update policy gates from store
  useEffect(() => {
    setPolicyGates(prev => prev.map(gate => {
      switch (gate.name) {
        case 'Latency P99':
          return {
            ...gate,
            current: store.health.decisionLatencyP99,
            status: store.health.decisionLatencyP99 > gate.threshold ? 'triggered' : 'active',
          };
        case 'Land Rate':
          return {
            ...gate,
            current: store.health.bundleLandRate,
            status: store.health.bundleLandRate < gate.threshold ? 'triggered' : 'active',
          };
        default:
          return gate;
      }
    }));
  }, [store.health]);

  // Generate risk alerts
  useEffect(() => {
    const interval = setInterval(() => {
      const rand = Math.random();
      if (rand < 0.1) { // 10% chance of alert
        const alertTypes: RiskAlert['type'][] = ['honeypot', 'adversary', 'slippage', 'frontrun', 'policy'];
        const severities: RiskAlert['severity'][] = ['low', 'medium', 'high', 'critical'];
        
        const type = alertTypes[Math.floor(Math.random() * alertTypes.length)];
        const severity = rand < 0.02 ? 'critical' : 
                        rand < 0.05 ? 'high' :
                        rand < 0.08 ? 'medium' : 'low';

        const messages: Record<RiskAlert['type'], string> = {
          honeypot: 'Potential honeypot detected in token 0x7a3...',
          adversary: 'New adversary bot detected competing for routes',
          slippage: 'High slippage detected on Raydium pool',
          frontrun: 'Possible frontrun attempt on transaction',
          policy: 'Policy gate threshold approaching limit',
        };

        const newAlert: RiskAlert = {
          id: `alert-${Date.now()}`,
          type,
          severity,
          message: messages[type],
          timestamp: Date.now(),
        };

        setAlerts(prev => [newAlert, ...prev].slice(0, 10));
      }
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  // Generate token risks
  useEffect(() => {
    const tokens = [
      '0x7a3f...', '0x9b2c...', '0x1d8e...', '0x4f6a...', '0x8c3d...',
    ];
    
    const interval = setInterval(() => {
      const risks: Record<string, number> = {};
      tokens.forEach(token => {
        risks[token] = Math.random();
      });
      setTokenRisks(risks);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getSeverityColor = (severity: RiskAlert['severity']) => {
    switch (severity) {
      case 'critical': return 'text-red-500 bg-red-900/20';
      case 'high': return 'text-orange-500 bg-orange-900/20';
      case 'medium': return 'text-yellow-500 bg-yellow-900/20';
      case 'low': return 'text-blue-500 bg-blue-900/20';
    }
  };

  const getAlertIcon = (type: RiskAlert['type']) => {
    switch (type) {
      case 'honeypot': return '🍯';
      case 'adversary': return '⚔️';
      case 'slippage': return '📉';
      case 'frontrun': return '🏃';
      case 'policy': return '🚧';
    }
  };

  const getRiskColor = (risk: number) => {
    if (risk < 0.3) return 'bg-green-600';
    if (risk < 0.6) return 'bg-yellow-600';
    if (risk < 0.8) return 'bg-orange-600';
    return 'bg-red-600';
  };

  const handleKillSwitch = async () => {
    if (!killSwitchArmed) return;
    
    if (confirm('Activate kill switch? This will stop all trading immediately.')) {
      try {
        const response = await fetch('http://localhost:8000/api/control/kill-switch', {
          method: 'POST',
        });
        if (response.ok) {
          setKillSwitchArmed(false);
          console.log('Kill switch activated');
        }
      } catch (error) {
        console.error('Failed to activate kill switch:', error);
      }
    }
  };

  return (
    <div className="h-full flex flex-col">
      <h3 className="text-sm font-semibold text-zinc-400 mb-2">RISK MONITOR</h3>
      
      {/* Adversary Density Heatmap */}
      <div className="mb-2">
        <div className="text-[10px] text-zinc-600 mb-1">ADVERSARY DENSITY</div>
        <div className="grid grid-cols-10 gap-[1px]">
          {adversaryDensity.flat().map((density, idx) => (
            <div
              key={idx}
              className="h-3 transition-all duration-500"
              style={{
                backgroundColor: `rgba(239, 68, 68, ${density * 0.8})`,
              }}
            />
          ))}
        </div>
      </div>

      {/* Token Risk Scores */}
      <div className="mb-2">
        <div className="text-[10px] text-zinc-600 mb-1">TOKEN RISKS</div>
        <div className="space-y-1">
          {Object.entries(tokenRisks).slice(0, 3).map(([token, risk]) => (
            <div key={token} className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-zinc-500 w-16">{token}</span>
              <div className="flex-1 h-2 bg-zinc-900 rounded-full overflow-hidden">
                <div 
                  className={`h-full transition-all duration-300 ${getRiskColor(risk)}`}
                  style={{ width: `${risk * 100}%` }}
                />
              </div>
              <span className="text-[10px] font-mono text-zinc-400 w-8 text-right">
                {(risk * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Policy Gates */}
      <div className="mb-2">
        <div className="text-[10px] text-zinc-600 mb-1">POLICY GATES</div>
        <div className="grid grid-cols-2 gap-1 text-[10px]">
          {policyGates.map(gate => (
            <div 
              key={gate.name}
              className={`flex items-center justify-between px-1 py-0.5 rounded ${
                gate.status === 'triggered' ? 'bg-red-900/30 text-red-400' :
                gate.status === 'active' ? 'bg-zinc-900 text-zinc-400' :
                'bg-zinc-900/50 text-zinc-600'
              }`}
            >
              <span className="truncate">{gate.name}</span>
              <span className="font-mono">
                {gate.current.toFixed(0)}/{gate.threshold}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Alerts */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <div className="text-[10px] text-zinc-600 mb-1">ALERTS</div>
        <div className="space-y-1 overflow-y-auto max-h-[60px]">
          {alerts.map(alert => (
            <div 
              key={alert.id}
              className={`flex items-center gap-1 px-1 py-0.5 rounded text-[10px] ${getSeverityColor(alert.severity)}`}
            >
              <span>{getAlertIcon(alert.type)}</span>
              <span className="flex-1 truncate">{alert.message}</span>
              <span className="text-[9px] opacity-60">
                {new Date(alert.timestamp).toTimeString().slice(0, 8)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Kill Switch */}
      <div className="mt-2 pt-2 border-t border-zinc-800">
        <button
          onClick={handleKillSwitch}
          disabled={!killSwitchArmed}
          className={`w-full py-2 rounded font-bold text-xs transition-all ${
            killSwitchArmed 
              ? 'bg-red-900 hover:bg-red-800 text-red-300 border border-red-700' 
              : 'bg-zinc-900 text-zinc-600 border border-zinc-800 cursor-not-allowed'
          }`}
        >
          {killSwitchArmed ? 'KILL SWITCH ARMED' : 'KILL SWITCH ACTIVATED'}
        </button>
      </div>
    </div>
  );
}