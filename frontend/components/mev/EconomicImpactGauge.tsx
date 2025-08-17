'use client';

import { memo, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ImpactMetric {
  label: string;
  value: number;
  change: number;
  unit: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export const EconomicImpactGauge = memo(() => {
  const [metrics, setMetrics] = useState<ImpactMetric[]>([
    {
      label: 'Daily Extraction',
      value: 2847293,
      change: 12.4,
      unit: 'USD',
      severity: 'high'
    },
    {
      label: 'Victim Count',
      value: 18472,
      change: -5.2,
      unit: 'wallets',
      severity: 'critical'
    },
    {
      label: 'Avg Loss/Victim',
      value: 154.23,
      change: 8.7,
      unit: 'USD',
      severity: 'medium'
    },
    {
      label: 'Network Impact',
      value: 0.82,
      change: 0.03,
      unit: '%',
      severity: 'medium'
    },
    {
      label: 'Gas Overhead',
      value: 487293,
      change: 15.2,
      unit: 'USD',
      severity: 'high'
    },
    {
      label: 'Failed TXs',
      value: 3.2,
      change: -0.8,
      unit: '%',
      severity: 'low'
    }
  ]);
  
  const [gaugeValue, setGaugeValue] = useState(0);
  const [targetValue, setTargetValue] = useState(75);
  
  // Animate gauge
  useEffect(() => {
    const interval = setInterval(() => {
      setTargetValue(40 + Math.random() * 50);
    }, 3000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Smooth gauge animation
  useEffect(() => {
    const timer = setInterval(() => {
      setGaugeValue(prev => {
        const diff = targetValue - prev;
        if (Math.abs(diff) < 0.5) return targetValue;
        return prev + diff * 0.1;
      });
    }, 16); // 60fps
    
    return () => clearInterval(timer);
  }, [targetValue]);
  
  // Update metrics periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => prev.map(metric => ({
        ...metric,
        value: metric.value * (0.95 + Math.random() * 0.1),
        change: (Math.random() - 0.5) * 20
      })));
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  const getGaugeColor = (value: number) => {
    if (value < 30) return '#10b981';
    if (value < 50) return '#f59e0b';
    if (value < 70) return '#ef4444';
    return '#dc2626';
  };
  
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'text-green-500';
      case 'medium': return 'text-yellow-500';
      case 'high': return 'text-orange-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };
  
  return (
    <div className="space-y-6">
      {/* Main Gauge */}
      <div className="relative h-48">
        <svg className="w-full h-full" viewBox="0 0 200 120">
          {/* Background arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#374151"
            strokeWidth="10"
            strokeLinecap="round"
          />
          
          {/* Value arc with GPU acceleration */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={getGaugeColor(gaugeValue)}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${gaugeValue * 2.51} 251`}
            style={{
              transition: 'stroke-dasharray 100ms linear, stroke 500ms ease',
              transform: 'translateZ(0)',
              willChange: 'stroke-dasharray'
            }}
          />
          
          {/* Needle */}
          <g
            style={{
              transform: `rotate(${-90 + gaugeValue * 1.8}deg)`,
              transformOrigin: '100px 100px',
              transition: 'transform 100ms linear',
              willChange: 'transform'
            }}
          >
            <line
              x1="100"
              y1="100"
              x2="100"
              y2="30"
              stroke="#fff"
              strokeWidth="2"
            />
            <circle cx="100" cy="100" r="4" fill="#fff" />
          </g>
          
          {/* Value text */}
          <text
            x="100"
            y="85"
            textAnchor="middle"
            className="fill-white text-2xl font-bold"
          >
            {gaugeValue.toFixed(0)}%
          </text>
          
          <text
            x="100"
            y="105"
            textAnchor="middle"
            className="fill-gray-400 text-xs"
          >
            Economic Impact Score
          </text>
        </svg>
        
        {/* Scale labels */}
        <div className="absolute bottom-0 left-4 text-xs text-gray-500">0</div>
        <div className="absolute bottom-0 right-4 text-xs text-gray-500">100</div>
      </div>
      
      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-3">
        {metrics.map((metric, index) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="bg-gray-900/50 rounded-lg p-3 border border-gray-800"
          >
            <div className="flex justify-between items-start mb-1">
              <span className="text-xs text-gray-500">{metric.label}</span>
              <span className={`text-xs ${getSeverityColor(metric.severity)}`}>
                ●
              </span>
            </div>
            
            <div className="flex items-baseline gap-2">
              <span className="text-lg font-bold">
                {metric.unit === 'USD' ? '$' : ''}
                {metric.value >= 1000 
                  ? `${(metric.value / 1000).toFixed(1)}k`
                  : metric.value.toFixed(metric.unit === '%' ? 2 : 0)
                }
                {metric.unit === '%' ? '%' : ''}
              </span>
              
              <span className={`text-xs flex items-center ${
                metric.change > 0 ? 'text-red-400' : 'text-green-400'
              }`}>
                {metric.change > 0 ? '↑' : '↓'}
                {Math.abs(metric.change).toFixed(1)}%
              </span>
            </div>
            
            {metric.unit !== 'USD' && metric.unit !== '%' && (
              <span className="text-xs text-gray-600">{metric.unit}</span>
            )}
          </motion.div>
        ))}
      </div>
      
      {/* Alert Banner */}
      <AnimatePresence>
        {gaugeValue > 70 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-red-900/20 border border-red-800 rounded-lg p-3"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="text-xs text-red-400">
                HIGH IMPACT DETECTED: Unusual MEV extraction activity
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Historical Trend */}
      <div className="mt-4">
        <div className="text-xs text-gray-500 mb-2">24h Trend</div>
        <div className="h-8 flex items-end gap-1">
          {Array.from({ length: 24 }, (_, i) => {
            const height = 20 + Math.random() * 60;
            const isRecent = i >= 20;
            return (
              <div
                key={i}
                className={`flex-1 rounded-t transition-all duration-500 ${
                  isRecent ? 'bg-red-500/60' : 'bg-gray-700'
                }`}
                style={{
                  height: `${height}%`,
                  transform: 'translateZ(0)',
                  willChange: 'height'
                }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
});

EconomicImpactGauge.displayName = 'EconomicImpactGauge';