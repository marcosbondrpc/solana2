'use client';

import { memo, useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface Metric {
  label: string;
  value: string | number;
  change: number;
  icon: string;
  color: string;
  unit?: string;
}

export const MetricsOverview = memo(() => {
  const [metrics, setMetrics] = useState<Metric[]>([
    {
      label: 'Active Sandwiches',
      value: 42,
      change: 15.2,
      icon: '🥪',
      color: 'from-red-600 to-pink-600',
      unit: 'attacks'
    },
    {
      label: 'Detection Rate',
      value: 94.7,
      change: 2.3,
      icon: '🎯',
      color: 'from-green-600 to-emerald-600',
      unit: '%'
    },
    {
      label: 'Entities Tracked',
      value: 1847,
      change: 8.4,
      icon: '👁',
      color: 'from-blue-600 to-cyan-600',
      unit: 'wallets'
    },
    {
      label: 'Network Coverage',
      value: 87.3,
      change: -1.2,
      icon: '🌐',
      color: 'from-purple-600 to-indigo-600',
      unit: '%'
    },
    {
      label: 'Avg Response Time',
      value: 7.2,
      change: -12.5,
      icon: '⚡',
      color: 'from-yellow-600 to-orange-600',
      unit: 'ms'
    },
    {
      label: 'Data Ingestion',
      value: 235,
      change: 5.7,
      icon: '📊',
      color: 'from-teal-600 to-green-600',
      unit: 'k/s'
    }
  ]);
  
  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => prev.map(metric => ({
        ...metric,
        value: typeof metric.value === 'number' 
          ? metric.value * (0.95 + Math.random() * 0.1)
          : metric.value,
        change: (Math.random() - 0.5) * 30
      })));
    }, 3000);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {metrics.map((metric, index) => (
        <motion.div
          key={metric.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          whileHover={{ scale: 1.05 }}
          className="relative overflow-hidden rounded-xl bg-gray-900/50 backdrop-blur-sm border border-gray-800 p-4"
        >
          {/* Gradient Background */}
          <div 
            className={`absolute inset-0 opacity-10 bg-gradient-to-br ${metric.color}`}
          />
          
          {/* Icon */}
          <div className="text-2xl mb-2">{metric.icon}</div>
          
          {/* Label */}
          <div className="text-xs text-gray-500 mb-1">{metric.label}</div>
          
          {/* Value */}
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold">
              {typeof metric.value === 'number' 
                ? metric.value >= 1000 
                  ? `${(metric.value / 1000).toFixed(1)}k`
                  : metric.value.toFixed(metric.unit === '%' || metric.unit === 'ms' ? 1 : 0)
                : metric.value
              }
            </span>
            {metric.unit && (
              <span className="text-xs text-gray-500">{metric.unit}</span>
            )}
          </div>
          
          {/* Change Indicator */}
          <div className="mt-2 flex items-center gap-1">
            <span className={`text-xs flex items-center ${
              metric.change > 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {metric.change > 0 ? '↑' : '↓'}
              {Math.abs(metric.change).toFixed(1)}%
            </span>
            <span className="text-xs text-gray-600">24h</span>
          </div>
          
          {/* Sparkline */}
          <div className="absolute bottom-0 left-0 right-0 h-8 opacity-20">
            <svg className="w-full h-full" preserveAspectRatio="none">
              <polyline
                points={Array.from({ length: 20 }, (_, i) => 
                  `${i * 5},${20 - Math.random() * 20}`
                ).join(' ')}
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
                className="text-white"
              />
            </svg>
          </div>
        </motion.div>
      ))}
    </div>
  );
});

MetricsOverview.displayName = 'MetricsOverview';