'use client';

import { useEffect, useState } from 'react';
import { clsx } from 'clsx';
import { motion } from 'framer-motion';
import { ClickHouseQueries } from '../../lib/clickhouse';

interface LatencyBucket {
  latency_bucket: number;
  count: number;
  success_rate: number;
}

export default function LatencyHeatmap() {
  const [distribution, setDistribution] = useState<LatencyBucket[]>([]);
  const [maxCount, setMaxCount] = useState(0);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await ClickHouseQueries.getLatencyDistribution();
        setDistribution(data);
        setMaxCount(Math.max(...data.map(d => d.count)));
      } catch (err) {
        console.error('Failed to fetch latency distribution:', err);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);
  
  const getColor = (bucket: LatencyBucket) => {
    const intensity = maxCount > 0 ? bucket.count / maxCount : 0;
    
    if (bucket.latency_bucket <= 8) {
      // Excellent latency - green
      return `rgba(34, 197, 94, ${0.2 + intensity * 0.8})`;
    } else if (bucket.latency_bucket <= 20) {
      // Good latency - yellow
      return `rgba(234, 179, 8, ${0.2 + intensity * 0.8})`;
    } else {
      // Poor latency - red
      return `rgba(239, 68, 68, ${0.2 + intensity * 0.8})`;
    }
  };
  
  return (
    <div className="glass rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-zinc-400">Latency Distribution</h3>
        <div className="flex items-center gap-2 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded bg-green-500" />
            <span className="text-zinc-500">≤8ms</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded bg-yellow-500" />
            <span className="text-zinc-500">≤20ms</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded bg-red-500" />
            <span className="text-zinc-500">>20ms</span>
          </div>
        </div>
      </div>
      
      <div className="space-y-2">
        {distribution.map((bucket) => (
          <div key={bucket.latency_bucket} className="flex items-center gap-2">
            <span className="text-xs text-zinc-500 w-12 text-right">
              {bucket.latency_bucket}-{bucket.latency_bucket + 5}ms
            </span>
            <div className="flex-1 h-6 bg-zinc-900 rounded overflow-hidden relative">
              <motion.div
                className="absolute inset-y-0 left-0 rounded"
                style={{ backgroundColor: getColor(bucket) }}
                initial={{ width: 0 }}
                animate={{ 
                  width: maxCount > 0 ? `${(bucket.count / maxCount) * 100}%` : '0%' 
                }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              >
                <div className="absolute inset-0 flex items-center px-2">
                  <span className="text-xs text-white/80 font-medium">
                    {bucket.count}
                  </span>
                </div>
              </motion.div>
            </div>
            <span className={clsx(
              "text-xs w-12",
              bucket.success_rate >= 65 ? "text-green-400" :
              bucket.success_rate >= 55 ? "text-yellow-400" : "text-red-400"
            )}>
              {bucket.success_rate.toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
      
      {distribution.length === 0 && (
        <div className="text-center py-8 text-zinc-600 text-sm">
          No latency data available
        </div>
      )}
    </div>
  );
}