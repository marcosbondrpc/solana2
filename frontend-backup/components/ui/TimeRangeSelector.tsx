'use client';

import { memo, useState } from 'react';
import { motion } from 'framer-motion';

const TIME_RANGES = [
  { label: '1H', value: '1h' },
  { label: '6H', value: '6h' },
  { label: '24H', value: '24h' },
  { label: '7D', value: '7d' },
  { label: '30D', value: '30d' }
];

export const TimeRangeSelector = memo(() => {
  const [selected, setSelected] = useState('24h');
  
  return (
    <div className="flex items-center gap-1 bg-gray-900/50 rounded-lg p-1">
      {TIME_RANGES.map((range) => (
        <button
          key={range.value}
          onClick={() => setSelected(range.value)}
          className={`relative px-3 py-1.5 text-xs font-medium transition-colors ${
            selected === range.value
              ? 'text-white'
              : 'text-gray-500 hover:text-gray-300'
          }`}
        >
          {selected === range.value && (
            <motion.div
              layoutId="timeRangeIndicator"
              className="absolute inset-0 bg-gray-800 rounded"
              transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            />
          )}
          <span className="relative z-10">{range.label}</span>
        </button>
      ))}
    </div>
  );
});

TimeRangeSelector.displayName = 'TimeRangeSelector';