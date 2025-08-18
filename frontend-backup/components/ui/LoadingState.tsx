'use client';

import { memo } from 'react';
import { motion } from 'framer-motion';

export const LoadingState = memo(() => {
  return (
    <div className="flex items-center justify-center p-8">
      <div className="relative">
        {/* Spinning rings */}
        <motion.div
          className="w-16 h-16 border-4 border-gray-800 border-t-blue-500 rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute inset-0 w-16 h-16 border-4 border-gray-800 border-r-purple-500 rounded-full"
          animate={{ rotate: -360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        />
        
        {/* Center dot */}
        <motion.div
          className="absolute inset-0 flex items-center justify-center"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 1, repeat: Infinity }}
        >
          <div className="w-2 h-2 bg-white rounded-full" />
        </motion.div>
      </div>
    </div>
  );
});

LoadingState.displayName = 'LoadingState';