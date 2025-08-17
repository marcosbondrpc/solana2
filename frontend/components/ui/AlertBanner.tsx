'use client';

import { memo } from 'react';
import { motion } from 'framer-motion';

interface AlertBannerProps {
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
  onClose?: () => void;
}

export const AlertBanner = memo(({ type, message, onClose }: AlertBannerProps) => {
  const styles = {
    info: {
      bg: 'bg-blue-900/20',
      border: 'border-blue-800',
      text: 'text-blue-400',
      icon: 'ℹ️'
    },
    warning: {
      bg: 'bg-yellow-900/20',
      border: 'border-yellow-800',
      text: 'text-yellow-400',
      icon: '⚠️'
    },
    error: {
      bg: 'bg-red-900/20',
      border: 'border-red-800',
      text: 'text-red-400',
      icon: '⛔'
    },
    success: {
      bg: 'bg-green-900/20',
      border: 'border-green-800',
      text: 'text-green-400',
      icon: '✅'
    }
  };
  
  const style = styles[type];
  
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${style.bg} ${style.border} border rounded-lg p-4 flex items-center justify-between`}
    >
      <div className="flex items-center gap-3">
        <span className="text-xl">{style.icon}</span>
        <span className={`text-sm ${style.text}`}>{message}</span>
      </div>
      
      {onClose && (
        <button
          onClick={onClose}
          className={`${style.text} hover:opacity-70 transition-opacity`}
        >
          ✕
        </button>
      )}
    </motion.div>
  );
});

AlertBanner.displayName = 'AlertBanner';