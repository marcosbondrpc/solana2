import React from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

// Linear Progress Bar
export interface ProgressProps {
  value: number;
  max?: number;
  variant?: 'primary' | 'success' | 'warning' | 'danger' | 'gradient';
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
  striped?: boolean;
  className?: string;
}

export const Progress: React.FC<ProgressProps> = ({
  value,
  max = 100,
  variant = 'primary',
  size = 'md',
  showLabel = false,
  label,
  animated = true,
  striped = false,
  className
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  
  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-4'
  };
  
  const variantClasses = {
    primary: 'from-cyan-500 to-blue-500',
    success: 'from-green-500 to-emerald-500',
    warning: 'from-amber-500 to-orange-500',
    danger: 'from-red-500 to-pink-500',
    gradient: 'from-purple-500 via-pink-500 to-cyan-500'
  };
  
  return (
    <div className={cn("relative", className)}>
      {(showLabel || label) && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-400">{label || 'Progress'}</span>
          <span className="text-sm font-medium text-white">{percentage.toFixed(0)}%</span>
        </div>
      )}
      <div className={cn(
        "w-full bg-gray-800 rounded-full overflow-hidden",
        sizeClasses[size]
      )}>
        <motion.div
          className={cn(
            "h-full bg-gradient-to-r rounded-full relative overflow-hidden",
            variantClasses[variant]
          )}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ 
            duration: animated ? 1 : 0, 
            ease: "easeOut" 
          }}
        >
          {striped && (
            <div 
              className="absolute inset-0 opacity-20"
              style={{
                backgroundImage: `repeating-linear-gradient(
                  45deg,
                  transparent,
                  transparent 10px,
                  rgba(255,255,255,0.1) 10px,
                  rgba(255,255,255,0.1) 20px
                )`,
                animation: animated ? 'progress-stripes 1s linear infinite' : undefined
              }}
            />
          )}
          {animated && percentage > 0 && percentage < 100 && (
            <div className="absolute right-0 top-0 h-full w-8 bg-gradient-to-r from-transparent to-white/20 animate-shimmer" />
          )}
        </motion.div>
      </div>
    </div>
  );
};

// Circular Progress
export interface CircularProgressProps {
  value: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  variant?: 'primary' | 'success' | 'warning' | 'danger' | 'gradient';
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
  className?: string;
}

export const CircularProgress: React.FC<CircularProgressProps> = ({
  value,
  max = 100,
  size = 120,
  strokeWidth = 8,
  variant = 'primary',
  showLabel = true,
  label,
  animated = true,
  className
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;
  
  const colors = {
    primary: ['#00D4FF', '#0080FF'],
    success: ['#00FF88', '#00CC6B'],
    warning: ['#FFB800', '#FF9500'],
    danger: ['#FF3366', '#FF0040'],
    gradient: ['#9945FF', '#00D4FF']
  };
  
  const [startColor, endColor] = colors[variant];
  const gradientId = `gradient-${Math.random().toString(36).substr(2, 9)}`;
  
  return (
    <div className={cn("relative inline-flex items-center justify-center", className)}>
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
      >
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={startColor} />
            <stop offset="100%" stopColor={endColor} />
          </linearGradient>
        </defs>
        
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth={strokeWidth}
          fill="none"
        />
        
        {/* Progress circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={`url(#${gradientId})`}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ 
            duration: animated ? 1 : 0, 
            ease: "easeOut" 
          }}
          style={{
            strokeDasharray: circumference
          }}
        />
      </svg>
      
      {showLabel && (
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold text-white">
            {percentage.toFixed(0)}%
          </span>
          {label && (
            <span className="text-xs text-gray-400 mt-1">{label}</span>
          )}
        </div>
      )}
    </div>
  );
};

// Multi-segment Progress
export interface SegmentedProgressProps {
  segments: {
    value: number;
    color: string;
    label?: string;
  }[];
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  showLabels?: boolean;
  animated?: boolean;
  className?: string;
}

export const SegmentedProgress: React.FC<SegmentedProgressProps> = ({
  segments,
  max = 100,
  size = 'md',
  showLabels = false,
  animated = true,
  className
}) => {
  const sizeClasses = {
    sm: 'h-2',
    md: 'h-4',
    lg: 'h-6'
  };
  
  const total = segments.reduce((sum, segment) => sum + segment.value, 0);
  
  return (
    <div className={cn("relative", className)}>
      {showLabels && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-400">Total</span>
          <span className="text-sm font-medium text-white">
            {total} / {max}
          </span>
        </div>
      )}
      <div className={cn(
        "w-full bg-gray-800 rounded-full overflow-hidden flex",
        sizeClasses[size]
      )}>
        {segments.map((segment, index) => {
          const width = (segment.value / max) * 100;
          return (
            <motion.div
              key={index}
              className="h-full relative"
              style={{ background: segment.color }}
              initial={{ width: 0 }}
              animate={{ width: `${width}%` }}
              transition={{ 
                duration: animated ? 0.5 : 0, 
                delay: animated ? index * 0.1 : 0,
                ease: "easeOut" 
              }}
            >
              {segment.label && size === 'lg' && width > 10 && (
                <span className="absolute inset-0 flex items-center justify-center text-xs text-white font-medium">
                  {segment.label}
                </span>
              )}
            </motion.div>
          );
        })}
      </div>
      {showLabels && segments.length > 1 && (
        <div className="flex flex-wrap gap-4 mt-3">
          {segments.map((segment, index) => (
            <div key={index} className="flex items-center gap-2">
              <div 
                className="w-3 h-3 rounded-full"
                style={{ background: segment.color }}
              />
              <span className="text-xs text-gray-400">
                {segment.label || `Segment ${index + 1}`}: {segment.value}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Loading Dots
export const LoadingDots: React.FC<{ className?: string }> = ({ className }) => {
  return (
    <div className={cn("flex gap-1", className)}>
      {[0, 1, 2].map((index) => (
        <motion.div
          key={index}
          className="w-2 h-2 bg-cyan-500 rounded-full"
          animate={{
            y: [0, -10, 0],
            opacity: [0.5, 1, 0.5]
          }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
            delay: index * 0.15
          }}
        />
      ))}
    </div>
  );
};

// CSS for striped animation (add to your global CSS)
const styles = `
@keyframes progress-stripes {
  from { background-position: 0 0; }
  to { background-position: 40px 40px; }
}
`;

if (typeof document !== 'undefined') {
  const styleSheet = document.createElement("style");
  styleSheet.textContent = styles;
  document.head.appendChild(styleSheet);
}