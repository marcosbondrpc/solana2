import React from 'react';
import { cn } from '@/lib/utils';

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'text' | 'circular' | 'rectangular' | 'rounded';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className,
  variant = 'text',
  width,
  height,
  animation = 'pulse',
  ...props
}) => {
  const variantClasses = {
    text: 'rounded h-4',
    circular: 'rounded-full',
    rectangular: 'rounded-none',
    rounded: 'rounded-lg'
  };
  
  const animationClasses = {
    pulse: 'animate-pulse',
    wave: 'skeleton-wave',
    none: ''
  };
  
  const style: React.CSSProperties = {
    width: width || (variant === 'circular' ? '40px' : '100%'),
    height: height || (variant === 'circular' ? '40px' : variant === 'text' ? '20px' : '100px')
  };
  
  return (
    <div
      className={cn(
        "bg-gradient-to-r from-gray-800 via-gray-700 to-gray-800",
        "bg-[length:200%_100%]",
        variantClasses[variant],
        animationClasses[animation],
        className
      )}
      style={style}
      {...props}
    />
  );
};

// Skeleton Card Component
export const SkeletonCard: React.FC<{ className?: string }> = ({ className }) => {
  return (
    <div className={cn("p-6 space-y-4 bg-color-surface rounded-lg border border-color-border", className)}>
      <div className="flex items-center justify-between">
        <Skeleton variant="text" width="40%" height={24} />
        <Skeleton variant="circular" width={32} height={32} />
      </div>
      <Skeleton variant="text" width="100%" />
      <Skeleton variant="text" width="80%" />
      <Skeleton variant="text" width="60%" />
    </div>
  );
};

// Skeleton Table Component
export const SkeletonTable: React.FC<{ rows?: number; columns?: number; className?: string }> = ({ 
  rows = 5, 
  columns = 4,
  className 
}) => {
  return (
    <div className={cn("overflow-hidden rounded-lg border border-color-border", className)}>
      <div className="bg-color-surface">
        {/* Header */}
        <div className="border-b border-color-border p-4">
          <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
            {Array.from({ length: columns }).map((_, i) => (
              <Skeleton key={i} variant="text" height={20} />
            ))}
          </div>
        </div>
        {/* Rows */}
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div key={rowIndex} className="border-b border-color-border last:border-0 p-4">
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
              {Array.from({ length: columns }).map((_, colIndex) => (
                <Skeleton 
                  key={colIndex} 
                  variant="text" 
                  height={16}
                  width={colIndex === 0 ? "60%" : "80%"}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Skeleton List Component
export const SkeletonList: React.FC<{ items?: number; className?: string }> = ({ 
  items = 5,
  className 
}) => {
  return (
    <div className={cn("space-y-3", className)}>
      {Array.from({ length: items }).map((_, index) => (
        <div key={index} className="flex items-center gap-4 p-4 bg-color-surface rounded-lg border border-color-border">
          <Skeleton variant="circular" width={48} height={48} />
          <div className="flex-1 space-y-2">
            <Skeleton variant="text" width="30%" height={20} />
            <Skeleton variant="text" width="50%" height={16} />
          </div>
          <Skeleton variant="rounded" width={80} height={32} />
        </div>
      ))}
    </div>
  );
};

// Skeleton Metric Card
export const SkeletonMetric: React.FC<{ className?: string }> = ({ className }) => {
  return (
    <div className={cn(
      "p-6 bg-gradient-to-br from-gray-800/50 to-gray-900/50",
      "border border-gray-700 rounded-xl",
      className
    )}>
      <div className="flex items-start justify-between mb-4">
        <div className="space-y-2">
          <Skeleton variant="text" width={100} height={16} />
          <Skeleton variant="text" width={120} height={32} />
        </div>
        <Skeleton variant="rounded" width={48} height={48} />
      </div>
      <div className="flex items-center gap-2">
        <Skeleton variant="text" width={60} height={14} />
        <Skeleton variant="text" width={80} height={14} />
      </div>
    </div>
  );
};

// Skeleton Chart
export const SkeletonChart: React.FC<{ className?: string; height?: number }> = ({ 
  className,
  height = 300 
}) => {
  return (
    <div className={cn("relative", className)}>
      <Skeleton variant="rounded" height={height} animation="wave" />
      <div className="absolute bottom-4 left-4 right-4 flex justify-between">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} variant="text" width={30} height={12} />
        ))}
      </div>
    </div>
  );
};

// Dashboard Loading State
export const DashboardSkeleton: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <Skeleton variant="text" width={200} height={32} />
        <Skeleton variant="text" width={100} height={24} />
      </div>
      
      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <SkeletonMetric key={i} />
        ))}
      </div>
      
      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-color-surface rounded-lg border border-color-border p-6">
          <Skeleton variant="text" width={150} height={24} className="mb-4" />
          <SkeletonChart height={250} />
        </div>
        <div className="bg-color-surface rounded-lg border border-color-border p-6">
          <Skeleton variant="text" width={150} height={24} className="mb-4" />
          <SkeletonChart height={250} />
        </div>
      </div>
      
      {/* Activity List */}
      <div className="bg-color-surface rounded-lg border border-color-border p-6">
        <Skeleton variant="text" width={180} height={24} className="mb-4" />
        <SkeletonList items={3} />
      </div>
    </div>
  );
};

// Add wave animation CSS
const styles = `
@keyframes skeleton-wave {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
}

.skeleton-wave {
  animation: skeleton-wave 2s ease-in-out infinite;
}
`;

if (typeof document !== 'undefined') {
  const styleSheet = document.createElement("style");
  styleSheet.textContent = styles;
  document.head.appendChild(styleSheet);
}