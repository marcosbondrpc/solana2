import React from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger' | 'info' | 'purple' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  animated?: boolean;
  pulse?: boolean;
  icon?: React.ReactNode;
  onRemove?: () => void;
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ 
    className, 
    variant = 'default', 
    size = 'md',
    animated = false,
    pulse = false,
    icon,
    onRemove,
    children,
    ...props 
  }, ref) => {
    const sizeClasses = {
      sm: "px-2 py-0.5 text-xs",
      md: "px-2.5 py-1 text-sm",
      lg: "px-3 py-1.5 text-base"
    };
    
    const variantClasses = {
      default: "bg-gray-800 text-gray-300 border-gray-700",
      primary: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
      success: "bg-green-500/20 text-green-400 border-green-500/30",
      warning: "bg-amber-500/20 text-amber-400 border-amber-500/30",
      danger: "bg-red-500/20 text-red-400 border-red-500/30",
      info: "bg-blue-500/20 text-blue-400 border-blue-500/30",
      purple: "bg-purple-500/20 text-purple-400 border-purple-500/30",
      outline: "bg-transparent text-gray-300 border-gray-600"
    };
    
    const badgeContent = (
      <span
        ref={ref}
        className={cn(
          "inline-flex items-center gap-1.5 font-medium rounded-full border backdrop-blur-sm",
          "transition-all duration-300",
          sizeClasses[size],
          variantClasses[variant],
          pulse && "animate-pulse-glow",
          className
        )}
        {...props}
      >
        {icon && <span className="flex-shrink-0">{icon}</span>}
        {children}
        {onRemove && (
          <button
            onClick={onRemove}
            className="ml-1 -mr-1 hover:opacity-75 transition-opacity"
            aria-label="Remove"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </span>
    );
    
    if (animated) {
      return (
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0, opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          {badgeContent}
        </motion.div>
      );
    }
    
    return badgeContent;
  }
);
Badge.displayName = 'Badge';

// Status Badge with live indicator
export interface StatusBadgeProps extends Omit<BadgeProps, 'variant'> {
  status: 'online' | 'offline' | 'busy' | 'away' | 'error';
  showIndicator?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  showIndicator = true,
  children,
  ...props
}) => {
  const statusConfig = {
    online: { variant: 'success' as const, label: 'Online', color: 'bg-green-400' },
    offline: { variant: 'default' as const, label: 'Offline', color: 'bg-gray-400' },
    busy: { variant: 'danger' as const, label: 'Busy', color: 'bg-red-400' },
    away: { variant: 'warning' as const, label: 'Away', color: 'bg-amber-400' },
    error: { variant: 'danger' as const, label: 'Error', color: 'bg-red-400' }
  };
  
  const config = statusConfig[status];
  
  return (
    <Badge variant={config.variant} {...props}>
      {showIndicator && (
        <span className="relative flex h-2 w-2">
          {status === 'online' && (
            <span className={cn(
              "absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping",
              config.color
            )} />
          )}
          <span className={cn(
            "relative inline-flex rounded-full h-2 w-2",
            config.color
          )} />
        </span>
      )}
      {children || config.label}
    </Badge>
  );
};

// Badge Group Component
export interface BadgeGroupProps {
  children: React.ReactNode;
  className?: string;
  max?: number;
}

export const BadgeGroup: React.FC<BadgeGroupProps> = ({ 
  children, 
  className,
  max 
}) => {
  const items = React.Children.toArray(children);
  const visibleItems = max ? items.slice(0, max) : items;
  const hiddenCount = max && items.length > max ? items.length - max : 0;
  
  return (
    <div className={cn("inline-flex items-center gap-2 flex-wrap", className)}>
      {visibleItems}
      {hiddenCount > 0 && (
        <Badge variant="default" size="sm">
          +{hiddenCount}
        </Badge>
      )}
    </div>
  );
};

// Notification Badge
export interface NotificationBadgeProps {
  count: number;
  max?: number;
  variant?: 'primary' | 'danger';
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
  children: React.ReactNode;
}

export const NotificationBadge: React.FC<NotificationBadgeProps> = ({
  count,
  max = 99,
  variant = 'danger',
  position = 'top-right',
  children
}) => {
  const positionClasses = {
    'top-right': '-top-2 -right-2',
    'top-left': '-top-2 -left-2',
    'bottom-right': '-bottom-2 -right-2',
    'bottom-left': '-bottom-2 -left-2'
  };
  
  const variantClasses = {
    primary: 'bg-gradient-to-r from-cyan-500 to-blue-500',
    danger: 'bg-gradient-to-r from-red-500 to-pink-500'
  };
  
  if (count <= 0) return <>{children}</>;
  
  return (
    <div className="relative inline-block">
      {children}
      <motion.span
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        className={cn(
          "absolute flex items-center justify-center",
          "min-w-[20px] h-5 px-1.5 rounded-full",
          "text-xs font-bold text-white",
          variantClasses[variant],
          positionClasses[position]
        )}
      >
        {count > max ? `${max}+` : count}
      </motion.span>
    </div>
  );
};