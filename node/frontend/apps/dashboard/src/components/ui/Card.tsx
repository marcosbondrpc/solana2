import React from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'glass' | 'gradient' | 'outlined';
  hover?: boolean;
  glow?: boolean;
  animated?: boolean;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'glass', hover = true, glow = false, animated = true, children, ...props }, ref) => {
    const baseClasses = "rounded-lg overflow-hidden transition-all duration-300";
    
    const variantClasses = {
      default: "bg-color-surface-solid border border-color-border",
      glass: "glass",
      gradient: "gradient-border",
      outlined: "border-2 border-color-primary bg-transparent"
    };
    
    const hoverClasses = hover ? "hover-lift hover:shadow-glow-sm" : "";
    const glowClasses = glow ? "glow-primary" : "";
    
    const cardContent = (
      <div
        ref={ref}
        className={cn(
          baseClasses,
          variantClasses[variant],
          hoverClasses,
          glowClasses,
          className
        )}
        style={{
          background: variant === 'glass' 
            ? 'linear-gradient(135deg, rgba(20, 22, 28, 0.8) 0%, rgba(20, 22, 28, 0.6) 100%)' 
            : undefined,
          backdropFilter: variant === 'glass' ? 'blur(20px)' : undefined,
          WebkitBackdropFilter: variant === 'glass' ? 'blur(20px)' : undefined,
          border: variant === 'glass' ? '1px solid rgba(255, 255, 255, 0.08)' : undefined,
        }}
        {...props}
      >
        {children}
      </div>
    );
    
    if (animated) {
      return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          {cardContent}
        </motion.div>
      );
    }
    
    return cardContent;
  }
);
Card.displayName = 'Card';

export const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("flex flex-col space-y-1.5 p-6", className)}
      {...props}
    />
  )
);
CardHeader.displayName = 'CardHeader';

export const CardTitle = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn(
        "text-2xl font-semibold leading-none tracking-tight",
        "bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent",
        className
      )}
      {...props}
    />
  )
);
CardTitle.displayName = 'CardTitle';

export const CardDescription = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p
      ref={ref}
      className={cn("text-sm text-gray-400", className)}
      {...props}
    />
  )
);
CardDescription.displayName = 'CardDescription';

export const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
  )
);
CardContent.displayName = 'CardContent';

export const CardFooter = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("flex items-center p-6 pt-0", className)}
      {...props}
    />
  )
);
CardFooter.displayName = 'CardFooter';

// Metric Card Component for dashboard stats
export interface MetricCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string;
  value: string | number;
  change?: {
    value: string | number;
    trend: 'up' | 'down' | 'neutral';
  };
  icon?: React.ReactNode;
  color?: 'primary' | 'success' | 'warning' | 'danger' | 'purple';
}

export const MetricCard = React.forwardRef<HTMLDivElement, MetricCardProps>(
  ({ title, value, change, icon, color = 'primary', className, ...props }, ref) => {
    const colorClasses = {
      primary: 'from-cyan-500/20 to-blue-500/20 border-cyan-500/30',
      success: 'from-green-500/20 to-emerald-500/20 border-green-500/30',
      warning: 'from-amber-500/20 to-orange-500/20 border-amber-500/30',
      danger: 'from-red-500/20 to-pink-500/20 border-red-500/30',
      purple: 'from-purple-500/20 to-indigo-500/20 border-purple-500/30',
    };
    
    const trendColors = {
      up: 'text-green-400',
      down: 'text-red-400',
      neutral: 'text-gray-400',
    };
    
    return (
      <motion.div
        ref={ref}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        whileHover={{ y: -4, transition: { duration: 0.2 } }}
        className={cn(
          "relative overflow-hidden rounded-xl p-6",
          "bg-gradient-to-br",
          colorClasses[color],
          "border backdrop-blur-xl",
          "transition-all duration-300",
          "hover:shadow-2xl hover:shadow-cyan-500/10",
          className
        )}
      >
        {/* Background gradient effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-transparent via-transparent to-black/20" />
        
        {/* Content */}
        <div className="relative z-10">
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-sm font-medium text-gray-400 mb-1">{title}</p>
              <p className="text-3xl font-bold text-white">
                {typeof value === 'number' ? value.toLocaleString() : value}
              </p>
            </div>
            {icon && (
              <div className="p-3 rounded-lg bg-white/5 backdrop-blur">
                {icon}
              </div>
            )}
          </div>
          
          {change && (
            <div className="flex items-center gap-2 text-sm">
              <span className={cn("font-medium", trendColors[change.trend])}>
                {change.trend === 'up' ? '↑' : change.trend === 'down' ? '↓' : '→'}
                {' '}{change.value}
              </span>
              <span className="text-gray-500">vs last period</span>
            </div>
          )}
        </div>
        
        {/* Animated background particles */}
        <div className="absolute inset-0 opacity-30">
          <div className="absolute w-32 h-32 bg-cyan-500 rounded-full blur-3xl animate-pulse" 
               style={{ top: '-20%', right: '-10%' }} />
          <div className="absolute w-24 h-24 bg-purple-500 rounded-full blur-3xl animate-pulse" 
               style={{ bottom: '-15%', left: '-5%', animationDelay: '1s' }} />
        </div>
      </motion.div>
    );
  }
);
MetricCard.displayName = 'MetricCard';