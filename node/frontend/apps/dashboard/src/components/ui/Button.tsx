import React from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'success' | 'gradient';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
  animated?: boolean;
  glow?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ 
    className, 
    variant = 'primary', 
    size = 'md', 
    loading = false,
    icon,
    iconPosition = 'left',
    fullWidth = false,
    animated = true,
    glow = false,
    disabled,
    children,
    ...props 
  }, ref) => {
    const baseClasses = "relative inline-flex items-center justify-center font-semibold transition-all duration-300 rounded-lg overflow-hidden group";
    
    const sizeClasses = {
      sm: "px-3 py-1.5 text-xs gap-1.5",
      md: "px-4 py-2 text-sm gap-2",
      lg: "px-6 py-3 text-base gap-2.5",
      xl: "px-8 py-4 text-lg gap-3"
    };
    
    const variantClasses = {
      primary: `
        bg-gradient-to-r from-cyan-500 to-blue-500 text-white
        hover:from-cyan-400 hover:to-blue-400
        active:from-cyan-600 active:to-blue-600
        disabled:from-gray-600 disabled:to-gray-700
        ${glow ? 'shadow-lg shadow-cyan-500/25' : ''}
      `,
      secondary: `
        bg-color-surface border border-color-border text-color-text-primary
        hover:bg-color-surface-hover hover:border-color-primary
        active:bg-color-surface-solid
        disabled:bg-gray-800 disabled:border-gray-700
      `,
      ghost: `
        bg-transparent text-color-text-primary
        hover:bg-white/5 hover:text-color-primary
        active:bg-white/10
        disabled:text-gray-600
      `,
      danger: `
        bg-gradient-to-r from-red-500 to-pink-500 text-white
        hover:from-red-400 hover:to-pink-400
        active:from-red-600 active:to-pink-600
        disabled:from-gray-600 disabled:to-gray-700
        ${glow ? 'shadow-lg shadow-red-500/25' : ''}
      `,
      success: `
        bg-gradient-to-r from-green-500 to-emerald-500 text-white
        hover:from-green-400 hover:to-emerald-400
        active:from-green-600 active:to-emerald-600
        disabled:from-gray-600 disabled:to-gray-700
        ${glow ? 'shadow-lg shadow-green-500/25' : ''}
      `,
      gradient: `
        bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 text-white
        hover:from-purple-400 hover:via-pink-400 hover:to-cyan-400
        active:from-purple-600 active:via-pink-600 active:to-cyan-600
        disabled:from-gray-600 disabled:via-gray-700 disabled:to-gray-600
        ${glow ? 'shadow-lg shadow-purple-500/25' : ''}
      `
    };
    
    const isDisabled = disabled || loading;
    
    const buttonContent = (
      <button
        ref={ref}
        className={cn(
          baseClasses,
          sizeClasses[size],
          variantClasses[variant],
          fullWidth && "w-full",
          isDisabled && "cursor-not-allowed opacity-50",
          className
        )}
        disabled={isDisabled}
        {...props}
      >
        {/* Ripple effect overlay */}
        <span className="absolute inset-0 bg-white opacity-0 group-hover:opacity-10 transition-opacity duration-300" />
        
        {/* Shimmer effect for gradient variant */}
        {variant === 'gradient' && (
          <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent 
                         -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
        )}
        
        {/* Button content */}
        <span className="relative flex items-center gap-2">
          {loading ? (
            <span className="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
          ) : (
            <>
              {icon && iconPosition === 'left' && <span className="flex-shrink-0">{icon}</span>}
              {children}
              {icon && iconPosition === 'right' && <span className="flex-shrink-0">{icon}</span>}
            </>
          )}
        </span>
      </button>
    );
    
    if (animated && !isDisabled) {
      return (
        <motion.div
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          transition={{ duration: 0.1 }}
        >
          {buttonContent}
        </motion.div>
      );
    }
    
    return buttonContent;
  }
);
Button.displayName = 'Button';

// Icon Button Component
export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  animated?: boolean;
}

export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ 
    className, 
    variant = 'ghost', 
    size = 'md', 
    loading = false,
    animated = true,
    disabled,
    children,
    ...props 
  }, ref) => {
    const sizeClasses = {
      sm: "w-8 h-8 text-sm",
      md: "w-10 h-10 text-base",
      lg: "w-12 h-12 text-lg"
    };
    
    const variantClasses = {
      primary: "bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:from-cyan-400 hover:to-blue-400",
      secondary: "bg-color-surface border border-color-border text-color-text-primary hover:border-color-primary",
      ghost: "bg-transparent text-color-text-secondary hover:bg-white/5 hover:text-color-primary",
      danger: "bg-gradient-to-r from-red-500 to-pink-500 text-white hover:from-red-400 hover:to-pink-400"
    };
    
    const buttonContent = (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center rounded-lg transition-all duration-300",
          sizeClasses[size],
          variantClasses[variant],
          (disabled || loading) && "cursor-not-allowed opacity-50",
          className
        )}
        disabled={disabled || loading}
        {...props}
      >
        {loading ? (
          <span className="inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
        ) : (
          children
        )}
      </button>
    );
    
    if (animated && !disabled && !loading) {
      return (
        <motion.div
          whileHover={{ scale: 1.1, rotate: 5 }}
          whileTap={{ scale: 0.9 }}
          transition={{ duration: 0.1 }}
        >
          {buttonContent}
        </motion.div>
      );
    }
    
    return buttonContent;
  }
);
IconButton.displayName = 'IconButton';

// Toggle Button Component
export interface ToggleButtonProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export const ToggleButton: React.FC<ToggleButtonProps> = ({
  checked,
  onChange,
  label,
  disabled = false,
  size = 'md'
}) => {
  const sizeClasses = {
    sm: { track: 'w-8 h-4', thumb: 'w-3 h-3', translate: 'translate-x-4' },
    md: { track: 'w-12 h-6', thumb: 'w-5 h-5', translate: 'translate-x-6' },
    lg: { track: 'w-16 h-8', thumb: 'w-7 h-7', translate: 'translate-x-8' }
  };
  
  return (
    <label className="inline-flex items-center gap-3 cursor-pointer">
      <div className="relative">
        <input
          type="checkbox"
          className="sr-only"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
        />
        <div
          className={cn(
            "block rounded-full transition-all duration-300",
            sizeClasses[size].track,
            checked 
              ? "bg-gradient-to-r from-cyan-500 to-blue-500" 
              : "bg-gray-700",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        >
          <motion.div
            className={cn(
              "absolute top-0.5 left-0.5 bg-white rounded-full shadow-lg",
              sizeClasses[size].thumb
            )}
            animate={{
              x: checked ? sizeClasses[size].translate.replace('translate-x-', '') + 'px' : '0px'
            }}
            transition={{ type: "spring", stiffness: 500, damping: 30 }}
          />
        </div>
      </div>
      {label && (
        <span className={cn(
          "text-sm font-medium",
          disabled ? "text-gray-500" : "text-gray-300"
        )}>
          {label}
        </span>
      )}
    </label>
  );
};