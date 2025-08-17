import React, { useState } from 'react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  success?: string;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  variant?: 'default' | 'floating' | 'outlined';
  fullWidth?: boolean;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ 
    className, 
    label,
    error,
    success,
    icon,
    iconPosition = 'left',
    variant = 'default',
    fullWidth = false,
    type = 'text',
    ...props 
  }, ref) => {
    const [isFocused, setIsFocused] = useState(false);
    const [hasValue, setHasValue] = useState(!!props.value || !!props.defaultValue);
    
    const handleFocus = () => setIsFocused(true);
    const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
      setIsFocused(false);
      setHasValue(!!e.target.value);
      props.onBlur?.(e);
    };
    
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      setHasValue(!!e.target.value);
      props.onChange?.(e);
    };
    
    const baseInputClasses = cn(
      "w-full px-4 py-3 text-white bg-transparent border rounded-lg",
      "transition-all duration-300",
      "focus:outline-none focus:ring-2 focus:ring-cyan-500/50",
      "placeholder:text-gray-500",
      icon && iconPosition === 'left' && "pl-12",
      icon && iconPosition === 'right' && "pr-12",
      error && "border-red-500 focus:ring-red-500/50",
      success && "border-green-500 focus:ring-green-500/50",
      !error && !success && "border-gray-700 hover:border-gray-600 focus:border-cyan-500"
    );
    
    if (variant === 'floating' && label) {
      return (
        <div className={cn("relative", fullWidth && "w-full", className)}>
          <div className="relative">
            {icon && (
              <div className={cn(
                "absolute top-1/2 -translate-y-1/2 text-gray-400",
                iconPosition === 'left' ? "left-4" : "right-4"
              )}>
                {icon}
              </div>
            )}
            <input
              ref={ref}
              type={type}
              className={cn(
                baseInputClasses,
                "pt-6 pb-2",
                variant === 'floating' && "bg-color-surface"
              )}
              onFocus={handleFocus}
              onBlur={handleBlur}
              onChange={handleChange}
              {...props}
            />
            <motion.label
              className={cn(
                "absolute left-4 text-gray-400 pointer-events-none",
                "transition-all duration-300",
                icon && iconPosition === 'left' && "left-12"
              )}
              animate={{
                top: isFocused || hasValue ? "0.5rem" : "50%",
                fontSize: isFocused || hasValue ? "0.75rem" : "0.875rem",
                color: isFocused ? "#00D4FF" : error ? "#FF3366" : "#9CA3AF",
                y: isFocused || hasValue ? 0 : "-50%"
              }}
              transition={{ duration: 0.2 }}
            >
              {label}
            </motion.label>
          </div>
          <AnimatePresence>
            {(error || success) && (
              <motion.p
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className={cn(
                  "mt-1 text-xs",
                  error && "text-red-400",
                  success && "text-green-400"
                )}
              >
                {error || success}
              </motion.p>
            )}
          </AnimatePresence>
        </div>
      );
    }
    
    return (
      <div className={cn("relative", fullWidth && "w-full", className)}>
        {label && variant !== 'floating' && (
          <label className="block mb-2 text-sm font-medium text-gray-300">
            {label}
          </label>
        )}
        <div className="relative">
          {icon && (
            <div className={cn(
              "absolute top-1/2 -translate-y-1/2 text-gray-400",
              iconPosition === 'left' ? "left-4" : "right-4"
            )}>
              {icon}
            </div>
          )}
          <input
            ref={ref}
            type={type}
            className={cn(
              baseInputClasses,
              variant === 'outlined' && "bg-transparent",
              variant === 'default' && "bg-color-surface"
            )}
            {...props}
          />
        </div>
        <AnimatePresence>
          {(error || success) && (
            <motion.p
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className={cn(
                "mt-1 text-xs",
                error && "text-red-400",
                success && "text-green-400"
              )}
            >
              {error || success}
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    );
  }
);
Input.displayName = 'Input';

// Textarea Component
export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  success?: string;
  fullWidth?: boolean;
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, label, error, success, fullWidth = false, ...props }, ref) => {
    return (
      <div className={cn("relative", fullWidth && "w-full", className)}>
        {label && (
          <label className="block mb-2 text-sm font-medium text-gray-300">
            {label}
          </label>
        )}
        <textarea
          ref={ref}
          className={cn(
            "w-full px-4 py-3 text-white bg-color-surface border rounded-lg",
            "transition-all duration-300 resize-none",
            "focus:outline-none focus:ring-2 focus:ring-cyan-500/50",
            "placeholder:text-gray-500",
            error && "border-red-500 focus:ring-red-500/50",
            success && "border-green-500 focus:ring-green-500/50",
            !error && !success && "border-gray-700 hover:border-gray-600 focus:border-cyan-500"
          )}
          {...props}
        />
        <AnimatePresence>
          {(error || success) && (
            <motion.p
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className={cn(
                "mt-1 text-xs",
                error && "text-red-400",
                success && "text-green-400"
              )}
            >
              {error || success}
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    );
  }
);
Textarea.displayName = 'Textarea';

// Select Component
export interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
  options: { value: string; label: string }[];
  fullWidth?: boolean;
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, label, error, options, fullWidth = false, ...props }, ref) => {
    return (
      <div className={cn("relative", fullWidth && "w-full", className)}>
        {label && (
          <label className="block mb-2 text-sm font-medium text-gray-300">
            {label}
          </label>
        )}
        <select
          ref={ref}
          className={cn(
            "w-full px-4 py-3 text-white bg-color-surface border rounded-lg",
            "transition-all duration-300 appearance-none cursor-pointer",
            "focus:outline-none focus:ring-2 focus:ring-cyan-500/50",
            error && "border-red-500 focus:ring-red-500/50",
            !error && "border-gray-700 hover:border-gray-600 focus:border-cyan-500"
          )}
          {...props}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value} className="bg-gray-900">
              {option.label}
            </option>
          ))}
        </select>
        {/* Custom dropdown arrow */}
        <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none">
          <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
        <AnimatePresence>
          {error && (
            <motion.p
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="mt-1 text-xs text-red-400"
            >
              {error}
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    );
  }
);
Select.displayName = 'Select';