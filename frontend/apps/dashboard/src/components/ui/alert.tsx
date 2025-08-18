import React from 'react';
import { cn } from '@/lib/utils';

type AlertVariant = 'default' | 'primary' | 'success' | 'warning' | 'danger' | 'info' | 'purple' | 'destructive' | 'secondary' | 'outline';

export interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: AlertVariant;
}

export const Alert: React.FC<AlertProps> = ({ variant = 'default', className, children, ...props }) => {
  const variants: Record<AlertVariant, string> = {
    default: 'border-gray-700 bg-gray-900/60 text-gray-300',
    primary: 'border-cyan-500/30 bg-cyan-500/10 text-cyan-300',
    success: 'border-green-500/30 bg-green-500/10 text-green-300',
    warning: 'border-amber-500/30 bg-amber-500/10 text-amber-300',
    danger: 'border-red-500/30 bg-red-500/10 text-red-300',
    info: 'border-blue-500/30 bg-blue-500/10 text-blue-300',
    purple: 'border-purple-500/30 bg-purple-500/10 text-purple-300',
    destructive: 'border-rose-500/30 bg-rose-500/10 text-rose-300',
    secondary: 'border-gray-600 bg-gray-800 text-gray-300',
    outline: 'border-gray-600 bg-transparent text-gray-300',
  };
  return (
    <div className={cn('rounded-lg border p-4', variants[variant], className)} {...props}>
      {children}
    </div>
  );
};

export const AlertTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({ className, ...props }) => (
  <h5 className={cn('mb-1 font-semibold text-white', className)} {...props} />
);

export const AlertDescription: React.FC<React.HTMLAttributes<HTMLParagraphElement>> = ({ className, ...props }) => (
  <p className={cn('text-sm text-gray-300', className)} {...props} />
);