/**
 * Beautiful Toast Notification System
 * Provides elegant notifications for the MEV dashboard
 */

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, CheckCircle, AlertCircle, Info, XCircle } from 'lucide-react';

// Toast types
export type ToastType = 'success' | 'error' | 'warning' | 'info';
export type ToastPosition = 'top-left' | 'top-center' | 'top-right' | 'bottom-left' | 'bottom-center' | 'bottom-right';

interface Toast {
  id: string;
  type: ToastType;
  title?: string;
  message: string;
  duration?: number;
  onClose?: () => void;
}

interface ToasterContextType {
  toast: (toast: Omit<Toast, 'id'>) => void;
  dismiss: (id: string) => void;
  dismissAll: () => void;
}

const ToasterContext = createContext<ToasterContextType | null>(null);

export const useToast = () => {
  const context = useContext(ToasterContext);
  if (!context) {
    throw new Error('useToast must be used within ToasterProvider');
  }
  return context;
};

interface ToasterProps {
  position?: ToastPosition;
  maxToasts?: number;
  children?: React.ReactNode;
}

const toastIcons = {
  success: CheckCircle,
  error: XCircle,
  warning: AlertCircle,
  info: Info,
};

const toastColors = {
  success: 'from-green-500/20 to-emerald-500/20 border-green-500/30 text-green-400',
  error: 'from-red-500/20 to-rose-500/20 border-red-500/30 text-red-400',
  warning: 'from-yellow-500/20 to-amber-500/20 border-yellow-500/30 text-yellow-400',
  info: 'from-blue-500/20 to-cyan-500/20 border-blue-500/30 text-blue-400',
};

const positionClasses = {
  'top-left': 'top-6 left-6',
  'top-center': 'top-6 left-1/2 -translate-x-1/2',
  'top-right': 'top-6 right-6',
  'bottom-left': 'bottom-6 left-6',
  'bottom-center': 'bottom-6 left-1/2 -translate-x-1/2',
  'bottom-right': 'bottom-6 right-6',
};

const ToastComponent: React.FC<{ toast: Toast; onDismiss: (id: string) => void }> = ({ toast, onDismiss }) => {
  const Icon = toastIcons[toast.type];
  const colorClass = toastColors[toast.type];

  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(() => {
        onDismiss(toast.id);
      }, toast.duration);

      return () => clearTimeout(timer);
    }
  }, [toast, onDismiss]);

  return (
    <motion.div
      initial={{ opacity: 0, y: -20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 20, scale: 0.95 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className={`relative overflow-hidden rounded-xl border bg-gradient-to-r ${colorClass} p-4 shadow-2xl backdrop-blur-xl`}
      style={{
        minWidth: '320px',
        maxWidth: '420px',
      }}
    >
      {/* Background glow effect */}
      <motion.div
        className="absolute inset-0 opacity-30"
        animate={{
          background: [
            'radial-gradient(circle at 0% 0%, currentColor 0%, transparent 50%)',
            'radial-gradient(circle at 100% 100%, currentColor 0%, transparent 50%)',
            'radial-gradient(circle at 0% 0%, currentColor 0%, transparent 50%)',
          ],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: 'linear',
        }}
      />

      <div className="relative flex items-start space-x-3">
        <div className="flex-shrink-0">
          <Icon className="h-5 w-5" />
        </div>
        
        <div className="flex-1">
          {toast.title && (
            <h4 className="mb-1 font-semibold text-white">{toast.title}</h4>
          )}
          <p className="text-sm text-gray-300">{toast.message}</p>
        </div>

        <button
          onClick={() => onDismiss(toast.id)}
          className="flex-shrink-0 rounded-lg p-1 text-gray-400 transition-colors hover:bg-white/10 hover:text-white"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Progress bar for timed toasts */}
      {toast.duration && toast.duration > 0 && (
        <motion.div
          className="absolute bottom-0 left-0 h-1 bg-current opacity-30"
          initial={{ width: '100%' }}
          animate={{ width: '0%' }}
          transition={{ duration: toast.duration / 1000, ease: 'linear' }}
        />
      )}
    </motion.div>
  );
};

export const Toaster: React.FC<ToasterProps> = ({ 
  position = 'bottom-right', 
  maxToasts = 5,
  children 
}) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const toast = useCallback((newToast: Omit<Toast, 'id'>) => {
    const id = `toast-${Date.now()}-${Math.random()}`;
    const toastWithId: Toast = {
      ...newToast,
      id,
      duration: newToast.duration ?? 5000,
    };

    setToasts((prev) => {
      const updated = [...prev, toastWithId];
      // Limit number of toasts
      if (updated.length > maxToasts) {
        return updated.slice(-maxToasts);
      }
      return updated;
    });
  }, [maxToasts]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const dismissAll = useCallback(() => {
    setToasts([]);
  }, []);

  return (
    <ToasterContext.Provider value={{ toast, dismiss, dismissAll }}>
      {children}
      <div
        className={`fixed z-50 ${positionClasses[position]}`}
        style={{ pointerEvents: 'none' }}
      >
        <div className="space-y-3" style={{ pointerEvents: 'auto' }}>
          <AnimatePresence mode="sync">
            {toasts.map((toast) => (
              <ToastComponent key={toast.id} toast={toast} onDismiss={dismiss} />
            ))}
          </AnimatePresence>
        </div>
      </div>
    </ToasterContext.Provider>
  );
};

// Convenience methods
export const toast = {
  success: (message: string, options?: Partial<Toast>) => {
    const context = useContext(ToasterContext);
    context?.toast({ ...options, type: 'success', message });
  },
  error: (message: string, options?: Partial<Toast>) => {
    const context = useContext(ToasterContext);
    context?.toast({ ...options, type: 'error', message });
  },
  warning: (message: string, options?: Partial<Toast>) => {
    const context = useContext(ToasterContext);
    context?.toast({ ...options, type: 'warning', message });
  },
  info: (message: string, options?: Partial<Toast>) => {
    const context = useContext(ToasterContext);
    context?.toast({ ...options, type: 'info', message });
  },
};