/**
 * Advanced Notification System
 * Real-time alerts with beautiful animations
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle,
  AlertCircle,
  XCircle,
  Info,
  Zap,
  TrendingUp,
  DollarSign,
  X,
  Bell,
  BellOff,
} from 'lucide-react';

export interface Notification {
  id: string;
  type: 'success' | 'warning' | 'error' | 'info' | 'mev' | 'profit';
  title: string;
  message: string;
  timestamp: number;
  persistent?: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface NotificationSystemProps {
  notifications: Notification[];
  onDismiss: (id: string) => void;
  onClearAll: () => void;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
}

const notificationIcons = {
  success: CheckCircle,
  warning: AlertCircle,
  error: XCircle,
  info: Info,
  mev: Zap,
  profit: DollarSign,
};

const notificationColors = {
  success: {
    bg: 'from-green-500/20 to-emerald-500/10',
    border: 'border-green-500/30',
    icon: 'text-green-400',
    text: 'text-green-300',
  },
  warning: {
    bg: 'from-yellow-500/20 to-amber-500/10',
    border: 'border-yellow-500/30',
    icon: 'text-yellow-400',
    text: 'text-yellow-300',
  },
  error: {
    bg: 'from-red-500/20 to-rose-500/10',
    border: 'border-red-500/30',
    icon: 'text-red-400',
    text: 'text-red-300',
  },
  info: {
    bg: 'from-blue-500/20 to-cyan-500/10',
    border: 'border-blue-500/30',
    icon: 'text-blue-400',
    text: 'text-blue-300',
  },
  mev: {
    bg: 'from-purple-500/20 to-pink-500/10',
    border: 'border-purple-500/30',
    icon: 'text-purple-400',
    text: 'text-purple-300',
  },
  profit: {
    bg: 'from-green-500/20 to-emerald-500/10',
    border: 'border-green-500/30',
    icon: 'text-green-400',
    text: 'text-green-300',
  },
};

const NotificationItem = ({ notification, onDismiss }: { notification: Notification; onDismiss: () => void }) => {
  const Icon = notificationIcons[notification.type];
  const colors = notificationColors[notification.type];
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    if (!notification.persistent) {
      const duration = 5000;
      const interval = 50;
      const decrement = (100 / duration) * interval;

      const timer = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev - decrement;
          if (newProgress <= 0) {
            clearInterval(timer);
            onDismiss();
            return 0;
          }
          return newProgress;
        });
      }, interval);

      return () => clearInterval(timer);
    }
  }, [notification.persistent, onDismiss]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 300, scale: 0.9 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 300, scale: 0.9 }}
      whileHover={{ scale: 1.02 }}
      className={`relative overflow-hidden rounded-xl bg-gradient-to-br ${colors.bg} backdrop-blur-sm border ${colors.border} p-4 shadow-2xl min-w-[320px] max-w-[400px]`}
    >
      {/* Background shimmer effect */}
      <div className="absolute inset-0 opacity-30">
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
          animate={{ x: [-400, 400] }}
          transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
        />
      </div>

      <div className="relative z-10">
        <div className="flex items-start space-x-3">
          <div className={`p-2 rounded-lg bg-black/20 ${colors.icon}`}>
            <Icon className="h-5 w-5" />
          </div>
          
          <div className="flex-1">
            <div className="flex items-start justify-between">
              <div>
                <h4 className="font-semibold text-white">{notification.title}</h4>
                <p className={`text-sm mt-1 ${colors.text}`}>{notification.message}</p>
                
                {notification.action && (
                  <motion.button
                    whileTap={{ scale: 0.95 }}
                    onClick={notification.action.onClick}
                    className="mt-2 px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-medium transition-colors"
                  >
                    {notification.action.label}
                  </motion.button>
                )}
              </div>
              
              <motion.button
                whileTap={{ scale: 0.9 }}
                onClick={onDismiss}
                className="ml-2 p-1 rounded-lg hover:bg-white/10 transition-colors"
              >
                <X className={`h-4 w-4 ${colors.icon}`} />
              </motion.button>
            </div>
            
            <div className="mt-2 text-xs text-gray-500">
              {new Date(notification.timestamp).toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>

      {/* Progress bar */}
      {!notification.persistent && (
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-black/20">
          <motion.div
            className={`h-full bg-gradient-to-r ${colors.bg}`}
            initial={{ width: '100%' }}
            animate={{ width: `${progress}%` }}
            transition={{ type: 'linear' }}
            style={{
              filter: 'brightness(2)',
            }}
          />
        </div>
      )}
    </motion.div>
  );
};

export function NotificationSystem({
  notifications,
  onDismiss,
  onClearAll,
  position = 'top-right',
}: NotificationSystemProps) {
  const [muted, setMuted] = useState(false);

  const positionClasses = {
    'top-right': 'top-6 right-6',
    'top-left': 'top-6 left-6',
    'bottom-right': 'bottom-6 right-6',
    'bottom-left': 'bottom-6 left-6',
  };

  // Play sound for new notifications
  useEffect(() => {
    if (!muted && notifications.length > 0) {
      const latestNotification = notifications[0];
      if (latestNotification && Date.now() - latestNotification.timestamp < 1000) {
        console.log('Playing notification sound');
      }
    }
  }, [notifications, muted]);

  return (
    <>
      {/* Notification container */}
      <div className={`fixed ${positionClasses[position]} z-50 space-y-3`}>
        {/* Header controls */}
        {notifications.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-end space-x-2 mb-2"
          >
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => setMuted(!muted)}
              className="p-2 rounded-lg bg-gray-900/80 backdrop-blur-sm border border-gray-700 hover:bg-gray-800/80 transition-colors"
            >
              {muted ? (
                <BellOff className="h-4 w-4 text-gray-400" />
              ) : (
                <Bell className="h-4 w-4 text-gray-400" />
              )}
            </motion.button>
            
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={onClearAll}
              className="px-3 py-1.5 rounded-lg bg-gray-900/80 backdrop-blur-sm border border-gray-700 hover:bg-gray-800/80 text-xs text-gray-400 font-medium transition-colors"
            >
              Clear All
            </motion.button>
          </motion.div>
        )}

        {/* Notifications */}
        <AnimatePresence mode="sync">
          {notifications.slice(0, 5).map((notification) => (
            <NotificationItem
              key={notification.id}
              notification={notification}
              onDismiss={() => onDismiss(notification.id)}
            />
          ))}
        </AnimatePresence>
      </div>

      {/* Badge for hidden notifications */}
      {notifications.length > 5 && (
        <motion.div
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          className={`fixed ${positionClasses[position]} z-40`}
          style={{ marginTop: notifications.slice(0, 5).length * 100 + 80 }}
        >
          <div className="px-3 py-2 rounded-lg bg-gray-900/80 backdrop-blur-sm border border-gray-700 text-sm text-gray-400">
            +{notifications.length - 5} more notifications
          </div>
        </motion.div>
      )}
    </>
  );
}