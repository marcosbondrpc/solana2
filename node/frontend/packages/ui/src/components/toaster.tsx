import { useEffect, useState } from 'react';

interface Toast {
  id: string;
  title?: string;
  description?: string;
  variant?: 'default' | 'success' | 'error' | 'warning';
}

// Simple Toaster component for notifications
export function Toaster({ position = 'bottom-right' }: { position?: string }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Listen for custom toast events
  useEffect(() => {
    const handleToast = (event: CustomEvent) => {
      const newToast: Toast = {
        id: Date.now().toString(),
        ...event.detail,
      };
      
      setToasts(prev => [...prev, newToast]);
      
      // Auto-remove after 5 seconds
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== newToast.id));
      }, 5000);
    };

    window.addEventListener('show-toast' as any, handleToast);
    return () => window.removeEventListener('show-toast' as any, handleToast);
  }, []);

  const positionClasses = {
    'top-left': 'top-4 left-4',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-right': 'bottom-4 right-4',
  }[position] || 'bottom-4 right-4';

  const variantClasses = {
    default: 'bg-gray-800 border-gray-700 text-white',
    success: 'bg-green-900/90 border-green-700 text-green-100',
    error: 'bg-red-900/90 border-red-700 text-red-100',
    warning: 'bg-yellow-900/90 border-yellow-700 text-yellow-100',
  };

  if (toasts.length === 0) return null;

  return (
    <div className={`fixed ${positionClasses} z-50 flex flex-col gap-2`}>
      {toasts.map(toast => (
        <div
          key={toast.id}
          className={`
            min-w-[300px] max-w-md p-4 rounded-lg border backdrop-blur-sm
            shadow-lg animate-in slide-in-from-bottom-2 fade-in duration-200
            ${variantClasses[toast.variant || 'default']}
          `}
        >
          {toast.title && (
            <h3 className="font-semibold text-sm mb-1">{toast.title}</h3>
          )}
          {toast.description && (
            <p className="text-sm opacity-90">{toast.description}</p>
          )}
          <button
            onClick={() => setToasts(prev => prev.filter(t => t.id !== toast.id))}
            className="absolute top-2 right-2 text-current opacity-50 hover:opacity-100"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}