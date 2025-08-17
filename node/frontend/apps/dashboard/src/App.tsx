import { Suspense, lazy, useEffect, useState, useDeferredValue, useTransition, startTransition } from 'react';
import { createBrowserRouter, RouterProvider, Navigate, Outlet } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ErrorBoundary } from 'react-error-boundary';
import { Toaster } from './components/ui';
import { WebSocketProvider } from './providers/WebSocketProvider';
import { ThemeProvider } from './providers/ThemeProvider';
import { AuthProvider } from './providers/AuthProvider';
import { PerformanceMonitor } from './components/PerformanceMonitor';
import { LoadingScreen } from './components/LoadingScreen';
import { ErrorFallback } from './components/ErrorFallback';
import { Layout } from './components/Layout';

// Lazy load pages for code splitting
const DashboardPage = lazy(() => import('./pages/Dashboard'));
const NodePage = lazy(() => import('./pages/Node'));
const ScrapperPage = lazy(() => import('./pages/Scrapper'));
const MEVPage = lazy(() => import('./pages/MEV'));
const ArbitragePage = lazy(() => import('./pages/Arbitrage'));
const JitoPage = lazy(() => import('./pages/Jito'));
const AnalyticsPage = lazy(() => import('./pages/Analytics'));
const SettingsPage = lazy(() => import('./pages/Settings'));
const MonitoringPage = lazy(() => import('./pages/Monitoring'));

// Configure React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      gcTime: 10 * 60 * 1000,
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      refetchOnWindowFocus: false,
      refetchOnReconnect: 'always',
    },
    mutations: {
      retry: 2,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
});

// WebSocket configuration with proper environment variables and fallback
const wsConfig = {
  url: (import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8085') + '/ws/node-metrics',
  protocols: ['protobuf', 'json'],
  reconnectInterval: 5000, // Increased from 2000ms to prevent rapid reconnection attempts
  maxReconnectAttempts: 5,
  heartbeatInterval: 30000,
  enableCompression: true,
  binaryType: 'arraybuffer' as const,
};

// Layout wrapper component for router with concurrent features
function LayoutWrapper() {
  const [isPending, startTransition] = useTransition();
  
  return (
    <Layout>
      <Suspense 
        fallback={
          <LoadingScreen 
            message={isPending ? 'Loading new content...' : 'Loading...'} 
          />
        }
      >
        <div style={{ opacity: isPending ? 0.6 : 1, transition: 'opacity 0.2s' }}>
          <Outlet />
        </div>
      </Suspense>
    </Layout>
  );
}

// Create router with v7 future flags
const router = createBrowserRouter(
  [
    {
      path: '/',
      element: <LayoutWrapper />,
      errorElement: <ErrorFallback />,
      children: [
        { index: true, element: <Navigate to="/dashboard" replace /> },
        { path: 'dashboard', element: <DashboardPage /> },
        { path: 'node', element: <NodePage /> },
        { path: 'scrapper', element: <ScrapperPage /> },
        { path: 'mev', element: <MEVPage /> },
        { path: 'arbitrage', element: <ArbitragePage /> },
        { path: 'jito', element: <JitoPage /> },
        { path: 'analytics', element: <AnalyticsPage /> },
        { path: 'monitoring', element: <MonitoringPage /> },
        { path: 'settings', element: <SettingsPage /> },
        { path: '*', element: <Navigate to="/dashboard" replace /> },
      ],
    },
  ],
  {
    future: {
      v7_startTransition: true,
      v7_relativeSplatPath: true,
      v7_fetcherPersist: true,
      v7_normalizeFormMethod: true,
      v7_partialHydration: true,
      v7_skipActionErrorRevalidation: true,
    },
  }
);

// Performance monitoring integration
const enablePerformanceMonitoring = import.meta.env.VITE_ENABLE_PERF_MONITOR !== 'false';

export function App() {
  const [showPerfMonitor, setShowPerfMonitor] = useState(false);
  
  // Enable performance monitoring in development
  useEffect(() => {
    if (enablePerformanceMonitoring && import.meta.env.DEV) {
      // Toggle with Ctrl+Shift+P
      const handleKeyPress = (e: KeyboardEvent) => {
        if (e.ctrlKey && e.shiftKey && e.key === 'P') {
          setShowPerfMonitor(prev => !prev);
        }
      };
      window.addEventListener('keydown', handleKeyPress);
      return () => window.removeEventListener('keydown', handleKeyPress);
    }
  }, []);
  useEffect(() => {
    // Register service worker for PWA
    if ('serviceWorker' in navigator && import.meta.env.PROD) {
      navigator.serviceWorker.register('/sw.js').catch(console.error);
    }

    // Performance monitoring
    if (import.meta.env.PROD && typeof window !== 'undefined') {
      // Log Web Vitals
      import('web-vitals').then(({ onCLS, onFID, onFCP, onLCP, onTTFB }) => {
        onCLS(console.log);
        onFID(console.log);
        onFCP(console.log);
        onLCP(console.log);
        onTTFB(console.log);
      });
    }
  }, []);

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback} onReset={() => window.location.reload()}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <AuthProvider>
            <WebSocketProvider config={wsConfig}>
              <RouterProvider router={router} />
              <Toaster position="bottom-right" />
              {import.meta.env.DEV && (
                <>
                  <ReactQueryDevtools initialIsOpen={false} />
                  {showPerfMonitor && <PerformanceMonitor />}
                </>
              )}
            </WebSocketProvider>
          </AuthProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}