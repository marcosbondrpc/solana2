import React from 'react';
import ReactDOM from 'react-dom/client';
import { MEVDashboardUltra } from './components/MEVDashboardUltra';
import './styles/global.css';

// Initialize performance monitoring
if ('performance' in window && 'measure' in window.performance) {
  window.performance.mark('app-init-start');
}

// Enable concurrent features
const rootElement = document.getElementById('root');
if (!rootElement) throw new Error('Root element not found');

const root = ReactDOM.createRoot(rootElement);

// Prefetch critical resources
const prefetchResources = async () => {
  const resources = [
    '/assets/protobuf-worker.js',
    '/assets/gpu-worker.js',
    '/assets/shared-memory-worker.js'
  ];
  
  await Promise.all(
    resources.map(url => 
      fetch(url, { priority: 'high' as RequestPriority }).catch(() => {})
    )
  );
};

// Initialize service worker for offline support
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register('/sw.js').catch(console.error);
}

// Initialize SharedArrayBuffer if available
if (typeof SharedArrayBuffer !== 'undefined') {
  console.log('SharedArrayBuffer enabled - Zero-copy mode active');
  (window as any).sharedMemoryEnabled = true;
} else {
  console.warn('SharedArrayBuffer not available - Fallback mode');
  (window as any).sharedMemoryEnabled = false;
}

// Start prefetching
prefetchResources();

// Render with concurrent mode
root.render(
  <React.StrictMode>
    <React.Suspense fallback={<div className="loading">Loading MEV Dashboard...</div>}>
      <MEVDashboardUltra />
    </React.Suspense>
  </React.StrictMode>
);

// Mark performance
if ('performance' in window && 'measure' in window.performance) {
  window.performance.mark('app-init-end');
  window.performance.measure('app-init', 'app-init-start', 'app-init-end');
}