import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import './styles/globals.css';

// Enable performance profiling in development
if (import.meta.env.DEV) {
  if (typeof window !== 'undefined') {
    // @ts-ignore
    window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
      supportsFiber: true,
      rendererPackageName: 'react-dom',
      rendererConfig: {},
      overrideHookState: null,
      overrideProps: null,
      setSuspenseHandler: null,
      scheduleUpdate: null,
      currentDispatcherRef: null,
      findHostInstanceByFiber: () => null,
      findFiberByHostInstance: () => null,
      findHostInstancesForRefresh: null,
      scheduleRefresh: null,
      scheduleRoot: null,
      setRefreshHandler: null,
      getCurrentFiber: null,
    };
  }
}

// Mount the app
const container = document.getElementById('root');
if (!container) throw new Error('Failed to find the root element');

const root = createRoot(container);

root.render(
  <StrictMode>
    <App />
  </StrictMode>
);