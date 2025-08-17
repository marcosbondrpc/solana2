import { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ENABLE_OPS } from './flags';
import OpsLayout from './layouts/OpsLayout';
import { OpsHome, NodePage, ScraperPage, ArbitragePage, MevPage, StatsPage, ConfigPage } from './routes/ops/routes';
const MEVDetectionDashboard = lazy(() => import('./pages/MEVDetectionDashboard'));

export default function App() {
  if (!ENABLE_OPS) {
    return <MEVDetectionDashboard />;
  }
  return (
    <BrowserRouter>
      <Suspense fallback={<div className="loading">Loading...</div>}>
        <Routes>
          <Route path="/" element={<MEVDetectionDashboard />} />
          <Route path="/ops" element={<OpsLayout />}>
            <Route index element={<OpsHome />} />
            <Route path="node" element={<NodePage />} />
            <Route path="scraper" element={<ScraperPage />} />
            <Route path="arbitrage" element={<ArbitragePage />} />
            <Route path="mev" element={<MevPage />} />
            <Route path="stats" element={<StatsPage />} />
            <Route path="config" element={<ConfigPage />} />
          </Route>
          <Route path="*" element={<Navigate to={ENABLE_OPS ? '/ops' : '/'} replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}