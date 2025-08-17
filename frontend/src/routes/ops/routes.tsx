import { lazy } from 'react';

export const OpsHome = lazy(() => import('./index'));
export const NodePage = lazy(() => import('./node'));
export const ScraperPage = lazy(() => import('./scraper'));
export const ArbitragePage = lazy(() => import('./arbitrage'));
export const MevPage = lazy(() => import('./mev'));
export const StatsPage = lazy(() => import('./stats'));
export const ConfigPage = lazy(() => import('./config'));