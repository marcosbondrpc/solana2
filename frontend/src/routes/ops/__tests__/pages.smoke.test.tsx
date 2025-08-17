import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import OpsLayout from '../../../layouts/OpsLayout';
import OpsHome from '../index';
import NodePage from '../node';
import ScraperPage from '../scraper';
import ArbitragePage from '../arbitrage';
import MevPage from '../mev';
import StatsPage from '../stats';
import ConfigPage from '../config';

function renderRoute(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/ops" element={<OpsLayout />}>
          <Route index element={<OpsHome />} />
          <Route path="node" element={<NodePage />} />
          <Route path="scraper" element={<ScraperPage />} />
          <Route path="arbitrage" element={<ArbitragePage />} />
          <Route path="mev" element={<MevPage />} />
          <Route path="stats" element={<StatsPage />} />
          <Route path="config" element={<ConfigPage />} />
        </Route>
      </Routes>
    </MemoryRouter>
  );
}

describe('Ops pages smoke', () => {
  it('renders Ops home', () => {
    renderRoute('/ops');
    expect(screen.getByText(/Operations Dashboard|System Health/i)).toBeInTheDocument();
  });
  it('renders Node page', () => {
    renderRoute('/ops/node');
    expect(screen.getAllByText(/Node|TPS|Latency/i).length).toBeGreaterThan(0);
  });
  it('renders Scraper page', () => {
    renderRoute('/ops/scraper');
    expect(screen.getAllByText(/Running Jobs|Filters|Jobs/i).length).toBeGreaterThan(0);
  });
  it('renders Arbitrage page', () => {
    renderRoute('/ops/arbitrage');
    expect(screen.getByText(/Opportunities|Filters/i)).toBeInTheDocument();
  });
  it('renders MEV page', () => {
    renderRoute('/ops/mev');
    expect(screen.getByText(/Bundles|Alerts|Profit/i)).toBeInTheDocument();
  });
  it('renders Stats page', () => {
    renderRoute('/ops/stats');
    expect(screen.getByText(/TPS|Latency|Top/i)).toBeInTheDocument();
  });
  it('renders Config page', () => {
    renderRoute('/ops/config');
    expect(screen.getByText(/Endpoints|Thresholds|Feature Flags/i)).toBeInTheDocument();
  });
});