import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import OpsNav from '../OpsNav';

describe('OpsNav', () => {
  it('shows Ops menu items', () => {
    render(<MemoryRouter><OpsNav /></MemoryRouter>);
    ['Dashboard','Node','Scraper','Arbitrage','MEV','Stats','Config'].forEach(label => {
      expect(screen.getByText(label)).toBeInTheDocument();
    });
  });
});