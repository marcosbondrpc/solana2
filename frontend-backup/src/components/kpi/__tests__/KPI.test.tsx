import { render, screen } from '@testing-library/react';
import KPI from '../KPI';

describe('KPI', () => {
  it('renders label and value', () => {
    render(<KPI label="Throughput" value={1234} />);
    expect(screen.getByText('Throughput')).toBeInTheDocument();
    expect(screen.getByText(/1\.23K|1,234|1234/)).toBeInTheDocument();
  });
  it('renders delta tone', () => {
    render(<KPI label="Delta" value={10} delta={0.1} />);
    expect(screen.getByText(/\+10\.00%/)).toBeInTheDocument();
  });
});