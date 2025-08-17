import { Link, NavLink } from 'react-router-dom';

export default function OpsNav() {
  const items = [
    { to: '/ops', label: 'Dashboard', end: true },
    { to: '/ops/node', label: 'Node' },
    { to: '/ops/scraper', label: 'Scraper' },
    { to: '/ops/arbitrage', label: 'Arbitrage' },
    { to: '/ops/mev', label: 'MEV' },
    { to: '/ops/stats', label: 'Stats' },
    { to: '/ops/config', label: 'Config' }
  ];
  return (
    <nav className="h-full w-56 shrink-0 border-r border-zinc-200 dark:border-zinc-800 bg-zinc-50/40 dark:bg-zinc-900/40">
      <div className="px-4 py-3 text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">Ops</div>
      <ul className="px-2 space-y-1">
        {items.map(i => (
          <li key={i.to}>
            <NavLink
              to={i.to}
              end={(i as any).end}
              className={({ isActive }) =>
                `block rounded px-3 py-2 text-sm ${isActive ? 'bg-brand-600 text-white' : 'hover:bg-zinc-100 dark:hover:bg-zinc-800'}`
              }
            >
              {i.label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}