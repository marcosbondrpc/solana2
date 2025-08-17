import { Outlet } from 'react-router-dom';
import OpsNav from '../components/nav/OpsNav';

export default function OpsLayout() {
  return (
    <div className="min-h-screen flex bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
      <OpsNav />
      <div className="flex-1 flex flex-col">
        <header className="h-14 border-b border-zinc-200 dark:border-zinc-800 flex items-center justify-between px-4">
          <div className="font-medium">Solana MEV Ops</div>
          <div className="flex items-center gap-2">
            <button className="text-sm px-3 py-1.5 rounded border border-zinc-300 dark:border-zinc-700">Theme</button>
            <button className="text-sm px-3 py-1.5 rounded border border-zinc-300 dark:border-zinc-700">Profile</button>
          </div>
        </header>
        <main className="p-4">
          <Outlet />
        </main>
      </div>
    </div>
  );
}