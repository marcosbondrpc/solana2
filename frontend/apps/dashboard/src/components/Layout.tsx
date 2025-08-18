import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../providers/AuthProvider';
import { useTheme } from '../providers/ThemeProvider';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const { user, disconnect } = useAuth();
  const { toggleTheme, resolvedTheme } = useTheme();

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: '📊' },
    { name: 'Node', href: '/node', icon: '🖥️' },
    { name: 'Scrapper', href: '/scrapper', icon: '🗄️' },
    { name: 'MEV', href: '/mev', icon: '💎' },
    { name: 'Arbitrage', href: '/arbitrage', icon: '🔄' },
    { name: 'Jito', href: '/jito', icon: '📦' },
    { name: 'Analytics', href: '/analytics', icon: '📈' },
    { name: 'Monitoring', href: '/monitoring', icon: '🔍' },
    { name: 'Settings', href: '/settings', icon: '⚙️' },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                  Solana MEV Dashboard
                </h1>
              </div>
            </div>

            {/* Right side actions */}
            <div className="flex items-center space-x-4">
              {/* Theme Toggle */}
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors"
                aria-label="Toggle theme"
              >
                {resolvedTheme === 'dark' ? '🌙' : '☀️'}
              </button>

              {/* User Info */}
              {user && (
                <div className="flex items-center space-x-3">
                  <div className="text-sm">
                    <p className="text-gray-400">Connected</p>
                    <p className="font-mono text-xs">
                      {user.address.slice(0, 4)}...{user.address.slice(-4)}
                    </p>
                  </div>
                  <button
                    onClick={disconnect}
                    className="px-3 py-1 text-sm bg-red-500 hover:bg-red-600 rounded transition-colors"
                  >
                    Disconnect
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar Navigation */}
        <nav className="w-64 bg-gray-800 min-h-[calc(100vh-4rem)] border-r border-gray-700">
          <div className="px-3 py-4">
            <ul className="space-y-1">
              {navigation.map((item) => (
                <li key={item.name}>
                  <Link
                    to={item.href}
                    className={`
                      flex items-center px-3 py-2 rounded-lg transition-colors
                      ${
                        isActive(item.href)
                          ? 'bg-gradient-to-r from-purple-500/20 to-blue-500/20 text-white border border-purple-500/30'
                          : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                      }
                    `}
                  >
                    <span className="mr-3 text-lg">{item.icon}</span>
                    <span className="font-medium">{item.name}</span>
                    {isActive(item.href) && (
                      <span className="ml-auto w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    )}
                  </Link>
                </li>
              ))}
            </ul>

            {/* Stats Section */}
            <div className="mt-8 px-3">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                Network Stats
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">TPS</span>
                  <span className="text-green-400">3,245</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Slot</span>
                  <span className="text-blue-400">234,567,890</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">MEV Profit</span>
                  <span className="text-purple-400">+$12,345</span>
                </div>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1">
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}