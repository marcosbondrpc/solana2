import { useEffect, useState } from 'react';

const STORAGE_KEY = 'theme';
type Mode = 'light' | 'dark';

export function useTheme(): [Mode, () => void] {
  const [mode, setMode] = useState<Mode>(() => (localStorage.getItem(STORAGE_KEY) as Mode) || 'light');
  useEffect(() => {
    const root = document.documentElement;
    if (mode === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem(STORAGE_KEY, mode);
  }, [mode]);
  const toggle = () => setMode(m => (m === 'dark' ? 'light' : 'dark'));
  return [mode, toggle];
}