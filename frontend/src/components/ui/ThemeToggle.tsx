import { useTheme } from '../../theme/useTheme';

export default function ThemeToggle() {
  const [mode, toggle] = useTheme();
  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      className="text-sm px-3 py-1.5 rounded border border-zinc-300 dark:border-zinc-700"
    >
      {mode === 'dark' ? 'Dark' : 'Light'}
    </button>
  );
}