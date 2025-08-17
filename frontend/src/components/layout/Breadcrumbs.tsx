<![CDATA[
import { Link, useLocation } from 'react-router-dom';

export default function Breadcrumbs() {
  const { pathname } = useLocation();
  const parts = pathname.split('/').filter(Boolean);
  const crumbs = parts.map((_, i) => '/' + parts.slice(0, i + 1).join('/'));
  return (
    <nav aria-label="Breadcrumb" className="text-sm text-zinc-500 dark:text-zinc-400">
      <ol className="flex items-center gap-2">
        <li><Link to="/" className="hover:underline">Home</Link></li>
        {crumbs.map((href, i) => {
          const label = parts[i] || '';
          return (
            <li key={href} className="flex items-center gap-2">
              <span>/</span>
              <Link to={href} className="capitalize hover:underline">
                {label || 'ops'}
              </Link>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
]]>