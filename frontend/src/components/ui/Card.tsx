import { PropsWithChildren } from 'react';

export function Card(props: PropsWithChildren<{ title?: string; className?: string }>) {
  const { title, className, children } = props;
  return (
    <section className={`rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-900/50 ${className || ''}`}>
      {title ? <header className="px-4 py-2 border-b border-zinc-200 dark:border-zinc-800 font-medium">{title}</header> : null}
      <div className="p-4">{children}</div>
    </section>
  );
}