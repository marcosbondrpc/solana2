import React, { createContext, useContext, useState, PropsWithChildren } from 'react';
import { cn } from '@/lib/utils';

interface TabsContextValue {
  value: string;
  setValue: (v: string) => void;
}
const TabsContext = createContext<TabsContextValue | null>(null);

export interface TabsProps extends PropsWithChildren {
  value?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
  className?: string;
}
export const Tabs: React.FC<TabsProps> = ({ value, defaultValue, onValueChange, className, children }) => {
  const [internal, setInternal] = useState<string>(defaultValue || value || '');
  const current = value !== undefined ? value : internal;
  const setValue = (v: string) => {
    if (onValueChange) onValueChange(v);
    if (value === undefined) setInternal(v);
  };
  return (
    <TabsContext.Provider value={{ value: current, setValue }}>
      <div className={cn(className)}>{children}</div>
    </TabsContext.Provider>
  );
};

export const TabsList: React.FC<PropsWithChildren<{ className?: string }>> = ({ className, children }) => {
  return <div className={cn('flex gap-2', className)}>{children}</div>;
};

export const TabsTrigger: React.FC<PropsWithChildren<{ value: string; className?: string }>> = ({ value, className, children }) => {
  const ctx = useContext(TabsContext)!;
  const active = ctx.value === value;
  return (
    <button
      className={cn(
        'px-3 py-2 rounded-md transition-colors',
        active ? 'bg-white/10 text-white' : 'text-gray-400 hover:text-white hover:bg-white/5',
        className
      )}
      onClick={() => ctx.setValue(value)}
      type="button"
    >
      {children}
    </button>
  );
};

export const TabsContent: React.FC<PropsWithChildren<{ value: string; className?: string }>> = ({ value, className, children }) => {
  const ctx = useContext(TabsContext)!;
  if (ctx.value !== value) return null;
  return <div className={cn(className)}>{children}</div>;
};