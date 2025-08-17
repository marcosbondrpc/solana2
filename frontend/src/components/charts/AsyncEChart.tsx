import { lazy, Suspense } from 'react';
const ReactECharts = lazy(() => import('echarts-for-react'));
export default function AsyncEChart(props: any) {
  return (
    <Suspense fallback=<div className="h-40 animate-pulse bg-zinc-100 dark:bg-zinc-800 rounded" />>
      <ReactECharts {...props} />
    </Suspense>
  );
}