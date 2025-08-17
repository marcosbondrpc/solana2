import AsyncEChart from './AsyncEChart';

export default function BarChart({ data, height = 200, title }: { data: { name: string; value: number }[]; height?: number; title?: string }) {
  const option = {
    title: title ? { text: title, left: 'center', textStyle: { fontSize: 12 } } : undefined,
    tooltip: { trigger: 'axis' },
    grid: { left: 8, right: 8, top: title ? 28 : 8, bottom: 24, containLabel: true },
    xAxis: { type: 'category', data: data.map(d => d.name), axisLabel: { color: '#888', rotate: 30 } },
    yAxis: { type: 'value', axisLabel: { color: '#888' }, splitLine: { lineStyle: { color: 'rgba(0,0,0,0.08)' } } },
    series: [{ type: 'bar', data: data.map(d => d.value), itemStyle: { color: '#6f5cff' } }]
  };
  return <AsyncEChart option={option} style={{ height }} notMerge />;
}