import AsyncEChart from './AsyncEChart';

export default function Sparkline({ data, height = 64, color = '#6f5cff' }: { data: number[]; height?: number; color?: string }) {
  const option = {
    grid: { left: 0, right: 0, top: 0, bottom: 0 },
    xAxis: { type: 'category', show: false, data: data.map((_, i) => i) },
    yAxis: { type: 'value', show: false },
    series: [{ type: 'line', smooth: true, symbol: 'none', lineStyle: { color }, areaStyle: { color, opacity: 0.2 }, data }]
  };
  return <AsyncEChart option={option} style={{ height }} notMerge />;
}