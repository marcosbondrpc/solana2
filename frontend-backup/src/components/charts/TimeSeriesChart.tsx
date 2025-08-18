import AsyncEChart from './AsyncEChart';

type Point = { t: number | string; v: number };
export default function TimeSeriesChart({ data, height = 200, title }: { data: Point[]; height?: number; title?: string }) {
  const option = {
    title: title ? { text: title, left: 'center', textStyle: { fontSize: 12 } } : undefined,
    tooltip: { trigger: 'axis' },
    grid: { left: 8, right: 8, top: title ? 28 : 8, bottom: 24, containLabel: true },
    xAxis: { type: 'time', axisLabel: { color: '#888' } },
    yAxis: { type: 'value', axisLabel: { color: '#888' }, splitLine: { lineStyle: { color: 'rgba(0,0,0,0.08)' } } },
    series: [{ type: 'line', smooth: true, symbol: 'none', areaStyle: {}, data: data.map(p => [p.t, p.v]) }]
  };
  return <AsyncEChart option={option} style={{ height }} notMerge />;
}