'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useNodeStore } from '@/lib/store';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { format } from 'date-fns';

interface ChartContainerProps {
  title: string;
  description: string;
  dataKey: 'slotHistory' | 'cpuHistory' | 'memoryHistory' | 'networkHistory' | 'tpsHistory';
  color?: string;
  type?: 'line' | 'area';
}

export function ChartContainer({
  title,
  description,
  dataKey,
  color = 'hsl(var(--primary))',
  type = 'area',
}: ChartContainerProps) {
  const data = useNodeStore((state) => state[dataKey]);

  const formattedData = data.map((item) => ({
    time: format(item.time, 'HH:mm:ss'),
    value: 'slot' in item ? item.slot : 
           'usage' in item ? item.usage :
           'tps' in item ? item.tps :
           'rx' in item ? item.rx : 0,
    ...(dataKey === 'networkHistory' && 'tx' in item ? { tx: item.tx } : {}),
  }));

  if (formattedData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] flex items-center justify-center text-muted-foreground">
            Waiting for data...
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          {type === 'line' ? (
            <LineChart data={formattedData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="time"
                className="text-xs"
                tick={{ fill: 'hsl(var(--muted-foreground))' }}
              />
              <YAxis
                className="text-xs"
                tick={{ fill: 'hsl(var(--muted-foreground))' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--background))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke={color}
                strokeWidth={2}
                dot={false}
              />
              {dataKey === 'networkHistory' && (
                <Line
                  type="monotone"
                  dataKey="tx"
                  stroke="hsl(var(--accent))"
                  strokeWidth={2}
                  dot={false}
                />
              )}
            </LineChart>
          ) : (
            <AreaChart data={formattedData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="time"
                className="text-xs"
                tick={{ fill: 'hsl(var(--muted-foreground))' }}
              />
              <YAxis
                className="text-xs"
                tick={{ fill: 'hsl(var(--muted-foreground))' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--background))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke={color}
                fill={color}
                fillOpacity={0.2}
                strokeWidth={2}
              />
              {dataKey === 'networkHistory' && (
                <Area
                  type="monotone"
                  dataKey="tx"
                  stroke="hsl(var(--accent))"
                  fill="hsl(var(--accent))"
                  fillOpacity={0.2}
                  strokeWidth={2}
                />
              )}
            </AreaChart>
          )}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}