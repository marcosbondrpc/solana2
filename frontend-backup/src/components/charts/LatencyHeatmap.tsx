/**
 * High-performance latency heatmap using @visx
 * Renders 60fps with thousands of data points
 */

import React, { useMemo } from 'react';
import { scaleLinear, scaleTime } from '@visx/scale';
import { HeatmapRect } from '@visx/heatmap';
import { Group } from '@visx/group';
import { AxisBottom, AxisLeft } from '@visx/axis';

interface LatencyHeatmapProps {
  data: number[][];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  colorScheme?: 'green' | 'blue' | 'red' | 'viridis';
}

export function LatencyHeatmap({
  data,
  width = 800,
  height = 400,
  margin = { top: 20, right: 20, bottom: 40, left: 60 },
  colorScheme = 'viridis'
}: LatencyHeatmapProps) {
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Transform data for heatmap
  const heatmapData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    return data.map((row, i) => ({
      bin: i,
      bins: row.map((value, j) => ({
        bin: j,
        count: value
      }))
    }));
  }, [data]);

  // Scales
  const xScale = useMemo(
    () => scaleLinear({
      domain: [0, data[0]?.length || 60],
      range: [0, innerWidth]
    }),
    [data, innerWidth]
  );

  const yScale = useMemo(
    () => scaleLinear({
      domain: [0, data.length || 24],
      range: [0, innerHeight]
    }),
    [data, innerHeight]
  );

  // Color scale based on scheme
  const colorScale = useMemo(() => {
    const max = Math.max(...data.flat());
    
    const colors = {
      green: ['#0d9488', '#10b981', '#34d399', '#86efac', '#bbf7d0'],
      blue: ['#1e40af', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd'],
      red: ['#991b1b', '#dc2626', '#ef4444', '#f87171', '#fca5a5'],
      viridis: ['#440154', '#3b528b', '#21908c', '#5dc863', '#fde725']
    };
    
    return scaleLinear({
      domain: [0, max * 0.25, max * 0.5, max * 0.75, max],
      range: colors[colorScheme]
    });
  }, [data, colorScheme]);

  // Bin dimensions
  const binWidth = innerWidth / (data[0]?.length || 60);
  const binHeight = innerHeight / (data.length || 24);

  return (
    <svg width={width} height={height}>
      <Group left={margin.left} top={margin.top}>
        <HeatmapRect
          data={heatmapData}
          xScale={(d) => xScale(d) ?? 0}
          yScale={(d) => yScale(d) ?? 0}
          colorScale={colorScale}
          binWidth={binWidth}
          binHeight={binHeight}
        >
          {(heatmap) =>
            heatmap.map((heatmapBins) =>
              heatmapBins.map((bin) => (
                <rect
                  key={`heatmap-rect-${bin.row}-${bin.column}`}
                  className="cursor-pointer transition-opacity hover:opacity-80"
                  width={bin.width}
                  height={bin.height}
                  x={bin.x}
                  y={bin.y}
                  fill={bin.color}
                  fillOpacity={bin.opacity}
                  onClick={() => {
                    console.log('Latency bin clicked:', {
                      time: bin.column,
                      percentile: bin.row,
                      value: bin.datum
                    });
                  }}
                />
              ))
            )
          }
        </HeatmapRect>

        <AxisBottom
          top={innerHeight}
          scale={xScale}
          numTicks={10}
          label="Time (minutes)"
          labelProps={{
            fill: '#9ca3af',
            textAnchor: 'middle',
            fontSize: 12
          }}
          tickLabelProps={() => ({
            fill: '#9ca3af',
            fontSize: 10,
            textAnchor: 'middle'
          })}
          stroke="#4b5563"
        />

        <AxisLeft
          scale={yScale}
          numTicks={6}
          label="Latency Percentile"
          labelProps={{
            fill: '#9ca3af',
            textAnchor: 'middle',
            fontSize: 12
          }}
          tickLabelProps={() => ({
            fill: '#9ca3af',
            fontSize: 10,
            textAnchor: 'end',
            dy: '0.33em'
          })}
          tickFormat={(value) => `P${value}`}
          stroke="#4b5563"
        />

        {/* Legend */}
        <Group left={innerWidth - 100} top={0}>
          <rect
            width={20}
            height={100}
            fill="url(#gradient)"
            stroke="#4b5563"
            strokeWidth={1}
          />
          <text x={25} y={10} fontSize={10} fill="#9ca3af">
            {Math.max(...data.flat()).toFixed(0)}ms
          </text>
          <text x={25} y={95} fontSize={10} fill="#9ca3af">
            0ms
          </text>
        </Group>
      </Group>

      {/* Gradient definition for legend */}
      <defs>
        <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
          {colorScheme === 'viridis' && (
            <>
              <stop offset="0%" stopColor="#fde725" />
              <stop offset="25%" stopColor="#5dc863" />
              <stop offset="50%" stopColor="#21908c" />
              <stop offset="75%" stopColor="#3b528b" />
              <stop offset="100%" stopColor="#440154" />
            </>
          )}
          {colorScheme === 'green' && (
            <>
              <stop offset="0%" stopColor="#bbf7d0" />
              <stop offset="100%" stopColor="#0d9488" />
            </>
          )}
          {colorScheme === 'blue' && (
            <>
              <stop offset="0%" stopColor="#93c5fd" />
              <stop offset="100%" stopColor="#1e40af" />
            </>
          )}
          {colorScheme === 'red' && (
            <>
              <stop offset="0%" stopColor="#fca5a5" />
              <stop offset="100%" stopColor="#991b1b" />
            </>
          )}
        </linearGradient>
      </defs>
    </svg>
  );
}