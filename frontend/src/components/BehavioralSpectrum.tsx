import React, { useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { motion, AnimatePresence } from 'framer-motion';

interface EntityProfile {
  address: string;
  attackStyle: 'surgical' | 'shotgun' | 'hybrid';
  riskAppetite: number; // 0-100
  feePosture: 'aggressive' | 'conservative' | 'adaptive';
  uptimeCadence: number[]; // 24-hour activity pattern
  successRate: number;
  avgLatency: number;
  volumeSOL: number;
  sandwichCount: number;
  arbitrageCount: number;
  liquidationCount: number;
  jitCount: number;
}

interface BehavioralSpectrumProps {
  profiles: EntityProfile[];
  selectedAddress?: string;
  onAddressSelect?: (address: string) => void;
}

const DIMENSIONS = [
  'Attack Style',
  'Risk Appetite',
  'Fee Posture',
  'Success Rate',
  'Latency',
  'Volume',
  'Sandwich',
  'Arbitrage',
  'Liquidation',
  'JIT'
];

export const BehavioralSpectrum: React.FC<BehavioralSpectrumProps> = ({
  profiles,
  selectedAddress,
  onAddressSelect
}) => {
  const radarRef = useRef<SVGSVGElement>(null);
  const heatmapRef = useRef<HTMLDivElement>(null);

  // Normalize entity data for radar chart
  const normalizeProfile = (profile: EntityProfile) => {
    return {
      address: profile.address,
      values: [
        profile.attackStyle === 'surgical' ? 100 : profile.attackStyle === 'shotgun' ? 0 : 50,
        profile.riskAppetite,
        profile.feePosture === 'aggressive' ? 100 : profile.feePosture === 'conservative' ? 0 : 50,
        profile.successRate,
        100 - (profile.avgLatency / 100), // Invert so lower latency = higher score
        Math.min(100, profile.volumeSOL / 10000), // Normalize to 100
        Math.min(100, profile.sandwichCount / 100),
        Math.min(100, profile.arbitrageCount / 100),
        Math.min(100, profile.liquidationCount / 50),
        Math.min(100, profile.jitCount / 200)
      ]
    };
  };

  // Draw radar chart with D3
  useEffect(() => {
    if (!radarRef.current || profiles.length === 0) return;

    const svg = d3.select(radarRef.current);
    svg.selectAll('*').remove();

    const width = 500;
    const height = 500;
    const radius = Math.min(width, height) / 2 - 40;
    const angleSlice = (Math.PI * 2) / DIMENSIONS.length;

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    // Scales
    const rScale = d3.scaleLinear().domain([0, 100]).range([0, radius]);

    // Grid circles
    const levels = 5;
    for (let level = 0; level < levels; level++) {
      g.append('circle')
        .attr('r', (radius / levels) * (level + 1))
        .style('fill', 'none')
        .style('stroke', '#333')
        .style('stroke-opacity', 0.3);
    }

    // Axis lines
    const axes = g.selectAll('.axis').data(DIMENSIONS).enter().append('g').attr('class', 'axis');

    axes
      .append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', (d, i) => rScale(100) * Math.cos(angleSlice * i - Math.PI / 2))
      .attr('y2', (d, i) => rScale(100) * Math.sin(angleSlice * i - Math.PI / 2))
      .style('stroke', '#444')
      .style('stroke-width', '1px');

    // Axis labels
    axes
      .append('text')
      .attr('class', 'legend')
      .style('font-size', '11px')
      .style('fill', '#888')
      .attr('text-anchor', 'middle')
      .attr('x', (d, i) => rScale(120) * Math.cos(angleSlice * i - Math.PI / 2))
      .attr('y', (d, i) => rScale(120) * Math.sin(angleSlice * i - Math.PI / 2))
      .text(d => d);

    // Plot data
    const line = d3.lineRadial<number>()
      .radius(d => rScale(d))
      .angle((d, i) => angleSlice * i)
      .curve(d3.curveLinearClosed);

    const colorScale = d3.scaleOrdinal()
      .domain(profiles.map(p => p.address))
      .range(['#00ff88', '#ff00ff', '#00ffff', '#ffff00', '#ff6600', '#6666ff']);

    profiles.forEach((profile, idx) => {
      const normalized = normalizeProfile(profile);
      const color = colorScale(profile.address) as string;

      // Area
      g.append('path')
        .datum(normalized.values)
        .attr('d', line as any)
        .style('fill', color)
        .style('fill-opacity', selectedAddress === profile.address ? 0.3 : 0.1)
        .style('stroke', color)
        .style('stroke-width', selectedAddress === profile.address ? 2 : 1)
        .style('stroke-opacity', selectedAddress === profile.address ? 1 : 0.5)
        .style('cursor', 'pointer')
        .on('click', () => onAddressSelect?.(profile.address))
        .on('mouseover', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .style('fill-opacity', 0.4)
            .style('stroke-width', 2);
        })
        .on('mouseout', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .style('fill-opacity', selectedAddress === profile.address ? 0.3 : 0.1)
            .style('stroke-width', selectedAddress === profile.address ? 2 : 1);
        });

      // Data points
      normalized.values.forEach((value, i) => {
        g.append('circle')
          .attr('cx', rScale(value) * Math.cos(angleSlice * i - Math.PI / 2))
          .attr('cy', rScale(value) * Math.sin(angleSlice * i - Math.PI / 2))
          .attr('r', 3)
          .style('fill', color)
          .style('fill-opacity', 0.8);
      });
    });
  }, [profiles, selectedAddress, onAddressSelect]);

  // Risk appetite heatmap
  const riskHeatmap = useMemo(() => {
    const gridSize = 10;
    const grid = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
    
    profiles.forEach(profile => {
      const x = Math.floor((profile.riskAppetite / 100) * (gridSize - 1));
      const y = Math.floor((profile.successRate / 100) * (gridSize - 1));
      grid[y][x]++;
    });

    return grid;
  }, [profiles]);

  // Uptime cadence visualization
  const uptimePatterns = useMemo(() => {
    const hourlyActivity = Array(24).fill(0);
    profiles.forEach(profile => {
      profile.uptimeCadence.forEach((activity, hour) => {
        hourlyActivity[hour] += activity;
      });
    });
    return hourlyActivity.map(a => a / profiles.length);
  }, [profiles]);

  return (
    <div className="space-y-6">
      {/* Radar Chart */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          Entity Behavioral Profiles
        </h3>
        <div className="flex justify-center">
          <svg ref={radarRef}></svg>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {profiles.map((profile, idx) => (
            <button
              key={profile.address}
              onClick={() => onAddressSelect?.(profile.address)}
              className={`px-3 py-1 rounded-full text-xs font-mono transition-all ${
                selectedAddress === profile.address
                  ? 'bg-gradient-to-r from-cyan-600 to-purple-600 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
              }`}
            >
              {profile.address.slice(0, 6)}...{profile.address.slice(-4)}
            </button>
          ))}
        </div>
      </div>

      {/* Risk Appetite Heatmap */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          Risk Appetite vs Success Rate
        </h3>
        <div className="relative" ref={heatmapRef}>
          <div className="grid grid-cols-10 gap-1">
            {riskHeatmap.map((row, y) => (
              <div key={y} className="flex gap-1">
                {row.map((value, x) => (
                  <div
                    key={`${x}-${y}`}
                    className="w-8 h-8 rounded transition-all hover:scale-110"
                    style={{
                      backgroundColor: value > 0 
                        ? `rgba(0, 255, 136, ${Math.min(1, value * 0.3)})`
                        : 'rgba(255, 255, 255, 0.05)',
                      boxShadow: value > 2 
                        ? '0 0 10px rgba(0, 255, 136, 0.5)'
                        : 'none'
                    }}
                    title={`Risk: ${x * 10}%, Success: ${(9 - y) * 10}%, Count: ${value}`}
                  />
                ))}
              </div>
            ))}
          </div>
          <div className="absolute -bottom-6 left-0 right-0 text-center text-xs text-zinc-500">
            Risk Appetite →
          </div>
          <div className="absolute top-0 bottom-0 -left-6 flex items-center">
            <span className="text-xs text-zinc-500 -rotate-90">Success Rate →</span>
          </div>
        </div>
      </div>

      {/* Uptime Cadence Pattern */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          24-Hour Activity Pattern
        </h3>
        <div className="flex items-end justify-between h-32 gap-1">
          {uptimePatterns.map((activity, hour) => (
            <motion.div
              key={hour}
              className="flex-1 bg-gradient-to-t from-cyan-600 to-purple-600 rounded-t hover:opacity-80 transition-opacity"
              initial={{ height: 0 }}
              animate={{ height: `${activity * 100}%` }}
              transition={{ duration: 0.5, delay: hour * 0.02 }}
              title={`${hour}:00 - Activity: ${(activity * 100).toFixed(1)}%`}
            />
          ))}
        </div>
        <div className="flex justify-between mt-2">
          <span className="text-xs text-zinc-500">00:00</span>
          <span className="text-xs text-zinc-500">06:00</span>
          <span className="text-xs text-zinc-500">12:00</span>
          <span className="text-xs text-zinc-500">18:00</span>
          <span className="text-xs text-zinc-500">23:59</span>
        </div>
      </div>

      {/* Attack Style Distribution */}
      <div className="grid grid-cols-3 gap-4">
        {['surgical', 'shotgun', 'hybrid'].map(style => {
          const count = profiles.filter(p => p.attackStyle === style).length;
          const percentage = (count / profiles.length) * 100;
          return (
            <div key={style} className="glass rounded-xl p-4">
              <div className="text-2xl font-bold text-zinc-100">
                {count}
              </div>
              <div className="text-sm text-zinc-400 capitalize">{style}</div>
              <div className="mt-2 h-2 bg-zinc-800 rounded-full overflow-hidden">
                <motion.div
                  className={`h-full ${
                    style === 'surgical' ? 'bg-green-500' :
                    style === 'shotgun' ? 'bg-red-500' : 'bg-yellow-500'
                  }`}
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ duration: 1 }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Fee Posture Analysis */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          Fee Posture Distribution
        </h3>
        <div className="space-y-3">
          {['aggressive', 'conservative', 'adaptive'].map(posture => {
            const entities = profiles.filter(p => p.feePosture === posture);
            const avgVolume = entities.reduce((acc, e) => acc + e.volumeSOL, 0) / entities.length || 0;
            
            return (
              <div key={posture} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${
                    posture === 'aggressive' ? 'bg-red-500' :
                    posture === 'conservative' ? 'bg-blue-500' : 'bg-purple-500'
                  }`} />
                  <span className="capitalize text-zinc-300">{posture}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-zinc-500">
                    {entities.length} entities
                  </span>
                  <span className="text-sm text-zinc-400">
                    Avg: {avgVolume.toFixed(2)} SOL
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};