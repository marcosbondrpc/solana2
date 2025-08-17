import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';

interface ImpactData {
  date: Date;
  extracted: number;
  tips: number;
  fees: number;
  congestion: number;
  victims: number;
  avgVictimLoss: number;
}

interface AggregatedMetrics {
  period: '7d' | '30d' | '90d';
  totalExtracted: number;
  totalTips: number;
  totalFees: number;
  avgCongestion: number;
  totalVictims: number;
  topExtractors: { address: string; amount: number }[];
  venueBreakdown: { venue: string; volume: number; percentage: number }[];
}

interface EconomicImpactProps {
  data?: ImpactData[];
  metrics?: AggregatedMetrics[];
}

export const EconomicImpact: React.FC<EconomicImpactProps> = ({
  data = [],
  metrics = []
}) => {
  const extractionChartRef = useRef<SVGSVGElement>(null);
  const burdenChartRef = useRef<SVGSVGElement>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<'7d' | '30d' | '90d'>('30d');
  const [animatedTotals, setAnimatedTotals] = useState({ extracted: 0, victims: 0, congestion: 0 });

  // Generate mock data if not provided
  const impactData = useMemo(() => {
    if (data.length > 0) return data;
    
    const days = selectedPeriod === '7d' ? 7 : selectedPeriod === '30d' ? 30 : 90;
    return Array(days).fill(0).map((_, i) => ({
      date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000),
      extracted: Math.random() * 10000 + 1000,
      tips: Math.random() * 1000 + 100,
      fees: Math.random() * 500 + 50,
      congestion: Math.random() * 100,
      victims: Math.floor(Math.random() * 500 + 50),
      avgVictimLoss: Math.random() * 50 + 5
    }));
  }, [data, selectedPeriod]);

  // Generate aggregated metrics if not provided
  const aggregatedMetrics = useMemo(() => {
    if (metrics.length > 0) return metrics;
    
    return [
      {
        period: '7d' as const,
        totalExtracted: 45678,
        totalTips: 5678,
        totalFees: 2345,
        avgCongestion: 67,
        totalVictims: 1234,
        topExtractors: [
          { address: 'B91piB...JoFpi', amount: 12345 },
          { address: '6gAnjd...y338', amount: 9876 },
          { address: 'CaCZgx...rH7q', amount: 7654 }
        ],
        venueBreakdown: [
          { venue: 'Raydium', volume: 45000, percentage: 45 },
          { venue: 'PumpSwap', volume: 35000, percentage: 35 },
          { venue: 'Orca', volume: 20000, percentage: 20 }
        ]
      },
      {
        period: '30d' as const,
        totalExtracted: 234567,
        totalTips: 34567,
        totalFees: 12345,
        avgCongestion: 72,
        totalVictims: 5678,
        topExtractors: [
          { address: 'D9Akv6...XWEC', amount: 56789 },
          { address: 'E6YoRP...mfLi', amount: 45678 },
          { address: 'GGG4BB...cNQ', amount: 34567 }
        ],
        venueBreakdown: [
          { venue: 'Raydium', volume: 120000, percentage: 42 },
          { venue: 'PumpSwap', volume: 100000, percentage: 35 },
          { venue: 'Orca', volume: 65000, percentage: 23 }
        ]
      },
      {
        period: '90d' as const,
        totalExtracted: 789012,
        totalTips: 123456,
        totalFees: 45678,
        avgCongestion: 78,
        totalVictims: 23456,
        topExtractors: [
          { address: 'EAJ1DP...CYMW', amount: 234567 },
          { address: '2brXWR...PZTy', amount: 189012 },
          { address: 'B91piB...JoFpi', amount: 156789 }
        ],
        venueBreakdown: [
          { venue: 'Raydium', volume: 400000, percentage: 40 },
          { venue: 'PumpSwap', volume: 380000, percentage: 38 },
          { venue: 'Orca', volume: 220000, percentage: 22 }
        ]
      }
    ];
  }, [metrics]);

  const currentMetrics = aggregatedMetrics.find(m => m.period === selectedPeriod) || aggregatedMetrics[0];

  // Animate totals
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimatedTotals(prev => ({
        extracted: Math.min(prev.extracted + currentMetrics.totalExtracted / 50, currentMetrics.totalExtracted),
        victims: Math.min(prev.victims + currentMetrics.totalVictims / 50, currentMetrics.totalVictims),
        congestion: Math.min(prev.congestion + 1, currentMetrics.avgCongestion)
      }));
    }, 50);

    return () => clearInterval(interval);
  }, [currentMetrics]);

  // Draw SOL extraction visualization
  useEffect(() => {
    if (!extractionChartRef.current || impactData.length === 0) return;

    const svg = d3.select(extractionChartRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 80, bottom: 50, left: 70 };
    const width = 700 - margin.left - margin.right;
    const height = 350 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(impactData, d => d.date) as [Date, Date])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(impactData, d => d.extracted) as number])
      .nice()
      .range([height, 0]);

    // Grid
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      .call(
        d3.axisBottom(xScale)
          .tickSize(-height)
          .tickFormat(() => '')
      )
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    g.append('g')
      .attr('class', 'grid')
      .call(
        d3.axisLeft(yScale)
          .tickSize(-width)
          .tickFormat(() => '')
      )
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    // Area gradient
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'area-gradient')
      .attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', 0).attr('y1', height)
      .attr('x2', 0).attr('y2', 0);

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#ff00ff')
      .attr('stop-opacity', 0.1);

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#00ff88')
      .attr('stop-opacity', 0.6);

    // Area
    const area = d3.area<ImpactData>()
      .x(d => xScale(d.date))
      .y0(height)
      .y1(d => yScale(d.extracted))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(impactData)
      .attr('fill', 'url(#area-gradient)')
      .attr('d', area);

    // Line
    const line = d3.line<ImpactData>()
      .x(d => xScale(d.date))
      .y(d => yScale(d.extracted))
      .curve(d3.curveMonotoneX);

    const path = g.append('path')
      .datum(impactData)
      .attr('fill', 'none')
      .attr('stroke', '#00ff88')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Animate line drawing
    const totalLength = (path.node() as SVGPathElement).getTotalLength();
    path
      .attr('stroke-dasharray', `${totalLength} ${totalLength}`)
      .attr('stroke-dashoffset', totalLength)
      .transition()
      .duration(2000)
      .ease(d3.easeLinear)
      .attr('stroke-dashoffset', 0);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%m/%d') as any))
      .style('color', '#666');

    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => `${d3.format('.1s')(d)} SOL`))
      .style('color', '#666');

    // Labels
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - height / 2)
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('fill', '#888')
      .style('font-size', '12px')
      .text('SOL Extracted');

    // Add moving average
    const movingAvg = impactData.map((d, i) => {
      const start = Math.max(0, i - 3);
      const subset = impactData.slice(start, i + 1);
      const avg = subset.reduce((sum, d) => sum + d.extracted, 0) / subset.length;
      return { date: d.date, value: avg };
    });

    g.append('path')
      .datum(movingAvg)
      .attr('fill', 'none')
      .attr('stroke', '#ff00ff')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '5,5')
      .attr('opacity', 0.7)
      .attr('d', d3.line<{ date: Date; value: number }>()
        .x(d => xScale(d.date))
        .y(d => yScale(d.value))
        .curve(d3.curveMonotoneX)
      );

  }, [impactData]);

  // Draw tip/fee burden chart
  useEffect(() => {
    if (!burdenChartRef.current || impactData.length === 0) return;

    const svg = d3.select(burdenChartRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 80, bottom: 50, left: 70 };
    const width = 700 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const stack = d3.stack<ImpactData>()
      .keys(['tips', 'fees'])
      .order(d3.stackOrderNone)
      .offset(d3.stackOffsetNone);

    const series = stack(impactData);

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(impactData, d => d.date) as [Date, Date])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(series, s => d3.max(s, d => d[1])) as number])
      .nice()
      .range([height, 0]);

    const color = d3.scaleOrdinal()
      .domain(['tips', 'fees'])
      .range(['#00ffff', '#ffff00']);

    // Area generator
    const area = d3.area<any>()
      .x(d => xScale(d.data.date))
      .y0(d => yScale(d[0]))
      .y1(d => yScale(d[1]))
      .curve(d3.curveMonotoneX);

    // Draw stacked areas
    g.selectAll('.layer')
      .data(series)
      .enter()
      .append('path')
      .attr('class', 'layer')
      .attr('fill', d => color(d.key) as string)
      .attr('opacity', 0.7)
      .attr('d', area);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%m/%d') as any))
      .style('color', '#666');

    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => `${d3.format('.1s')(d)} SOL`))
      .style('color', '#666');

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${width - 60}, 20)`);

    ['tips', 'fees'].forEach((key, i) => {
      const legendRow = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      legendRow.append('rect')
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', color(key) as string);

      legendRow.append('text')
        .attr('x', 15)
        .attr('y', 9)
        .style('font-size', '12px')
        .style('fill', '#888')
        .text(key.charAt(0).toUpperCase() + key.slice(1));
    });

  }, [impactData]);

  return (
    <div className="space-y-6">
      {/* Period Selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-zinc-100">Economic Impact Analysis</h2>
        <div className="flex gap-2">
          {(['7d', '30d', '90d'] as const).map(period => (
            <button
              key={period}
              onClick={() => setSelectedPeriod(period)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedPeriod === period
                  ? 'bg-gradient-to-r from-cyan-600 to-purple-600 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
              }`}
            >
              {period === '7d' ? '7 Days' : period === '30d' ? '30 Days' : '90 Days'}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-4 gap-4">
        <motion.div
          className="glass rounded-xl p-6"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-3xl font-bold text-zinc-100">
            {animatedTotals.extracted.toFixed(0)}
          </div>
          <div className="text-sm text-zinc-400 mt-1">SOL Extracted</div>
          <div className="text-xs text-green-400 mt-2">
            +{((currentMetrics.totalExtracted / 1000) * 12.5).toFixed(1)}% vs prev
          </div>
        </motion.div>

        <motion.div
          className="glass rounded-xl p-6"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-3xl font-bold text-zinc-100">
            {animatedTotals.victims.toFixed(0)}
          </div>
          <div className="text-sm text-zinc-400 mt-1">Total Victims</div>
          <div className="text-xs text-red-400 mt-2">
            Avg loss: {(currentMetrics.totalExtracted / currentMetrics.totalVictims).toFixed(1)} SOL
          </div>
        </motion.div>

        <motion.div
          className="glass rounded-xl p-6"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-3xl font-bold text-zinc-100">
            {currentMetrics.totalTips.toFixed(0)}
          </div>
          <div className="text-sm text-zinc-400 mt-1">Tips Paid</div>
          <div className="text-xs text-yellow-400 mt-2">
            {((currentMetrics.totalTips / currentMetrics.totalExtracted) * 100).toFixed(1)}% of extracted
          </div>
        </motion.div>

        <motion.div
          className="glass rounded-xl p-6"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-3xl font-bold text-zinc-100">
            {animatedTotals.congestion.toFixed(0)}%
          </div>
          <div className="text-sm text-zinc-400 mt-1">Avg Congestion</div>
          <div className="text-xs text-orange-400 mt-2">
            Peak: {(currentMetrics.avgCongestion * 1.3).toFixed(0)}%
          </div>
        </motion.div>
      </div>

      {/* SOL Extraction Chart */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          SOL Extraction Over Time
        </h3>
        <svg ref={extractionChartRef}></svg>
      </div>

      {/* Tip/Fee Burden Analysis */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          Network Burden Analysis (Tips & Fees)
        </h3>
        <svg ref={burdenChartRef}></svg>
      </div>

      {/* Top Extractors */}
      <div className="grid grid-cols-2 gap-4">
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">
            Top Value Extractors
          </h3>
          <div className="space-y-3">
            {currentMetrics.topExtractors.map((extractor, i) => (
              <div key={extractor.address} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                    i === 0 ? 'bg-yellow-500/20 text-yellow-400' :
                    i === 1 ? 'bg-zinc-500/20 text-zinc-400' :
                    'bg-orange-500/20 text-orange-400'
                  }`}>
                    {i + 1}
                  </div>
                  <span className="font-mono text-sm text-zinc-300">
                    {extractor.address}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-semibold text-zinc-100">
                    {extractor.amount.toLocaleString()} SOL
                  </div>
                  <div className="text-xs text-zinc-500">
                    {((extractor.amount / currentMetrics.totalExtracted) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Venue Breakdown */}
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">
            Extraction by Venue
          </h3>
          <div className="space-y-3">
            {currentMetrics.venueBreakdown.map(venue => (
              <div key={venue.venue} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-zinc-300">{venue.venue}</span>
                  <span className="text-sm font-semibold text-zinc-100">
                    {venue.volume.toLocaleString()} SOL
                  </span>
                </div>
                <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-cyan-500 to-purple-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${venue.percentage}%` }}
                    transition={{ duration: 1, delay: 0.1 }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Congestion Externality Metrics */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          Network Congestion Impact
        </h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-zinc-900/50 rounded-lg p-4">
            <div className="text-2xl font-bold text-red-400">
              {(currentMetrics.avgCongestion * 1.2).toFixed(0)}%
            </div>
            <div className="text-sm text-zinc-400 mt-1">Peak Congestion</div>
            <div className="text-xs text-zinc-500 mt-2">During MEV spikes</div>
          </div>
          <div className="bg-zinc-900/50 rounded-lg p-4">
            <div className="text-2xl font-bold text-yellow-400">
              {(currentMetrics.totalFees / 1000).toFixed(1)}k SOL
            </div>
            <div className="text-sm text-zinc-400 mt-1">Extra Fees Paid</div>
            <div className="text-xs text-zinc-500 mt-2">Due to congestion</div>
          </div>
          <div className="bg-zinc-900/50 rounded-lg p-4">
            <div className="text-2xl font-bold text-orange-400">
              {(currentMetrics.totalVictims * 0.3).toFixed(0)}
            </div>
            <div className="text-sm text-zinc-400 mt-1">Failed Transactions</div>
            <div className="text-xs text-zinc-500 mt-2">Per day average</div>
          </div>
        </div>
      </div>
    </div>
  );
};