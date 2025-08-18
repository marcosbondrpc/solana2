import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { motion, AnimatePresence } from 'framer-motion';

interface AnomalyData {
  timestamp: Date;
  landingRate: number;
  expectedRate: number;
  deviation: number;
  isAnomaly: boolean;
  confidence: number;
}

interface LatencyDistribution {
  bucket: number;
  count: number;
  isOutlier: boolean;
  skewness: number;
}

interface WalletCoordination {
  wallet1: string;
  wallet2: string;
  correlation: number;
  simultaneousActions: number;
  timeDelta: number; // milliseconds
}

interface OrderingQuirk {
  pattern: string;
  frequency: number;
  significance: number;
  examples: { tx1: string; tx2: string; timing: number }[];
}

interface HypothesisTestingProps {
  anomalies?: AnomalyData[];
  latencyDist?: LatencyDistribution[];
  walletCoordination?: WalletCoordination[];
  orderingQuirks?: OrderingQuirk[];
}

export const HypothesisTesting: React.FC<HypothesisTestingProps> = ({
  anomalies = [],
  latencyDist = [],
  walletCoordination = [],
  orderingQuirks = []
}) => {
  const anomalyChartRef = useRef<SVGSVGElement>(null);
  const latencyChartRef = useRef<SVGSVGElement>(null);
  const coordinationRef = useRef<SVGSVGElement>(null);
  const [selectedHypothesis, setSelectedHypothesis] = useState<string>('landing-rate');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.8);
  const [animatedStats, setAnimatedStats] = useState({ anomalies: 0, coordinated: 0, quirks: 0 });

  // Generate mock data if not provided
  const anomalyData = useMemo(() => {
    if (anomalies.length > 0) return anomalies;
    
    return Array(100).fill(0).map((_, i) => {
      const expectedRate = 65 + Math.sin(i / 10) * 5;
      const actualRate = expectedRate + (Math.random() - 0.5) * 20;
      const deviation = Math.abs(actualRate - expectedRate);
      const isAnomaly = deviation > 10;
      
      return {
        timestamp: new Date(Date.now() - (100 - i) * 60 * 60 * 1000),
        landingRate: actualRate,
        expectedRate,
        deviation,
        isAnomaly,
        confidence: isAnomaly ? 0.7 + Math.random() * 0.3 : Math.random() * 0.5
      };
    });
  }, [anomalies]);

  const latencyDistribution = useMemo(() => {
    if (latencyDist.length > 0) return latencyDist;
    
    return Array(50).fill(0).map((_, i) => {
      const bucket = i * 2; // 0-100ms in 2ms buckets
      const normalCount = Math.exp(-Math.pow(bucket - 20, 2) / 200) * 100;
      const outlierProb = bucket > 60 ? 0.1 : 0.01;
      const isOutlier = Math.random() < outlierProb;
      
      return {
        bucket,
        count: normalCount + (isOutlier ? Math.random() * 50 : 0),
        isOutlier,
        skewness: (bucket - 20) / 20
      };
    });
  }, [latencyDist]);

  const walletCoords = useMemo(() => {
    if (walletCoordination.length > 0) return walletCoordination;
    
    const addresses = [
      'B91piB...JoFpi', '6gAnjd...y338', 'CaCZgx...rH7q',
      'D9Akv6...XWEC', 'E6YoRP...mfLi', 'GGG4BB...cNQ'
    ];
    
    const coords: WalletCoordination[] = [];
    for (let i = 0; i < addresses.length; i++) {
      for (let j = i + 1; j < addresses.length; j++) {
        const correlation = Math.random();
        if (correlation > 0.6) {
          coords.push({
            wallet1: addresses[i],
            wallet2: addresses[j],
            correlation,
            simultaneousActions: Math.floor(Math.random() * 100),
            timeDelta: Math.random() * 1000
          });
        }
      }
    }
    return coords;
  }, [walletCoordination]);

  const quirks = useMemo(() => {
    if (orderingQuirks.length > 0) return orderingQuirks;
    
    return [
      {
        pattern: 'Back-to-back sandwich',
        frequency: 87,
        significance: 0.95,
        examples: Array(3).fill(0).map(() => ({
          tx1: `tx_${Math.random().toString(36).substr(2, 9)}`,
          tx2: `tx_${Math.random().toString(36).substr(2, 9)}`,
          timing: Math.random() * 100
        }))
      },
      {
        pattern: 'JIT before arbitrage',
        frequency: 62,
        significance: 0.88,
        examples: Array(3).fill(0).map(() => ({
          tx1: `tx_${Math.random().toString(36).substr(2, 9)}`,
          tx2: `tx_${Math.random().toString(36).substr(2, 9)}`,
          timing: Math.random() * 50
        }))
      },
      {
        pattern: 'Coordinated liquidation',
        frequency: 34,
        significance: 0.92,
        examples: Array(3).fill(0).map(() => ({
          tx1: `tx_${Math.random().toString(36).substr(2, 9)}`,
          tx2: `tx_${Math.random().toString(36).substr(2, 9)}`,
          timing: Math.random() * 200
        }))
      }
    ];
  }, [orderingQuirks]);

  // Animate statistics
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimatedStats(prev => ({
        anomalies: Math.min(prev.anomalies + 1, anomalyData.filter(a => a.isAnomaly).length),
        coordinated: Math.min(prev.coordinated + 1, walletCoords.length),
        quirks: Math.min(prev.quirks + 1, quirks.reduce((sum, q) => sum + q.frequency, 0))
      }));
    }, 50);

    return () => clearInterval(interval);
  }, [anomalyData, walletCoords, quirks]);

  // Draw landing rate anomaly chart
  useEffect(() => {
    if (!anomalyChartRef.current || anomalyData.length === 0) return;

    const svg = d3.select(anomalyChartRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 700 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(anomalyData, d => d.timestamp) as [Date, Date])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([height, 0]);

    // Expected rate area
    const expectedArea = d3.area<AnomalyData>()
      .x(d => xScale(d.timestamp))
      .y0(d => yScale(d.expectedRate - 5))
      .y1(d => yScale(d.expectedRate + 5))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(anomalyData)
      .attr('fill', '#00ff88')
      .attr('opacity', 0.1)
      .attr('d', expectedArea);

    // Actual rate line
    const line = d3.line<AnomalyData>()
      .x(d => xScale(d.timestamp))
      .y(d => yScale(d.landingRate))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(anomalyData)
      .attr('fill', 'none')
      .attr('stroke', '#00ffff')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Anomaly points
    g.selectAll('.anomaly')
      .data(anomalyData.filter(d => d.isAnomaly && d.confidence > confidenceThreshold))
      .enter()
      .append('circle')
      .attr('class', 'anomaly')
      .attr('cx', d => xScale(d.timestamp))
      .attr('cy', d => yScale(d.landingRate))
      .attr('r', 4)
      .attr('fill', '#ff00ff')
      .attr('opacity', d => d.confidence)
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 6);
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 4);
      });

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M') as any))
      .style('color', '#666');

    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => `${d}%`))
      .style('color', '#666');

    // Labels
    g.append('text')
      .attr('x', width / 2)
      .attr('y', -5)
      .style('text-anchor', 'middle')
      .style('fill', '#888')
      .style('font-size', '12px')
      .text('Landing Rate Anomalies');

  }, [anomalyData, confidenceThreshold]);

  // Draw latency distribution
  useEffect(() => {
    if (!latencyChartRef.current || latencyDistribution.length === 0) return;

    const svg = d3.select(latencyChartRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 700 - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, 100])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(latencyDistribution, d => d.count) as number])
      .range([height, 0]);

    // Bars
    g.selectAll('.bar')
      .data(latencyDistribution)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.bucket))
      .attr('y', d => yScale(d.count))
      .attr('width', width / latencyDistribution.length - 1)
      .attr('height', d => height - yScale(d.count))
      .attr('fill', d => d.isOutlier ? '#ff00ff' : '#00ff88')
      .attr('opacity', d => d.isOutlier ? 0.8 : 0.6);

    // Skewness indicator
    const skewness = d3.mean(latencyDistribution, d => d.skewness) || 0;
    g.append('text')
      .attr('x', width - 100)
      .attr('y', 20)
      .style('fill', Math.abs(skewness) > 0.5 ? '#ff00ff' : '#00ff88')
      .style('font-size', '12px')
      .text(`Skewness: ${skewness.toFixed(2)}`);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `${d}ms`))
      .style('color', '#666');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .style('color', '#666');

  }, [latencyDistribution]);

  // Draw wallet coordination network
  useEffect(() => {
    if (!coordinationRef.current || walletCoords.length === 0) return;

    const svg = d3.select(coordinationRef.current);
    svg.selectAll('*').remove();

    const width = 500;
    const height = 400;

    // Create nodes from wallets
    const wallets = new Set<string>();
    walletCoords.forEach(coord => {
      wallets.add(coord.wallet1);
      wallets.add(coord.wallet2);
    });

    const nodes = Array.from(wallets).map(wallet => ({ id: wallet }));
    const links = walletCoords.map(coord => ({
      source: coord.wallet1,
      target: coord.wallet2,
      value: coord.correlation
    }));

    // Force simulation
    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2));

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g');

    // Links
    const link = g.selectAll('.link')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'link')
      .style('stroke', '#00ff88')
      .style('stroke-opacity', d => d.value)
      .style('stroke-width', d => d.value * 5);

    // Nodes
    const node = g.selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node');

    node.append('circle')
      .attr('r', 20)
      .style('fill', '#ff00ff')
      .style('opacity', 0.8);

    node.append('text')
      .attr('dy', '.35em')
      .style('text-anchor', 'middle')
      .style('fill', 'white')
      .style('font-size', '10px')
      .text(d => d.id.slice(0, 6));

    // Update positions
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

  }, [walletCoords]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-zinc-100">
          Statistical Hypothesis Testing
        </h2>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-zinc-400">
            Confidence Threshold:
            <input
              type="range"
              min="0.5"
              max="1"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              className="w-32"
            />
            <span className="text-zinc-300">{(confidenceThreshold * 100).toFixed(0)}%</span>
          </label>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4">
        <motion.div
          className="glass rounded-xl p-6"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-3xl font-bold text-red-400">
            {animatedStats.anomalies}
          </div>
          <div className="text-sm text-zinc-400 mt-1">Anomalies Detected</div>
          <div className="text-xs text-zinc-500 mt-2">
            Above {(confidenceThreshold * 100).toFixed(0)}% confidence
          </div>
        </motion.div>

        <motion.div
          className="glass rounded-xl p-6"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-3xl font-bold text-purple-400">
            {animatedStats.coordinated}
          </div>
          <div className="text-sm text-zinc-400 mt-1">Coordinated Wallets</div>
          <div className="text-xs text-zinc-500 mt-2">
            Correlation {'>'} 0.6
          </div>
        </motion.div>

        <motion.div
          className="glass rounded-xl p-6"
          whileHover={{ scale: 1.02 }}
        >
          <div className="text-3xl font-bold text-cyan-400">
            {animatedStats.quirks}
          </div>
          <div className="text-sm text-zinc-400 mt-1">Ordering Quirks</div>
          <div className="text-xs text-zinc-500 mt-2">
            Total occurrences
          </div>
        </motion.div>
      </div>

      {/* Hypothesis Tabs */}
      <div className="flex gap-2 mb-4">
        {['landing-rate', 'latency-skew', 'wallet-coord', 'ordering'].map(hyp => (
          <button
            key={hyp}
            onClick={() => setSelectedHypothesis(hyp)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedHypothesis === hyp
                ? 'bg-gradient-to-r from-cyan-600 to-purple-600 text-white'
                : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
            }`}
          >
            {hyp.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
          </button>
        ))}
      </div>

      {/* Landing Rate Anomalies */}
      {selectedHypothesis === 'landing-rate' && (
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">
            Bundle Landing Rate Anomaly Detection
          </h3>
          <svg ref={anomalyChartRef}></svg>
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="bg-zinc-900/50 rounded-lg p-3">
              <div className="text-sm text-zinc-400">Expected Range</div>
              <div className="text-lg font-semibold text-green-400">60-70%</div>
            </div>
            <div className="bg-zinc-900/50 rounded-lg p-3">
              <div className="text-sm text-zinc-400">Anomaly Rate</div>
              <div className="text-lg font-semibold text-red-400">
                {((anomalyData.filter(a => a.isAnomaly).length / anomalyData.length) * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-zinc-900/50 rounded-lg p-3">
              <div className="text-sm text-zinc-400">Max Deviation</div>
              <div className="text-lg font-semibold text-yellow-400">
                {Math.max(...anomalyData.map(a => a.deviation)).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Latency Distribution Skew */}
      {selectedHypothesis === 'latency-skew' && (
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">
            Latency Distribution & Skewness Analysis
          </h3>
          <svg ref={latencyChartRef}></svg>
          <div className="mt-4 grid grid-cols-4 gap-4">
            <div className="bg-zinc-900/50 rounded-lg p-3">
              <div className="text-sm text-zinc-400">P50 Latency</div>
              <div className="text-lg font-semibold text-green-400">20ms</div>
            </div>
            <div className="bg-zinc-900/50 rounded-lg p-3">
              <div className="text-sm text-zinc-400">P95 Latency</div>
              <div className="text-lg font-semibold text-yellow-400">45ms</div>
            </div>
            <div className="bg-zinc-900/50 rounded-lg p-3">
              <div className="text-sm text-zinc-400">P99 Latency</div>
              <div className="text-lg font-semibold text-red-400">78ms</div>
            </div>
            <div className="bg-zinc-900/50 rounded-lg p-3">
              <div className="text-sm text-zinc-400">Outliers</div>
              <div className="text-lg font-semibold text-purple-400">
                {latencyDistribution.filter(d => d.isOutlier).length}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Wallet Fleet Coordination */}
      {selectedHypothesis === 'wallet-coord' && (
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">
            Wallet Fleet Coordination Network
          </h3>
          <div className="flex justify-center">
            <svg ref={coordinationRef}></svg>
          </div>
          <div className="mt-4 space-y-2">
            {walletCoords.slice(0, 5).map((coord, i) => (
              <div key={i} className="flex items-center justify-between bg-zinc-900/50 rounded-lg p-3">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm text-zinc-300">{coord.wallet1}</span>
                  <span className="text-zinc-500">↔</span>
                  <span className="font-mono text-sm text-zinc-300">{coord.wallet2}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-zinc-400">
                    Correlation: <span className="text-cyan-400">{(coord.correlation * 100).toFixed(0)}%</span>
                  </span>
                  <span className="text-sm text-zinc-400">
                    Actions: <span className="text-purple-400">{coord.simultaneousActions}</span>
                  </span>
                  <span className="text-sm text-zinc-400">
                    Δt: <span className="text-yellow-400">{coord.timeDelta.toFixed(0)}ms</span>
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ordering Quirk Detection */}
      {selectedHypothesis === 'ordering' && (
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">
            Transaction Ordering Quirks
          </h3>
          <div className="space-y-4">
            {quirks.map((quirk, i) => (
              <div key={i} className="bg-zinc-900/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-semibold text-zinc-100">{quirk.pattern}</h4>
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-zinc-400">
                      Frequency: <span className="text-cyan-400">{quirk.frequency}</span>
                    </span>
                    <span className="text-sm text-zinc-400">
                      Significance: <span className={quirk.significance > 0.9 ? 'text-red-400' : 'text-yellow-400'}>
                        {(quirk.significance * 100).toFixed(0)}%
                      </span>
                    </span>
                  </div>
                </div>
                <div className="space-y-1">
                  {quirk.examples.slice(0, 2).map((ex, j) => (
                    <div key={j} className="flex items-center gap-3 text-xs">
                      <span className="font-mono text-zinc-500">{ex.tx1}</span>
                      <span className="text-zinc-600">→</span>
                      <span className="font-mono text-zinc-500">{ex.tx2}</span>
                      <span className="text-zinc-400">({ex.timing.toFixed(0)}ms)</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};