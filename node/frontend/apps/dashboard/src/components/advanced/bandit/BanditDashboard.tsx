import { useEffect, useRef, useState, useCallback } from 'react';
import { proxy, useSnapshot, subscribe } from 'valtio';
import * as d3 from 'd3';

// Bandit state with lock-free updates via Valtio
const banditState = proxy({
  arms: new Map<string, {
    id: string;
    pulls: number;
    rewards: number;
    avgReward: number;
    ucb: number;
    thompsonSample: number;
    lastUpdate: number;
    route: 'Direct' | 'Jito';
    tipLadder: number;
    landRate: number;
    ev: number;
  }>(),
  
  totalPulls: 0,
  totalRewards: 0,
  currentPolicy: 'ucb' as 'ucb' | 'thompson' | 'epsilon-greedy' | 'contextual',
  explorationRate: 0.1,
  
  realtimeMetrics: {
    landsPerMinute: 0,
    evPerMinute: 0,
    tipEfficiency: 0,
    routeBalance: 0.5
  },
  
  leaderHeatmap: new Map<string, Map<string, number>>(),
  tipLadderDistribution: [] as number[],
  
  historicalData: [] as Array<{
    timestamp: number;
    armId: string;
    reward: number;
    landed: boolean;
    policy: string;
  }>
});

// High-performance circular buffer for time series
class TimeSeriesBuffer {
  private buffer: Float32Array;
  private timestamps: Float32Array;
  private writePos = 0;
  private capacity: number;
  
  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Float32Array(capacity);
    this.timestamps = new Float32Array(capacity);
  }
  
  push(timestamp: number, value: number): void {
    const idx = this.writePos % this.capacity;
    this.timestamps[idx] = timestamp;
    this.buffer[idx] = value;
    this.writePos++;
  }
  
  getRecent(count: number): Array<[number, number]> {
    const result: Array<[number, number]> = [];
    const start = Math.max(0, this.writePos - count);
    for (let i = start; i < this.writePos; i++) {
      const idx = i % this.capacity;
      result.push([this.timestamps[idx], this.buffer[idx]]);
    }
    return result;
  }
}

const evTimeSeries = new TimeSeriesBuffer(10000);
const landRateTimeSeries = new TimeSeriesBuffer(10000);

// UCB1 calculation
function calculateUCB(avgReward: number, totalPulls: number, armPulls: number): number {
  if (armPulls === 0) return Infinity;
  const explorationTerm = Math.sqrt(2 * Math.log(totalPulls) / armPulls);
  return avgReward + explorationTerm;
}

// Thompson Sampling Beta distribution
function thompsonSample(successes: number, failures: number): number {
  // Simple beta distribution sampling
  const alpha = successes + 1;
  const beta = failures + 1;
  
  // Using gamma distribution to sample from beta
  const gammaAlpha = sampleGamma(alpha);
  const gammaBeta = sampleGamma(beta);
  
  return gammaAlpha / (gammaAlpha + gammaBeta);
}

function sampleGamma(shape: number): number {
  // Marsaglia and Tsang method
  if (shape < 1) {
    return sampleGamma(shape + 1) * Math.pow(Math.random(), 1 / shape);
  }
  
  const d = shape - 1/3;
  const c = 1 / Math.sqrt(9 * d);
  
  while (true) {
    const x = normalRandom();
    const v = Math.pow(1 + c * x, 3);
    
    if (v > 0 && Math.log(Math.random()) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
      return d * v;
    }
  }
}

function normalRandom(): number {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export default function BanditDashboard() {
  const state = useSnapshot(banditState);
  const d3ContainerRef = useRef<HTMLDivElement>(null);
  const heatmapRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);
  
  // WebSocket connection for real-time bandit events
  useEffect(() => {
    const ws = new WebSocket(process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/bandit');
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateBanditState(data);
      } catch (error) {
        console.error('Failed to parse bandit event:', error);
      }
    };
    
    return () => ws.close();
  }, []);
  
  const updateBanditState = useCallback((event: any) => {
    const armId = `${event.route}_${event.arm}`;
    
    if (!banditState.arms.has(armId)) {
      banditState.arms.set(armId, {
        id: armId,
        pulls: 0,
        rewards: 0,
        avgReward: 0,
        ucb: 0,
        thompsonSample: 0,
        lastUpdate: Date.now(),
        route: event.route,
        tipLadder: event.arm,
        landRate: 0,
        ev: 0
      });
    }
    
    const arm = banditState.arms.get(armId)!;
    arm.pulls++;
    arm.rewards += event.payoff;
    arm.avgReward = arm.rewards / arm.pulls;
    arm.landRate = event.landed ? (arm.landRate * (arm.pulls - 1) + 1) / arm.pulls : 
                                   (arm.landRate * (arm.pulls - 1)) / arm.pulls;
    arm.ev = event.ev_sol;
    arm.lastUpdate = Date.now();
    
    // Update UCB and Thompson sampling values
    banditState.totalPulls++;
    banditState.totalRewards += event.payoff;
    
    arm.ucb = calculateUCB(arm.avgReward, banditState.totalPulls, arm.pulls);
    
    const successes = Math.round(arm.landRate * arm.pulls);
    const failures = arm.pulls - successes;
    arm.thompsonSample = thompsonSample(successes, failures);
    
    // Update time series
    evTimeSeries.push(Date.now(), event.ev_sol);
    landRateTimeSeries.push(Date.now(), event.landed ? 1 : 0);
    
    // Update leader heatmap
    if (!banditState.leaderHeatmap.has(event.leader)) {
      banditState.leaderHeatmap.set(event.leader, new Map());
    }
    const leaderMap = banditState.leaderHeatmap.get(event.leader)!;
    leaderMap.set(event.route, (leaderMap.get(event.route) || 0) + (event.landed ? 1 : 0));
    
    // Update historical data (keep last 1000)
    banditState.historicalData.push({
      timestamp: Date.now(),
      armId,
      reward: event.payoff,
      landed: event.landed,
      policy: event.policy
    });
    
    if (banditState.historicalData.length > 1000) {
      banditState.historicalData.shift();
    }
    
    // Update real-time metrics
    updateRealtimeMetrics();
  }, []);
  
  const updateRealtimeMetrics = useCallback(() => {
    const recentData = banditState.historicalData.slice(-100);
    const timeWindow = 60000; // 1 minute
    const now = Date.now();
    
    const recentMinuteData = recentData.filter(d => now - d.timestamp < timeWindow);
    
    banditState.realtimeMetrics.landsPerMinute = recentMinuteData.filter(d => d.landed).length;
    banditState.realtimeMetrics.evPerMinute = recentMinuteData.reduce((sum, d) => sum + d.reward, 0);
    
    // Calculate tip efficiency
    const avgTip = Array.from(banditState.arms.values())
      .reduce((sum, arm) => sum + arm.tipLadder * arm.pulls, 0) / banditState.totalPulls;
    const avgReward = banditState.totalRewards / banditState.totalPulls;
    banditState.realtimeMetrics.tipEfficiency = avgTip > 0 ? avgReward / avgTip : 0;
    
    // Calculate route balance
    const directPulls = Array.from(banditState.arms.values())
      .filter(arm => arm.route === 'Direct')
      .reduce((sum, arm) => sum + arm.pulls, 0);
    banditState.realtimeMetrics.routeBalance = directPulls / banditState.totalPulls;
  }, []);
  
  // D3.js visualization for arm quality
  useEffect(() => {
    if (!d3ContainerRef.current) return;
    
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;
    
    // Clear previous chart
    d3.select(d3ContainerRef.current).selectAll('*').remove();
    
    const svg = d3.select(d3ContainerRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, Math.max(...Array.from(state.arms.values()).map(a => a.pulls))])
      .range([0, width]);
    
    const yScale = d3.scaleLinear()
      .domain([0, Math.max(...Array.from(state.arms.values()).map(a => a.avgReward))])
      .range([height, 0]);
    
    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).ticks(10))
      .append('text')
      .attr('x', width / 2)
      .attr('y', 35)
      .style('fill', '#00ff00')
      .text('Pulls');
    
    g.append('g')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -35)
      .attr('x', -height / 2)
      .style('fill', '#00ff00')
      .text('Average Reward');
    
    // Plot arms
    const colorScale = d3.scaleOrdinal<string, string>()
      .domain(['Direct', 'Jito'])
      .range(['#00ff00', '#00aaff']);
    
    g.selectAll('.arm-circle')
      .data(Array.from(state.arms.values()))
      .enter()
      .append('circle')
      .attr('class', 'arm-circle')
      .attr('cx', d => xScale(d.pulls))
      .attr('cy', d => yScale(d.avgReward))
      .attr('r', d => Math.sqrt(d.ucb) * 3)
      .style('fill', d => colorScale(d.route))
      .style('opacity', 0.7)
      .style('stroke', '#fff')
      .style('stroke-width', 1);
    
    // Add UCB confidence bounds
    g.selectAll('.ucb-line')
      .data(Array.from(state.arms.values()))
      .enter()
      .append('line')
      .attr('class', 'ucb-line')
      .attr('x1', d => xScale(d.pulls))
      .attr('y1', d => yScale(d.avgReward))
      .attr('x2', d => xScale(d.pulls))
      .attr('y2', d => yScale(d.ucb))
      .style('stroke', '#ffff00')
      .style('stroke-width', 1)
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.5);
    
  }, [state.arms]);
  
  // Heatmap for leader performance
  useEffect(() => {
    if (!heatmapRef.current) return;
    
    const width = 400;
    const height = 300;
    const margin = { top: 30, right: 30, bottom: 30, left: 80 };
    
    d3.select(heatmapRef.current).selectAll('*').remove();
    
    const svg = d3.select(heatmapRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    const leaders = Array.from(state.leaderHeatmap.keys()).slice(0, 10);
    const routes = ['Direct', 'Jito'];
    
    const xScale = d3.scaleBand()
      .domain(routes)
      .range([margin.left, width - margin.right])
      .padding(0.1);
    
    const yScale = d3.scaleBand()
      .domain(leaders)
      .range([margin.top, height - margin.bottom])
      .padding(0.1);
    
    const colorScale = d3.scaleSequential(d3.interpolateYlOrRd)
      .domain([0, 1]);
    
    // Create heatmap cells
    leaders.forEach(leader => {
      const leaderData = state.leaderHeatmap.get(leader);
      if (!leaderData) return;
      
      routes.forEach(route => {
        const value = leaderData.get(route) || 0;
        
        svg.append('rect')
          .attr('x', xScale(route)!)
          .attr('y', yScale(leader)!)
          .attr('width', xScale.bandwidth())
          .attr('height', yScale.bandwidth())
          .style('fill', colorScale(value))
          .style('stroke', '#000')
          .style('stroke-width', 1);
      });
    });
    
    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${margin.top})`)
      .call(d3.axisTop(xScale))
      .style('color', '#00ff00');
    
    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale))
      .style('color', '#00ff00')
      .selectAll('text')
      .style('font-size', '10px');
    
  }, [state.leaderHeatmap]);
  
  const selectArm = useCallback(() => {
    const arms = Array.from(banditState.arms.values());
    if (arms.length === 0) return null;
    
    switch (banditState.currentPolicy) {
      case 'ucb':
        return arms.reduce((best, arm) => arm.ucb > best.ucb ? arm : best);
      
      case 'thompson':
        return arms.reduce((best, arm) => arm.thompsonSample > best.thompsonSample ? arm : best);
      
      case 'epsilon-greedy':
        if (Math.random() < banditState.explorationRate) {
          return arms[Math.floor(Math.random() * arms.length)];
        }
        return arms.reduce((best, arm) => arm.avgReward > best.avgReward ? arm : best);
      
      case 'contextual':
        // Contextual bandit with features
        // This would use additional context like slot, leader, network conditions
        return arms.reduce((best, arm) => {
          const contextScore = arm.avgReward * (1 + arm.landRate) * (1 / (1 + arm.tipLadder));
          const bestScore = best.avgReward * (1 + best.landRate) * (1 / (1 + best.tipLadder));
          return contextScore > bestScore ? arm : best;
        });
      
      default:
        return arms[0];
    }
  }, []);
  
  return (
    <div className="bandit-dashboard" style={{
      padding: '24px',
      backgroundColor: '#0a0a0a',
      color: '#00ff00',
      fontFamily: 'Monaco, monospace',
      minHeight: '100vh'
    }}>
      <h1 style={{ fontSize: '2.5rem', marginBottom: '30px', textShadow: '0 0 15px #00ff00' }}>
        Bandit Optimizer Dashboard
      </h1>
      
      {/* Control Panel */}
      <div style={{
        display: 'flex',
        gap: '20px',
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <div>
          <label>Policy: </label>
          <select
            value={banditState.currentPolicy}
            onChange={(e) => banditState.currentPolicy = e.target.value as any}
            style={{
              background: '#000',
              color: '#00ff00',
              border: '1px solid #00ff00',
              padding: '5px',
              borderRadius: '4px'
            }}
          >
            <option value="ucb">UCB1</option>
            <option value="thompson">Thompson Sampling</option>
            <option value="epsilon-greedy">Epsilon-Greedy</option>
            <option value="contextual">Contextual Bandit</option>
          </select>
        </div>
        
        <div>
          <label>Exploration Rate: </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={banditState.explorationRate}
            onChange={(e) => banditState.explorationRate = parseFloat(e.target.value)}
            style={{ verticalAlign: 'middle' }}
          />
          <span style={{ marginLeft: '10px' }}>{(banditState.explorationRate * 100).toFixed(0)}%</span>
        </div>
      </div>
      
      {/* Real-time Metrics */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '20px',
        marginBottom: '30px'
      }}>
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Lands/Min</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
            {state.realtimeMetrics.landsPerMinute}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>EV/Min (SOL)</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
            {state.realtimeMetrics.evPerMinute.toFixed(4)}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Tip Efficiency</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
            {(state.realtimeMetrics.tipEfficiency * 100).toFixed(1)}%
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Route Balance</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
            {(state.realtimeMetrics.routeBalance * 100).toFixed(0)}% Direct
          </div>
        </div>
      </div>
      
      {/* Arm Performance Chart */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h2>Arm Performance & UCB Bounds</h2>
        <div ref={d3ContainerRef} />
      </div>
      
      {/* Leader Heatmap */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h2>Leader × Route Performance Heatmap</h2>
        <div ref={heatmapRef} />
      </div>
      
      {/* Arms Table */}
      <div style={{
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h2>Arm Statistics</h2>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #00ff00' }}>
              <th style={{ padding: '10px', textAlign: 'left' }}>Arm ID</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Route</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Tip Ladder</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Pulls</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Avg Reward</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Land Rate</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>UCB Score</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Thompson Score</th>
            </tr>
          </thead>
          <tbody>
            {Array.from(state.arms.values())
              .sort((a, b) => b.ucb - a.ucb)
              .slice(0, 10)
              .map(arm => (
                <tr key={arm.id} style={{ borderBottom: '1px solid #00ff0044' }}>
                  <td style={{ padding: '10px' }}>{arm.id}</td>
                  <td style={{ padding: '10px' }}>{arm.route}</td>
                  <td style={{ padding: '10px' }}>{arm.tipLadder.toFixed(2)}</td>
                  <td style={{ padding: '10px' }}>{arm.pulls}</td>
                  <td style={{ padding: '10px' }}>{arm.avgReward.toFixed(6)}</td>
                  <td style={{ padding: '10px' }}>{(arm.landRate * 100).toFixed(1)}%</td>
                  <td style={{ padding: '10px' }}>{arm.ucb.toFixed(6)}</td>
                  <td style={{ padding: '10px' }}>{arm.thompsonSample.toFixed(6)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}