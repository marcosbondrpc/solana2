import { useEffect, useRef, useState, useCallback } from 'react';
import { proxy, useSnapshot } from 'valtio';
import * as d3 from 'd3';

const heatmapState = proxy({
  slots: [] as Array<{
    slot: number;
    leader: string;
    submissions: Array<{
      timestamp: number;
      phase: number; // 0-400ms within slot
      landed: boolean;
      route: 'Direct' | 'Jito';
      tip: number;
      ev: number;
      dscp: number; // DSCP marking value
    }>;
  }>,
  
  leaderStats: new Map<string, {
    totalSlots: number;
    landRate: number;
    avgEv: number;
    optimalPhase: number;
    dscpSuccess: number;
  }>(),
  
  phaseStats: Array.from({ length: 8 }, () => ({
    submissions: 0,
    lands: 0,
    landRate: 0,
    avgEv: 0
  })),
  
  currentSlot: 0,
  currentLeader: '',
  currentPhase: 0,
  
  config: {
    slotDuration: 400, // ms
    phaseCount: 8, // divide slot into 8 phases of 50ms each
    targetPhase: 2, // optimal submission phase (100-150ms)
    dscpEnabled: true
  }
});

// Calculate phase from timestamp within slot
function getPhase(timestamp: number, slotStart: number, slotDuration: number, phaseCount: number): number {
  const offset = timestamp - slotStart;
  return Math.floor((offset / slotDuration) * phaseCount);
}

export default function LeaderPhaseHeatmap() {
  const state = useSnapshot(heatmapState);
  const heatmapRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [selectedLeader, setSelectedLeader] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  // Fetch slot data
  useEffect(() => {
    const fetchSlotData = async () => {
      try {
        const response = await fetch('/api/slots/recent?limit=100');
        const data = await response.json();
        
        // Process slot data
        heatmapState.slots = data.slots;
        heatmapState.currentSlot = data.currentSlot;
        heatmapState.currentLeader = data.currentLeader;
        
        // Calculate leader stats
        updateLeaderStats();
        updatePhaseStats();
        
      } catch (error) {
        console.error('Failed to fetch slot data:', error);
      }
    };
    
    if (autoRefresh) {
      fetchSlotData();
      const interval = setInterval(fetchSlotData, 2000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);
  
  // WebSocket for real-time submissions
  useEffect(() => {
    const ws = new WebSocket(process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/submissions');
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Add submission to current slot
        const currentSlot = heatmapState.slots.find(s => s.slot === data.slot);
        if (currentSlot) {
          currentSlot.submissions.push({
            timestamp: data.timestamp,
            phase: getPhase(data.timestamp, data.slotStart, heatmapState.config.slotDuration, heatmapState.config.phaseCount),
            landed: data.landed,
            route: data.route,
            tip: data.tip,
            ev: data.ev,
            dscp: data.dscp || 0
          });
        }
        
        // Update current phase
        const now = Date.now();
        const slotStart = Math.floor(now / heatmapState.config.slotDuration) * heatmapState.config.slotDuration;
        heatmapState.currentPhase = getPhase(now, slotStart, heatmapState.config.slotDuration, heatmapState.config.phaseCount);
        
      } catch (error) {
        console.error('Failed to parse submission:', error);
      }
    };
    
    return () => ws.close();
  }, []);
  
  const updateLeaderStats = useCallback(() => {
    const stats = new Map<string, any>();
    
    heatmapState.slots.forEach(slot => {
      if (!stats.has(slot.leader)) {
        stats.set(slot.leader, {
          totalSlots: 0,
          lands: 0,
          totalEv: 0,
          phaseDistribution: new Array(heatmapState.config.phaseCount).fill(0),
          dscpLands: 0,
          dscpTotal: 0
        });
      }
      
      const leaderStat = stats.get(slot.leader);
      leaderStat.totalSlots++;
      
      slot.submissions.forEach(sub => {
        if (sub.landed) leaderStat.lands++;
        leaderStat.totalEv += sub.ev;
        leaderStat.phaseDistribution[sub.phase]++;
        
        if (sub.dscp > 0) {
          leaderStat.dscpTotal++;
          if (sub.landed) leaderStat.dscpLands++;
        }
      });
    });
    
    // Calculate final stats
    stats.forEach((value, key) => {
      const landRate = value.lands / Math.max(value.totalSlots, 1);
      const avgEv = value.totalEv / Math.max(value.totalSlots, 1);
      const optimalPhase = value.phaseDistribution.indexOf(Math.max(...value.phaseDistribution));
      const dscpSuccess = value.dscpTotal > 0 ? value.dscpLands / value.dscpTotal : 0;
      
      heatmapState.leaderStats.set(key, {
        totalSlots: value.totalSlots,
        landRate,
        avgEv,
        optimalPhase,
        dscpSuccess
      });
    });
  }, []);
  
  const updatePhaseStats = useCallback(() => {
    const phases = Array.from({ length: heatmapState.config.phaseCount }, () => ({
      submissions: 0,
      lands: 0,
      totalEv: 0
    }));
    
    heatmapState.slots.forEach(slot => {
      slot.submissions.forEach(sub => {
        phases[sub.phase].submissions++;
        if (sub.landed) phases[sub.phase].lands++;
        phases[sub.phase].totalEv += sub.ev;
      });
    });
    
    heatmapState.phaseStats = phases.map(phase => ({
      submissions: phase.submissions,
      lands: phase.lands,
      landRate: phase.submissions > 0 ? phase.lands / phase.submissions : 0,
      avgEv: phase.submissions > 0 ? phase.totalEv / phase.submissions : 0
    }));
  }, []);
  
  // D3.js heatmap visualization
  useEffect(() => {
    if (!heatmapRef.current || heatmapState.slots.length === 0) return;
    
    const margin = { top: 50, right: 50, bottom: 50, left: 100 };
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;
    
    // Clear previous chart
    d3.select(heatmapRef.current).selectAll('*').remove();
    
    const svg = d3.select(heatmapRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Get unique leaders (top 20)
    const leaders = Array.from(new Set(heatmapState.slots.map(s => s.leader))).slice(0, 20);
    const phases = Array.from({ length: heatmapState.config.phaseCount }, (_, i) => i);
    
    // Scales
    const xScale = d3.scaleBand()
      .domain(phases.map(String))
      .range([0, width])
      .padding(0.05);
    
    const yScale = d3.scaleBand()
      .domain(leaders)
      .range([0, height])
      .padding(0.05);
    
    // Color scale
    const colorScale = d3.scaleSequential()
      .interpolator(d3.interpolateViridis)
      .domain([0, 1]);
    
    // Create heatmap data
    const heatmapData: any[] = [];
    leaders.forEach(leader => {
      phases.forEach(phase => {
        const slots = heatmapState.slots.filter(s => s.leader === leader);
        let lands = 0;
        let total = 0;
        
        slots.forEach(slot => {
          const phaseSubmissions = slot.submissions.filter(s => s.phase === phase);
          total += phaseSubmissions.length;
          lands += phaseSubmissions.filter(s => s.landed).length;
        });
        
        heatmapData.push({
          leader,
          phase,
          landRate: total > 0 ? lands / total : 0,
          total
        });
      });
    });
    
    // Draw cells
    g.selectAll('.cell')
      .data(heatmapData)
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => xScale(String(d.phase))!)
      .attr('y', d => yScale(d.leader)!)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .style('fill', d => colorScale(d.landRate))
      .style('stroke', d => d.phase === heatmapState.config.targetPhase ? '#ffff00' : '#000')
      .style('stroke-width', d => d.phase === heatmapState.config.targetPhase ? 2 : 1)
      .on('mouseover', function(event, d) {
        // Tooltip
        const tooltip = d3.select('body').append('div')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.9)')
          .style('color', '#00ff00')
          .style('padding', '10px')
          .style('border', '1px solid #00ff00')
          .style('border-radius', '4px')
          .style('pointer-events', 'none')
          .style('font-size', '12px');
        
        tooltip.html(`
          Leader: ${d.leader.slice(0, 8)}...<br/>
          Phase: ${d.phase * 50}-${(d.phase + 1) * 50}ms<br/>
          Land Rate: ${(d.landRate * 100).toFixed(1)}%<br/>
          Submissions: ${d.total}
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', function() {
        d3.selectAll('div').remove();
      })
      .on('click', function(event, d) {
        setSelectedLeader(d.leader);
      });
    
    // X axis (phases)
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `${parseInt(d) * 50}ms`))
      .style('color', '#00ff00');
    
    // Y axis (leaders)
    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => d.slice(0, 8) + '...'))
      .style('color', '#00ff00');
    
    // Title
    g.append('text')
      .attr('x', width / 2)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('fill', '#00ff00')
      .text('Leader × Phase Land Rate Heatmap');
    
    // Legend
    const legendWidth = 200;
    const legendHeight = 20;
    
    const legendScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, legendWidth]);
    
    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d => `${(d.valueOf() * 100).toFixed(0)}%`);
    
    const legend = svg.append('g')
      .attr('transform', `translate(${margin.left + width - legendWidth}, ${margin.top - 40})`);
    
    // Legend gradient
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'legend-gradient');
    
    gradient.selectAll('stop')
      .data(d3.range(0, 1.1, 0.1))
      .enter()
      .append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => colorScale(d));
    
    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#legend-gradient)');
    
    legend.append('g')
      .attr('transform', `translate(0,${legendHeight})`)
      .call(legendAxis)
      .style('color', '#00ff00');
    
  }, [heatmapState.slots, heatmapState.config]);
  
  // Phase timeline visualization
  useEffect(() => {
    if (!timelineRef.current) return;
    
    const width = 800;
    const height = 150;
    const margin = { top: 20, right: 50, bottom: 30, left: 50 };
    
    d3.select(timelineRef.current).selectAll('*').remove();
    
    const svg = d3.select(timelineRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const xScale = d3.scaleBand()
      .domain(Array.from({ length: heatmapState.config.phaseCount }, (_, i) => String(i)))
      .range([0, width - margin.left - margin.right])
      .padding(0.1);
    
    const yScale = d3.scaleLinear()
      .domain([0, Math.max(...heatmapState.phaseStats.map(p => p.landRate))])
      .range([height - margin.top - margin.bottom, 0]);
    
    // Bars
    g.selectAll('.bar')
      .data(heatmapState.phaseStats)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', (d, i) => xScale(String(i))!)
      .attr('y', d => yScale(d.landRate))
      .attr('width', xScale.bandwidth())
      .attr('height', d => height - margin.top - margin.bottom - yScale(d.landRate))
      .style('fill', (d, i) => i === heatmapState.config.targetPhase ? '#ffff00' : 
                               i === heatmapState.currentPhase ? '#ff9900' : '#00ff00')
      .style('opacity', 0.7);
    
    // Current phase indicator
    g.append('line')
      .attr('x1', xScale(String(heatmapState.currentPhase))! + xScale.bandwidth() / 2)
      .attr('y1', 0)
      .attr('x2', xScale(String(heatmapState.currentPhase))! + xScale.bandwidth() / 2)
      .attr('y2', height - margin.top - margin.bottom)
      .style('stroke', '#ff0000')
      .style('stroke-width', 2)
      .style('stroke-dasharray', '5,5');
    
    // X axis
    g.append('g')
      .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `${parseInt(d) * 50}-${(parseInt(d) + 1) * 50}ms`))
      .style('color', '#00ff00')
      .selectAll('text')
      .style('font-size', '10px')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');
    
    // Y axis
    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => `${(d.valueOf() * 100).toFixed(0)}%`))
      .style('color', '#00ff00');
    
  }, [heatmapState.phaseStats, heatmapState.currentPhase]);
  
  return (
    <div style={{
      padding: '24px',
      backgroundColor: '#0a0a0a',
      color: '#00ff00',
      fontFamily: 'Monaco, monospace',
      minHeight: '100vh'
    }}>
      <h1 style={{ fontSize: '2.5rem', marginBottom: '30px', textShadow: '0 0 15px #00ff00' }}>
        Leader Phase Heatmap & DSCP Analysis
      </h1>
      
      {/* Current Status */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '15px',
        marginBottom: '30px'
      }}>
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4>Current Slot</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {state.currentSlot}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4>Current Leader</h4>
          <div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>
            {state.currentLeader.slice(0, 8)}...
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4>Current Phase</h4>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
            {state.currentPhase} / {state.config.phaseCount - 1}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h4>DSCP Mode</h4>
          <div style={{
            fontSize: '1.5rem',
            fontWeight: 'bold',
            color: state.config.dscpEnabled ? '#00ff00' : '#ff0000'
          }}>
            {state.config.dscpEnabled ? 'ENABLED' : 'DISABLED'}
          </div>
        </div>
      </div>
      
      {/* Controls */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)',
        display: 'flex',
        gap: '20px',
        alignItems: 'center'
      }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
          />
          Auto Refresh
        </label>
        
        <label style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <input
            type="checkbox"
            checked={heatmapState.config.dscpEnabled}
            onChange={(e) => heatmapState.config.dscpEnabled = e.target.checked}
          />
          DSCP Marking
        </label>
        
        <label>
          Target Phase:
          <select
            value={heatmapState.config.targetPhase}
            onChange={(e) => heatmapState.config.targetPhase = parseInt(e.target.value)}
            style={{
              marginLeft: '10px',
              padding: '5px',
              background: '#000',
              color: '#00ff00',
              border: '1px solid #00ff00',
              borderRadius: '4px'
            }}
          >
            {Array.from({ length: heatmapState.config.phaseCount }, (_, i) => (
              <option key={i} value={i}>
                Phase {i} ({i * 50}-{(i + 1) * 50}ms)
              </option>
            ))}
          </select>
        </label>
      </div>
      
      {/* Heatmap */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <div ref={heatmapRef} />
      </div>
      
      {/* Phase Timeline */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h3>Phase Land Rate Distribution</h3>
        <div ref={timelineRef} />
      </div>
      
      {/* Leader Statistics */}
      {selectedLeader && (
        <div style={{
          marginBottom: '30px',
          padding: '20px',
          border: '2px solid #ffff00',
          borderRadius: '8px',
          background: 'rgba(255, 255, 0, 0.05)'
        }}>
          <h3>Leader Details: {selectedLeader}</h3>
          {(() => {
            const stats = heatmapState.leaderStats.get(selectedLeader);
            if (!stats) return null;
            
            return (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginTop: '15px' }}>
                <div>
                  <h4>Total Slots</h4>
                  <div style={{ fontSize: '1.5rem' }}>{stats.totalSlots}</div>
                </div>
                <div>
                  <h4>Land Rate</h4>
                  <div style={{ fontSize: '1.5rem' }}>{(stats.landRate * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <h4>Avg EV (SOL)</h4>
                  <div style={{ fontSize: '1.5rem' }}>{stats.avgEv.toFixed(4)}</div>
                </div>
                <div>
                  <h4>Optimal Phase</h4>
                  <div style={{ fontSize: '1.5rem' }}>
                    {stats.optimalPhase} ({stats.optimalPhase * 50}-{(stats.optimalPhase + 1) * 50}ms)
                  </div>
                </div>
                <div>
                  <h4>DSCP Success Rate</h4>
                  <div style={{ fontSize: '1.5rem' }}>{(stats.dscpSuccess * 100).toFixed(1)}%</div>
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}