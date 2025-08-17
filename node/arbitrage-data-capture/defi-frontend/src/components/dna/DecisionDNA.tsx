import { useEffect, useRef, useState, useCallback } from 'react';
import { proxy, useSnapshot } from 'valtio';
import * as d3 from 'd3';

// Decision DNA state
const dnaState = proxy({
  decisions: [] as Array<{
    id: string;
    fingerprint: string;
    timestamp: number;
    modelId: string;
    featureHash: string;
    policy: string;
    route: string;
    tip: number;
    ev: number;
    landed: boolean;
    parentFingerprint: string | null;
    children: string[];
    depth: number;
  }>,
  
  lineageMap: new Map<string, Set<string>>(),
  merkleRoots: [] as Array<{
    date: string;
    root: string;
    count: number;
  }>,
  
  selectedFingerprint: null as string | null,
  searchQuery: '',
  
  stats: {
    totalDecisions: 0,
    uniqueModels: new Set<string>(),
    avgDepth: 0,
    maxDepth: 0,
    landRateByModel: new Map<string, number>(),
    evByPolicy: new Map<string, number>()
  }
});

// Blake3 hash function (simplified for demo - use actual blake3 in production)
async function blake3Hash(data: string): Promise<string> {
  const encoder = new TextEncoder();
  const dataBuffer = encoder.encode(data);
  const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// Generate decision fingerprint
async function generateFingerprint(
  modelId: string,
  features: any,
  policy: string,
  route: string,
  tip: number
): Promise<string> {
  const featureStr = JSON.stringify(features);
  const data = `${modelId}|${featureStr}|${policy}|${route}|${tip}`;
  return await blake3Hash(data);
}

// Build Merkle tree
function buildMerkleTree(hashes: string[]): string {
  if (hashes.length === 0) return '';
  if (hashes.length === 1) return hashes[0];
  
  const pairs = [];
  for (let i = 0; i < hashes.length; i += 2) {
    const left = hashes[i];
    const right = hashes[i + 1] || hashes[i];
    pairs.push(blake3Hash(left + right));
  }
  
  return Promise.all(pairs).then(newHashes => buildMerkleTree(newHashes));
}

export default function DecisionDNA() {
  const state = useSnapshot(dnaState);
  const graphRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [selectedDecision, setSelectedDecision] = useState<any>(null);
  
  // WebSocket for real-time decision events
  useEffect(() => {
    const ws = new WebSocket(process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/dna');
    
    ws.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);
        await addDecision(data);
      } catch (error) {
        console.error('Failed to parse DNA event:', error);
      }
    };
    
    return () => ws.close();
  }, []);
  
  const addDecision = useCallback(async (data: any) => {
    const fingerprint = await generateFingerprint(
      data.modelId,
      data.features,
      data.policy,
      data.route,
      data.tip
    );
    
    const decision = {
      id: `decision_${Date.now()}_${Math.random()}`,
      fingerprint,
      timestamp: Date.now(),
      modelId: data.modelId,
      featureHash: await blake3Hash(JSON.stringify(data.features)),
      policy: data.policy,
      route: data.route,
      tip: data.tip,
      ev: data.ev,
      landed: data.landed || false,
      parentFingerprint: data.parentFingerprint || null,
      children: [],
      depth: 0
    };
    
    // Update parent-child relationships
    if (decision.parentFingerprint) {
      const parent = dnaState.decisions.find(d => d.fingerprint === decision.parentFingerprint);
      if (parent) {
        parent.children.push(fingerprint);
        decision.depth = parent.depth + 1;
      }
    }
    
    // Add to state
    dnaState.decisions.push(decision);
    
    // Update lineage map
    if (!dnaState.lineageMap.has(decision.modelId)) {
      dnaState.lineageMap.set(decision.modelId, new Set());
    }
    dnaState.lineageMap.get(decision.modelId)!.add(fingerprint);
    
    // Update stats
    updateStats();
    
    // Keep only last 1000 decisions for performance
    if (dnaState.decisions.length > 1000) {
      dnaState.decisions.shift();
    }
  }, []);
  
  const updateStats = useCallback(() => {
    const stats = dnaState.stats;
    stats.totalDecisions = dnaState.decisions.length;
    stats.uniqueModels = new Set(dnaState.decisions.map(d => d.modelId));
    
    // Calculate average and max depth
    const depths = dnaState.decisions.map(d => d.depth);
    stats.avgDepth = depths.reduce((a, b) => a + b, 0) / depths.length || 0;
    stats.maxDepth = Math.max(...depths, 0);
    
    // Calculate land rate by model
    stats.landRateByModel.clear();
    const modelGroups = new Map<string, { landed: number; total: number }>();
    
    dnaState.decisions.forEach(d => {
      if (!modelGroups.has(d.modelId)) {
        modelGroups.set(d.modelId, { landed: 0, total: 0 });
      }
      const group = modelGroups.get(d.modelId)!;
      group.total++;
      if (d.landed) group.landed++;
    });
    
    modelGroups.forEach((value, key) => {
      stats.landRateByModel.set(key, value.landed / value.total);
    });
    
    // Calculate EV by policy
    stats.evByPolicy.clear();
    const policyGroups = new Map<string, number[]>();
    
    dnaState.decisions.forEach(d => {
      if (!policyGroups.has(d.policy)) {
        policyGroups.set(d.policy, []);
      }
      policyGroups.get(d.policy)!.push(d.ev);
    });
    
    policyGroups.forEach((values, key) => {
      const avgEv = values.reduce((a, b) => a + b, 0) / values.length;
      stats.evByPolicy.set(key, avgEv);
    });
  }, []);
  
  // D3.js lineage graph visualization
  useEffect(() => {
    if (!graphRef.current || dnaState.decisions.length === 0) return;
    
    const width = 800;
    const height = 600;
    
    // Clear previous graph
    d3.select(graphRef.current).selectAll('*').remove();
    
    const svg = d3.select(graphRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    // Create hierarchical data structure
    const rootDecisions = dnaState.decisions.filter(d => !d.parentFingerprint);
    
    const buildTree = (decision: any): any => ({
      name: decision.fingerprint.slice(0, 8),
      data: decision,
      children: decision.children.map((childFp: string) => {
        const child = dnaState.decisions.find(d => d.fingerprint === childFp);
        return child ? buildTree(child) : null;
      }).filter(Boolean)
    });
    
    const treeData = {
      name: 'Root',
      children: rootDecisions.map(buildTree)
    };
    
    // Create tree layout
    const treeLayout = d3.tree<any>()
      .size([width - 100, height - 100]);
    
    const root = d3.hierarchy(treeData);
    const treeNodes = treeLayout(root);
    
    // Add container with zoom
    const g = svg.append('g')
      .attr('transform', 'translate(50, 50)');
    
    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svg.call(zoom as any);
    
    // Draw links
    g.selectAll('.link')
      .data(treeNodes.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkVertical()
        .x((d: any) => d.x)
        .y((d: any) => d.y) as any)
      .style('fill', 'none')
      .style('stroke', '#00ff00')
      .style('stroke-width', 1)
      .style('opacity', 0.6);
    
    // Draw nodes
    const nodes = g.selectAll('.node')
      .data(treeNodes.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    
    // Node circles
    nodes.append('circle')
      .attr('r', (d: any) => {
        if (!d.data.data) return 3;
        return d.data.data.landed ? 8 : 5;
      })
      .style('fill', (d: any) => {
        if (!d.data.data) return '#666';
        if (d.data.data.landed) return '#00ff00';
        if (d.data.data.route === 'Jito') return '#00aaff';
        return '#ff9900';
      })
      .style('stroke', '#fff')
      .style('stroke-width', 1)
      .style('cursor', 'pointer')
      .on('click', (event, d: any) => {
        if (d.data.data) {
          setSelectedDecision(d.data.data);
          dnaState.selectedFingerprint = d.data.data.fingerprint;
        }
      });
    
    // Node labels
    nodes.append('text')
      .attr('dy', -10)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', '#00ff00')
      .text((d: any) => d.data.name);
    
  }, [dnaState.decisions]);
  
  // Timeline visualization
  useEffect(() => {
    if (!timelineRef.current || dnaState.decisions.length === 0) return;
    
    const width = 800;
    const height = 200;
    const margin = { top: 20, right: 30, bottom: 30, left: 50 };
    
    d3.select(timelineRef.current).selectAll('*').remove();
    
    const svg = d3.select(timelineRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Time scale
    const timeExtent = d3.extent(dnaState.decisions, d => d.timestamp) as [number, number];
    const xScale = d3.scaleTime()
      .domain(timeExtent)
      .range([0, width - margin.left - margin.right]);
    
    // Model scale
    const models = Array.from(new Set(dnaState.decisions.map(d => d.modelId)));
    const yScale = d3.scaleBand()
      .domain(models)
      .range([0, height - margin.top - margin.bottom])
      .padding(0.1);
    
    // Color scale for policies
    const colorScale = d3.scaleOrdinal()
      .domain(['ucb', 'thompson', 'epsilon-greedy', 'contextual'])
      .range(['#00ff00', '#00aaff', '#ff9900', '#ff00ff']);
    
    // Draw decisions as circles on timeline
    g.selectAll('.timeline-decision')
      .data(dnaState.decisions)
      .enter()
      .append('circle')
      .attr('class', 'timeline-decision')
      .attr('cx', d => xScale(d.timestamp))
      .attr('cy', d => (yScale(d.modelId) || 0) + yScale.bandwidth() / 2)
      .attr('r', d => d.landed ? 4 : 2)
      .style('fill', d => colorScale(d.policy) as string)
      .style('opacity', 0.7)
      .on('click', (event, d) => {
        setSelectedDecision(d);
        dnaState.selectedFingerprint = d.fingerprint;
      });
    
    // X axis
    g.append('g')
      .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(10))
      .style('color', '#00ff00');
    
    // Y axis
    g.append('g')
      .call(d3.axisLeft(yScale))
      .style('color', '#00ff00');
    
  }, [dnaState.decisions]);
  
  // Compute daily Merkle root
  const computeDailyMerkleRoot = useCallback(async () => {
    const today = new Date().toISOString().split('T')[0];
    const todayDecisions = dnaState.decisions.filter(d => {
      const decisionDate = new Date(d.timestamp).toISOString().split('T')[0];
      return decisionDate === today;
    });
    
    if (todayDecisions.length === 0) return;
    
    const hashes = todayDecisions.map(d => d.fingerprint);
    const root = await buildMerkleTree(hashes);
    
    dnaState.merkleRoots.push({
      date: today,
      root,
      count: todayDecisions.length
    });
    
    // Keep only last 30 days
    if (dnaState.merkleRoots.length > 30) {
      dnaState.merkleRoots.shift();
    }
  }, []);
  
  // Search functionality
  const searchDecision = useCallback((query: string) => {
    dnaState.searchQuery = query;
    
    if (query.length < 3) {
      dnaState.selectedFingerprint = null;
      return;
    }
    
    const found = dnaState.decisions.find(d => 
      d.fingerprint.includes(query) ||
      d.modelId.includes(query) ||
      d.featureHash.includes(query)
    );
    
    if (found) {
      dnaState.selectedFingerprint = found.fingerprint;
      setSelectedDecision(found);
    }
  }, []);
  
  return (
    <div className="decision-dna" style={{
      padding: '24px',
      backgroundColor: '#0a0a0a',
      color: '#00ff00',
      fontFamily: 'Monaco, monospace',
      minHeight: '100vh'
    }}>
      <h1 style={{ fontSize: '2.5rem', marginBottom: '30px', textShadow: '0 0 15px #00ff00' }}>
        Decision DNA Lineage Explorer
      </h1>
      
      {/* Search Bar */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <input
          type="text"
          placeholder="Search by fingerprint, model ID, or feature hash..."
          value={state.searchQuery}
          onChange={(e) => searchDecision(e.target.value)}
          style={{
            width: '100%',
            padding: '10px',
            background: '#000',
            color: '#00ff00',
            border: '1px solid #00ff00',
            borderRadius: '4px',
            fontSize: '14px'
          }}
        />
      </div>
      
      {/* Statistics */}
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
          <h3>Total Decisions</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
            {state.stats.totalDecisions}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Unique Models</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
            {state.stats.uniqueModels.size}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Avg Depth</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
            {state.stats.avgDepth.toFixed(1)}
          </div>
        </div>
        
        <div style={{
          padding: '15px',
          border: '1px solid #00ff00',
          borderRadius: '8px',
          background: 'rgba(0, 255, 0, 0.05)'
        }}>
          <h3>Max Depth</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
            {state.stats.maxDepth}
          </div>
        </div>
      </div>
      
      {/* Lineage Graph */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h2>Decision Lineage Graph</h2>
        <div ref={graphRef} style={{ width: '100%', height: '600px' }} />
      </div>
      
      {/* Timeline */}
      <div style={{
        marginBottom: '30px',
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h2>Decision Timeline</h2>
        <div ref={timelineRef} style={{ width: '100%', height: '200px' }} />
      </div>
      
      {/* Selected Decision Details */}
      {selectedDecision && (
        <div style={{
          marginBottom: '30px',
          padding: '20px',
          border: '2px solid #ffff00',
          borderRadius: '8px',
          background: 'rgba(255, 255, 0, 0.05)'
        }}>
          <h2>Selected Decision Details</h2>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '10px' }}>
            <div>Fingerprint:</div>
            <div style={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
              {selectedDecision.fingerprint}
            </div>
            
            <div>Model ID:</div>
            <div>{selectedDecision.modelId}</div>
            
            <div>Policy:</div>
            <div>{selectedDecision.policy}</div>
            
            <div>Route:</div>
            <div>{selectedDecision.route}</div>
            
            <div>Tip (SOL):</div>
            <div>{selectedDecision.tip.toFixed(6)}</div>
            
            <div>EV (SOL):</div>
            <div>{selectedDecision.ev.toFixed(6)}</div>
            
            <div>Landed:</div>
            <div style={{ color: selectedDecision.landed ? '#00ff00' : '#ff0000' }}>
              {selectedDecision.landed ? 'Yes' : 'No'}
            </div>
            
            <div>Depth:</div>
            <div>{selectedDecision.depth}</div>
            
            <div>Children:</div>
            <div>{selectedDecision.children.length}</div>
            
            <div>Timestamp:</div>
            <div>{new Date(selectedDecision.timestamp).toLocaleString()}</div>
          </div>
        </div>
      )}
      
      {/* Merkle Roots */}
      <div style={{
        padding: '20px',
        border: '1px solid #00ff00',
        borderRadius: '8px',
        background: 'rgba(0, 255, 0, 0.05)'
      }}>
        <h2>Daily Merkle Roots (Audit Trail)</h2>
        <button
          onClick={computeDailyMerkleRoot}
          style={{
            marginBottom: '20px',
            padding: '10px 20px',
            background: '#00ff00',
            color: '#000',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          Compute Today's Root
        </button>
        
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #00ff00' }}>
              <th style={{ padding: '10px', textAlign: 'left' }}>Date</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Root Hash</th>
              <th style={{ padding: '10px', textAlign: 'left' }}>Decision Count</th>
            </tr>
          </thead>
          <tbody>
            {state.merkleRoots.slice(-10).reverse().map(root => (
              <tr key={root.date} style={{ borderBottom: '1px solid #00ff0044' }}>
                <td style={{ padding: '10px' }}>{root.date}</td>
                <td style={{ padding: '10px', fontFamily: 'monospace', fontSize: '12px' }}>
                  {root.root.slice(0, 32)}...
                </td>
                <td style={{ padding: '10px' }}>{root.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}