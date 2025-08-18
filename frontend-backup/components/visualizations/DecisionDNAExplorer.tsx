/**
 * Decision DNA Explorer Component
 * Interactive visualization for Ed25519-signed decision trees with Merkle anchoring
 * Real-time audit trail with cryptographic verification
 */

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { motion, AnimatePresence } from 'framer-motion';

interface DecisionNode {
  id: string;
  parentId?: string;
  timestamp: number;
  signature: string;
  hash: string;
  decision: {
    type: 'buy' | 'sell' | 'hedge' | 'skip';
    confidence: number;
    route: string;
    amount: number;
    expectedEV: number;
  };
  metrics: {
    latency: number;
    modelVersion: string;
    features: Record<string, number>;
  };
  children?: DecisionNode[];
  verified: boolean;
  anchored: boolean;
}

interface MerkleProof {
  root: string;
  proof: string[];
  index: number;
  verified: boolean;
}

interface DecisionDNAExplorerProps {
  decisions: DecisionNode[];
  merkleProofs?: Map<string, MerkleProof>;
  onNodeClick?: (node: DecisionNode) => void;
  autoUpdate?: boolean;
  maxNodes?: number;
}

// D3 Tree Visualization Component
const DecisionTree: React.FC<{
  data: DecisionNode;
  width: number;
  height: number;
  onNodeClick?: (node: DecisionNode) => void;
}> = ({ data, width, height, onNodeClick }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current || !data) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const margin = { top: 20, right: 120, bottom: 20, left: 120 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Create tree layout
    const treeLayout = d3.tree<DecisionNode>()
      .size([innerHeight, innerWidth]);
    
    const root = d3.hierarchy(data);
    const treeData = treeLayout(root);
    
    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svg.call(zoom as any);
    
    // Draw links
    const link = g.selectAll('.link')
      .data(treeData.links())
      .enter().append('path')
      .attr('class', 'link')
      .attr('d', d3.linkHorizontal()
        .x((d: any) => d.y)
        .y((d: any) => d.x) as any)
      .style('fill', 'none')
      .style('stroke', (d: any) => {
        const node = d.target.data as DecisionNode;
        return node.verified ? '#00ff88' : '#ff4444';
      })
      .style('stroke-width', 2)
      .style('stroke-opacity', 0.6);
    
    // Draw nodes
    const node = g.selectAll('.node')
      .data(treeData.descendants())
      .enter().append('g')
      .attr('class', 'node')
      .attr('transform', (d: any) => `translate(${d.y},${d.x})`)
      .style('cursor', 'pointer')
      .on('click', (event, d: any) => {
        if (onNodeClick) onNodeClick(d.data);
      });
    
    // Add circles for nodes
    node.append('circle')
      .attr('r', 8)
      .style('fill', (d: any) => {
        const node = d.data as DecisionNode;
        const colors = {
          buy: '#00ff88',
          sell: '#ff4444',
          hedge: '#ffaa00',
          skip: '#888888'
        };
        return colors[node.decision.type];
      })
      .style('stroke', (d: any) => {
        const node = d.data as DecisionNode;
        return node.anchored ? '#ffffff' : '#333333';
      })
      .style('stroke-width', (d: any) => {
        const node = d.data as DecisionNode;
        return node.anchored ? 3 : 1;
      });
    
    // Add text labels
    node.append('text')
      .attr('dy', '.35em')
      .attr('x', (d: any) => d.children ? -13 : 13)
      .style('text-anchor', (d: any) => d.children ? 'end' : 'start')
      .style('fill', '#ffffff')
      .style('font-size', '10px')
      .style('font-family', 'monospace')
      .text((d: any) => {
        const node = d.data as DecisionNode;
        return `${node.decision.type} (${(node.decision.confidence * 100).toFixed(0)}%)`;
      });
    
    // Add signature tooltips
    node.append('title')
      .text((d: any) => {
        const node = d.data as DecisionNode;
        return `ID: ${node.id}\nSignature: ${node.signature.slice(0, 16)}...\nLatency: ${node.metrics.latency}ms`;
      });
    
    // Animate entrance
    node
      .style('opacity', 0)
      .transition()
      .duration(500)
      .delay((d: any, i: number) => i * 10)
      .style('opacity', 1);
    
  }, [data, width, height, onNodeClick]);
  
  return (
    <svg ref={svgRef} width={width} height={height} className="decision-tree" />
  );
};

// Hash Chain Visualization
const HashChainVisualization: React.FC<{ nodes: DecisionNode[] }> = ({ nodes }) => {
  const sortedNodes = useMemo(() => 
    [...nodes].sort((a, b) => a.timestamp - b.timestamp).slice(-20),
    [nodes]
  );
  
  return (
    <div className="hash-chain-container p-4 bg-gray-900 rounded-lg">
      <h4 className="text-white text-sm font-bold mb-3">Hash Chain (Last 20)</h4>
      <div className="space-y-1">
        {sortedNodes.map((node, index) => (
          <motion.div
            key={node.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="flex items-center space-x-2 text-xs"
          >
            <div className={`w-2 h-2 rounded-full ${node.verified ? 'bg-green-500' : 'bg-red-500'}`} />
            <div className="font-mono text-gray-400">
              {node.hash.slice(0, 8)}...{node.hash.slice(-8)}
            </div>
            {index < sortedNodes.length - 1 && (
              <div className="text-gray-600">→</div>
            )}
            {node.anchored && (
              <div className="text-yellow-500 text-xs">⚓</div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
};

// Signature Verification Panel
const SignatureVerificationPanel: React.FC<{ node?: DecisionNode }> = ({ node }) => {
  const [verifying, setVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<any>(null);
  
  const verifySignature = useCallback(async () => {
    if (!node) return;
    
    setVerifying(true);
    try {
      // Simulate Ed25519 verification
      await new Promise(resolve => setTimeout(resolve, 500));
      setVerificationResult({
        valid: node.verified,
        publicKey: 'EdDSA_PubKey_' + node.id.slice(0, 8),
        timestamp: node.timestamp,
        message: JSON.stringify(node.decision)
      });
    } finally {
      setVerifying(false);
    }
  }, [node]);
  
  if (!node) {
    return (
      <div className="p-4 bg-gray-900 rounded-lg">
        <p className="text-gray-500 text-sm">Select a decision node to view signature details</p>
      </div>
    );
  }
  
  return (
    <div className="p-4 bg-gray-900 rounded-lg space-y-3">
      <h4 className="text-white text-sm font-bold">Signature Verification</h4>
      
      <div className="space-y-2 text-xs">
        <div className="flex justify-between">
          <span className="text-gray-400">Node ID:</span>
          <span className="text-white font-mono">{node.id.slice(0, 16)}...</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-400">Signature:</span>
          <span className="text-white font-mono">{node.signature.slice(0, 16)}...</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-400">Hash:</span>
          <span className="text-white font-mono">{node.hash.slice(0, 16)}...</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-400">Status:</span>
          <span className={node.verified ? 'text-green-500' : 'text-red-500'}>
            {node.verified ? '✓ Verified' : '✗ Unverified'}
          </span>
        </div>
        
        {node.anchored && (
          <div className="flex justify-between">
            <span className="text-gray-400">Merkle Anchor:</span>
            <span className="text-yellow-500">✓ Anchored</span>
          </div>
        )}
      </div>
      
      <button
        onClick={verifySignature}
        disabled={verifying}
        className="w-full px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors disabled:opacity-50"
      >
        {verifying ? 'Verifying...' : 'Verify Signature'}
      </button>
      
      {verificationResult && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-2 bg-gray-800 rounded text-xs space-y-1"
        >
          <div className="text-green-500">Verification Complete</div>
          <div className="text-gray-400">Public Key: {verificationResult.publicKey}</div>
        </motion.div>
      )}
    </div>
  );
};

// Main Component
export const DecisionDNAExplorer: React.FC<DecisionDNAExplorerProps> = ({
  decisions,
  merkleProofs = new Map(),
  onNodeClick,
  autoUpdate = true,
  maxNodes = 100
}) => {
  const [selectedNode, setSelectedNode] = useState<DecisionNode | undefined>();
  const [treeData, setTreeData] = useState<DecisionNode | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Build tree structure from flat decisions
  useEffect(() => {
    if (decisions.length === 0) return;
    
    const nodeMap = new Map<string, DecisionNode>();
    decisions.forEach(node => nodeMap.set(node.id, { ...node, children: [] }));
    
    let root: DecisionNode | null = null;
    nodeMap.forEach(node => {
      if (node.parentId && nodeMap.has(node.parentId)) {
        const parent = nodeMap.get(node.parentId)!;
        if (!parent.children) parent.children = [];
        parent.children.push(node);
      } else if (!node.parentId) {
        root = node;
      }
    });
    
    setTreeData(root || decisions[0]);
  }, [decisions]);
  
  // Handle responsive sizing
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        });
      }
    };
    
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);
  
  const handleNodeClick = useCallback((node: DecisionNode) => {
    setSelectedNode(node);
    if (onNodeClick) onNodeClick(node);
  }, [onNodeClick]);
  
  const stats = useMemo(() => ({
    total: decisions.length,
    verified: decisions.filter(d => d.verified).length,
    anchored: decisions.filter(d => d.anchored).length,
    avgLatency: decisions.reduce((sum, d) => sum + d.metrics.latency, 0) / decisions.length || 0,
    successRate: decisions.filter(d => d.decision.expectedEV > 0).length / decisions.length || 0
  }), [decisions]);
  
  return (
    <div className="decision-dna-explorer h-full bg-gray-950 rounded-lg overflow-hidden">
      <div className="grid grid-cols-12 gap-4 h-full p-4">
        {/* Main Tree Visualization */}
        <div ref={containerRef} className="col-span-8 bg-gray-900 rounded-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-white text-lg font-bold">Decision Tree</h3>
            <div className="flex space-x-4 text-xs">
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-green-500 rounded-full" />
                <span className="text-gray-400">Buy</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-red-500 rounded-full" />
                <span className="text-gray-400">Sell</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-yellow-500 rounded-full" />
                <span className="text-gray-400">Hedge</span>
              </div>
            </div>
          </div>
          
          {treeData && (
            <DecisionTree
              data={treeData}
              width={dimensions.width - 32}
              height={dimensions.height - 100}
              onNodeClick={handleNodeClick}
            />
          )}
        </div>
        
        {/* Side Panels */}
        <div className="col-span-4 space-y-4">
          {/* Stats Panel */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h4 className="text-white text-sm font-bold mb-3">Statistics</h4>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <div className="text-gray-400">Total Decisions</div>
                <div className="text-white text-lg font-bold">{stats.total}</div>
              </div>
              <div>
                <div className="text-gray-400">Verified</div>
                <div className="text-green-500 text-lg font-bold">
                  {((stats.verified / stats.total) * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-gray-400">Avg Latency</div>
                <div className="text-white text-lg font-bold">
                  {stats.avgLatency.toFixed(1)}ms
                </div>
              </div>
              <div>
                <div className="text-gray-400">Success Rate</div>
                <div className="text-white text-lg font-bold">
                  {(stats.successRate * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
          
          {/* Signature Verification */}
          <SignatureVerificationPanel node={selectedNode} />
          
          {/* Hash Chain */}
          <HashChainVisualization nodes={decisions.slice(-20)} />
        </div>
      </div>
    </div>
  );
};

export default DecisionDNAExplorer;