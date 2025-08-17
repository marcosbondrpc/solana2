import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { motion, AnimatePresence } from 'framer-motion';

interface DecisionEvent {
  id: string;
  timestamp: Date;
  dnaHash: string;
  signature: string;
  features: {
    model: string;
    confidence: number;
    latency: number;
    landingRate: number;
    profitEstimate: number;
  };
  merkleProof: string[];
  blockHeight: number;
  verified: boolean;
}

interface MerkleNode {
  hash: string;
  left?: MerkleNode;
  right?: MerkleNode;
  level: number;
  x?: number;
  y?: number;
}

interface AuditEntry {
  timestamp: Date;
  action: string;
  actor: string;
  signature: string;
  previousHash: string;
  metadata: Record<string, any>;
}

interface DecisionDNAProps {
  events?: DecisionEvent[];
  auditTrail?: AuditEntry[];
  onEventSelect?: (eventId: string) => void;
}

export const DecisionDNA: React.FC<DecisionDNAProps> = ({
  events = [],
  auditTrail = [],
  onEventSelect
}) => {
  const merkleTreeRef = useRef<SVGSVGElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [selectedEvent, setSelectedEvent] = useState<string | null>(null);
  const [verificationStatus, setVerificationStatus] = useState<Record<string, boolean>>({});
  const [animatedHashes, setAnimatedHashes] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  // Generate mock data if not provided
  const decisionEvents = useMemo(() => {
    if (events.length > 0) return events;
    
    return Array(50).fill(0).map((_, i) => ({
      id: `event_${i}`,
      timestamp: new Date(Date.now() - (50 - i) * 60 * 60 * 1000),
      dnaHash: `0x${Math.random().toString(16).substr(2, 64)}`,
      signature: `sig_${Math.random().toString(36).substr(2, 32)}`,
      features: {
        model: ['GNN', 'Transformer', 'Hybrid'][Math.floor(Math.random() * 3)],
        confidence: 0.7 + Math.random() * 0.3,
        latency: 5 + Math.random() * 15,
        landingRate: 60 + Math.random() * 20,
        profitEstimate: Math.random() * 1000 - 100
      },
      merkleProof: Array(4).fill(0).map(() => 
        `0x${Math.random().toString(16).substr(2, 64)}`
      ),
      blockHeight: 150000000 + i * 1000,
      verified: Math.random() > 0.1
    }));
  }, [events]);

  const auditEntries = useMemo(() => {
    if (auditTrail.length > 0) return auditTrail;
    
    return Array(20).fill(0).map((_, i) => ({
      timestamp: new Date(Date.now() - (20 - i) * 2 * 60 * 60 * 1000),
      action: ['DETECT', 'VERIFY', 'ANCHOR', 'AUDIT'][Math.floor(Math.random() * 4)],
      actor: `system_${Math.floor(Math.random() * 5)}`,
      signature: `audit_sig_${Math.random().toString(36).substr(2, 32)}`,
      previousHash: i > 0 ? `0x${Math.random().toString(16).substr(2, 64)}` : '0x0',
      metadata: {
        events_processed: Math.floor(Math.random() * 1000),
        anomalies_detected: Math.floor(Math.random() * 50),
        confidence_avg: 0.7 + Math.random() * 0.3
      }
    }));
  }, [auditTrail]);

  // Build merkle tree structure
  const merkleTree = useMemo(() => {
    const leaves = decisionEvents.slice(0, 8).map(e => ({
      hash: e.dnaHash.slice(0, 8),
      level: 0
    } as MerkleNode));

    const buildTree = (nodes: MerkleNode[], level: number): MerkleNode => {
      if (nodes.length === 1) return nodes[0];
      
      const pairs: MerkleNode[] = [];
      for (let i = 0; i < nodes.length; i += 2) {
        const left = nodes[i];
        const right = nodes[i + 1] || left;
        pairs.push({
          hash: `${left.hash.slice(0, 4)}${right.hash.slice(0, 4)}`,
          left,
          right,
          level: level + 1
        });
      }
      
      return buildTree(pairs, level + 1);
    };

    return buildTree(leaves, 0);
  }, [decisionEvents]);

  // Animate DNA hashes
  useEffect(() => {
    const interval = setInterval(() => {
      const randomEvent = decisionEvents[Math.floor(Math.random() * decisionEvents.length)];
      if (randomEvent && !animatedHashes.includes(randomEvent.id)) {
        setAnimatedHashes(prev => [...prev.slice(-10), randomEvent.id]);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [decisionEvents, animatedHashes]);

  // Verify signatures simulation
  useEffect(() => {
    const interval = setInterval(() => {
      const unverified = decisionEvents.find(e => !verificationStatus[e.id]);
      if (unverified) {
        setVerificationStatus(prev => ({
          ...prev,
          [unverified.id]: true
        }));
      }
    }, 500);

    return () => clearInterval(interval);
  }, [decisionEvents, verificationStatus]);

  // Draw Merkle tree
  useEffect(() => {
    if (!merkleTreeRef.current || !merkleTree) return;

    const svg = d3.select(merkleTreeRef.current);
    svg.selectAll('*').remove();

    const width = 600;
    const height = 400;
    const nodeRadius = 25;

    svg.attr('width', width).attr('height', height);

    // Calculate positions
    const calculatePositions = (node: MerkleNode, x: number, y: number, spread: number) => {
      node.x = x;
      node.y = y;
      
      if (node.left) {
        calculatePositions(node.left, x - spread, y + 80, spread / 2);
      }
      if (node.right && node.right !== node.left) {
        calculatePositions(node.right, x + spread, y + 80, spread / 2);
      }
    };

    calculatePositions(merkleTree, width / 2, 50, width / 4);

    // Draw edges
    const drawEdges = (node: MerkleNode) => {
      if (node.left && node.x && node.y) {
        svg.append('line')
          .attr('x1', node.x)
          .attr('y1', node.y)
          .attr('x2', node.left.x!)
          .attr('y2', node.left.y!)
          .attr('stroke', '#00ff88')
          .attr('stroke-width', 1)
          .attr('opacity', 0.5);
        drawEdges(node.left);
      }
      if (node.right && node.right !== node.left && node.x && node.y) {
        svg.append('line')
          .attr('x1', node.x)
          .attr('y1', node.y)
          .attr('x2', node.right.x!)
          .attr('y2', node.right.y!)
          .attr('stroke', '#00ff88')
          .attr('stroke-width', 1)
          .attr('opacity', 0.5);
        drawEdges(node.right);
      }
    };

    drawEdges(merkleTree);

    // Draw nodes
    const drawNodes = (node: MerkleNode) => {
      if (node.x && node.y) {
        const g = svg.append('g')
          .attr('transform', `translate(${node.x},${node.y})`);

        g.append('circle')
          .attr('r', nodeRadius)
          .attr('fill', node.level === 0 ? '#ff00ff' : '#00ffff')
          .attr('opacity', 0.8)
          .attr('stroke', '#fff')
          .attr('stroke-width', 1);

        g.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', '.35em')
          .style('fill', 'white')
          .style('font-size', '10px')
          .style('font-family', 'monospace')
          .text(node.hash);

        if (node.left) drawNodes(node.left);
        if (node.right && node.right !== node.left) drawNodes(node.right);
      }
    };

    drawNodes(merkleTree);

  }, [merkleTree]);

  // Filter events based on search
  const filteredEvents = useMemo(() => {
    if (!searchQuery) return decisionEvents;
    return decisionEvents.filter(e => 
      e.dnaHash.includes(searchQuery) || 
      e.signature.includes(searchQuery) ||
      e.id.includes(searchQuery)
    );
  }, [decisionEvents, searchQuery]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-zinc-100">
          Decision DNA & Cryptographic Verification
        </h2>
        <div className="flex items-center gap-4">
          <input
            type="text"
            placeholder="Search by hash or signature..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="px-4 py-2 bg-zinc-800 text-zinc-300 rounded-lg text-sm border border-zinc-700 focus:outline-none focus:border-cyan-500 w-64"
          />
        </div>
      </div>

      {/* Verification Stats */}
      <div className="grid grid-cols-4 gap-4">
        <motion.div className="glass rounded-xl p-6" whileHover={{ scale: 1.02 }}>
          <div className="text-3xl font-bold text-green-400">
            {Object.values(verificationStatus).filter(v => v).length}
          </div>
          <div className="text-sm text-zinc-400 mt-1">Verified Decisions</div>
          <div className="text-xs text-zinc-500 mt-2">Ed25519 signatures</div>
        </motion.div>

        <motion.div className="glass rounded-xl p-6" whileHover={{ scale: 1.02 }}>
          <div className="text-3xl font-bold text-cyan-400">
            {decisionEvents.length}
          </div>
          <div className="text-sm text-zinc-400 mt-1">Total Events</div>
          <div className="text-xs text-zinc-500 mt-2">Last 24 hours</div>
        </motion.div>

        <motion.div className="glass rounded-xl p-6" whileHover={{ scale: 1.02 }}>
          <div className="text-3xl font-bold text-purple-400">
            {auditEntries.length}
          </div>
          <div className="text-sm text-zinc-400 mt-1">Audit Entries</div>
          <div className="text-xs text-zinc-500 mt-2">Immutable chain</div>
        </motion.div>

        <motion.div className="glass rounded-xl p-6" whileHover={{ scale: 1.02 }}>
          <div className="text-3xl font-bold text-yellow-400">
            {Math.max(...decisionEvents.map(e => e.blockHeight))}
          </div>
          <div className="text-sm text-zinc-400 mt-1">Latest Block</div>
          <div className="text-xs text-zinc-500 mt-2">Solana mainnet</div>
        </motion.div>
      </div>

      {/* Merkle Tree Visualization */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          Merkle Tree Anchoring
        </h3>
        <div className="flex justify-center">
          <svg ref={merkleTreeRef}></svg>
        </div>
        <div className="mt-4 text-center text-sm text-zinc-400">
          Daily merkle root anchored to Solana blockchain
        </div>
      </div>

      {/* Event Timeline */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          Decision Event Timeline
        </h3>
        <div className="space-y-2 max-h-96 overflow-y-auto" ref={timelineRef}>
          {filteredEvents.slice(0, 20).map((event, i) => (
            <motion.div
              key={event.id}
              className={`bg-zinc-900/50 rounded-lg p-4 cursor-pointer transition-all ${
                selectedEvent === event.id ? 'ring-2 ring-cyan-500' : ''
              } ${animatedHashes.includes(event.id) ? 'animate-pulse' : ''}`}
              onClick={() => {
                setSelectedEvent(event.id);
                onEventSelect?.(event.id);
              }}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-xs text-zinc-500">
                      {event.timestamp.toLocaleTimeString()}
                    </span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      verificationStatus[event.id]
                        ? 'bg-green-500/20 text-green-400'
                        : 'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {verificationStatus[event.id] ? 'Verified' : 'Pending'}
                    </span>
                    <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded text-xs">
                      {event.features.model}
                    </span>
                  </div>
                  <div className="font-mono text-xs text-zinc-400 mb-1">
                    DNA: {event.dnaHash.slice(0, 20)}...{event.dnaHash.slice(-8)}
                  </div>
                  <div className="flex items-center gap-4 text-xs text-zinc-500">
                    <span>Confidence: {(event.features.confidence * 100).toFixed(0)}%</span>
                    <span>Latency: {event.features.latency.toFixed(1)}ms</span>
                    <span>Landing: {event.features.landingRate.toFixed(0)}%</span>
                    <span className={event.features.profitEstimate > 0 ? 'text-green-400' : 'text-red-400'}>
                      Profit: {event.features.profitEstimate > 0 ? '+' : ''}{event.features.profitEstimate.toFixed(1)} SOL
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-zinc-500">Block</div>
                  <div className="font-mono text-sm text-zinc-400">{event.blockHeight}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Audit Trail */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          Immutable Audit Trail
        </h3>
        <div className="space-y-2">
          {auditEntries.slice(0, 5).map((entry, i) => (
            <div key={i} className="flex items-center justify-between bg-zinc-900/50 rounded-lg p-3">
              <div className="flex items-center gap-4">
                <div className={`w-2 h-2 rounded-full ${
                  entry.action === 'DETECT' ? 'bg-cyan-500' :
                  entry.action === 'VERIFY' ? 'bg-green-500' :
                  entry.action === 'ANCHOR' ? 'bg-purple-500' :
                  'bg-yellow-500'
                }`} />
                <div>
                  <div className="text-sm text-zinc-300">
                    {entry.action} by {entry.actor}
                  </div>
                  <div className="text-xs text-zinc-500">
                    {entry.timestamp.toLocaleString()}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono text-xs text-zinc-500">
                  {entry.signature.slice(0, 12)}...
                </div>
                <div className="text-xs text-zinc-600">
                  Events: {entry.metadata.events_processed}
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center justify-center">
          <button className="px-4 py-2 bg-zinc-800 text-zinc-400 rounded-lg text-sm hover:bg-zinc-700 transition-colors">
            View Full Audit History
          </button>
        </div>
      </div>

      {/* Feature Hash Explorer */}
      <AnimatePresence>
        {selectedEvent && (
          <motion.div
            className="glass rounded-xl p-6"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            {(() => {
              const event = decisionEvents.find(e => e.id === selectedEvent);
              if (!event) return null;

              return (
                <>
                  <h3 className="text-lg font-semibold text-zinc-100 mb-4">
                    Feature Hash Details
                  </h3>
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-sm font-medium text-zinc-400 mb-2">DNA Hash</h4>
                      <div className="font-mono text-xs text-zinc-300 break-all bg-zinc-900/50 rounded p-3">
                        {event.dnaHash}
                      </div>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-zinc-400 mb-2">Signature</h4>
                      <div className="font-mono text-xs text-zinc-300 break-all bg-zinc-900/50 rounded p-3">
                        {event.signature}
                      </div>
                    </div>
                  </div>
                  <div className="mt-4">
                    <h4 className="text-sm font-medium text-zinc-400 mb-2">Merkle Proof</h4>
                    <div className="space-y-1">
                      {event.merkleProof.map((proof, i) => (
                        <div key={i} className="font-mono text-xs text-zinc-500">
                          [{i}] {proof.slice(0, 32)}...{proof.slice(-8)}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="mt-4 grid grid-cols-5 gap-3">
                    <div className="bg-zinc-900/50 rounded-lg p-2 text-center">
                      <div className="text-xs text-zinc-500">Model</div>
                      <div className="text-sm text-zinc-300">{event.features.model}</div>
                    </div>
                    <div className="bg-zinc-900/50 rounded-lg p-2 text-center">
                      <div className="text-xs text-zinc-500">Confidence</div>
                      <div className="text-sm text-zinc-300">{(event.features.confidence * 100).toFixed(0)}%</div>
                    </div>
                    <div className="bg-zinc-900/50 rounded-lg p-2 text-center">
                      <div className="text-xs text-zinc-500">Latency</div>
                      <div className="text-sm text-zinc-300">{event.features.latency.toFixed(1)}ms</div>
                    </div>
                    <div className="bg-zinc-900/50 rounded-lg p-2 text-center">
                      <div className="text-xs text-zinc-500">Landing</div>
                      <div className="text-sm text-zinc-300">{event.features.landingRate.toFixed(0)}%</div>
                    </div>
                    <div className="bg-zinc-900/50 rounded-lg p-2 text-center">
                      <div className="text-xs text-zinc-500">Profit</div>
                      <div className={`text-sm ${event.features.profitEstimate > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {event.features.profitEstimate.toFixed(1)} SOL
                      </div>
                    </div>
                  </div>
                </>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};