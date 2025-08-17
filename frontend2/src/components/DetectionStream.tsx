import React, { useEffect, useState, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactFlow, { 
  Node, 
  Edge, 
  Background, 
  Controls, 
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { theme } from '../theme';
import { DetectionEvent } from '../services/websocket';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';

dayjs.extend(relativeTime);

interface Props {
  events: DetectionEvent[];
  maxEvents?: number;
}

const DetectionStream: React.FC<Props> = ({ events, maxEvents = 100 }) => {
  const [selectedEvent, setSelectedEvent] = useState<DetectionEvent | null>(null);
  const [filterSeverity, setFilterSeverity] = useState<string>('ALL');
  const [flowNodes, setFlowNodes, onNodesChange] = useNodesState([]);
  const [flowEdges, setFlowEdges, onEdgesChange] = useEdgesState([]);
  const streamRef = useRef<HTMLDivElement>(null);

  const severityColors = {
    LOW: '#00ff41',
    MEDIUM: '#ffaa00',
    HIGH: '#ff6600',
    CRITICAL: '#ff0040',
  };

  const typeIcons = {
    SANDWICH: '🥪',
    FRONTRUN: '🏃',
    BACKRUN: '🔄',
    ARBITRAGE: '💱',
    LIQUIDATION: '💀',
  };

  const filteredEvents = useMemo(() => {
    const filtered = filterSeverity === 'ALL' 
      ? events 
      : events.filter(e => e.severity === filterSeverity);
    return filtered.slice(0, maxEvents);
  }, [events, filterSeverity, maxEvents]);

  // Update flow graph when events change
  useEffect(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const entityMap = new Map<string, { x: number; y: number; count: number }>();

    filteredEvents.slice(0, 20).forEach((event, idx) => {
      const attackerPos = entityMap.get(event.actors.attacker) || {
        x: Math.random() * 600,
        y: Math.random() * 400,
        count: 0,
      };
      
      if (!entityMap.has(event.actors.attacker)) {
        nodes.push({
          id: `attacker-${event.actors.attacker}`,
          type: 'default',
          position: attackerPos,
          data: { 
            label: `${event.actors.attacker.slice(0, 6)}...`,
          },
          style: {
            background: theme.colors.danger,
            color: theme.colors.text.primary,
            border: `2px solid ${severityColors[event.severity]}`,
            borderRadius: theme.borderRadius.md,
            fontSize: theme.fontSize.xs,
            boxShadow: `0 0 20px ${severityColors[event.severity]}66`,
          },
        });
        entityMap.set(event.actors.attacker, { ...attackerPos, count: 1 });
      } else {
        entityMap.set(event.actors.attacker, { 
          ...attackerPos, 
          count: attackerPos.count + 1 
        });
      }

      if (event.actors.victim) {
        const victimPos = {
          x: attackerPos.x + (Math.random() - 0.5) * 200,
          y: attackerPos.y + (Math.random() - 0.5) * 200,
        };

        nodes.push({
          id: `victim-${event.id}`,
          type: 'default',
          position: victimPos,
          data: { 
            label: `${event.actors.victim.slice(0, 6)}...`,
          },
          style: {
            background: theme.colors.bg.tertiary,
            color: theme.colors.text.secondary,
            border: `1px solid ${theme.colors.border.glass}`,
            borderRadius: theme.borderRadius.md,
            fontSize: theme.fontSize.xs,
          },
        });

        edges.push({
          id: `edge-${event.id}`,
          source: `attacker-${event.actors.attacker}`,
          target: `victim-${event.id}`,
          type: 'smoothstep',
          animated: true,
          style: {
            stroke: severityColors[event.severity],
            strokeWidth: 2,
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: severityColors[event.severity],
          },
          label: typeIcons[event.type],
          labelStyle: {
            fontSize: 16,
          },
        });
      }
    });

    setFlowNodes(nodes);
    setFlowEdges(edges);
  }, [filteredEvents, setFlowNodes, setFlowEdges]);

  const EventCard: React.FC<{ event: DetectionEvent }> = ({ event }) => (
    <motion.div
      layout
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      whileHover={{ scale: 1.02 }}
      onClick={() => setSelectedEvent(event)}
      style={{
        background: theme.colors.bg.tertiary,
        borderLeft: `4px solid ${severityColors[event.severity]}`,
        borderRadius: theme.borderRadius.md,
        padding: theme.spacing.md,
        marginBottom: theme.spacing.sm,
        cursor: 'pointer',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Animated background gradient */}
      <motion.div
        animate={{
          background: [
            `linear-gradient(90deg, ${severityColors[event.severity]}11 0%, transparent 100%)`,
            `linear-gradient(90deg, transparent 0%, ${severityColors[event.severity]}11 100%)`,
            `linear-gradient(90deg, ${severityColors[event.severity]}11 0%, transparent 100%)`,
          ],
        }}
        transition={{ duration: 3, repeat: Infinity }}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          pointerEvents: 'none',
        }}
      />

      <div style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <span style={{ fontSize: theme.fontSize.xl }}>{typeIcons[event.type]}</span>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <span style={{ 
                  color: theme.colors.text.primary,
                  fontWeight: 'bold',
                  fontSize: theme.fontSize.base,
                }}>
                  {event.type}
                </span>
                <span style={{
                  padding: `2px ${theme.spacing.xs}`,
                  background: severityColors[event.severity],
                  color: theme.colors.bg.primary,
                  borderRadius: theme.borderRadius.sm,
                  fontSize: theme.fontSize.xs,
                  fontWeight: 'bold',
                }}>
                  {event.severity}
                </span>
              </div>
              <span style={{ 
                color: theme.colors.text.muted, 
                fontSize: theme.fontSize.xs,
              }}>
                {dayjs(event.timestamp).fromNow()}
              </span>
            </div>
          </div>
          
          <div style={{ textAlign: 'right' }}>
            <div style={{ 
              color: theme.colors.primary,
              fontWeight: 'bold',
              fontSize: theme.fontSize.lg,
            }}>
              ${event.metrics.profitEstimate.toFixed(2)}
            </div>
            <div style={{ 
              color: theme.colors.text.muted,
              fontSize: theme.fontSize.xs,
            }}>
              {(event.metrics.confidence * 100).toFixed(1)}% confidence
            </div>
          </div>
        </div>

        <div style={{ 
          marginTop: theme.spacing.sm,
          display: 'flex',
          gap: theme.spacing.sm,
          fontSize: theme.fontSize.xs,
          color: theme.colors.text.secondary,
        }}>
          <span>Block: {event.blockHeight}</span>
          <span>•</span>
          <span>Venue: {event.venue}</span>
          <span>•</span>
          <span>Latency: {event.metrics.latency}ms</span>
        </div>

        <div style={{
          marginTop: theme.spacing.xs,
          fontSize: theme.fontSize.xs,
          color: theme.colors.text.muted,
          fontFamily: 'monospace',
        }}>
          TX: {event.txHash.slice(0, 10)}...{event.txHash.slice(-8)}
        </div>
      </div>
    </motion.div>
  );

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: theme.spacing.lg,
        height: '600px',
      }}
    >
      {/* Event Stream */}
      <div style={{
        background: theme.colors.bg.secondary,
        borderRadius: theme.borderRadius.lg,
        padding: theme.spacing.lg,
        display: 'flex',
        flexDirection: 'column',
      }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          marginBottom: theme.spacing.md,
        }}>
          <h2 style={{ 
            color: theme.colors.text.primary,
            fontSize: theme.fontSize.xl,
          }}>
            Detection Stream
          </h2>
          
          <div style={{ display: 'flex', gap: theme.spacing.xs }}>
            {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map(severity => (
              <motion.button
                key={severity}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setFilterSeverity(severity)}
                style={{
                  padding: `${theme.spacing.xs} ${theme.spacing.sm}`,
                  background: filterSeverity === severity 
                    ? theme.colors.primary 
                    : theme.colors.bg.tertiary,
                  color: filterSeverity === severity 
                    ? theme.colors.bg.primary 
                    : theme.colors.text.secondary,
                  border: 'none',
                  borderRadius: theme.borderRadius.sm,
                  fontSize: theme.fontSize.xs,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
              >
                {severity}
              </motion.button>
            ))}
          </div>
        </div>

        <div 
          ref={streamRef}
          style={{ 
            flex: 1,
            overflowY: 'auto',
            overflowX: 'hidden',
            paddingRight: theme.spacing.sm,
          }}
        >
          <AnimatePresence mode="popLayout">
            {filteredEvents.map(event => (
              <EventCard key={event.id} event={event} />
            ))}
          </AnimatePresence>
        </div>

        {/* Live indicator */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center',
          gap: theme.spacing.xs,
          marginTop: theme.spacing.sm,
        }}>
          <motion.div
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: theme.colors.primary,
              boxShadow: `0 0 10px ${theme.colors.primary}`,
            }}
          />
          <span style={{ 
            color: theme.colors.text.muted, 
            fontSize: theme.fontSize.xs,
          }}>
            Live • {filteredEvents.length} events
          </span>
        </div>
      </div>

      {/* Transaction Flow Graph */}
      <div style={{
        background: theme.colors.bg.secondary,
        borderRadius: theme.borderRadius.lg,
        padding: theme.spacing.lg,
        position: 'relative',
      }}>
        <h2 style={{ 
          color: theme.colors.text.primary,
          fontSize: theme.fontSize.xl,
          marginBottom: theme.spacing.md,
        }}>
          Attack Relationship Graph
        </h2>
        
        <div style={{ height: 'calc(100% - 40px)' }}>
          <ReactFlow
            nodes={flowNodes}
            edges={flowEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            style={{
              background: theme.colors.bg.primary,
              borderRadius: theme.borderRadius.md,
            }}
          >
            <Background color={theme.colors.border.glass} gap={20} />
            <Controls 
              style={{
                background: theme.colors.bg.tertiary,
                border: `1px solid ${theme.colors.border.glass}`,
              }}
            />
            <MiniMap 
              style={{
                background: theme.colors.bg.tertiary,
                border: `1px solid ${theme.colors.border.glass}`,
              }}
              nodeColor={node => node.style?.background || theme.colors.primary}
            />
          </ReactFlow>
        </div>
      </div>

      {/* Event Details Modal */}
      <AnimatePresence>
        {selectedEvent && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0,0,0,0.8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000,
            }}
            onClick={() => setSelectedEvent(null)}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              onClick={(e) => e.stopPropagation()}
              style={{
                background: theme.colors.bg.secondary,
                borderRadius: theme.borderRadius.lg,
                padding: theme.spacing.xl,
                maxWidth: '600px',
                width: '90%',
                border: `1px solid ${severityColors[selectedEvent.severity]}`,
                boxShadow: `0 0 40px ${severityColors[selectedEvent.severity]}44`,
              }}
            >
              <h3 style={{ 
                color: theme.colors.text.primary,
                fontSize: theme.fontSize['2xl'],
                marginBottom: theme.spacing.lg,
              }}>
                {typeIcons[selectedEvent.type]} {selectedEvent.type} Detection
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.md }}>
                <div>
                  <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                    Severity
                  </span>
                  <p style={{ 
                    color: severityColors[selectedEvent.severity],
                    fontWeight: 'bold',
                  }}>
                    {selectedEvent.severity}
                  </p>
                </div>
                <div>
                  <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                    Profit Estimate
                  </span>
                  <p style={{ color: theme.colors.primary }}>
                    ${selectedEvent.metrics.profitEstimate.toFixed(2)}
                  </p>
                </div>
                <div>
                  <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                    Confidence
                  </span>
                  <p style={{ color: theme.colors.text.primary }}>
                    {(selectedEvent.metrics.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                    Gas Used
                  </span>
                  <p style={{ color: theme.colors.text.primary }}>
                    {selectedEvent.metrics.gasUsed}
                  </p>
                </div>
              </div>

              <div style={{ marginTop: theme.spacing.lg }}>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  Transaction Hash
                </span>
                <p style={{ 
                  color: theme.colors.secondary,
                  fontFamily: 'monospace',
                  fontSize: theme.fontSize.sm,
                  wordBreak: 'break-all',
                }}>
                  {selectedEvent.txHash}
                </p>
              </div>

              <div style={{ marginTop: theme.spacing.md }}>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  Ed25519 Signature
                </span>
                <p style={{ 
                  color: theme.colors.text.secondary,
                  fontFamily: 'monospace',
                  fontSize: theme.fontSize.xs,
                  wordBreak: 'break-all',
                }}>
                  {selectedEvent.signatures.ed25519}
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default DetectionStream;