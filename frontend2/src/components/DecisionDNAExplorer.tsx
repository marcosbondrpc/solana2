import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Box, Sphere } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';
import { theme } from '../theme';
import dayjs from 'dayjs';

interface DNANode {
  id: string;
  hash: string;
  timestamp: number;
  features: string[];
  signature: string;
  parent?: string;
  children: string[];
  depth: number;
  confidence: number;
}

interface Props {
  nodes: DNANode[];
  selectedNodeId?: string;
  onNodeSelect?: (nodeId: string) => void;
}

// 3D Merkle Tree Node Component
const TreeNode: React.FC<{
  node: DNANode;
  position: [number, number, number];
  isSelected: boolean;
  onClick: () => void;
}> = ({ node, position, isSelected, onClick }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01;
      if (isSelected || hovered) {
        meshRef.current.scale.setScalar(1.2 + Math.sin(state.clock.elapsedTime * 2) * 0.1);
      } else {
        meshRef.current.scale.setScalar(1);
      }
    }
  });

  const color = useMemo(() => {
    if (isSelected) return '#00ff41';
    if (node.confidence > 0.9) return '#00ffff';
    if (node.confidence > 0.7) return '#ffaa00';
    return '#ff0040';
  }, [isSelected, node.confidence]);

  return (
    <group position={position}>
      <Box
        ref={meshRef}
        args={[0.5, 0.5, 0.5]}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isSelected || hovered ? 0.5 : 0.2}
          metalness={0.8}
          roughness={0.2}
        />
      </Box>
      {(isSelected || hovered) && (
        <Text
          position={[0, 0.8, 0]}
          fontSize={0.15}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
        >
          {node.id.slice(0, 8)}
        </Text>
      )}
    </group>
  );
};

// 3D Merkle Tree Visualization
const MerkleTree3D: React.FC<{
  nodes: DNANode[];
  selectedNodeId?: string;
  onNodeSelect?: (nodeId: string) => void;
}> = ({ nodes, selectedNodeId, onNodeSelect }) => {
  const positions = useMemo(() => {
    const pos: Record<string, [number, number, number]> = {};
    const maxDepth = Math.max(...nodes.map(n => n.depth));
    
    nodes.forEach((node) => {
      const depthNodes = nodes.filter(n => n.depth === node.depth);
      const index = depthNodes.indexOf(node);
      const spread = 10 / (node.depth + 1);
      
      pos[node.id] = [
        (index - depthNodes.length / 2) * spread,
        (maxDepth - node.depth) * 2,
        node.depth * -1.5,
      ];
    });
    
    return pos;
  }, [nodes]);

  const edges = useMemo(() => {
    const edgeList: Array<[THREE.Vector3, THREE.Vector3]> = [];
    
    nodes.forEach(node => {
      if (node.parent) {
        const parentNode = nodes.find(n => n.id === node.parent);
        if (parentNode) {
          const start = new THREE.Vector3(...positions[node.parent]);
          const end = new THREE.Vector3(...positions[node.id]);
          edgeList.push([start, end]);
        }
      }
    });
    
    return edgeList;
  }, [nodes, positions]);

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#00ffff" />
      
      {/* Render edges */}
      {edges.map((edge, idx) => (
        <Line
          key={idx}
          points={edge}
          color="#00ff4133"
          lineWidth={1}
          transparent
          opacity={0.5}
        />
      ))}
      
      {/* Render nodes */}
      {nodes.map(node => (
        <TreeNode
          key={node.id}
          node={node}
          position={positions[node.id]}
          isSelected={selectedNodeId === node.id}
          onClick={() => onNodeSelect?.(node.id)}
        />
      ))}
      
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        autoRotate={true}
        autoRotateSpeed={0.5}
      />
    </>
  );
};

const DecisionDNAExplorer: React.FC<Props> = ({ nodes, selectedNodeId, onNodeSelect }) => {
  const [selectedNode, setSelectedNode] = useState<DNANode | null>(null);
  const [verificationStatus, setVerificationStatus] = useState<'pending' | 'verified' | 'failed'>('pending');
  const [viewMode, setViewMode] = useState<'3d' | 'timeline' | 'features'>('3d');

  useEffect(() => {
    if (selectedNodeId) {
      const node = nodes.find(n => n.id === selectedNodeId);
      if (node) {
        setSelectedNode(node);
        // Simulate signature verification
        setTimeout(() => {
          setVerificationStatus(Math.random() > 0.1 ? 'verified' : 'failed');
        }, 500);
      }
    }
  }, [selectedNodeId, nodes]);

  const handleNodeSelect = (nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId);
    setSelectedNode(node || null);
    setVerificationStatus('pending');
    onNodeSelect?.(nodeId);
  };

  const auditTrail = useMemo(() => {
    if (!selectedNode) return [];
    
    const trail: DNANode[] = [];
    let current: DNANode | undefined = selectedNode;
    
    while (current) {
      trail.push(current);
      current = nodes.find(n => n.id === current?.parent);
    }
    
    return trail.reverse();
  }, [selectedNode, nodes]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      style={{
        background: theme.colors.bg.secondary,
        borderRadius: theme.borderRadius.lg,
        padding: theme.spacing.lg,
        height: '800px',
      }}
    >
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        marginBottom: theme.spacing.lg,
      }}>
        <h2 style={{ 
          color: theme.colors.text.primary,
          fontSize: theme.fontSize['2xl'],
          textShadow: theme.effects.neon.text,
        }}>
          Decision DNA Explorer
        </h2>

        <div style={{ display: 'flex', gap: theme.spacing.sm }}>
          {['3d', 'timeline', 'features'].map(mode => (
            <motion.button
              key={mode}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setViewMode(mode as any)}
              style={{
                padding: `${theme.spacing.sm} ${theme.spacing.md}`,
                background: viewMode === mode ? theme.colors.primary : theme.colors.bg.tertiary,
                color: viewMode === mode ? theme.colors.bg.primary : theme.colors.text.secondary,
                border: 'none',
                borderRadius: theme.borderRadius.md,
                fontSize: theme.fontSize.sm,
                cursor: 'pointer',
                textTransform: 'capitalize',
              }}
            >
              {mode === '3d' ? '3D Tree' : mode}
            </motion.button>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: theme.spacing.lg, height: 'calc(100% - 60px)' }}>
        {/* Main Visualization Area */}
        <div style={{
          background: theme.colors.bg.tertiary,
          borderRadius: theme.borderRadius.md,
          overflow: 'hidden',
          position: 'relative',
        }}>
          <AnimatePresence mode="wait">
            {viewMode === '3d' && (
              <motion.div
                key="3d"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{ width: '100%', height: '100%' }}
              >
                <Canvas camera={{ position: [0, 5, 15], fov: 60 }}>
                  <MerkleTree3D
                    nodes={nodes}
                    selectedNodeId={selectedNode?.id}
                    onNodeSelect={handleNodeSelect}
                  />
                </Canvas>
              </motion.div>
            )}

            {viewMode === 'timeline' && (
              <motion.div
                key="timeline"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{ 
                  padding: theme.spacing.lg,
                  height: '100%',
                  overflowY: 'auto',
                }}
              >
                <h3 style={{ color: theme.colors.text.primary, marginBottom: theme.spacing.md }}>
                  Audit Trail Timeline
                </h3>
                {auditTrail.map((node, idx) => (
                  <motion.div
                    key={node.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      marginBottom: theme.spacing.md,
                    }}
                  >
                    <div style={{
                      width: '40px',
                      height: '40px',
                      borderRadius: '50%',
                      background: `linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary})`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      marginRight: theme.spacing.md,
                      flexShrink: 0,
                    }}>
                      {idx + 1}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ 
                        color: theme.colors.text.primary,
                        fontFamily: 'monospace',
                        fontSize: theme.fontSize.sm,
                      }}>
                        {node.hash.slice(0, 16)}...
                      </div>
                      <div style={{ 
                        color: theme.colors.text.muted,
                        fontSize: theme.fontSize.xs,
                      }}>
                        {dayjs(node.timestamp).format('YYYY-MM-DD HH:mm:ss.SSS')}
                      </div>
                    </div>
                    {idx < auditTrail.length - 1 && (
                      <div style={{
                        position: 'absolute',
                        left: '20px',
                        top: '40px',
                        width: '2px',
                        height: '60px',
                        background: theme.colors.border.primary,
                      }} />
                    )}
                  </motion.div>
                ))}
              </motion.div>
            )}

            {viewMode === 'features' && selectedNode && (
              <motion.div
                key="features"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{ 
                  padding: theme.spacing.lg,
                  height: '100%',
                  overflowY: 'auto',
                }}
              >
                <h3 style={{ color: theme.colors.text.primary, marginBottom: theme.spacing.md }}>
                  Feature Hash Components
                </h3>
                {selectedNode.features.map((feature, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    style={{
                      background: theme.colors.bg.primary,
                      borderRadius: theme.borderRadius.sm,
                      padding: theme.spacing.sm,
                      marginBottom: theme.spacing.sm,
                      fontFamily: 'monospace',
                      fontSize: theme.fontSize.xs,
                      color: theme.colors.secondary,
                      wordBreak: 'break-all',
                    }}
                  >
                    {feature}
                  </motion.div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Details Panel */}
        <div style={{
          background: theme.colors.bg.tertiary,
          borderRadius: theme.borderRadius.md,
          padding: theme.spacing.md,
          display: 'flex',
          flexDirection: 'column',
        }}>
          <h3 style={{ 
            color: theme.colors.text.primary,
            fontSize: theme.fontSize.lg,
            marginBottom: theme.spacing.md,
          }}>
            Node Details
          </h3>

          {selectedNode ? (
            <>
              <div style={{ marginBottom: theme.spacing.md }}>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  Node ID
                </span>
                <p style={{ 
                  color: theme.colors.text.primary,
                  fontFamily: 'monospace',
                  fontSize: theme.fontSize.sm,
                  wordBreak: 'break-all',
                }}>
                  {selectedNode.id}
                </p>
              </div>

              <div style={{ marginBottom: theme.spacing.md }}>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  Hash
                </span>
                <p style={{ 
                  color: theme.colors.secondary,
                  fontFamily: 'monospace',
                  fontSize: theme.fontSize.xs,
                  wordBreak: 'break-all',
                }}>
                  {selectedNode.hash}
                </p>
              </div>

              <div style={{ marginBottom: theme.spacing.md }}>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  Ed25519 Signature
                </span>
                <p style={{ 
                  color: theme.colors.text.secondary,
                  fontFamily: 'monospace',
                  fontSize: theme.fontSize.xs,
                  wordBreak: 'break-all',
                }}>
                  {selectedNode.signature}
                </p>
              </div>

              <div style={{ marginBottom: theme.spacing.md }}>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  Verification Status
                </span>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  marginTop: theme.spacing.xs,
                }}>
                  <motion.div
                    animate={{ 
                      scale: verificationStatus === 'pending' ? [1, 1.2, 1] : 1,
                    }}
                    transition={{ duration: 1, repeat: verificationStatus === 'pending' ? Infinity : 0 }}
                    style={{
                      width: '12px',
                      height: '12px',
                      borderRadius: '50%',
                      background: verificationStatus === 'verified' 
                        ? theme.colors.primary
                        : verificationStatus === 'failed'
                        ? theme.colors.danger
                        : theme.colors.warning,
                      boxShadow: `0 0 10px ${
                        verificationStatus === 'verified' 
                          ? theme.colors.primary
                          : verificationStatus === 'failed'
                          ? theme.colors.danger
                          : theme.colors.warning
                      }`,
                    }}
                  />
                  <span style={{ 
                    color: verificationStatus === 'verified' 
                      ? theme.colors.primary
                      : verificationStatus === 'failed'
                      ? theme.colors.danger
                      : theme.colors.warning,
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    fontSize: theme.fontSize.sm,
                  }}>
                    {verificationStatus}
                  </span>
                </div>
              </div>

              <div style={{ marginBottom: theme.spacing.md }}>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  Confidence Score
                </span>
                <div style={{ marginTop: theme.spacing.xs }}>
                  <div style={{
                    height: '8px',
                    background: theme.colors.bg.primary,
                    borderRadius: theme.borderRadius.sm,
                    overflow: 'hidden',
                  }}>
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${selectedNode.confidence * 100}%` }}
                      transition={{ duration: 0.5 }}
                      style={{
                        height: '100%',
                        background: `linear-gradient(90deg, ${theme.colors.primary}, ${theme.colors.secondary})`,
                        boxShadow: `0 0 10px ${theme.colors.primary}`,
                      }}
                    />
                  </div>
                  <p style={{ 
                    color: theme.colors.text.primary,
                    fontSize: theme.fontSize.sm,
                    marginTop: theme.spacing.xs,
                  }}>
                    {(selectedNode.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              <div style={{ marginBottom: theme.spacing.md }}>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  Timestamp
                </span>
                <p style={{ 
                  color: theme.colors.text.primary,
                  fontSize: theme.fontSize.sm,
                }}>
                  {dayjs(selectedNode.timestamp).format('YYYY-MM-DD HH:mm:ss.SSS')}
                </p>
              </div>

              <div>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  Tree Depth
                </span>
                <p style={{ 
                  color: theme.colors.text.primary,
                  fontSize: theme.fontSize.sm,
                }}>
                  Level {selectedNode.depth}
                </p>
              </div>
            </>
          ) : (
            <div style={{ 
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: theme.colors.text.muted,
            }}>
              Select a node to view details
            </div>
          )}

          {/* Merkle Root Anchor Status */}
          <div style={{
            marginTop: 'auto',
            padding: theme.spacing.md,
            background: theme.colors.bg.primary,
            borderRadius: theme.borderRadius.md,
            border: `1px solid ${theme.colors.border.primary}`,
          }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <span style={{ 
                color: theme.colors.text.secondary,
                fontSize: theme.fontSize.xs,
              }}>
                Daily Merkle Anchor
              </span>
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
            </div>
            <p style={{ 
              color: theme.colors.text.muted,
              fontSize: theme.fontSize.xs,
              marginTop: theme.spacing.xs,
            }}>
              Next anchor in 3h 42m
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default DecisionDNAExplorer;