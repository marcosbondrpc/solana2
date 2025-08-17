import React, { useMemo, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { theme } from '../theme';
import { EntityProfile } from '../services/websocket';

interface Props {
  entities: EntityProfile[];
  focusEntity?: string;
}

const EntityBehavioralSpectrum: React.FC<Props> = ({ entities, focusEntity }) => {
  const [selectedEntity, setSelectedEntity] = useState<EntityProfile | null>(null);
  const [comparisonMode, setComparisonMode] = useState(false);

  const radarData = useMemo(() => {
    if (!selectedEntity) return [];

    return [
      { metric: 'Risk Appetite', value: selectedEntity.riskAppetite * 100, fullMark: 100 },
      { metric: 'Success Rate', value: selectedEntity.successRate, fullMark: 100 },
      { metric: 'Volume', value: Math.min(selectedEntity.attackVolume / 1000, 100), fullMark: 100 },
      { metric: 'Uptime', value: Math.min(selectedEntity.uptime.hours / 24, 100), fullMark: 100 },
      { metric: 'Avg Profit', value: Math.min(selectedEntity.avgProfit / 10000, 100), fullMark: 100 },
    ];
  }, [selectedEntity]);

  const styleDistribution = useMemo(() => {
    const styles = entities.reduce((acc, entity) => {
      acc[entity.style] = (acc[entity.style] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(styles).map(([style, count]) => ({
      style,
      count,
      percentage: (count / entities.length) * 100,
    }));
  }, [entities]);

  const feePostureMap = useMemo(() => {
    const colorMap = {
      AGGRESSIVE: theme.colors.danger,
      CONSERVATIVE: theme.colors.primary,
      ADAPTIVE: theme.colors.secondary,
    };

    return entities.map(entity => ({
      ...entity,
      color: colorMap[entity.feePosture],
    }));
  }, [entities]);

  useEffect(() => {
    if (focusEntity) {
      const entity = entities.find(e => e.address === focusEntity);
      if (entity) setSelectedEntity(entity);
    }
  }, [focusEntity, entities]);

  const HeatMap: React.FC = () => {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);

    useEffect(() => {
      if (!canvasRef.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Create heat map visualization
      const width = canvas.width;
      const height = canvas.height;

      // Clear canvas
      ctx.fillStyle = theme.colors.bg.primary;
      ctx.fillRect(0, 0, width, height);

      // Draw heat map grid
      const gridSize = 20;
      const maxRisk = Math.max(...entities.map(e => e.riskAppetite));

      entities.forEach((entity, idx) => {
        const x = (idx % gridSize) * (width / gridSize);
        const y = Math.floor(idx / gridSize) * (height / gridSize);
        
        const intensity = entity.riskAppetite / maxRisk;
        const hue = 120 - (intensity * 120); // Green to red
        
        ctx.fillStyle = `hsla(${hue}, 100%, 50%, ${0.3 + intensity * 0.7})`;
        ctx.fillRect(x, y, width / gridSize - 2, height / gridSize - 2);
        
        // Add glow effect for high-risk entities
        if (intensity > 0.7) {
          ctx.shadowBlur = 10;
          ctx.shadowColor = `hsla(${hue}, 100%, 50%, 1)`;
          ctx.fillRect(x, y, width / gridSize - 2, height / gridSize - 2);
          ctx.shadowBlur = 0;
        }
      });
    }, []);

    return (
      <canvas 
        ref={canvasRef} 
        width={400} 
        height={300}
        style={{ width: '100%', height: '100%' }}
      />
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      style={{
        background: theme.colors.bg.secondary,
        borderRadius: theme.borderRadius.lg,
        padding: theme.spacing.lg,
        border: `1px solid ${theme.colors.border.glass}`,
        backdropFilter: 'blur(10px)',
      }}
    >
      <div style={{ marginBottom: theme.spacing.lg }}>
        <h2 style={{ 
          color: theme.colors.text.primary,
          fontSize: theme.fontSize['2xl'],
          marginBottom: theme.spacing.md,
          textShadow: theme.effects.neon.text,
        }}>
          Entity Behavioral Spectrum
        </h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.lg }}>
        {/* Radar Chart */}
        <div style={{
          background: theme.colors.bg.tertiary,
          borderRadius: theme.borderRadius.md,
          padding: theme.spacing.md,
        }}>
          <h3 style={{ color: theme.colors.text.secondary, marginBottom: theme.spacing.sm }}>
            Attack Profile Analysis
          </h3>
          {selectedEntity ? (
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid 
                  gridType="polygon"
                  stroke={theme.colors.border.glass}
                />
                <PolarAngleAxis 
                  dataKey="metric"
                  tick={{ fill: theme.colors.text.muted, fontSize: 10 }}
                />
                <PolarRadiusAxis 
                  domain={[0, 100]}
                  tick={{ fill: theme.colors.text.muted, fontSize: 8 }}
                />
                <Radar
                  name={selectedEntity.address.slice(0, 8) + '...'}
                  dataKey="value"
                  stroke={theme.colors.primary}
                  fill={theme.colors.primary}
                  fillOpacity={0.3}
                />
                <Tooltip
                  contentStyle={{
                    background: theme.colors.bg.glass,
                    border: `1px solid ${theme.colors.border.primary}`,
                    borderRadius: theme.borderRadius.sm,
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ 
              height: 300, 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              color: theme.colors.text.muted,
            }}>
              Select an entity to view profile
            </div>
          )}
        </div>

        {/* Risk Heat Map */}
        <div style={{
          background: theme.colors.bg.tertiary,
          borderRadius: theme.borderRadius.md,
          padding: theme.spacing.md,
        }}>
          <h3 style={{ color: theme.colors.text.secondary, marginBottom: theme.spacing.sm }}>
            Risk Appetite Heat Map
          </h3>
          <HeatMap />
        </div>

        {/* Style Distribution */}
        <div style={{
          background: theme.colors.bg.tertiary,
          borderRadius: theme.borderRadius.md,
          padding: theme.spacing.md,
        }}>
          <h3 style={{ color: theme.colors.text.secondary, marginBottom: theme.spacing.sm }}>
            Attack Style Distribution
          </h3>
          {styleDistribution.map((item, idx) => (
            <motion.div
              key={item.style}
              initial={{ width: 0 }}
              animate={{ width: `${item.percentage}%` }}
              transition={{ delay: idx * 0.1, duration: 0.5 }}
              style={{
                marginBottom: theme.spacing.sm,
              }}
            >
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                marginBottom: theme.spacing.xs,
                color: theme.colors.text.secondary,
                fontSize: theme.fontSize.sm,
              }}>
                <span>{item.style}</span>
                <span>{item.count} ({item.percentage.toFixed(1)}%)</span>
              </div>
              <div style={{
                height: '20px',
                background: `linear-gradient(90deg, ${theme.colors.primary} 0%, ${theme.colors.secondary} 100%)`,
                borderRadius: theme.borderRadius.sm,
                boxShadow: theme.effects.glow.primary,
              }} />
            </motion.div>
          ))}
        </div>

        {/* Fee Posture */}
        <div style={{
          background: theme.colors.bg.tertiary,
          borderRadius: theme.borderRadius.md,
          padding: theme.spacing.md,
        }}>
          <h3 style={{ color: theme.colors.text.secondary, marginBottom: theme.spacing.sm }}>
            Fee Posture Distribution
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {feePostureMap.slice(0, 10).map((entity) => (
              <motion.div
                key={entity.address}
                whileHover={{ scale: 1.02 }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: theme.spacing.sm,
                  background: theme.colors.bg.primary,
                  borderRadius: theme.borderRadius.sm,
                  border: `1px solid ${entity.color}33`,
                  cursor: 'pointer',
                }}
                onClick={() => setSelectedEntity(entity)}
              >
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: entity.color,
                  marginRight: theme.spacing.sm,
                  boxShadow: `0 0 10px ${entity.color}`,
                }} />
                <span style={{ 
                  color: theme.colors.text.muted,
                  fontSize: theme.fontSize.xs,
                  fontFamily: 'monospace',
                }}>
                  {entity.address.slice(0, 6)}...{entity.address.slice(-4)}
                </span>
                <span style={{
                  marginLeft: 'auto',
                  color: entity.color,
                  fontSize: theme.fontSize.xs,
                  fontWeight: 'bold',
                }}>
                  {entity.feePosture}
                </span>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Entity Details Panel */}
      {selectedEntity && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          style={{
            marginTop: theme.spacing.lg,
            padding: theme.spacing.md,
            background: theme.colors.bg.tertiary,
            borderRadius: theme.borderRadius.md,
            border: `1px solid ${theme.colors.border.primary}`,
          }}
        >
          <h3 style={{ color: theme.colors.primary, marginBottom: theme.spacing.sm }}>
            Entity Details: {selectedEntity.address}
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: theme.spacing.md }}>
            <div>
              <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>Style</span>
              <p style={{ color: theme.colors.text.primary }}>{selectedEntity.style}</p>
            </div>
            <div>
              <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>Success Rate</span>
              <p style={{ color: theme.colors.text.primary }}>{selectedEntity.successRate.toFixed(1)}%</p>
            </div>
            <div>
              <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>Avg Profit</span>
              <p style={{ color: theme.colors.text.primary }}>${selectedEntity.avgProfit.toFixed(2)}</p>
            </div>
            <div>
              <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>Uptime Pattern</span>
              <p style={{ color: theme.colors.text.primary }}>{selectedEntity.uptime.pattern}</p>
            </div>
          </div>
          <div style={{ marginTop: theme.spacing.sm }}>
            <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>Preferred Venues</span>
            <div style={{ display: 'flex', gap: theme.spacing.xs, marginTop: theme.spacing.xs }}>
              {selectedEntity.preferredVenues.map(venue => (
                <span key={venue} style={{
                  padding: `${theme.spacing.xs} ${theme.spacing.sm}`,
                  background: theme.colors.bg.primary,
                  borderRadius: theme.borderRadius.sm,
                  color: theme.colors.secondary,
                  fontSize: theme.fontSize.xs,
                }}>
                  {venue}
                </span>
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default EntityBehavioralSpectrum;