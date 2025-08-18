'use client';

import { memo, useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';

interface EntityArchetype {
  name: string;
  metrics: {
    speed: number;        // 0-100
    volume: number;       // 0-100
    sophistication: number; // 0-100
    aggression: number;  // 0-100
    efficiency: number;  // 0-100
    stealth: number;     // 0-100
  };
  count: number;
  color: string;
  totalProfit: number;
}

const ARCHETYPES: EntityArchetype[] = [
  {
    name: 'Empire',
    metrics: {
      speed: 95,
      volume: 98,
      sophistication: 92,
      aggression: 75,
      efficiency: 96,
      stealth: 60
    },
    count: 12,
    color: '#ef4444',
    totalProfit: 2847293
  },
  {
    name: 'Warlord',
    metrics: {
      speed: 88,
      volume: 72,
      sophistication: 80,
      aggression: 95,
      efficiency: 85,
      stealth: 45
    },
    count: 34,
    color: '#f59e0b',
    totalProfit: 1293847
  },
  {
    name: 'Guerrilla',
    metrics: {
      speed: 78,
      volume: 45,
      sophistication: 65,
      aggression: 60,
      efficiency: 70,
      stealth: 90
    },
    count: 156,
    color: '#10b981',
    totalProfit: 493827
  },
  {
    name: 'Phantom',
    metrics: {
      speed: 70,
      volume: 30,
      sophistication: 55,
      aggression: 40,
      efficiency: 65,
      stealth: 98
    },
    count: 287,
    color: '#8b5cf6',
    totalProfit: 183749
  }
];

export const ArchetypeRadar = memo(() => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedArchetype, setSelectedArchetype] = useState<string | null>(null);
  const [animationPhase, setAnimationPhase] = useState(0);
  
  useEffect(() => {
    if (!svgRef.current) return;
    
    const width = 400;
    const height = 400;
    const margin = 60;
    const radius = Math.min(width, height) / 2 - margin;
    
    // Clear previous render
    d3.select(svgRef.current).selectAll('*').remove();
    
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);
    
    const g = svg.append('g')
      .attr('transform', `translate(${width/2},${height/2})`);
    
    // Define scales
    const angleScale = d3.scaleLinear()
      .domain([0, 6])
      .range([0, 2 * Math.PI]);
    
    const radiusScale = d3.scaleLinear()
      .domain([0, 100])
      .range([0, radius]);
    
    // Define axes
    const axes = Object.keys(ARCHETYPES[0].metrics);
    
    // Draw grid circles with GPU acceleration hint
    const gridLevels = [20, 40, 60, 80, 100];
    g.selectAll('.grid-circle')
      .data(gridLevels)
      .enter()
      .append('circle')
      .attr('class', 'grid-circle')
      .attr('r', d => radiusScale(d))
      .style('fill', 'none')
      .style('stroke', '#374151')
      .style('stroke-width', 0.5)
      .style('stroke-dasharray', '2,2')
      .style('will-change', 'transform');
    
    // Draw axes lines
    axes.forEach((axis, i) => {
      const angle = angleScale(i) - Math.PI / 2;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      
      g.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', x)
        .attr('y2', y)
        .style('stroke', '#4b5563')
        .style('stroke-width', 1);
      
      // Add axis labels
      const labelRadius = radius + 20;
      const labelX = Math.cos(angle) * labelRadius;
      const labelY = Math.sin(angle) * labelRadius;
      
      g.append('text')
        .attr('x', labelX)
        .attr('y', labelY)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .style('fill', '#9ca3af')
        .style('font-size', '12px')
        .style('text-transform', 'capitalize')
        .text(axis);
    });
    
    // Line generator
    const lineGenerator = d3.lineRadial<number>()
      .angle((d, i) => angleScale(i))
      .radius(d => radiusScale(d))
      .curve(d3.curveLinearClosed);
    
    // Draw archetype polygons
    ARCHETYPES.forEach((archetype, archetypeIndex) => {
      const values = Object.values(archetype.metrics);
      
      // Create gradient for each archetype
      const gradientId = `gradient-${archetype.name}`;
      const gradient = svg.append('defs')
        .append('radialGradient')
        .attr('id', gradientId);
      
      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', archetype.color)
        .attr('stop-opacity', 0.6);
      
      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', archetype.color)
        .attr('stop-opacity', 0.1);
      
      // Draw polygon
      const polygon = g.append('path')
        .datum(values)
        .attr('class', `archetype-polygon archetype-${archetype.name}`)
        .attr('d', lineGenerator)
        .style('fill', `url(#${gradientId})`)
        .style('stroke', archetype.color)
        .style('stroke-width', 2)
        .style('opacity', selectedArchetype === null || selectedArchetype === archetype.name ? 1 : 0.2)
        .style('cursor', 'pointer')
        .style('will-change', 'transform, opacity')
        .on('mouseenter', function() {
          setSelectedArchetype(archetype.name);
          d3.select(this)
            .transition()
            .duration(200)
            .style('stroke-width', 3)
            .style('opacity', 1);
        })
        .on('mouseleave', function() {
          setSelectedArchetype(null);
          d3.select(this)
            .transition()
            .duration(200)
            .style('stroke-width', 2)
            .style('opacity', 1);
        });
      
      // Add dots at vertices
      values.forEach((value, i) => {
        const angle = angleScale(i) - Math.PI / 2;
        const r = radiusScale(value);
        const x = Math.cos(angle) * r;
        const y = Math.sin(angle) * r;
        
        g.append('circle')
          .attr('cx', x)
          .attr('cy', y)
          .attr('r', 3)
          .style('fill', archetype.color)
          .style('stroke', '#1f2937')
          .style('stroke-width', 1)
          .style('opacity', selectedArchetype === null || selectedArchetype === archetype.name ? 1 : 0.2);
      });
    });
    
    // Animate rotation for visual effect
    const rotateAnimation = () => {
      g.transition()
        .duration(30000)
        .ease(d3.easeLinear)
        .attr('transform', `translate(${width/2},${height/2}) rotate(360)`)
        .on('end', () => {
          g.attr('transform', `translate(${width/2},${height/2})`);
          rotateAnimation();
        });
    };
    
    // Start subtle rotation
    rotateAnimation();
    
  }, [selectedArchetype]);
  
  // Update animation
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimationPhase(prev => (prev + 1) % 360);
    }, 50);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="relative">
      {/* Legend */}
      <div className="absolute top-0 right-0 space-y-2">
        {ARCHETYPES.map((archetype) => (
          <motion.div
            key={archetype.name}
            className={`flex items-center gap-2 px-3 py-1 rounded cursor-pointer transition-all ${
              selectedArchetype === archetype.name ? 'bg-gray-800' : 'hover:bg-gray-900'
            }`}
            onClick={() => setSelectedArchetype(
              selectedArchetype === archetype.name ? null : archetype.name
            )}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: archetype.color }}
            />
            <div className="text-xs">
              <div className="font-medium">{archetype.name}</div>
              <div className="text-gray-500">
                {archetype.count} entities • ${(archetype.totalProfit / 1000).toFixed(0)}k
              </div>
            </div>
          </motion.div>
        ))}
      </div>
      
      {/* SVG Container */}
      <div className="flex justify-center items-center">
        <svg 
          ref={svgRef}
          className="transform-gpu"
          style={{
            filter: 'drop-shadow(0 0 20px rgba(59, 130, 246, 0.3))',
            transform: `rotate(${animationPhase * 0.1}deg)`
          }}
        />
      </div>
      
      {/* Stats */}
      {selectedArchetype && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-4 bg-gray-900 rounded-lg"
        >
          <h4 className="text-sm font-medium mb-2">
            {selectedArchetype} Metrics
          </h4>
          <div className="grid grid-cols-3 gap-2 text-xs">
            {Object.entries(
              ARCHETYPES.find(a => a.name === selectedArchetype)?.metrics || {}
            ).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <span className="text-gray-500 capitalize">{key}:</span>
                <span className="font-mono">{value}%</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
});

ArchetypeRadar.displayName = 'ArchetypeRadar';