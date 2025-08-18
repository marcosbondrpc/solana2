'use client';

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useArchetypeData } from '@/hooks/useArchetypeData';

interface ArchetypeMetrics {
  empire: {
    sophistication: number;
    volume: number;
    efficiency: number;
    consistency: number;
    innovation: number;
  };
  warlord: {
    sophistication: number;
    volume: number;
    efficiency: number;
    consistency: number;
    innovation: number;
  };
  guerrilla: {
    sophistication: number;
    volume: number;
    efficiency: number;
    consistency: number;
    innovation: number;
  };
}

export function ArchetypeRadar() {
  const svgRef = useRef<SVGSVGElement>(null);
  const { data, isLoading } = useArchetypeData();
  
  useEffect(() => {
    if (!data || !svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const width = 400;
    const height = 400;
    const margin = 60;
    const radius = Math.min(width, height) / 2 - margin;
    
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);
    
    // Define axes
    const axes = ['Sophistication', 'Volume', 'Efficiency', 'Consistency', 'Innovation'];
    const angleSlice = (Math.PI * 2) / axes.length;
    
    // Scale
    const rScale = d3.scaleLinear().domain([0, 100]).range([0, radius]);
    
    // Grid circles
    const levels = 5;
    for (let level = 1; level <= levels; level++) {
      g.append('circle')
        .attr('r', (radius / levels) * level)
        .style('fill', 'none')
        .style('stroke', '#374151')
        .style('stroke-opacity', 0.3)
        .style('stroke-width', 0.5);
    }
    
    // Grid lines
    const axis = g.selectAll('.axis').data(axes).enter().append('g').attr('class', 'axis');
    
    axis
      .append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', (d, i) => rScale(100) * Math.cos(angleSlice * i - Math.PI / 2))
      .attr('y2', (d, i) => rScale(100) * Math.sin(angleSlice * i - Math.PI / 2))
      .style('stroke', '#374151')
      .style('stroke-opacity', 0.3)
      .style('stroke-width', 0.5);
    
    // Axis labels
    axis
      .append('text')
      .attr('x', (d, i) => (rScale(100) + 20) * Math.cos(angleSlice * i - Math.PI / 2))
      .attr('y', (d, i) => (rScale(100) + 20) * Math.sin(angleSlice * i - Math.PI / 2))
      .style('font-size', '12px')
      .style('fill', '#9CA3AF')
      .style('text-anchor', 'middle')
      .text((d) => d);
    
    // Data preparation
    const archetypes = [
      {
        name: 'Empire',
        color: '#845ec2',
        data: [
          data.empire.sophistication,
          data.empire.volume,
          data.empire.efficiency,
          data.empire.consistency,
          data.empire.innovation,
        ],
      },
      {
        name: 'Warlord',
        color: '#ff9671',
        data: [
          data.warlord.sophistication,
          data.warlord.volume,
          data.warlord.efficiency,
          data.warlord.consistency,
          data.warlord.innovation,
        ],
      },
      {
        name: 'Guerrilla',
        color: '#ffc75f',
        data: [
          data.guerrilla.sophistication,
          data.guerrilla.volume,
          data.guerrilla.efficiency,
          data.guerrilla.consistency,
          data.guerrilla.innovation,
        ],
      },
    ];
    
    // Draw radar areas
    const radarLine = d3
      .lineRadial<number>()
      .radius((d) => rScale(d))
      .angle((d, i) => angleSlice * i)
      .curve(d3.curveLinearClosed);
    
    archetypes.forEach((archetype) => {
      g.append('path')
        .datum(archetype.data)
        .attr('d', radarLine)
        .style('fill', archetype.color)
        .style('fill-opacity', 0.2)
        .style('stroke', archetype.color)
        .style('stroke-width', 2);
      
      // Add dots
      g.selectAll(`.dot-${archetype.name}`)
        .data(archetype.data)
        .enter()
        .append('circle')
        .attr('cx', (d, i) => rScale(d) * Math.cos(angleSlice * i - Math.PI / 2))
        .attr('cy', (d, i) => rScale(d) * Math.sin(angleSlice * i - Math.PI / 2))
        .attr('r', 3)
        .style('fill', archetype.color);
    });
    
    // Legend
    const legend = svg
      .append('g')
      .attr('transform', `translate(${width - 100}, 20)`);
    
    archetypes.forEach((archetype, i) => {
      const legendRow = legend
        .append('g')
        .attr('transform', `translate(0, ${i * 20})`);
      
      legendRow
        .append('rect')
        .attr('width', 12)
        .attr('height', 12)
        .attr('fill', archetype.color);
      
      legendRow
        .append('text')
        .attr('x', 18)
        .attr('y', 9)
        .style('font-size', '12px')
        .style('fill', '#9CA3AF')
        .text(archetype.name);
    });
    
  }, [data]);
  
  if (isLoading) {
    return (
      <div className="h-[400px] flex items-center justify-center">
        <div className="text-muted-foreground">Loading archetype data...</div>
      </div>
    );
  }
  
  return (
    <div>
      <svg ref={svgRef} className="w-full h-full" />
      <div className="mt-4 space-y-2 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-mev-empire rounded" />
          <span className="text-muted-foreground">Empire: High sophistication, consistent performance</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-mev-warlord rounded" />
          <span className="text-muted-foreground">Warlord: Regional dominance, moderate innovation</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-mev-guerrilla rounded" />
          <span className="text-muted-foreground">Guerrilla: Opportunistic, high volume</span>
        </div>
      </div>
    </div>
  );
}