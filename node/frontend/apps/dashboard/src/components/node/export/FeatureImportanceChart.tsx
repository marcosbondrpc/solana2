import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Card } from '../../ui/card';
import { TrendingUp } from 'lucide-react';

interface FeatureImportanceChartProps {
  data: Array<{
    feature: string;
    importance: number;
    category: string;
  }>;
}

export const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data || data.length === 0) return;

    const margin = { top: 40, right: 150, bottom: 60, left: 200 };
    const width = 800 - margin.left - margin.right;
    const height = Math.max(400, data.length * 25) - margin.top - margin.bottom;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Sort data by importance
    const sortedData = [...data].sort((a, b) => b.importance - a.importance).slice(0, 20);

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(sortedData, d => d.importance) || 1])
      .range([0, width]);

    const yScale = d3.scaleBand()
      .domain(sortedData.map(d => d.feature))
      .range([0, height])
      .padding(0.2);

    // Create color scale for categories
    const categories = [...new Set(sortedData.map(d => d.category))];
    const colorScale = d3.scaleOrdinal()
      .domain(categories)
      .range(['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899']);

    // Add gradient definitions
    const defs = svg.append('defs');
    categories.forEach(category => {
      const gradient = defs.append('linearGradient')
        .attr('id', `gradient-${category.replace(/\s+/g, '-')}`)
        .attr('x1', '0%')
        .attr('x2', '100%')
        .attr('y1', '0%')
        .attr('y2', '0%');
      
      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', colorScale(category) as string)
        .attr('stop-opacity', 0.2);
      
      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', colorScale(category) as string)
        .attr('stop-opacity', 1);
    });

    // Add grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale)
        .ticks(5)
        .tickSize(-height)
        .tickFormat(() => ''))
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.1)
      .style('stroke', 'white');

    // Add bars
    const bars = g.selectAll('.bar')
      .data(sortedData)
      .enter().append('g')
      .attr('class', 'bar-group');

    bars.append('rect')
      .attr('class', 'bar-bg')
      .attr('x', 0)
      .attr('y', d => yScale(d.feature)!)
      .attr('width', width)
      .attr('height', yScale.bandwidth())
      .attr('fill', 'rgba(255, 255, 255, 0.02)')
      .attr('rx', 4);

    bars.append('rect')
      .attr('class', 'bar')
      .attr('x', 0)
      .attr('y', d => yScale(d.feature)!)
      .attr('width', 0)
      .attr('height', yScale.bandwidth())
      .attr('fill', d => `url(#gradient-${d.category.replace(/\s+/g, '-')})`)
      .attr('rx', 4)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('filter', 'brightness(1.2)');
        
        // Show tooltip
        const tooltip = g.append('g')
          .attr('class', 'tooltip-group');
        
        const rect = tooltip.append('rect')
          .attr('x', xScale(d.importance) + 5)
          .attr('y', yScale(d.feature)! + yScale.bandwidth() / 2 - 15)
          .attr('width', 120)
          .attr('height', 30)
          .attr('fill', 'rgba(0, 0, 0, 0.9)')
          .attr('stroke', '#00ffff')
          .attr('stroke-width', 1)
          .attr('rx', 4);
        
        tooltip.append('text')
          .attr('x', xScale(d.importance) + 65)
          .attr('y', yScale(d.feature)! + yScale.bandwidth() / 2 + 5)
          .attr('text-anchor', 'middle')
          .attr('fill', 'white')
          .attr('font-size', '12px')
          .text(`${(d.importance * 100).toFixed(1)}%`);
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('filter', 'none');
        
        g.selectAll('.tooltip-group').remove();
      })
      .transition()
      .duration(1000)
      .delay((d, i) => i * 50)
      .attr('width', d => xScale(d.importance));

    // Add value labels
    bars.append('text')
      .attr('class', 'value-label')
      .attr('x', d => xScale(d.importance) + 5)
      .attr('y', d => yScale(d.feature)! + yScale.bandwidth() / 2)
      .attr('dominant-baseline', 'middle')
      .attr('fill', 'rgba(255, 255, 255, 0.7)')
      .attr('font-size', '11px')
      .text(d => `${(d.importance * 100).toFixed(1)}%`)
      .style('opacity', 0)
      .transition()
      .duration(1000)
      .delay((d, i) => i * 50 + 500)
      .style('opacity', 1);

    // Add x axis
    const xAxis = g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale)
        .ticks(5)
        .tickFormat(d => `${(d as number * 100).toFixed(0)}%`));
    
    xAxis.selectAll('text')
      .attr('fill', 'rgba(255, 255, 255, 0.7)')
      .attr('font-size', '11px');
    
    xAxis.select('.domain')
      .attr('stroke', 'rgba(255, 255, 255, 0.2)');
    
    xAxis.selectAll('.tick line')
      .attr('stroke', 'rgba(255, 255, 255, 0.2)');

    // Add y axis
    const yAxis = g.append('g')
      .call(d3.axisLeft(yScale));
    
    yAxis.selectAll('text')
      .attr('fill', 'rgba(255, 255, 255, 0.9)')
      .attr('font-size', '11px')
      .text(d => {
        const text = d as string;
        return text.length > 25 ? text.substring(0, 25) + '...' : text;
      });
    
    yAxis.select('.domain')
      .attr('stroke', 'rgba(255, 255, 255, 0.2)');
    
    yAxis.selectAll('.tick line')
      .attr('stroke', 'rgba(255, 255, 255, 0.2)');

    // Add title
    svg.append('text')
      .attr('x', width / 2 + margin.left)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text('Feature Importance for MEV Prediction');

    // Add legend
    const legend = svg.append('g')
      .attr('transform', `translate(${width + margin.left + 20}, ${margin.top})`);

    const legendItems = legend.selectAll('.legend-item')
      .data(categories)
      .enter().append('g')
      .attr('class', 'legend-item')
      .attr('transform', (d, i) => `translate(0, ${i * 25})`);

    legendItems.append('rect')
      .attr('width', 15)
      .attr('height', 15)
      .attr('fill', d => colorScale(d) as string)
      .attr('rx', 3);

    legendItems.append('text')
      .attr('x', 20)
      .attr('y', 12)
      .attr('fill', 'rgba(255, 255, 255, 0.8)')
      .attr('font-size', '12px')
      .text(d => d);

  }, [data]);

  return (
    <Card className="p-6 bg-gray-800/30 border-gray-700/50">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-gradient-to-r from-cyan-500/10 to-blue-500/10">
          <TrendingUp className="w-5 h-5 text-cyan-400" />
        </div>
        <h3 className="text-lg font-medium text-white">Feature Importance Analysis</h3>
      </div>
      <div className="overflow-x-auto">
        <svg ref={svgRef}></svg>
      </div>
    </Card>
  );
};