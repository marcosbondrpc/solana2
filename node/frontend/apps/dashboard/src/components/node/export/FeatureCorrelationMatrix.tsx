import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Card } from '../../ui/Card';

interface FeatureCorrelationMatrixProps {
  data: {
    features: string[];
    matrix: number[][];
  };
}

export const FeatureCorrelationMatrix: React.FC<FeatureCorrelationMatrixProps> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data) return;

    const margin = { top: 100, right: 20, bottom: 20, left: 100 };
    const width = 600 - margin.left - margin.right;
    const height = 600 - margin.top - margin.bottom;
    const cellSize = Math.min(width, height) / data.features.length;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create color scale
    const colorScale = d3.scaleSequential()
      .domain([-1, 1])
      .interpolator(d3.interpolateRdBu)
      .clamp(true);

    // Create x and y scales
    const xScale = d3.scaleBand()
      .domain(data.features)
      .range([0, width])
      .padding(0.05);

    const yScale = d3.scaleBand()
      .domain(data.features)
      .range([0, height])
      .padding(0.05);

    // Create tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.9)')
      .style('color', 'white')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none');

    // Draw cells
    const cells = g.selectAll('.cell')
      .data(data.features.flatMap((row, i) => 
        data.features.map((col, j) => ({
          row,
          col,
          value: data.matrix[i][j],
          x: j,
          y: i
        }))
      ));

    cells.enter().append('rect')
      .attr('class', 'cell')
      .attr('x', d => xScale(d.col)!)
      .attr('y', d => yScale(d.row)!)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', 'rgba(255, 255, 255, 0.1)')
      .attr('stroke-width', 0.5)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke', '#00ffff')
          .attr('stroke-width', 2);
        
        tooltip.transition()
          .duration(200)
          .style('opacity', .9);
        
        tooltip.html(`
          <div style="font-weight: bold;">${d.row} × ${d.col}</div>
          <div>Correlation: ${d.value.toFixed(3)}</div>
        `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', 'rgba(255, 255, 255, 0.1)')
          .attr('stroke-width', 0.5);
        
        tooltip.transition()
          .duration(500)
          .style('opacity', 0);
      });

    // Add text labels for strong correlations
    const textThreshold = 0.7;
    cells.enter().append('text')
      .attr('x', d => xScale(d.col)! + xScale.bandwidth() / 2)
      .attr('y', d => yScale(d.row)! + yScale.bandwidth() / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', d => Math.abs(d.value) > textThreshold ? 'white' : 'none')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .text(d => Math.abs(d.value) > textThreshold ? d.value.toFixed(2) : '');

    // Add x axis labels
    g.selectAll('.x-label')
      .data(data.features)
      .enter().append('text')
      .attr('class', 'x-label')
      .attr('x', d => xScale(d)! + xScale.bandwidth() / 2)
      .attr('y', -10)
      .attr('text-anchor', 'end')
      .attr('fill', 'rgba(255, 255, 255, 0.7)')
      .attr('font-size', '11px')
      .attr('transform', d => `rotate(-45, ${xScale(d)! + xScale.bandwidth() / 2}, -10)`)
      .text(d => d.length > 15 ? d.substring(0, 15) + '...' : d);

    // Add y axis labels
    g.selectAll('.y-label')
      .data(data.features)
      .enter().append('text')
      .attr('class', 'y-label')
      .attr('x', -10)
      .attr('y', d => yScale(d)! + yScale.bandwidth() / 2)
      .attr('text-anchor', 'end')
      .attr('fill', 'rgba(255, 255, 255, 0.7)')
      .attr('font-size', '11px')
      .attr('dominant-baseline', 'middle')
      .text(d => d.length > 15 ? d.substring(0, 15) + '...' : d);

    // Add color legend
    const legendWidth = 200;
    const legendHeight = 20;
    
    const legendScale = d3.scaleLinear()
      .domain([-1, 1])
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d3.format('.1f'));

    const legend = svg.append('g')
      .attr('transform', `translate(${width + margin.left - legendWidth}, ${margin.top - 50})`);

    // Create gradient for legend
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'correlation-gradient')
      .attr('x1', '0%')
      .attr('x2', '100%')
      .attr('y1', '0%')
      .attr('y2', '0%');

    const steps = 20;
    for (let i = 0; i <= steps; i++) {
      const value = -1 + (2 * i / steps);
      gradient.append('stop')
        .attr('offset', `${(i / steps) * 100}%`)
        .attr('stop-color', colorScale(value));
    }

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#correlation-gradient)');

    legend.append('g')
      .attr('transform', `translate(0, ${legendHeight})`)
      .call(legendAxis)
      .selectAll('text')
      .attr('fill', 'rgba(255, 255, 255, 0.7)');

    legend.append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -5)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255, 255, 255, 0.9)')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .text('Correlation Coefficient');

    // Cleanup
    return () => {
      tooltip.remove();
    };
  }, [data]);

  return (
    <Card className="p-6 bg-gray-800/30 border-gray-700/50">
      <h3 className="text-lg font-medium text-white mb-4">Feature Correlation Matrix</h3>
      <div className="flex justify-center">
        <svg ref={svgRef}></svg>
      </div>
    </Card>
  );
};