import React, { useMemo, useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { motion, AnimatePresence } from 'framer-motion';

interface ModelMetrics {
  name: string;
  type: 'GNN' | 'Transformer' | 'Hybrid' | 'Ensemble';
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  latencyP50: number;
  latencyP95: number;
  latencyP99: number;
  confusionMatrix: {
    truePositive: number;
    falsePositive: number;
    trueNegative: number;
    falseNegative: number;
  };
  rocCurve: { fpr: number; tpr: number }[];
  predictions: { timestamp: number; predicted: boolean; actual: boolean }[];
}

interface DetectionModelsProps {
  models: ModelMetrics[];
  selectedModel?: string;
  onModelSelect?: (modelName: string) => void;
}

export const DetectionModels: React.FC<DetectionModelsProps> = ({
  models,
  selectedModel,
  onModelSelect
}) => {
  const rocRef = useRef<SVGSVGElement>(null);
  const [animatedMetrics, setAnimatedMetrics] = useState<{ [key: string]: number }>({});
  const [votingSimulation, setVotingSimulation] = useState<{ [key: string]: boolean }>({});

  // Animate metrics on mount
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimatedMetrics(prev => {
        const next = { ...prev };
        models.forEach(model => {
          if (!next[model.name] || next[model.name] < model.accuracy) {
            next[model.name] = Math.min((next[model.name] || 0) + 2, model.accuracy);
          }
        });
        return next;
      });
    }, 20);

    return () => clearInterval(interval);
  }, [models]);

  // Simulate ensemble voting
  useEffect(() => {
    const interval = setInterval(() => {
      const votes: { [key: string]: boolean } = {};
      models.forEach(model => {
        votes[model.name] = Math.random() > 0.3; // Simulate detection
      });
      setVotingSimulation(votes);
    }, 2000);

    return () => clearInterval(interval);
  }, [models]);

  // Draw ROC curves
  useEffect(() => {
    if (!rocRef.current || models.length === 0) return;

    const svg = d3.select(rocRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 500 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, width]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([height, 0]);

    // Grid
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      .call(
        d3.axisBottom(xScale)
          .tickSize(-height)
          .tickFormat(() => '')
      )
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    g.append('g')
      .attr('class', 'grid')
      .call(
        d3.axisLeft(yScale)
          .tickSize(-width)
          .tickFormat(() => '')
      )
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .style('color', '#666');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .style('color', '#666');

    // Axis labels
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - height / 2)
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('fill', '#888')
      .style('font-size', '12px')
      .text('True Positive Rate');

    g.append('text')
      .attr('transform', `translate(${width / 2}, ${height + margin.bottom})`)
      .style('text-anchor', 'middle')
      .style('fill', '#888')
      .style('font-size', '12px')
      .text('False Positive Rate');

    // Diagonal reference line
    g.append('line')
      .attr('x1', xScale(0))
      .attr('y1', yScale(0))
      .attr('x2', xScale(1))
      .attr('y2', yScale(1))
      .style('stroke', '#444')
      .style('stroke-dasharray', '5,5')
      .style('opacity', 0.5);

    // Color scale
    const colorScale = d3.scaleOrdinal()
      .domain(models.map(m => m.name))
      .range(['#00ff88', '#ff00ff', '#00ffff', '#ffff00']);

    // Draw ROC curves
    const line = d3.line<{ fpr: number; tpr: number }>()
      .x(d => xScale(d.fpr))
      .y(d => yScale(d.tpr))
      .curve(d3.curveMonotoneX);

    models.forEach(model => {
      const color = colorScale(model.name) as string;
      
      // ROC curve path
      const path = g.append('path')
        .datum(model.rocCurve)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', selectedModel === model.name ? 3 : 2)
        .attr('opacity', selectedModel === model.name ? 1 : 0.7)
        .attr('d', line)
        .style('cursor', 'pointer')
        .on('click', () => onModelSelect?.(model.name));

      // Animate path drawing
      const totalLength = (path.node() as SVGPathElement).getTotalLength();
      path
        .attr('stroke-dasharray', `${totalLength} ${totalLength}`)
        .attr('stroke-dashoffset', totalLength)
        .transition()
        .duration(2000)
        .ease(d3.easeLinear)
        .attr('stroke-dashoffset', 0);

      // AUC area fill
      g.append('path')
        .datum(model.rocCurve)
        .attr('fill', color)
        .attr('opacity', 0.1)
        .attr('d', d3.area<{ fpr: number; tpr: number }>()
          .x(d => xScale(d.fpr))
          .y0(height)
          .y1(d => yScale(d.tpr))
          .curve(d3.curveMonotoneX)
        );
    });

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${width - 120}, 20)`);

    models.forEach((model, i) => {
      const color = colorScale(model.name) as string;
      
      legend.append('rect')
        .attr('x', 0)
        .attr('y', i * 20)
        .attr('width', 10)
        .attr('height', 10)
        .style('fill', color);

      legend.append('text')
        .attr('x', 15)
        .attr('y', i * 20 + 9)
        .style('font-size', '11px')
        .style('fill', '#888')
        .text(`${model.name} (${model.auc.toFixed(3)})`);
    });
  }, [models, selectedModel, onModelSelect]);

  // Calculate ensemble prediction
  const ensemblePrediction = useMemo(() => {
    const votes = Object.values(votingSimulation).filter(v => v).length;
    const threshold = models.length / 2;
    return votes > threshold;
  }, [votingSimulation, models]);

  return (
    <div className="space-y-6">
      {/* Model Performance Overview */}
      <div className="grid grid-cols-4 gap-4">
        {models.map(model => (
          <motion.div
            key={model.name}
            className={`glass rounded-xl p-4 cursor-pointer transition-all ${
              selectedModel === model.name ? 'ring-2 ring-cyan-500' : ''
            }`}
            onClick={() => onModelSelect?.(model.name)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-zinc-100">{model.name}</h4>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                model.type === 'GNN' ? 'bg-green-500/20 text-green-400' :
                model.type === 'Transformer' ? 'bg-blue-500/20 text-blue-400' :
                model.type === 'Hybrid' ? 'bg-purple-500/20 text-purple-400' :
                'bg-yellow-500/20 text-yellow-400'
              }`}>
                {model.type}
              </span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-zinc-500">Accuracy</span>
                <span className="text-zinc-300">
                  {(animatedMetrics[model.name] || 0).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-cyan-500 to-purple-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${animatedMetrics[model.name] || 0}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* ROC Curves */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          ROC Curves & AUC Scores
        </h3>
        <div className="flex justify-center">
          <svg ref={rocRef}></svg>
        </div>
      </div>

      {/* Confusion Matrices */}
      <div className="grid grid-cols-2 gap-4">
        {models.filter(m => selectedModel ? m.name === selectedModel : true).map(model => (
          <div key={model.name} className="glass rounded-xl p-4">
            <h4 className="font-semibold text-zinc-100 mb-3">{model.name} Confusion Matrix</h4>
            <div className="grid grid-cols-2 gap-2">
              <div className="text-center">
                <div className="bg-green-500/20 rounded-lg p-4">
                  <div className="text-2xl font-bold text-green-400">
                    {model.confusionMatrix.truePositive}
                  </div>
                  <div className="text-xs text-zinc-500 mt-1">True Positive</div>
                </div>
              </div>
              <div className="text-center">
                <div className="bg-red-500/20 rounded-lg p-4">
                  <div className="text-2xl font-bold text-red-400">
                    {model.confusionMatrix.falsePositive}
                  </div>
                  <div className="text-xs text-zinc-500 mt-1">False Positive</div>
                </div>
              </div>
              <div className="text-center">
                <div className="bg-yellow-500/20 rounded-lg p-4">
                  <div className="text-2xl font-bold text-yellow-400">
                    {model.confusionMatrix.falseNegative}
                  </div>
                  <div className="text-xs text-zinc-500 mt-1">False Negative</div>
                </div>
              </div>
              <div className="text-center">
                <div className="bg-blue-500/20 rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-400">
                    {model.confusionMatrix.trueNegative}
                  </div>
                  <div className="text-xs text-zinc-500 mt-1">True Negative</div>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2 mt-3">
              <div className="text-center">
                <div className="text-xs text-zinc-500">Precision</div>
                <div className="text-sm font-semibold text-zinc-300">
                  {(model.precision * 100).toFixed(1)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-zinc-500">Recall</div>
                <div className="text-sm font-semibold text-zinc-300">
                  {(model.recall * 100).toFixed(1)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-zinc-500">F1 Score</div>
                <div className="text-sm font-semibold text-zinc-300">
                  {(model.f1Score * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Latency Histograms */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          Model Inference Latency
        </h3>
        <div className="space-y-4">
          {models.map(model => (
            <div key={model.name} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-medium text-zinc-300">{model.name}</span>
                <div className="flex gap-4 text-sm">
                  <span className="text-green-400">P50: {model.latencyP50}μs</span>
                  <span className="text-yellow-400">P95: {model.latencyP95}μs</span>
                  <span className="text-red-400">P99: {model.latencyP99}μs</span>
                </div>
              </div>
              <div className="h-8 bg-zinc-900 rounded-lg relative overflow-hidden">
                <motion.div
                  className="absolute left-0 top-0 bottom-0 bg-green-500/30"
                  initial={{ width: 0 }}
                  animate={{ width: `${(model.latencyP50 / 200) * 100}%` }}
                  transition={{ duration: 1 }}
                />
                <motion.div
                  className="absolute left-0 top-0 bottom-0 bg-yellow-500/30"
                  initial={{ width: 0 }}
                  animate={{ width: `${(model.latencyP95 / 200) * 100}%` }}
                  transition={{ duration: 1, delay: 0.2 }}
                />
                <motion.div
                  className="absolute left-0 top-0 bottom-0 bg-red-500/30"
                  initial={{ width: 0 }}
                  animate={{ width: `${(model.latencyP99 / 200) * 100}%` }}
                  transition={{ duration: 1, delay: 0.4 }}
                />
                <div className="absolute inset-0 flex items-center justify-center text-xs text-zinc-400">
                  Target: ≤100μs
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Ensemble Voting Visualization */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">
          Model Ensemble Voting (Live)
        </h3>
        <div className="grid grid-cols-5 gap-4">
          {models.map(model => (
            <div key={model.name} className="text-center">
              <div className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center transition-all ${
                votingSimulation[model.name]
                  ? 'bg-green-500/20 ring-2 ring-green-500'
                  : 'bg-zinc-800'
              }`}>
                <span className="text-2xl">
                  {votingSimulation[model.name] ? '✓' : '✗'}
                </span>
              </div>
              <div className="text-xs text-zinc-500 mt-2">{model.name}</div>
            </div>
          ))}
          <div className="text-center">
            <div className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center transition-all ${
              ensemblePrediction
                ? 'bg-gradient-to-r from-cyan-500 to-purple-500 ring-2 ring-white'
                : 'bg-zinc-800'
            }`}>
              <span className="text-2xl font-bold">
                {ensemblePrediction ? 'MEV' : 'OK'}
              </span>
            </div>
            <div className="text-xs text-zinc-500 mt-2">Ensemble</div>
          </div>
        </div>
        <div className="mt-4 text-center text-sm text-zinc-400">
          Threshold: {Math.ceil(models.length / 2)}/{models.length} votes for positive detection
        </div>
      </div>
    </div>
  );
};