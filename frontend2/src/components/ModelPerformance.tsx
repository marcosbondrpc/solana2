import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import { LineChart, HeatmapChart, ScatterChart } from 'echarts/charts';
import {
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  VisualMapComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { theme } from '../theme';
import { ModelMetrics } from '../services/websocket';

echarts.use([
  LineChart,
  HeatmapChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

interface Props {
  metrics: ModelMetrics[];
  historicalData?: {
    timestamp: number;
    metrics: ModelMetrics;
  }[];
}

const ModelPerformance: React.FC<Props> = ({ metrics, historicalData = [] }) => {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'roc' | 'confusion' | 'latency' | 'ensemble'>('roc');

  // ROC Curve data
  const rocCurveOption = useMemo(() => {
    const selectedMetrics = selectedModel 
      ? metrics.filter(m => m.layerId === selectedModel)
      : metrics;

    const rocData = selectedMetrics.map(m => {
      const fpr = m.confusionMatrix.fp / (m.confusionMatrix.fp + m.confusionMatrix.tn);
      const tpr = m.confusionMatrix.tp / (m.confusionMatrix.tp + m.confusionMatrix.fn);
      return [fpr, tpr, m.layerId];
    });

    return {
      backgroundColor: 'transparent',
      title: {
        text: 'ROC Curves',
        textStyle: {
          color: theme.colors.text.primary,
          fontSize: 16,
        },
      },
      tooltip: {
        trigger: 'item',
        backgroundColor: theme.colors.bg.glass,
        borderColor: theme.colors.border.primary,
        textStyle: {
          color: theme.colors.text.primary,
        },
        formatter: (params: any) => {
          return `${params.value[2]}<br/>FPR: ${params.value[0].toFixed(3)}<br/>TPR: ${params.value[1].toFixed(3)}`;
        },
      },
      xAxis: {
        name: 'False Positive Rate',
        nameTextStyle: {
          color: theme.colors.text.secondary,
        },
        axisLine: {
          lineStyle: {
            color: theme.colors.border.glass,
          },
        },
        splitLine: {
          lineStyle: {
            color: theme.colors.border.glass,
            opacity: 0.3,
          },
        },
      },
      yAxis: {
        name: 'True Positive Rate',
        nameTextStyle: {
          color: theme.colors.text.secondary,
        },
        axisLine: {
          lineStyle: {
            color: theme.colors.border.glass,
          },
        },
        splitLine: {
          lineStyle: {
            color: theme.colors.border.glass,
            opacity: 0.3,
          },
        },
      },
      series: [
        {
          type: 'line',
          data: [[0, 0], [1, 1]],
          lineStyle: {
            color: theme.colors.text.muted,
            type: 'dashed',
          },
          silent: true,
        },
        ...metrics.map((m, idx) => ({
          type: 'line',
          name: m.layerId,
          data: rocData.filter(d => d[2] === m.layerId),
          smooth: true,
          lineStyle: {
            width: 2,
            color: idx === 0 ? theme.colors.primary : 
                   idx === 1 ? theme.colors.secondary :
                   idx === 2 ? theme.colors.warning : theme.colors.danger,
            shadowBlur: 10,
            shadowColor: idx === 0 ? theme.colors.primary : 
                        idx === 1 ? theme.colors.secondary :
                        idx === 2 ? theme.colors.warning : theme.colors.danger,
          },
          areaStyle: {
            opacity: 0.1,
          },
        })),
      ],
    };
  }, [metrics, selectedModel]);

  // Confusion Matrix Heatmap
  const confusionMatrixOption = useMemo(() => {
    const selectedMetric = selectedModel 
      ? metrics.find(m => m.layerId === selectedModel)
      : metrics[0];

    if (!selectedMetric) return {};

    const matrix = selectedMetric.confusionMatrix;
    const data = [
      [0, 0, matrix.tn],
      [0, 1, matrix.fp],
      [1, 0, matrix.fn],
      [1, 1, matrix.tp],
    ];

    return {
      backgroundColor: 'transparent',
      title: {
        text: `Confusion Matrix - ${selectedMetric.layerId}`,
        textStyle: {
          color: theme.colors.text.primary,
          fontSize: 16,
        },
      },
      tooltip: {
        position: 'top',
        backgroundColor: theme.colors.bg.glass,
        borderColor: theme.colors.border.primary,
      },
      grid: {
        height: '50%',
        top: '20%',
      },
      xAxis: {
        type: 'category',
        data: ['Predicted Negative', 'Predicted Positive'],
        splitArea: {
          show: true,
        },
        axisLabel: {
          color: theme.colors.text.secondary,
        },
      },
      yAxis: {
        type: 'category',
        data: ['Actual Negative', 'Actual Positive'],
        splitArea: {
          show: true,
        },
        axisLabel: {
          color: theme.colors.text.secondary,
        },
      },
      visualMap: {
        min: 0,
        max: Math.max(matrix.tp, matrix.tn, matrix.fp, matrix.fn),
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '5%',
        inRange: {
          color: ['#0a0a0f', theme.colors.primary],
        },
        textStyle: {
          color: theme.colors.text.secondary,
        },
      },
      series: [{
        name: 'Confusion Matrix',
        type: 'heatmap',
        data: data,
        label: {
          show: true,
          color: theme.colors.text.primary,
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: theme.colors.primary,
          },
        },
      }],
    };
  }, [metrics, selectedModel]);

  // Latency Distribution
  const latencyOption = useMemo(() => {
    const latencyData = metrics.map(m => ({
      name: m.layerId,
      p50: m.latencyP50,
      p95: m.latencyP95,
      p99: m.latencyP99,
    }));

    return {
      backgroundColor: 'transparent',
      title: {
        text: 'Model Latency Distribution',
        textStyle: {
          color: theme.colors.text.primary,
          fontSize: 16,
        },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: theme.colors.bg.glass,
        borderColor: theme.colors.border.primary,
        textStyle: {
          color: theme.colors.text.primary,
        },
      },
      legend: {
        data: ['P50', 'P95', 'P99'],
        textStyle: {
          color: theme.colors.text.secondary,
        },
      },
      xAxis: {
        type: 'category',
        data: latencyData.map(d => d.name),
        axisLabel: {
          color: theme.colors.text.secondary,
          rotate: 45,
        },
      },
      yAxis: {
        type: 'value',
        name: 'Latency (ms)',
        nameTextStyle: {
          color: theme.colors.text.secondary,
        },
        axisLabel: {
          color: theme.colors.text.secondary,
        },
        splitLine: {
          lineStyle: {
            color: theme.colors.border.glass,
            opacity: 0.3,
          },
        },
      },
      series: [
        {
          name: 'P50',
          type: 'bar',
          data: latencyData.map(d => d.p50),
          itemStyle: {
            color: theme.colors.primary,
          },
        },
        {
          name: 'P95',
          type: 'bar',
          data: latencyData.map(d => d.p95),
          itemStyle: {
            color: theme.colors.secondary,
          },
        },
        {
          name: 'P99',
          type: 'bar',
          data: latencyData.map(d => d.p99),
          itemStyle: {
            color: theme.colors.warning,
          },
        },
      ],
    };
  }, [metrics]);

  // Ensemble Voting Visualization
  const ensembleOption = useMemo(() => {
    const voteData = metrics.map(m => ({
      value: m.accuracy * 100,
      name: m.layerId,
    }));

    return {
      backgroundColor: 'transparent',
      title: {
        text: 'Model Ensemble Voting Weight',
        textStyle: {
          color: theme.colors.text.primary,
          fontSize: 16,
        },
      },
      tooltip: {
        trigger: 'item',
        backgroundColor: theme.colors.bg.glass,
        borderColor: theme.colors.border.primary,
        formatter: '{b}: {c}% accuracy',
      },
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 10,
          borderColor: theme.colors.bg.primary,
          borderWidth: 2,
        },
        label: {
          show: true,
          position: 'outside',
          color: theme.colors.text.secondary,
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 16,
            fontWeight: 'bold',
            color: theme.colors.text.primary,
          },
          itemStyle: {
            shadowBlur: 20,
            shadowColor: theme.colors.primary,
          },
        },
        labelLine: {
          show: true,
          lineStyle: {
            color: theme.colors.border.glass,
          },
        },
        data: voteData,
        color: [
          theme.colors.primary,
          theme.colors.secondary,
          theme.colors.warning,
          theme.colors.danger,
        ],
      }],
    };
  }, [metrics]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      style={{
        background: theme.colors.bg.secondary,
        borderRadius: theme.borderRadius.lg,
        padding: theme.spacing.lg,
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
        }}>
          Model Performance Metrics
        </h2>

        <div style={{ display: 'flex', gap: theme.spacing.sm }}>
          {['roc', 'confusion', 'latency', 'ensemble'].map(mode => (
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
              {mode}
            </motion.button>
          ))}
        </div>
      </div>

      {/* Metrics Summary Cards */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: theme.spacing.md,
        marginBottom: theme.spacing.lg,
      }}>
        {metrics.map((metric) => (
          <motion.div
            key={metric.layerId}
            whileHover={{ scale: 1.02 }}
            onClick={() => setSelectedModel(metric.layerId)}
            style={{
              background: selectedModel === metric.layerId 
                ? `linear-gradient(135deg, ${theme.colors.primary}22, ${theme.colors.secondary}22)`
                : theme.colors.bg.tertiary,
              borderRadius: theme.borderRadius.md,
              padding: theme.spacing.md,
              border: selectedModel === metric.layerId 
                ? `1px solid ${theme.colors.primary}`
                : `1px solid ${theme.colors.border.glass}`,
              cursor: 'pointer',
            }}
          >
            <h3 style={{ 
              color: theme.colors.text.secondary,
              fontSize: theme.fontSize.sm,
              marginBottom: theme.spacing.xs,
            }}>
              {metric.layerId}
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.sm }}>
              <div>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  Accuracy
                </span>
                <p style={{ 
                  color: theme.colors.primary,
                  fontSize: theme.fontSize.lg,
                  fontWeight: 'bold',
                }}>
                  {(metric.accuracy * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  F1 Score
                </span>
                <p style={{ 
                  color: theme.colors.secondary,
                  fontSize: theme.fontSize.lg,
                  fontWeight: 'bold',
                }}>
                  {metric.f1Score.toFixed(3)}
                </p>
              </div>
              <div>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  P50 Latency
                </span>
                <p style={{ 
                  color: metric.latencyP50 < 8 ? theme.colors.primary : theme.colors.warning,
                  fontSize: theme.fontSize.base,
                }}>
                  {metric.latencyP50}ms
                </p>
              </div>
              <div>
                <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.xs }}>
                  ROC AUC
                </span>
                <p style={{ 
                  color: theme.colors.text.primary,
                  fontSize: theme.fontSize.base,
                }}>
                  {metric.rocAuc.toFixed(3)}
                </p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Visualization Area */}
      <AnimatePresence mode="wait">
        <motion.div
          key={viewMode}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          style={{
            background: theme.colors.bg.tertiary,
            borderRadius: theme.borderRadius.md,
            padding: theme.spacing.md,
            height: '400px',
          }}
        >
          {viewMode === 'roc' && (
            <ReactEChartsCore
              echarts={echarts}
              option={rocCurveOption}
              style={{ height: '100%', width: '100%' }}
              theme="dark"
            />
          )}
          {viewMode === 'confusion' && (
            <ReactEChartsCore
              echarts={echarts}
              option={confusionMatrixOption}
              style={{ height: '100%', width: '100%' }}
              theme="dark"
            />
          )}
          {viewMode === 'latency' && (
            <ReactEChartsCore
              echarts={echarts}
              option={latencyOption}
              style={{ height: '100%', width: '100%' }}
              theme="dark"
            />
          )}
          {viewMode === 'ensemble' && (
            <ReactEChartsCore
              echarts={echarts}
              option={ensembleOption}
              style={{ height: '100%', width: '100%' }}
              theme="dark"
            />
          )}
        </motion.div>
      </AnimatePresence>

      {/* Performance Targets Indicator */}
      <div style={{
        marginTop: theme.spacing.lg,
        padding: theme.spacing.md,
        background: theme.colors.bg.tertiary,
        borderRadius: theme.borderRadius.md,
        border: `1px solid ${theme.colors.border.glass}`,
      }}>
        <h3 style={{ 
          color: theme.colors.text.secondary,
          fontSize: theme.fontSize.base,
          marginBottom: theme.spacing.sm,
        }}>
          Performance Targets
        </h3>
        <div style={{ display: 'flex', gap: theme.spacing.xl }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: metrics.some(m => m.latencyP50 <= 8) 
                ? theme.colors.primary 
                : theme.colors.danger,
              boxShadow: metrics.some(m => m.latencyP50 <= 8)
                ? `0 0 10px ${theme.colors.primary}`
                : `0 0 10px ${theme.colors.danger}`,
            }} />
            <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.sm }}>
              P50 ≤ 8ms: {metrics.some(m => m.latencyP50 <= 8) ? 'PASS' : 'FAIL'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: metrics.some(m => m.latencyP99 <= 20) 
                ? theme.colors.primary 
                : theme.colors.danger,
              boxShadow: metrics.some(m => m.latencyP99 <= 20)
                ? `0 0 10px ${theme.colors.primary}`
                : `0 0 10px ${theme.colors.danger}`,
            }} />
            <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.sm }}>
              P99 ≤ 20ms: {metrics.some(m => m.latencyP99 <= 20) ? 'PASS' : 'FAIL'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: metrics.some(m => m.accuracy >= 0.95) 
                ? theme.colors.primary 
                : theme.colors.warning,
              boxShadow: metrics.some(m => m.accuracy >= 0.95)
                ? `0 0 10px ${theme.colors.primary}`
                : `0 0 10px ${theme.colors.warning}`,
            }} />
            <span style={{ color: theme.colors.text.muted, fontSize: theme.fontSize.sm }}>
              Accuracy ≥ 95%: {metrics.some(m => m.accuracy >= 0.95) ? 'PASS' : 'WARN'}
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ModelPerformance;