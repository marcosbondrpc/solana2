'use client';

import React, { useMemo, useState } from 'react';
import { useMEVStore, selectProfitTrend } from '@/stores/mev-store';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Progress } from '@/components/ui/Progress';
import ReactECharts from 'echarts-for-react';
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  Percent,
  BarChart3,
  PieChart,
  Activity,
  Target,
  Zap,
  Award,
  AlertTriangle,
  ChevronUp,
  ChevronDown
} from 'lucide-react';
import { motion } from 'framer-motion';

interface MetricCard {
  label: string;
  value: number;
  change: number;
  format: 'currency' | 'percent' | 'number';
  icon: React.ElementType;
  color: string;
}

const ProfitMetricCard: React.FC<MetricCard> = ({ label, value, change, format, icon: Icon, color }) => {
  const formatValue = (val: number) => {
    switch (format) {
      case 'currency':
        return `$${val.toFixed(2)}`;
      case 'percent':
        return `${val.toFixed(1)}%`;
      default:
        return val.toLocaleString();
    }
  };
  
  const isPositive = change >= 0;
  const ChangeIcon = isPositive ? ChevronUp : ChevronDown;
  const changeColor = isPositive ? 'text-green-500' : 'text-red-500';
  
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.2 }}
    >
      <Card className="p-4 hover:shadow-lg transition-all duration-200">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="text-sm text-gray-500 mb-1">{label}</div>
            <div className="text-2xl font-bold mb-2">{formatValue(value)}</div>
            <div className={`flex items-center gap-1 text-sm ${changeColor}`}>
              <ChangeIcon className="w-3 h-3" />
              <span>{Math.abs(change).toFixed(1)}%</span>
              <span className="text-gray-500">vs last hour</span>
            </div>
          </div>
          <Icon className={`w-8 h-8 ${color}`} />
        </div>
      </Card>
    </motion.div>
  );
};

export const ProfitDashboard: React.FC = () => {
  const { profitMetrics } = useMEVStore();
  const profitTrend = useMEVStore(selectProfitTrend);
  const [timeframe, setTimeframe] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [selectedDex, setSelectedDex] = useState<string>('all');
  
  // Calculate period changes
  const hourlyChange = ((profitMetrics.hourlyProfit - (profitMetrics.hourlyProfit * 0.9)) / 
    (profitMetrics.hourlyProfit * 0.9)) * 100;
  const dailyChange = ((profitMetrics.dailyProfit - (profitMetrics.dailyProfit * 0.85)) / 
    (profitMetrics.dailyProfit * 0.85)) * 100;
  
  const metrics: MetricCard[] = [
    {
      label: 'Total Profit',
      value: profitMetrics.totalProfit,
      change: 15.3,
      format: 'currency',
      icon: DollarSign,
      color: 'text-green-500'
    },
    {
      label: 'Net Profit',
      value: profitMetrics.netProfit,
      change: 12.8,
      format: 'currency',
      icon: TrendingUp,
      color: 'text-blue-500'
    },
    {
      label: 'ROI',
      value: profitMetrics.roi * 100,
      change: 8.5,
      format: 'percent',
      icon: Percent,
      color: 'text-purple-500'
    },
    {
      label: 'Success Rate',
      value: profitMetrics.successRate * 100,
      change: 3.2,
      format: 'percent',
      icon: Target,
      color: 'text-cyan-500'
    }
  ];
  
  // Profit trend chart
  const profitTrendOptions = useMemo(() => ({
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const data = params[0];
        return `${data.name}<br/>Profit: $${parseFloat(data.value).toFixed(2)}`;
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '10%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: profitTrend.map((_, i) => {
        const time = new Date(Date.now() - (profitTrend.length - i) * 60000);
        return time.toLocaleTimeString('en-US', { 
          hour: '2-digit', 
          minute: '2-digit',
          hour12: false 
        });
      }),
      axisLabel: {
        rotate: 45,
        fontSize: 10
      }
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '${value}'
      }
    },
    series: [{
      type: 'line',
      smooth: true,
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(16, 185, 129, 0.3)' },
            { offset: 1, color: 'rgba(16, 185, 129, 0.05)' }
          ]
        }
      },
      lineStyle: {
        color: '#10b981',
        width: 2
      },
      data: profitTrend.map(p => p.y.toFixed(2))
    }]
  }), [profitTrend]);
  
  // DEX profit distribution
  const dexProfitOptions = useMemo(() => {
    const dexData = Array.from(profitMetrics.profitByDex.entries())
      .map(([dex, profit]) => ({ name: dex, value: profit }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 6);
    
    return {
      tooltip: {
        trigger: 'item',
        formatter: '{b}: ${c} ({d}%)'
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        top: 'center'
      },
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['60%', '50%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 10,
          borderColor: '#fff',
          borderWidth: 2
        },
        label: {
          show: false,
          position: 'center'
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 16,
            fontWeight: 'bold'
          }
        },
        labelLine: {
          show: false
        },
        data: dexData
      }]
    };
  }, [profitMetrics.profitByDex]);
  
  // Token profit distribution
  const tokenProfitOptions = useMemo(() => {
    const tokenData = Array.from(profitMetrics.profitByToken.entries())
      .map(([token, profit]) => ({ token, profit }))
      .sort((a, b) => b.profit - a.profit)
      .slice(0, 10);
    
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'value',
        axisLabel: {
          formatter: '${value}'
        }
      },
      yAxis: {
        type: 'category',
        data: tokenData.map(t => t.token),
        axisLabel: {
          fontSize: 10
        }
      },
      series: [{
        type: 'bar',
        data: tokenData.map(t => ({
          value: t.profit.toFixed(2),
          itemStyle: {
            color: t.profit > 0 ? '#10b981' : '#ef4444'
          }
        })),
        barMaxWidth: 20
      }]
    };
  }, [profitMetrics.profitByToken]);
  
  // Performance indicators
  const performanceIndicators = [
    {
      label: 'Avg Profit/Trade',
      value: profitMetrics.avgProfit,
      target: 0.5,
      unit: '$'
    },
    {
      label: 'Max Profit',
      value: profitMetrics.maxProfit,
      target: 10,
      unit: '$'
    },
    {
      label: 'Gas Efficiency',
      value: profitMetrics.gasSpent > 0 ? 
        (profitMetrics.netProfit / profitMetrics.gasSpent) * 100 : 0,
      target: 150,
      unit: '%'
    }
  ];
  
  return (
    <div className="space-y-4">
      {/* Metric Cards */}
      <div className="grid grid-cols-4 gap-4">
        {metrics.map((metric) => (
          <ProfitMetricCard key={metric.label} {...metric} />
        ))}
      </div>
      
      {/* Time Period Stats */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-500">Hourly Profit</span>
            <Activity className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-2xl font-bold">${profitMetrics.hourlyProfit.toFixed(2)}</div>
          <div className={`flex items-center gap-1 text-sm mt-2 ${hourlyChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {hourlyChange >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {Math.abs(hourlyChange).toFixed(1)}% from prev hour
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-500">Daily Profit</span>
            <BarChart3 className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-2xl font-bold">${profitMetrics.dailyProfit.toFixed(2)}</div>
          <div className={`flex items-center gap-1 text-sm mt-2 ${dailyChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {dailyChange >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {Math.abs(dailyChange).toFixed(1)}% from yesterday
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-500">Gas Spent</span>
            <Zap className="w-4 h-4 text-yellow-500" />
          </div>
          <div className="text-2xl font-bold">${profitMetrics.gasSpent.toFixed(2)}</div>
          <div className="text-sm text-gray-500 mt-2">
            {((profitMetrics.gasSpent / profitMetrics.totalProfit) * 100).toFixed(1)}% of gross profit
          </div>
        </Card>
      </div>
      
      {/* Performance Indicators */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Award className="w-5 h-5" />
          Performance Indicators
        </h3>
        <div className="grid grid-cols-3 gap-6">
          {performanceIndicators.map((indicator) => (
            <div key={indicator.label}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-500">{indicator.label}</span>
                <span className="text-sm font-bold">
                  {indicator.unit === '$' && '$'}
                  {indicator.value.toFixed(2)}
                  {indicator.unit === '%' && '%'}
                </span>
              </div>
              <Progress 
                value={(indicator.value / indicator.target) * 100} 
                className="h-2"
              />
              <div className="text-xs text-gray-500 mt-1">
                Target: {indicator.unit === '$' && '$'}{indicator.target}{indicator.unit === '%' && '%'}
              </div>
            </div>
          ))}
        </div>
      </Card>
      
      {/* Profit Trend Chart */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Cumulative Profit Trend
          </h3>
          <div className="flex gap-1">
            {(['1h', '24h', '7d', '30d'] as const).map(tf => (
              <Button
                key={tf}
                size="sm"
                variant={timeframe === tf ? 'default' : 'outline'}
                onClick={() => setTimeframe(tf)}
              >
                {tf}
              </Button>
            ))}
          </div>
        </div>
        <ReactECharts option={profitTrendOptions} style={{ height: '300px' }} />
      </Card>
      
      {/* Profit Distribution */}
      <div className="grid grid-cols-2 gap-4">
        <Card className="p-4">
          <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
            <PieChart className="w-4 h-4" />
            Profit by DEX
          </h3>
          <ReactECharts option={dexProfitOptions} style={{ height: '300px' }} />
        </Card>
        
        <Card className="p-4">
          <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Top Profitable Tokens
          </h3>
          <ReactECharts option={tokenProfitOptions} style={{ height: '300px' }} />
        </Card>
      </div>
      
      {/* Risk Metrics */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          Risk Metrics
        </h3>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <div className="text-sm text-gray-500">Min Profit</div>
            <div className={`text-xl font-bold ${profitMetrics.minProfit < 0 ? 'text-red-500' : 'text-green-500'}`}>
              ${profitMetrics.minProfit.toFixed(2)}
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Profit Variance</div>
            <div className="text-xl font-bold">
              {((profitMetrics.maxProfit - profitMetrics.minProfit) / 2).toFixed(2)}
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Loss Trades</div>
            <div className="text-xl font-bold text-red-500">
              {(100 - profitMetrics.successRate * 100).toFixed(0)}%
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Risk Score</div>
            <Badge 
              variant={profitMetrics.successRate > 0.7 ? 'default' : 'destructive'}
              className="mt-1"
            >
              {profitMetrics.successRate > 0.7 ? 'LOW' : 
               profitMetrics.successRate > 0.5 ? 'MEDIUM' : 'HIGH'}
            </Badge>
          </div>
        </div>
      </Card>
    </div>
  );
};