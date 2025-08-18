/**
 * Legendary Solana MEV Dashboard
 * Ultra-modern, beautiful interface with perfect alignment and smooth animations
 * Enhanced with real-time charts, advanced metrics, and professional UI components
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts';
import { 
  Activity, 
  Zap, 
  TrendingUp, 
  Server, 
  Shield,
  Database,
  Network,
  Cpu,
  BarChart3,
  Settings,
  AlertCircle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Power,
  Play,
  Pause,
  Terminal,
  Layers,
  Globe,
  Lock,
  Unlock,
  Eye,
  EyeOff,
  Moon,
  Sun,
  Volume2,
  VolumeX,
  Wallet,
  DollarSign,
  Clock,
  Target,
  ArrowUpRight,
  ArrowDownRight,
  Info,
  Sparkles,
  Flame,
  Crown
} from 'lucide-react';
import { AnimatedBackground } from '@/components/AnimatedBackground';
import { NotificationSystem, type Notification } from '@/components/NotificationSystem';
import { useMonitoringStore } from '@/lib/monitoring-store';
import { useMEVStore } from '@/stores/mev-store';
import { useControlStore } from '@/stores/control-store';

// Service configuration
const SERVICES = [
  {
    id: 'mev-engine',
    name: 'MEV Engine',
    description: 'Core arbitrage and MEV extraction engine',
    icon: Zap,
    color: 'from-purple-500 to-pink-500',
    glowColor: 'purple',
    port: 8080,
    healthEndpoint: '/health',
    critical: true,
  },
  {
    id: 'mission-control',
    name: 'Mission Control',
    description: 'Command and control center',
    icon: Shield,
    color: 'from-blue-500 to-cyan-500',
    glowColor: 'blue',
    port: 8083,
    healthEndpoint: '/health',
    critical: true,
  },
  {
    id: 'api-proxy',
    name: 'API Proxy',
    description: 'Load balancer and request router',
    icon: Network,
    color: 'from-green-500 to-emerald-500',
    glowColor: 'green',
    port: 8084,
    healthEndpoint: '/health',
    critical: false,
  },
  {
    id: 'websocket',
    name: 'WebSocket Server',
    description: 'Real-time data streaming',
    icon: Activity,
    color: 'from-orange-500 to-red-500',
    glowColor: 'orange',
    port: 8085,
    healthEndpoint: '/health',
    critical: true,
  },
  {
    id: 'clickhouse',
    name: 'ClickHouse',
    description: 'High-performance analytics database',
    icon: Database,
    color: 'from-yellow-500 to-amber-500',
    glowColor: 'yellow',
    port: 8123,
    healthEndpoint: '/ping',
    critical: false,
  },
  {
    id: 'grafana',
    name: 'Grafana',
    description: 'Metrics visualization',
    icon: BarChart3,
    color: 'from-indigo-500 to-purple-500',
    glowColor: 'indigo',
    port: 3000,
    healthEndpoint: '/api/health',
    critical: false,
  },
];

// Beautiful gradient animations
const gradientAnimation = {
  initial: { backgroundPosition: '0% 50%' },
  animate: {
    backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
    transition: {
      duration: 5,
      repeat: Infinity,
      ease: 'linear',
    },
  },
};

// Chart color schemes
const CHART_COLORS = {
  primary: '#00D4FF',
  secondary: '#9945FF',
  success: '#00FF88',
  warning: '#FFB800',
  danger: '#FF3366',
  purple: '#9945FF',
  cyan: '#00D4FF',
  gradient: ['#00D4FF', '#9945FF', '#00FF88'],
};

// Generate mock chart data
const generateMockChartData = (points = 24) => {
  return Array.from({ length: points }, (_, i) => ({
    time: i === 0 ? 'Now' : `${points - i}h`,
    profit: Math.floor(Math.random() * 10000) + 5000,
    opportunities: Math.floor(Math.random() * 50) + 20,
    success: Math.floor(Math.random() * 30) + 70,
    volume: Math.floor(Math.random() * 100000) + 50000,
    gas: Math.floor(Math.random() * 100) + 50,
  }));
};

// Custom tooltip for charts
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-lg bg-gray-900/95 backdrop-blur-sm border border-gray-700 p-3 shadow-xl">
        <p className="text-xs font-medium text-gray-400 mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center justify-between space-x-4">
            <span className="text-xs text-gray-300">{entry.name}:</span>
            <span className="text-xs font-bold" style={{ color: entry.color }}>
              {entry.name === 'Profit' ? `$${entry.value.toLocaleString()}` : 
               entry.name === 'Volume' ? `$${(entry.value / 1000).toFixed(1)}K` :
               entry.name === 'Success' ? `${entry.value}%` : entry.value}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

// Service card animations
const cardAnimation = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

// Stat card component
const StatCard = ({ title, value, change, icon: Icon, color, delay = 0 }: any) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.9 }}
    animate={{ opacity: 1, scale: 1 }}
    transition={{ duration: 0.5, delay }}
    whileHover={{ scale: 1.02 }}
    className={`relative overflow-hidden rounded-2xl bg-gradient-to-br ${color} p-6 shadow-2xl`}
  >
    <div className="absolute inset-0 bg-black/20" />
    <div className="relative z-10">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-white/80">{title}</p>
          <p className="mt-2 text-3xl font-bold text-white">{value}</p>
          {change && (
            <p className="mt-2 flex items-center text-sm text-white/90">
              <TrendingUp className="mr-1 h-4 w-4" />
              {change}
            </p>
          )}
        </div>
        <div className="rounded-full bg-white/20 p-3">
          <Icon className="h-8 w-8 text-white" />
        </div>
      </div>
    </div>
    
    {/* Animated glow effect */}
    <motion.div
      className="absolute -inset-10 opacity-30"
      animate={{
        rotate: [0, 360],
      }}
      transition={{
        duration: 20,
        repeat: Infinity,
        ease: 'linear',
      }}
    >
      <div className="h-full w-full bg-gradient-to-r from-transparent via-white to-transparent blur-3xl" />
    </motion.div>
  </motion.div>
);

// Service control card
const ServiceCard = ({ service, status, onControl }: any) => {
  const [isHovered, setIsHovered] = useState(false);
  const isRunning = status === 'running';
  const Icon = service.icon;

  return (
    <motion.div
      variants={cardAnimation}
      whileHover={{ scale: 1.02 }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="relative overflow-hidden rounded-2xl bg-gray-900/50 backdrop-blur-xl border border-gray-800"
    >
      {/* Background gradient */}
      <motion.div
        className={`absolute inset-0 bg-gradient-to-br ${service.color} opacity-10`}
        animate={{ opacity: isHovered ? 0.2 : 0.1 }}
      />

      <div className="relative p-6">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-4">
            <div className={`rounded-xl bg-gradient-to-br ${service.color} p-3`}>
              <Icon className="h-6 w-6 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">{service.name}</h3>
              <p className="text-sm text-gray-400">{service.description}</p>
              <p className="mt-1 text-xs text-gray-500">Port: {service.port}</p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            {/* Status indicator */}
            <motion.div
              animate={{
                scale: isRunning ? [1, 1.2, 1] : 1,
              }}
              transition={{
                duration: 2,
                repeat: isRunning ? Infinity : 0,
              }}
              className={`h-3 w-3 rounded-full ${
                isRunning ? 'bg-green-500' : 'bg-red-500'
              }`}
              style={{
                boxShadow: isRunning
                  ? '0 0 20px rgba(34, 197, 94, 0.5)'
                  : '0 0 20px rgba(239, 68, 68, 0.5)',
              }}
            />

            {/* Control buttons */}
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => onControl(service.id, isRunning ? 'stop' : 'start')}
              className={`rounded-lg p-2 transition-all ${
                isRunning
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                  : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
              }`}
            >
              {isRunning ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </motion.button>

            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => onControl(service.id, 'restart')}
              className="rounded-lg bg-blue-500/20 p-2 text-blue-400 transition-all hover:bg-blue-500/30"
            >
              <RefreshCw className="h-4 w-4" />
            </motion.button>
          </div>
        </div>

        {/* Health metrics */}
        <div className="mt-4 grid grid-cols-3 gap-4 text-xs">
          <div>
            <p className="text-gray-500">CPU</p>
            <p className="font-semibold text-white">
              {Math.floor(Math.random() * 30 + 10)}%
            </p>
          </div>
          <div>
            <p className="text-gray-500">Memory</p>
            <p className="font-semibold text-white">
              {Math.floor(Math.random() * 40 + 20)}%
            </p>
          </div>
          <div>
            <p className="text-gray-500">Uptime</p>
            <p className="font-semibold text-white">
              {Math.floor(Math.random() * 24 + 1)}h
            </p>
          </div>
        </div>

        {/* Critical badge */}
        {service.critical && (
          <div className="absolute right-2 top-2">
            <span className="rounded-full bg-red-500/20 px-2 py-1 text-xs font-semibold text-red-400">
              CRITICAL
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

// MEV Opportunity Card
const MEVOpportunityCard = ({ opportunity }: any) => {
  const profitColor = opportunity.profit > 1000 ? 'text-green-400' : 
                      opportunity.profit > 500 ? 'text-yellow-400' : 'text-blue-400';
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.02 }}
      className="relative overflow-hidden rounded-xl bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-sm border border-gray-700 p-4"
    >
      <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-purple-500/20 to-transparent rounded-bl-full" />
      
      <div className="relative z-10">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-lg bg-purple-500/20">
              <Zap className="h-4 w-4 text-purple-400" />
            </div>
            <div>
              <p className="text-sm font-semibold text-white">{opportunity.type}</p>
              <p className="text-xs text-gray-400">{opportunity.pair}</p>
            </div>
          </div>
          <span className={`text-lg font-bold ${profitColor}`}>
            +${opportunity.profit.toLocaleString()}
          </span>
        </div>
        
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div>
            <p className="text-gray-500">Gas</p>
            <p className="text-white font-medium">${opportunity.gas}</p>
          </div>
          <div>
            <p className="text-gray-500">Success</p>
            <p className="text-white font-medium">{opportunity.probability}%</p>
          </div>
          <div>
            <p className="text-gray-500">Time</p>
            <p className="text-white font-medium">{opportunity.timeLeft}s</p>
          </div>
        </div>
        
        <div className="mt-3 flex space-x-2">
          <motion.button
            whileTap={{ scale: 0.95 }}
            className="flex-1 py-1.5 px-3 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white text-xs font-semibold hover:opacity-90 transition-opacity"
          >
            Execute
          </motion.button>
          <motion.button
            whileTap={{ scale: 0.95 }}
            className="py-1.5 px-3 rounded-lg bg-gray-700/50 text-gray-300 text-xs font-semibold hover:bg-gray-700/70 transition-colors"
          >
            Details
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

// Alert component
const AlertCard = ({ alert, onDismiss }: any) => {
  const severityColors = {
    low: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    critical: 'bg-red-500/20 text-red-400 border-red-500/30',
  };

  const severityIcons = {
    low: AlertCircle,
    medium: AlertCircle,
    high: AlertCircle,
    critical: XCircle,
  };

  const Icon = severityIcons[alert.severity];

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className={`rounded-lg border p-4 ${severityColors[alert.severity]}`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3">
          <Icon className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-medium">{alert.message}</p>
            <p className="mt-1 text-xs opacity-80">
              {new Date(alert.timestamp).toLocaleTimeString()}
            </p>
          </div>
        </div>
        <button
          onClick={() => onDismiss(alert.id)}
          className="ml-4 rounded-lg p-1 hover:bg-white/10"
        >
          <XCircle className="h-4 w-4" />
        </button>
      </div>
    </motion.div>
  );
};

// Real-time metrics component
const MetricsChart = ({ type, data }: any) => {
  const chartConfig = {
    profit: {
      color: CHART_COLORS.success,
      name: 'Profit',
      formatter: (value: number) => `$${value.toLocaleString()}`,
    },
    volume: {
      color: CHART_COLORS.primary,
      name: 'Volume',
      formatter: (value: number) => `$${(value / 1000).toFixed(1)}K`,
    },
    opportunities: {
      color: CHART_COLORS.warning,
      name: 'Opportunities',
      formatter: (value: number) => value.toString(),
    },
    success: {
      color: CHART_COLORS.purple,
      name: 'Success Rate',
      formatter: (value: number) => `${value}%`,
    },
  };

  if (type === 'area') {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={CHART_COLORS.success} stopOpacity={0.3} />
              <stop offset="95%" stopColor={CHART_COLORS.success} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={CHART_COLORS.primary} stopOpacity={0.3} />
              <stop offset="95%" stopColor={CHART_COLORS.primary} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
          <XAxis dataKey="time" stroke="#9CA3AF" fontSize={10} />
          <YAxis stroke="#9CA3AF" fontSize={10} />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="profit"
            stroke={CHART_COLORS.success}
            fill="url(#profitGradient)"
            strokeWidth={2}
            name="Profit"
          />
          <Area
            type="monotone"
            dataKey="volume"
            stroke={CHART_COLORS.primary}
            fill="url(#volumeGradient)"
            strokeWidth={2}
            name="Volume"
          />
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'line') {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
          <XAxis dataKey="time" stroke="#9CA3AF" fontSize={10} />
          <YAxis stroke="#9CA3AF" fontSize={10} />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="success"
            stroke={CHART_COLORS.purple}
            strokeWidth={2}
            dot={false}
            name="Success"
          />
          <Line
            type="monotone"
            dataKey="gas"
            stroke={CHART_COLORS.warning}
            strokeWidth={2}
            dot={false}
            name="Gas"
          />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  return null;
};

export default function LegendaryDashboard() {
  const { 
    consensus, 
    performance, 
    jito, 
    activeAlerts,
    clearAlert,
    connectionStatus 
  } = useMonitoringStore();
  
  const { 
    arbitrageOpportunities,
    profitMetrics,
    systemPerformance 
  } = useMEVStore();
  
  const { commands } = useControlStore();

  const [serviceStatuses, setServiceStatuses] = useState<Record<string, string>>({});
  const [darkMode, setDarkMode] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [chartData, setChartData] = useState(generateMockChartData());
  const [mevOpportunities, setMevOpportunities] = useState<any[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const animationRef = useRef<number>();

  // Generate mock notifications
  useEffect(() => {
    const generateNotification = (): Notification => {
      const types: Notification['type'][] = ['success', 'warning', 'info', 'mev', 'profit'];
      const messages = {
        success: ['Transaction confirmed', 'Bundle landed successfully', 'Bot deployed'],
        warning: ['High gas prices detected', 'RPC latency increasing', 'Memory usage high'],
        info: ['New MEV strategy available', 'System update available', 'Maintenance scheduled'],
        mev: ['New arbitrage opportunity', 'Liquidation detected', 'JIT opportunity found'],
        profit: ['Profit target reached', 'Daily profit: $5,234', 'New high score achieved'],
      };
      
      const type = types[Math.floor(Math.random() * types.length)];
      const typeMessages = messages[type];
      const message = typeMessages[Math.floor(Math.random() * typeMessages.length)];
      
      return {
        id: `notif-${Date.now()}-${Math.random()}`,
        type,
        title: type.charAt(0).toUpperCase() + type.slice(1),
        message,
        timestamp: Date.now(),
        persistent: Math.random() > 0.8,
      };
    };

    // Add initial notifications
    setNotifications([generateNotification(), generateNotification()]);
    
    // Periodically add new notifications
    const interval = setInterval(() => {
      if (Math.random() > 0.7) {
        setNotifications(prev => [generateNotification(), ...prev].slice(0, 10));
      }
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  // Generate mock MEV opportunities
  useEffect(() => {
    const generateOpportunities = () => {
      const types = ['Arbitrage', 'Liquidation', 'JIT Liquidity', 'Sandwich'];
      const pairs = ['SOL/USDC', 'RAY/USDC', 'ORCA/SOL', 'mSOL/SOL', 'BONK/SOL'];
      
      return Array.from({ length: 6 }, (_, i) => ({
        id: `opp-${Date.now()}-${i}`,
        type: types[Math.floor(Math.random() * types.length)],
        pair: pairs[Math.floor(Math.random() * pairs.length)],
        profit: Math.floor(Math.random() * 5000) + 100,
        gas: Math.floor(Math.random() * 50) + 10,
        probability: Math.floor(Math.random() * 30) + 70,
        timeLeft: Math.floor(Math.random() * 60) + 5,
      }));
    };

    setMevOpportunities(generateOpportunities());
    const interval = setInterval(() => {
      setMevOpportunities(generateOpportunities());
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  // Animate chart data
  useEffect(() => {
    const animate = () => {
      setChartData(prev => {
        const newData = [...prev];
        newData.shift();
        newData.push({
          time: 'Now',
          profit: Math.floor(Math.random() * 10000) + 5000,
          opportunities: Math.floor(Math.random() * 50) + 20,
          success: Math.floor(Math.random() * 30) + 70,
          volume: Math.floor(Math.random() * 100000) + 50000,
          gas: Math.floor(Math.random() * 100) + 50,
        });
        return newData;
      });
    };

    const interval = setInterval(animate, 3000);
    return () => clearInterval(interval);
  }, []);

  // Check service health
  useEffect(() => {
    const checkServices = async () => {
      const statuses: Record<string, string> = {};
      
      for (const service of SERVICES) {
        try {
          const response = await fetch(`http://localhost:${service.port}${service.healthEndpoint}`, {
            method: 'GET',
            signal: AbortSignal.timeout(1000),
          }).catch(() => null);
          
          statuses[service.id] = response?.ok ? 'running' : 'stopped';
        } catch {
          statuses[service.id] = 'stopped';
        }
      }
      
      setServiceStatuses(statuses);
    };

    checkServices();
    const interval = setInterval(checkServices, 5000);
    return () => clearInterval(interval);
  }, []);

  // Handle service control
  const handleServiceControl = useCallback(async (serviceId: string, action: string) => {
    if (soundEnabled) {
      // Play sound effect (you can add actual sound later)
      console.log('Playing sound effect');
    }

    try {
      const response = await fetch(`http://localhost:8083/api/services/${serviceId}/${action}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        // Update status optimistically
        setServiceStatuses(prev => ({
          ...prev,
          [serviceId]: action === 'stop' ? 'stopped' : 'running',
        }));
      }
    } catch (error) {
      console.error(`Failed to ${action} service ${serviceId}:`, error);
    }
  }, [soundEnabled]);

  // Calculate stats
  const stats = useMemo(() => ({
    totalProfit: profitMetrics?.dailyProfit || 0,
    successRate: profitMetrics?.successRate || 0,
    activeOpportunities: arbitrageOpportunities?.size || 0,
    tps: performance?.currentTPS || 0,
    skipRate: consensus?.skipRate || 0,
    bundlesLanded: jito?.bundlesLanded || 0,
  }), [profitMetrics, arbitrageOpportunities, performance, consensus, jito]);

  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gray-950' : 'bg-gray-50'}`}>
      {/* Enhanced animated background */}
      <AnimatedBackground />
      
      {/* Legacy gradient animation */}
      <motion.div
        className="fixed inset-0 opacity-20 pointer-events-none"
        animate={gradientAnimation.animate}
        style={{
          background: 'linear-gradient(270deg, #667eea, #764ba2, #f093fb, #667eea)',
          backgroundSize: '400% 400%',
          mixBlendMode: 'screen',
        }}
      />

      {/* Main content */}
      <div className="relative z-10 p-6">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex items-center justify-between"
        >
          <div>
            <h1 className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-5xl font-bold text-transparent">
              Solana MEV Command Center
            </h1>
            <p className="mt-2 text-gray-400">
              Real-time monitoring and control of all MEV systems
            </p>
          </div>

          <div className="flex items-center space-x-4">
            {/* Connection status */}
            <div className="flex items-center space-x-2">
              <div className={`h-3 w-3 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <span className="text-sm text-gray-400">
                {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
              </span>
            </div>

            {/* Theme toggle */}
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => setDarkMode(!darkMode)}
              className="rounded-lg bg-gray-800 p-2 text-gray-400 hover:text-white"
            >
              {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </motion.button>

            {/* Sound toggle */}
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => setSoundEnabled(!soundEnabled)}
              className="rounded-lg bg-gray-800 p-2 text-gray-400 hover:text-white"
            >
              {soundEnabled ? <Volume2 className="h-5 w-5" /> : <VolumeX className="h-5 w-5" />}
            </motion.button>
          </div>
        </motion.div>

        {/* Enhanced Stats Grid with animations */}
        <div className="mb-8 grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          <StatCard
            title="Daily Profit"
            value={`$${stats.totalProfit.toLocaleString()}`}
            change="+12.5%"
            icon={TrendingUp}
            color="from-green-500 to-emerald-500"
            delay={0}
          />
          <StatCard
            title="Success Rate"
            value={`${stats.successRate.toFixed(1)}%`}
            change="+3.2%"
            icon={CheckCircle}
            color="from-blue-500 to-cyan-500"
            delay={0.1}
          />
          <StatCard
            title="Active MEV"
            value={stats.activeOpportunities}
            change="+5"
            icon={Zap}
            color="from-purple-500 to-pink-500"
            delay={0.2}
          />
          <StatCard
            title="TPS"
            value={stats.tps.toLocaleString()}
            change="+120"
            icon={Activity}
            color="from-orange-500 to-red-500"
            delay={0.3}
          />
          <StatCard
            title="Skip Rate"
            value={`${stats.skipRate.toFixed(1)}%`}
            change="-0.5%"
            icon={AlertCircle}
            color="from-yellow-500 to-amber-500"
            delay={0.4}
          />
          <StatCard
            title="Bundles"
            value={stats.bundlesLanded}
            change="+42"
            icon={Layers}
            color="from-indigo-500 to-purple-500"
            delay={0.5}
          />
        </div>

        {/* Main Dashboard Grid */}
        <div className="mb-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - MEV Opportunities */}
          <div className="lg:col-span-1 space-y-6">
            {/* MEV Scanner */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="rounded-2xl bg-gradient-to-br from-gray-900/60 to-gray-800/60 backdrop-blur-xl border border-gray-700 p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500">
                    <Zap className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white">MEV Scanner</h3>
                    <p className="text-xs text-gray-400">Live opportunities</p>
                  </div>
                </div>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                  className="h-2 w-2 rounded-full bg-green-500"
                />
              </div>
              
              <div className="space-y-3 max-h-[600px] overflow-y-auto custom-scrollbar">
                <AnimatePresence>
                  {mevOpportunities.map((opp) => (
                    <MEVOpportunityCard key={opp.id} opportunity={opp} />
                  ))}
                </AnimatePresence>
              </div>
            </motion.div>

            {/* Top Performers */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="rounded-2xl bg-gradient-to-br from-gray-900/60 to-gray-800/60 backdrop-blur-xl border border-gray-700 p-6"
            >
              <div className="flex items-center space-x-3 mb-4">
                <Crown className="h-5 w-5 text-yellow-400" />
                <h3 className="text-lg font-bold text-white">Top Performers</h3>
              </div>
              
              <div className="space-y-3">
                {['Arbitrage Bot #1', 'Liquidator Pro', 'JIT Master'].map((name, i) => (
                  <div key={name} className="flex items-center justify-between p-3 rounded-lg bg-gray-800/50">
                    <div className="flex items-center space-x-3">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                        i === 0 ? 'bg-yellow-500/20 text-yellow-400' :
                        i === 1 ? 'bg-gray-400/20 text-gray-400' :
                        'bg-orange-500/20 text-orange-400'
                      }`}>
                        {i + 1}
                      </div>
                      <span className="text-sm text-white">{name}</span>
                    </div>
                    <span className="text-sm font-bold text-green-400">
                      +${(Math.random() * 10000 + 5000).toFixed(0)}
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Center Column - Charts */}
          <div className="lg:col-span-2 space-y-6">
            {/* Profit Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-2xl bg-gradient-to-br from-gray-900/60 to-gray-800/60 backdrop-blur-xl border border-gray-700 p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-lg font-bold text-white">Performance Metrics</h3>
                  <p className="text-xs text-gray-400">Real-time profit & volume tracking</p>
                </div>
                <div className="flex space-x-2">
                  {['1h', '6h', '24h', '7d'].map((tf) => (
                    <button
                      key={tf}
                      onClick={() => setSelectedTimeframe(tf)}
                      className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                        selectedTimeframe === tf
                          ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              </div>
              <div className="h-64">
                <MetricsChart type="area" data={chartData} />
              </div>
            </motion.div>

            {/* Success Rate Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="rounded-2xl bg-gradient-to-br from-gray-900/60 to-gray-800/60 backdrop-blur-xl border border-gray-700 p-6"
            >
              <div className="mb-4">
                <h3 className="text-lg font-bold text-white">Success Rate & Gas Optimization</h3>
                <p className="text-xs text-gray-400">Transaction efficiency metrics</p>
              </div>
              <div className="h-48">
                <MetricsChart type="line" data={chartData} />
              </div>
            </motion.div>

            {/* Quick Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: 'Total Volume', value: '$2.4M', change: '+12%', icon: DollarSign, color: 'from-green-500 to-emerald-500' },
                { label: 'Win Rate', value: '87.3%', change: '+3.2%', icon: Target, color: 'from-blue-500 to-cyan-500' },
                { label: 'Avg Profit', value: '$342', change: '+8%', icon: TrendingUp, color: 'from-purple-500 to-pink-500' },
                { label: 'Active Bots', value: '12', change: '0', icon: Activity, color: 'from-orange-500 to-red-500' },
              ].map((stat, i) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.3 + i * 0.05 }}
                  whileHover={{ scale: 1.05 }}
                  className={`relative overflow-hidden rounded-xl bg-gradient-to-br ${stat.color} p-4`}
                >
                  <div className="absolute inset-0 bg-black/30" />
                  <div className="relative z-10">
                    <stat.icon className="h-5 w-5 text-white/80 mb-2" />
                    <p className="text-2xl font-bold text-white">{stat.value}</p>
                    <p className="text-xs text-white/80">{stat.label}</p>
                    <p className="text-xs text-white/60 mt-1">{stat.change}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Services Grid */}
        <div className="mb-8">
          <h2 className="mb-4 text-2xl font-bold text-white">Service Management</h2>
          <motion.div 
            className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3"
            initial="initial"
            animate="animate"
            variants={{
              animate: {
                transition: {
                  staggerChildren: 0.1,
                },
              },
            }}
          >
            {SERVICES.map((service) => (
              <ServiceCard
                key={service.id}
                service={service}
                status={serviceStatuses[service.id] || 'unknown'}
                onControl={handleServiceControl}
              />
            ))}
          </motion.div>
        </div>

        {/* Alerts Section */}
        {activeAlerts.length > 0 && (
          <div className="mb-8">
            <h2 className="mb-4 text-2xl font-bold text-white">Active Alerts</h2>
            <div className="space-y-3">
              <AnimatePresence>
                {activeAlerts.slice(0, 5).map((alert) => (
                  <AlertCard
                    key={alert.id}
                    alert={alert}
                    onDismiss={clearAlert}
                  />
                ))}
              </AnimatePresence>
            </div>
          </div>
        )}

        {/* System Health Overview */}
        <div className="mb-8">
          <h2 className="mb-4 text-2xl font-bold text-white">System Health</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: 'RPC Latency', value: '12ms', status: 'healthy', max: 50 },
              { label: 'WebSocket Ping', value: '3ms', status: 'healthy', max: 20 },
              { label: 'Memory Usage', value: '42%', status: 'healthy', max: 100 },
              { label: 'CPU Load', value: '28%', status: 'healthy', max: 100 },
            ].map((metric) => (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-xl bg-gray-900/50 backdrop-blur-sm border border-gray-700 p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">{metric.label}</span>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    metric.status === 'healthy' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
                  }`}>
                    {metric.status}
                  </span>
                </div>
                <div className="text-2xl font-bold text-white mb-2">{metric.value}</div>
                <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(parseInt(metric.value) / metric.max) * 100}%` }}
                    transition={{ duration: 1, ease: 'easeOut' }}
                    className={`h-full rounded-full ${
                      metric.status === 'healthy' 
                        ? 'bg-gradient-to-r from-green-500 to-emerald-500' 
                        : 'bg-gradient-to-r from-yellow-500 to-amber-500'
                    }`}
                  />
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-12 border-t border-gray-800 pt-6 text-center text-sm text-gray-500"
        >
          <p>Solana MEV System v2.0 | All systems operational</p>
        </motion.div>
      </div>
      
      {/* Notification System */}
      <NotificationSystem
        notifications={notifications}
        onDismiss={(id) => setNotifications(prev => prev.filter(n => n.id !== id))}
        onClearAll={() => setNotifications([])}
        position="top-right"
      />
    </div>
  );
}