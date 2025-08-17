import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { useToast } from '../hooks/use-toast';
import { format, subDays, subHours, subMonths, startOfDay, endOfDay } from 'date-fns';
import { 
  Calendar, 
  Download, 
  Play, 
  Pause, 
  RefreshCw, 
  Database, 
  Activity,
  ArrowRight,
  Layers,
  Zap,
  AlertCircle,
  CheckCircle,
  Clock,
  Filter,
  TrendingUp,
  Archive,
  Settings,
  ChevronDown,
  ChevronUp,
  Cpu,
  HardDrive,
  BarChart3,
  Target
} from 'lucide-react';

// Types for Historical Capture
interface CaptureJob {
  id: string;
  type: 'capture' | 'arbitrage' | 'sandwich';
  status: 'pending' | 'running' | 'completed' | 'error' | 'cancelled';
  progress: number;
  startTime: Date;
  endTime?: Date;
  details: {
    granularity: 'day' | 'month' | 'year';
    dateRange: { start: Date; end: Date };
    programIds?: string[];
    slotsProcessed?: number;
    totalSlots?: number;
    blocksWritten?: number;
    bytesWritten?: number;
    currentSlot?: number;
    estimatedCompletion?: Date;
    errorMessage?: string;
  };
}

interface DatasetStats {
  rawData: {
    bytes: number;
    rows: number;
    partitions: number;
    lastUpdated: Date;
  };
  swaps: {
    rows: number;
    uniquePairs: number;
    totalVolume: number;
  };
  arbitrage: {
    opportunities: number;
    totalProfit: number;
    avgProfit: number;
    successRate: number;
  };
  sandwich: {
    attacks: number;
    totalExtracted: number;
    avgExtracted: number;
    victimCount: number;
  };
}

// Preset date ranges
const DATE_PRESETS = [
  { label: 'Last 24 Hours', value: '24h', getDates: () => ({ start: subHours(new Date(), 24), end: new Date() }) },
  { label: 'Last 7 Days', value: '7d', getDates: () => ({ start: subDays(new Date(), 7), end: new Date() }) },
  { label: 'Last 30 Days', value: '30d', getDates: () => ({ start: subDays(new Date(), 30), end: new Date() }) },
  { label: 'Last 90 Days', value: '90d', getDates: () => ({ start: subDays(new Date(), 90), end: new Date() }) },
  { label: 'Last 6 Months', value: '6m', getDates: () => ({ start: subMonths(new Date(), 6), end: new Date() }) },
];

// DEX Programs
const DEX_PROGRAMS = [
  { id: 'raydium', name: 'Raydium', address: '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8', color: 'from-purple-500 to-blue-500' },
  { id: 'orca', name: 'Orca', address: '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP', color: 'from-blue-500 to-cyan-500' },
  { id: 'phoenix', name: 'Phoenix', address: 'PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY', color: 'from-orange-500 to-red-500' },
  { id: 'meteora', name: 'Meteora', address: 'LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo', color: 'from-green-500 to-teal-500' },
  { id: 'whirlpool', name: 'Whirlpool', address: 'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc', color: 'from-indigo-500 to-purple-500' },
  { id: 'lifinity', name: 'Lifinity', address: 'EewxydAPCCVuNEyrVN68PuSYdQ7wKn27V9Gjeoi8dy3S', color: 'from-pink-500 to-rose-500' },
];

export default function HistoricalCapturePanel() {
  const { toast } = useToast();
  const wsRef = useRef<WebSocket | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isUnmountingRef = useRef(false);

  // State management
  const [granularity, setGranularity] = useState<'day' | 'month' | 'year'>('day');
  const [dateRange, setDateRange] = useState<{ start: Date; end: Date }>({
    start: subDays(new Date(), 7),
    end: new Date(),
  });
  const [selectedPrograms, setSelectedPrograms] = useState<string[]>([]);
  const [jobs, setJobs] = useState<CaptureJob[]>([]);
  const [stats, setStats] = useState<DatasetStats>({
    rawData: { bytes: 0, rows: 0, partitions: 0, lastUpdated: new Date() },
    swaps: { rows: 0, uniquePairs: 0, totalVolume: 0 },
    arbitrage: { opportunities: 0, totalProfit: 0, avgProfit: 0, successRate: 0 },
    sandwich: { attacks: 0, totalExtracted: 0, avgExtracted: 0, victimCount: 0 },
  });
  const [isConnected, setIsConnected] = useState(false);
  const [expandedJob, setExpandedJob] = useState<string | null>(null);
  const [isLoadingStats, setIsLoadingStats] = useState(false);

  // API endpoint configuration - use Vite environment variables
  const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8085';

  const connectWebSocket = useCallback(() => {
    if (isUnmountingRef.current) return;
    
    try {
      const wsUrl = (import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8085') + '/ws/scrapper-progress';
      console.log('[HistoricalCapture] Attempting WebSocket connection to:', wsUrl);
      
      if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
        wsRef.current.close();
      }
      
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('[HistoricalCapture] WebSocket connected');
        setIsConnected(true);
        // Clear any pending reconnect
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.warn('[HistoricalCapture] Failed to parse WebSocket message:', error);
        }
      };

      wsRef.current.onerror = (error) => {
        console.warn('[HistoricalCapture] WebSocket error occurred');
        setIsConnected(false);
      };

      wsRef.current.onclose = (event) => {
        console.log(`[HistoricalCapture] WebSocket closed: code=${event.code}, reason=${event.reason || 'No reason'}`);
        setIsConnected(false);
        
        // Only attempt to reconnect if:
        // 1. Component is still mounted
        // 2. It wasn't a normal closure
        // 3. We don't already have a reconnect scheduled
        if (!isUnmountingRef.current && 
            event.code !== 1000 && 
            event.code !== 1001 && 
            !reconnectTimeoutRef.current) {
          console.log('[HistoricalCapture] Scheduling reconnection in 5 seconds');
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null;
            connectWebSocket();
          }, 5000);
        }
      };
    } catch (error) {
      console.error('[HistoricalCapture] Failed to connect WebSocket:', error);
      setIsConnected(false);
      // Retry connection after 5 seconds
      if (!isUnmountingRef.current && !reconnectTimeoutRef.current) {
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectTimeoutRef.current = null;
          connectWebSocket();
        }, 5000);
      }
    }
  }, []);

  const handleWebSocketMessage = (data: any) => {
    if (data.type === 'job_update') {
      setJobs(prev => prev.map(job => 
        job.id === data.jobId 
          ? { ...job, ...data.update }
          : job
      ));
    } else if (data.type === 'stats_update') {
      setStats(data.stats);
    }
  };

  const fetchDatasetStats = async () => {
    if (isUnmountingRef.current) return;
    
    setIsLoadingStats(true);
    try {
      const response = await fetch(`${API_URL}/datasets/stats`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (response.ok) {
        const data = await response.json();
        if (!isUnmountingRef.current) {
          setStats(data);
        }
      } else {
        console.warn(`[HistoricalCapture] Failed to fetch stats: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      console.warn('[HistoricalCapture] Failed to fetch dataset stats:', error);
      // Don't show error to user for background polling failures
    } finally {
      if (!isUnmountingRef.current) {
        setIsLoadingStats(false);
      }
    }
  };

  // WebSocket connection for real-time updates
  useEffect(() => {
    isUnmountingRef.current = false;
    
    // Delay initial connection attempt to avoid race conditions
    const connectionTimer = setTimeout(() => {
      if (!isUnmountingRef.current) {
        connectWebSocket();
      }
    }, 100);
    
    fetchDatasetStats();
    
    // Poll for stats every 30 seconds (less aggressive)
    pollIntervalRef.current = setInterval(() => {
      if (!isUnmountingRef.current) {
        fetchDatasetStats();
      }
    }, 30000);

    return () => {
      isUnmountingRef.current = true;
      clearTimeout(connectionTimer);
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close(1000, 'Component unmounting');
      }
      
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [connectWebSocket]);

  const startCapture = async () => {
    const jobId = `capture_${Date.now()}`;
    const newJob: CaptureJob = {
      id: jobId,
      type: 'capture',
      status: 'pending',
      progress: 0,
      startTime: new Date(),
      details: {
        granularity,
        dateRange,
        programIds: selectedPrograms.length > 0 ? selectedPrograms : undefined,
      },
    };

    setJobs(prev => [newJob, ...prev]);

    try {
      const response = await fetch(`${API_URL}/capture/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobId,
          granularity,
          startDate: dateRange.start.toISOString(),
          endDate: dateRange.end.toISOString(),
          programIds: selectedPrograms,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        toast({
          title: 'Capture Started',
          description: `Job ${jobId} is now processing historical data`,
        });
        
        // Update job with server response
        setJobs(prev => prev.map(job => 
          job.id === jobId 
            ? { ...job, status: 'running', details: { ...job.details, ...data } }
            : job
        ));
      } else {
        throw new Error('Failed to start capture');
      }
    } catch (error: any) {
      console.error('Failed to start capture:', error);
      setJobs(prev => prev.map(job => 
        job.id === jobId 
          ? { ...job, status: 'error', details: { ...job.details, errorMessage: error.message } }
          : job
      ));
      toast({
        title: 'Capture Failed',
        description: 'Failed to start historical data capture',
        variant: 'destructive',
      });
    }
  };

  const convertToArbitrage = async () => {
    const jobId = `arbitrage_${Date.now()}`;
    const newJob: CaptureJob = {
      id: jobId,
      type: 'arbitrage',
      status: 'pending',
      progress: 0,
      startTime: new Date(),
      details: {
        granularity,
        dateRange,
      },
    };

    setJobs(prev => [newJob, ...prev]);

    try {
      const response = await fetch(`${API_URL}/convert/arbitrage/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId }),
      });

      if (response.ok) {
        toast({
          title: 'Arbitrage Conversion Started',
          description: 'Building normalized swaps and generating arbitrage labels',
        });
        setJobs(prev => prev.map(job => 
          job.id === jobId ? { ...job, status: 'running' } : job
        ));
      } else {
        throw new Error('Failed to start arbitrage conversion');
      }
    } catch (error: any) {
      console.error('Failed to start arbitrage conversion:', error);
      setJobs(prev => prev.map(job => 
        job.id === jobId 
          ? { ...job, status: 'error', details: { ...job.details, errorMessage: error.message } }
          : job
      ));
      toast({
        title: 'Conversion Failed',
        description: 'Failed to start arbitrage conversion',
        variant: 'destructive',
      });
    }
  };

  const convertToSandwich = async () => {
    const jobId = `sandwich_${Date.now()}`;
    const newJob: CaptureJob = {
      id: jobId,
      type: 'sandwich',
      status: 'pending',
      progress: 0,
      startTime: new Date(),
      details: {
        granularity,
        dateRange,
      },
    };

    setJobs(prev => [newJob, ...prev]);

    try {
      const response = await fetch(`${API_URL}/convert/sandwich/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId }),
      });

      if (response.ok) {
        toast({
          title: 'Sandwich Detection Started',
          description: 'Analyzing transaction patterns for sandwich attacks',
        });
        setJobs(prev => prev.map(job => 
          job.id === jobId ? { ...job, status: 'running' } : job
        ));
      } else {
        throw new Error('Failed to start sandwich detection');
      }
    } catch (error: any) {
      console.error('Failed to start sandwich detection:', error);
      setJobs(prev => prev.map(job => 
        job.id === jobId 
          ? { ...job, status: 'error', details: { ...job.details, errorMessage: error.message } }
          : job
      ));
      toast({
        title: 'Detection Failed',
        description: 'Failed to start sandwich detection',
        variant: 'destructive',
      });
    }
  };

  const cancelJob = async (jobId: string) => {
    try {
      const response = await fetch(`${API_URL}/jobs/${jobId}/cancel`, {
        method: 'POST',
      });

      if (response.ok) {
        setJobs(prev => prev.map(job => 
          job.id === jobId ? { ...job, status: 'cancelled' } : job
        ));
        toast({
          title: 'Job Cancelled',
          description: `Job ${jobId} has been cancelled`,
        });
      }
    } catch (error) {
      console.error('Failed to cancel job:', error);
      toast({
        title: 'Cancellation Failed',
        description: 'Failed to cancel the job',
        variant: 'destructive',
      });
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  const formatNumber = (num: number): string => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
    return num.toLocaleString();
  };

  const getStatusIcon = (status: CaptureJob['status']) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-4 h-4 text-gray-400" />;
      case 'running':
        return <Activity className="w-4 h-4 text-yellow-400 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      case 'cancelled':
        return <AlertCircle className="w-4 h-4 text-gray-400" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: CaptureJob['status']) => {
    switch (status) {
      case 'pending':
        return 'text-gray-400';
      case 'running':
        return 'text-yellow-400';
      case 'completed':
        return 'text-green-400';
      case 'error':
        return 'text-red-400';
      case 'cancelled':
        return 'text-gray-400';
      default:
        return 'text-gray-400';
    }
  };

  const estimateTimeRemaining = (job: CaptureJob): string => {
    if (job.status !== 'running' || job.progress === 0) return '--';
    
    const elapsed = Date.now() - job.startTime.getTime();
    const estimatedTotal = elapsed / (job.progress / 100);
    const remaining = estimatedTotal - elapsed;
    
    const hours = Math.floor(remaining / 3600000);
    const minutes = Math.floor((remaining % 3600000) / 60000);
    const seconds = Math.floor((remaining % 60000) / 1000);
    
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            Historical Capture Panel
          </h2>
          <p className="text-gray-400 mt-1">Capture and process historical blockchain data for MEV analysis</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge 
            variant="outline" 
            className={`${isConnected ? 'text-green-400 border-green-400' : 'text-red-400 border-red-400'}`}
          >
            {isConnected ? 'Connected' : 'Disconnected'}
          </Badge>
          <Button
            size="sm"
            variant="outline"
            onClick={fetchDatasetStats}
            disabled={isLoadingStats}
          >
            <RefreshCw className={`w-4 h-4 ${isLoadingStats ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {/* Control Panel */}
      <Card className="bg-gray-800/50 backdrop-blur border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Capture Configuration</h3>
        
        {/* Top Row Controls */}
        <div className="space-y-4">
          {/* Granularity Selector */}
          <div>
            <label className="text-sm text-gray-400 mb-2 block">Granularity</label>
            <div className="flex gap-2">
              {(['day', 'month', 'year'] as const).map((gran) => (
                <button
                  key={gran}
                  onClick={() => setGranularity(gran)}
                  className={`px-4 py-2 rounded-lg transition-all capitalize ${
                    granularity === gran
                      ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {gran}
                </button>
              ))}
            </div>
          </div>

          {/* Date Range */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-gray-400 mb-2 block">Start Date</label>
              <input
                type="datetime-local"
                value={format(dateRange.start, "yyyy-MM-dd'T'HH:mm")}
                onChange={(e) => setDateRange(prev => ({ ...prev, start: new Date(e.target.value) }))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="text-sm text-gray-400 mb-2 block">End Date</label>
              <input
                type="datetime-local"
                value={format(dateRange.end, "yyyy-MM-dd'T'HH:mm")}
                onChange={(e) => setDateRange(prev => ({ ...prev, end: new Date(e.target.value) }))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none"
              />
            </div>
          </div>

          {/* Preset Date Ranges */}
          <div className="flex gap-2 flex-wrap">
            {DATE_PRESETS.map((preset) => (
              <button
                key={preset.value}
                onClick={() => setDateRange(preset.getDates())}
                className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-md transition-colors"
              >
                {preset.label}
              </button>
            ))}
          </div>

          {/* Program ID Filter */}
          <div>
            <label className="text-sm text-gray-400 mb-2 block">DEX Programs</label>
            <div className="grid grid-cols-3 gap-2">
              {DEX_PROGRAMS.map((program) => (
                <button
                  key={program.id}
                  onClick={() => {
                    setSelectedPrograms(prev =>
                      prev.includes(program.address)
                        ? prev.filter(p => p !== program.address)
                        : [...prev, program.address]
                    );
                  }}
                  className={`px-3 py-2 rounded-lg transition-all text-sm ${
                    selectedPrograms.includes(program.address)
                      ? `bg-gradient-to-r ${program.color} text-white`
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {program.name}
                </button>
              ))}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4 border-t border-gray-700">
            <Button
              onClick={startCapture}
              className="bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600"
            >
              <Play className="w-4 h-4 mr-2" />
              Start Capture
            </Button>
            <Button
              onClick={convertToArbitrage}
              variant="outline"
              className="border-purple-500 text-purple-400 hover:bg-purple-500/10"
            >
              <Layers className="w-4 h-4 mr-2" />
              Convert → Arbitrage
            </Button>
            <Button
              onClick={convertToSandwich}
              variant="outline"
              className="border-orange-500 text-orange-400 hover:bg-orange-500/10"
            >
              <Target className="w-4 h-4 mr-2" />
              Convert → Sandwich
            </Button>
            <Button
              onClick={fetchDatasetStats}
              variant="outline"
              className="ml-auto"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh Stats
            </Button>
          </div>
        </div>
      </Card>

      {/* Jobs List */}
      <Card className="bg-gray-800/50 backdrop-blur border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Active Jobs</h3>
          <Badge variant="outline">{jobs.filter(j => j.status === 'running').length} Running</Badge>
        </div>
        
        <div className="space-y-3">
          {jobs.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No jobs running. Start a capture to begin.</p>
            </div>
          ) : (
            jobs.map((job) => (
              <div
                key={job.id}
                className="bg-gray-900/50 rounded-lg p-4 border border-gray-700 hover:border-gray-600 transition-colors"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(job.status)}
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-sm text-white">{job.id}</span>
                        <Badge variant="outline" className="text-xs">
                          {job.type}
                        </Badge>
                      </div>
                      <span className={`text-xs ${getStatusColor(job.status)}`}>
                        {job.status}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">
                      ETA: {estimateTimeRemaining(job)}
                    </span>
                    {job.status === 'running' && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => cancelJob(job.id)}
                        className="text-red-400 border-red-400 hover:bg-red-400/10"
                      >
                        Cancel
                      </Button>
                    )}
                    <button
                      onClick={() => setExpandedJob(expandedJob === job.id ? null : job.id)}
                      className="text-gray-400 hover:text-white transition-colors"
                    >
                      {expandedJob === job.id ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                {job.status === 'running' && (
                  <div className="space-y-2">
                    <Progress value={job.progress} className="h-2" />
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Progress: {job.progress.toFixed(1)}%</span>
                      {job.details.slotsProcessed && job.details.totalSlots && (
                        <span className="text-gray-400">
                          {formatNumber(job.details.slotsProcessed)} / {formatNumber(job.details.totalSlots)} slots
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {expandedJob === job.id && (
                  <div className="mt-4 pt-4 border-t border-gray-700">
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Granularity</p>
                        <p className="text-white font-mono">{job.details.granularity}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">Date Range</p>
                        <p className="text-white font-mono">
                          {format(job.details.dateRange.start, 'MMM dd')} - {format(job.details.dateRange.end, 'MMM dd')}
                        </p>
                      </div>
                      {job.details.blocksWritten !== undefined && (
                        <div>
                          <p className="text-gray-400">Blocks Written</p>
                          <p className="text-white font-mono">{formatNumber(job.details.blocksWritten)}</p>
                        </div>
                      )}
                      {job.details.bytesWritten !== undefined && (
                        <div>
                          <p className="text-gray-400">Data Written</p>
                          <p className="text-white font-mono">{formatBytes(job.details.bytesWritten)}</p>
                        </div>
                      )}
                      {job.details.currentSlot !== undefined && (
                        <div>
                          <p className="text-gray-400">Current Slot</p>
                          <p className="text-white font-mono">{formatNumber(job.details.currentSlot)}</p>
                        </div>
                      )}
                      {job.details.programIds && job.details.programIds.length > 0 && (
                        <div className="col-span-3">
                          <p className="text-gray-400 mb-1">Programs</p>
                          <div className="flex gap-2 flex-wrap">
                            {job.details.programIds.map(id => {
                              const program = DEX_PROGRAMS.find(p => p.address === id);
                              return program ? (
                                <Badge key={id} variant="outline" className="text-xs">
                                  {program.name}
                                </Badge>
                              ) : (
                                <Badge key={id} variant="outline" className="text-xs font-mono">
                                  {id.slice(0, 8)}...
                                </Badge>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      {job.details.errorMessage && (
                        <div className="col-span-3">
                          <p className="text-gray-400">Error</p>
                          <p className="text-red-400 text-xs">{job.details.errorMessage}</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </Card>

      {/* Dataset Stats */}
      <div className="grid grid-cols-4 gap-4">
        {/* Raw Data Stats */}
        <Card className="bg-gray-800/50 backdrop-blur border-gray-700 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <HardDrive className="w-5 h-5 text-blue-400" />
              <h4 className="text-sm font-semibold text-white">Raw Data</h4>
            </div>
            <Badge variant="outline" className="text-xs">
              {stats.rawData.partitions} partitions
            </Badge>
          </div>
          <div className="space-y-2">
            <div>
              <p className="text-2xl font-bold text-white">{formatBytes(stats.rawData.bytes)}</p>
              <p className="text-xs text-gray-400">{formatNumber(stats.rawData.rows)} rows</p>
            </div>
            <div className="text-xs text-gray-500">
              Updated {format(stats.rawData.lastUpdated, 'MMM dd HH:mm')}
            </div>
          </div>
        </Card>

        {/* Swaps Stats */}
        <Card className="bg-gray-800/50 backdrop-blur border-gray-700 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <ArrowRight className="w-5 h-5 text-green-400" />
              <h4 className="text-sm font-semibold text-white">Swaps</h4>
            </div>
            <Badge variant="outline" className="text-xs">
              {stats.swaps.uniquePairs} pairs
            </Badge>
          </div>
          <div className="space-y-2">
            <div>
              <p className="text-2xl font-bold text-white">{formatNumber(stats.swaps.rows)}</p>
              <p className="text-xs text-gray-400">Total swaps</p>
            </div>
            <div className="text-xs text-green-400">
              {formatNumber(stats.swaps.totalVolume)} SOL volume
            </div>
          </div>
        </Card>

        {/* Arbitrage Stats */}
        <Card className="bg-gray-800/50 backdrop-blur border-gray-700 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-purple-400" />
              <h4 className="text-sm font-semibold text-white">Arbitrage</h4>
            </div>
            <Badge variant="outline" className="text-xs">
              {(stats.arbitrage.successRate * 100).toFixed(1)}% success
            </Badge>
          </div>
          <div className="space-y-2">
            <div>
              <p className="text-2xl font-bold text-white">{formatNumber(stats.arbitrage.opportunities)}</p>
              <p className="text-xs text-gray-400">Opportunities found</p>
            </div>
            <div className="text-xs text-purple-400">
              {formatNumber(stats.arbitrage.totalProfit)} SOL profit
            </div>
          </div>
        </Card>

        {/* Sandwich Stats */}
        <Card className="bg-gray-800/50 backdrop-blur border-gray-700 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-orange-400" />
              <h4 className="text-sm font-semibold text-white">Sandwich</h4>
            </div>
            <Badge variant="outline" className="text-xs">
              {stats.sandwich.victimCount} victims
            </Badge>
          </div>
          <div className="space-y-2">
            <div>
              <p className="text-2xl font-bold text-white">{formatNumber(stats.sandwich.attacks)}</p>
              <p className="text-xs text-gray-400">Attacks detected</p>
            </div>
            <div className="text-xs text-orange-400">
              {formatNumber(stats.sandwich.totalExtracted)} SOL extracted
            </div>
          </div>
        </Card>
      </div>

      {/* Visual Chart Section */}
      <Card className="bg-gray-800/50 backdrop-blur border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Dataset Growth</h3>
          <div className="flex items-center gap-2">
            <Badge variant="outline">Last 7 Days</Badge>
            <Button size="sm" variant="outline">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
        
        {/* Placeholder for chart - would integrate with D3.js or Recharts */}
        <div className="h-64 bg-gray-900/50 rounded-lg flex items-center justify-center">
          <div className="text-center">
            <BarChart3 className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400 text-sm">Chart visualization would render here</p>
            <p className="text-gray-500 text-xs mt-1">Integrated with D3.js or Apache ECharts</p>
          </div>
        </div>
      </Card>
    </div>
  );
}