import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { proxy, useSnapshot } from 'valtio';
import * as d3 from 'd3';
import { Line, Bar, Heatmap } from '@ant-design/plots';
import { 
  Card, 
  Grid, 
  Statistic, 
  Progress, 
  Alert, 
  Badge, 
  Space, 
  Typography,
  Select,
  Button,
  Tabs,
  Tag,
  Tooltip,
  Row,
  Col,
  Divider
} from 'antd';
import {
  DashboardOutlined,
  RocketOutlined,
  ThunderboltOutlined,
  SafetyOutlined,
  ForkOutlined,
  DatabaseOutlined,
  MonitorOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  CloseCircleOutlined,
  SyncOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

// Global MEV state management with Valtio
const mevState = proxy({
  // System Health
  systemHealth: {
    overall: 'healthy' as 'healthy' | 'degraded' | 'critical',
    services: {
      clickhouse: { status: 'healthy', latency: 0 },
      kafka: { status: 'healthy', lag: 0 },
      grafana: { status: 'healthy', dashboards: 0 },
      mevAgent: { status: 'running', uptime: 0 },
      arbAgent: { status: 'running', uptime: 0 },
      control: { status: 'running', uptime: 0 }
    }
  },
  
  // Real-time Metrics
  metrics: {
    cpuUsage: [] as Array<{ time: number, value: number }>,
    memoryUsage: [] as Array<{ time: number, value: number }>,
    networkIO: [] as Array<{ time: number, in: number, out: number }>,
    diskIO: [] as Array<{ time: number, read: number, write: number }>,
    latencyP99: 0,
    throughput: 0,
    errorRate: 0
  },
  
  // MEV Performance
  mevPerformance: {
    landsPerMinute: 0,
    totalLanded: 0,
    evPerMinute: 0,
    totalEV: 0,
    profitToday: 0,
    profitWeek: 0,
    profitMonth: 0,
    successRate: 0,
    avgTipSize: 0,
    tipEfficiency: 0
  },
  
  // Bandit Optimization
  banditMetrics: {
    totalPulls: 0,
    bestArm: '',
    exploration: 0.1,
    exploitation: 0.9,
    armPerformance: [] as Array<{
      id: string,
      route: string,
      tipLadder: number,
      pulls: number,
      avgReward: number,
      ucb: number,
      landRate: number
    }>,
    convergenceRate: 0
  },
  
  // Decision DNA
  decisionDNA: {
    totalDecisions: 0,
    uniqueStrategies: 0,
    hashChainHeight: 0,
    hashChainVerified: true,
    currentHash: '',
    previousHash: '',
    lineage: [] as Array<{
      hash: string,
      timestamp: number,
      strategy: string,
      outcome: 'success' | 'failure',
      reward: number
    }>
  },
  
  // Lab Smoke Tests
  labTests: {
    lastRun: 0,
    status: 'idle' as 'idle' | 'running' | 'success' | 'failed',
    testsRun: 0,
    testsPassed: 0,
    testsFailed: 0,
    coverage: 0,
    results: [] as Array<{
      name: string,
      status: 'pass' | 'fail',
      duration: number,
      error?: string
    }>
  },
  
  // Alerts & Notifications
  alerts: [] as Array<{
    id: string,
    level: 'info' | 'warning' | 'error' | 'critical',
    source: string,
    message: string,
    timestamp: number,
    acknowledged: boolean
  }>,
  
  // ClickHouse Query Builder State
  queryBuilder: {
    selectedTable: '',
    selectedColumns: [] as string[],
    conditions: [] as Array<{ column: string, operator: string, value: string }>,
    groupBy: [] as string[],
    orderBy: { column: '', direction: 'ASC' as 'ASC' | 'DESC' },
    limit: 100
  },
  
  // Kafka Topics
  kafkaTopics: {
    'bandit-events-proto': { lag: 0, throughput: 0, errors: 0 },
    'realtime-proto': { lag: 0, throughput: 0, errors: 0 },
    'control-acks': { lag: 0, throughput: 0, errors: 0 },
    'mev-decisions': { lag: 0, throughput: 0, errors: 0 }
  }
});

// WebSocket connections
const wsConnections = new Map<string, WebSocket>();

// Initialize WebSocket connections
function initializeWebSockets() {
  const endpoints = {
    metrics: 'ws://localhost:8080/ws/metrics',
    bandit: 'ws://localhost:8080/ws/bandit',
    dna: 'ws://localhost:8080/ws/dna',
    control: 'ws://localhost:8080/ws/control',
    kafka: 'ws://localhost:8080/ws/kafka',
    clickhouse: 'ws://localhost:8080/ws/clickhouse'
  };
  
  Object.entries(endpoints).forEach(([key, url]) => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      console.log(`Connected to ${key} WebSocket`);
      ws.send(JSON.stringify({ action: 'subscribe', topic: key }));
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(key, data);
      } catch (error) {
        console.error(`Error parsing ${key} message:`, error);
      }
    };
    
    ws.onerror = (error) => {
      console.error(`WebSocket error for ${key}:`, error);
      mevState.alerts.push({
        id: `ws_error_${Date.now()}`,
        level: 'error',
        source: `WebSocket/${key}`,
        message: `Connection error: ${error}`,
        timestamp: Date.now(),
        acknowledged: false
      });
    };
    
    ws.onclose = () => {
      console.log(`Disconnected from ${key} WebSocket`);
      // Attempt reconnection after 5 seconds
      setTimeout(() => {
        if (!wsConnections.has(key)) {
          initializeWebSockets();
        }
      }, 5000);
    };
    
    wsConnections.set(key, ws);
  });
}

// Handle incoming WebSocket messages
function handleWebSocketMessage(source: string, data: any) {
  switch (source) {
    case 'metrics':
      updateSystemMetrics(data);
      break;
    case 'bandit':
      updateBanditMetrics(data);
      break;
    case 'dna':
      updateDecisionDNA(data);
      break;
    case 'control':
      updateControlStatus(data);
      break;
    case 'kafka':
      updateKafkaMetrics(data);
      break;
    case 'clickhouse':
      updateClickHouseStatus(data);
      break;
  }
}

// Update functions for different data sources
function updateSystemMetrics(data: any) {
  const now = Date.now();
  
  // Update CPU usage
  if (data.cpu !== undefined) {
    mevState.metrics.cpuUsage.push({ time: now, value: data.cpu });
    if (mevState.metrics.cpuUsage.length > 300) {
      mevState.metrics.cpuUsage.shift();
    }
  }
  
  // Update Memory usage
  if (data.memory !== undefined) {
    mevState.metrics.memoryUsage.push({ time: now, value: data.memory });
    if (mevState.metrics.memoryUsage.length > 300) {
      mevState.metrics.memoryUsage.shift();
    }
  }
  
  // Update Network IO
  if (data.network) {
    mevState.metrics.networkIO.push({ 
      time: now, 
      in: data.network.in, 
      out: data.network.out 
    });
    if (mevState.metrics.networkIO.length > 300) {
      mevState.metrics.networkIO.shift();
    }
  }
  
  // Update other metrics
  if (data.latencyP99) mevState.metrics.latencyP99 = data.latencyP99;
  if (data.throughput) mevState.metrics.throughput = data.throughput;
  if (data.errorRate) mevState.metrics.errorRate = data.errorRate;
}

function updateBanditMetrics(data: any) {
  if (data.totalPulls) mevState.banditMetrics.totalPulls = data.totalPulls;
  if (data.bestArm) mevState.banditMetrics.bestArm = data.bestArm;
  if (data.exploration) mevState.banditMetrics.exploration = data.exploration;
  if (data.exploitation) mevState.banditMetrics.exploitation = 1 - data.exploration;
  if (data.armPerformance) mevState.banditMetrics.armPerformance = data.armPerformance;
  if (data.convergenceRate) mevState.banditMetrics.convergenceRate = data.convergenceRate;
  
  // Update MEV performance from bandit data
  if (data.landsPerMinute) mevState.mevPerformance.landsPerMinute = data.landsPerMinute;
  if (data.evPerMinute) mevState.mevPerformance.evPerMinute = data.evPerMinute;
  if (data.tipEfficiency) mevState.mevPerformance.tipEfficiency = data.tipEfficiency;
}

function updateDecisionDNA(data: any) {
  if (data.totalDecisions) mevState.decisionDNA.totalDecisions = data.totalDecisions;
  if (data.uniqueStrategies) mevState.decisionDNA.uniqueStrategies = data.uniqueStrategies;
  if (data.hashChainHeight) mevState.decisionDNA.hashChainHeight = data.hashChainHeight;
  if (data.hashChainVerified !== undefined) mevState.decisionDNA.hashChainVerified = data.hashChainVerified;
  if (data.currentHash) mevState.decisionDNA.currentHash = data.currentHash;
  if (data.previousHash) mevState.decisionDNA.previousHash = data.previousHash;
  
  if (data.newDecision) {
    mevState.decisionDNA.lineage.unshift(data.newDecision);
    if (mevState.decisionDNA.lineage.length > 100) {
      mevState.decisionDNA.lineage.pop();
    }
  }
}

function updateControlStatus(data: any) {
  if (data.services) {
    Object.entries(data.services).forEach(([service, status]: [string, any]) => {
      if (mevState.systemHealth.services[service as keyof typeof mevState.systemHealth.services]) {
        mevState.systemHealth.services[service as keyof typeof mevState.systemHealth.services] = status;
      }
    });
  }
  
  // Update overall health
  const services = Object.values(mevState.systemHealth.services);
  const unhealthyCount = services.filter(s => s.status !== 'healthy' && s.status !== 'running').length;
  
  if (unhealthyCount === 0) {
    mevState.systemHealth.overall = 'healthy';
  } else if (unhealthyCount <= 2) {
    mevState.systemHealth.overall = 'degraded';
  } else {
    mevState.systemHealth.overall = 'critical';
  }
}

function updateKafkaMetrics(data: any) {
  if (data.topics) {
    Object.entries(data.topics).forEach(([topic, metrics]: [string, any]) => {
      if (mevState.kafkaTopics[topic as keyof typeof mevState.kafkaTopics]) {
        mevState.kafkaTopics[topic as keyof typeof mevState.kafkaTopics] = metrics;
      }
    });
  }
}

function updateClickHouseStatus(data: any) {
  if (data.status) mevState.systemHealth.services.clickhouse.status = data.status;
  if (data.latency) mevState.systemHealth.services.clickhouse.latency = data.latency;
}

// Main Component
export default function MEVControlCenter() {
  const state = useSnapshot(mevState);
  const [selectedTab, setSelectedTab] = useState('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30000);
  
  // Initialize WebSockets on mount
  useEffect(() => {
    initializeWebSockets();
    
    return () => {
      wsConnections.forEach(ws => ws.close());
      wsConnections.clear();
    };
  }, []);
  
  // Auto-refresh data
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(async () => {
      // Fetch latest data from API
      try {
        const responses = await Promise.all([
          fetch('/api/metrics'),
          fetch('/api/mev/performance'),
          fetch('/api/bandit/stats'),
          fetch('/api/dna/latest'),
          fetch('/api/lab/status')
        ]);
        
        const [metrics, mevPerf, banditStats, dnaData, labStatus] = await Promise.all(
          responses.map(r => r.json())
        );
        
        // Update state with fetched data
        Object.assign(mevState.metrics, metrics);
        Object.assign(mevState.mevPerformance, mevPerf);
        Object.assign(mevState.banditMetrics, banditStats);
        Object.assign(mevState.decisionDNA, dnaData);
        Object.assign(mevState.labTests, labStatus);
      } catch (error) {
        console.error('Error fetching data:', error);
        mevState.alerts.push({
          id: `fetch_error_${Date.now()}`,
          level: 'warning',
          source: 'API',
          message: 'Failed to fetch latest data',
          timestamp: Date.now(),
          acknowledged: false
        });
      }
    }, refreshInterval);
    
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Alt+1-5 to switch tabs
      if (e.altKey && e.key >= '1' && e.key <= '5') {
        const tabs = ['overview', 'bandit', 'dna', 'monitoring', 'lab'];
        const index = parseInt(e.key) - 1;
        const tab = tabs[index];
        if (tab) {
          setSelectedTab(tab);
        }
      }
      
      // Alt+R to toggle refresh
      if (e.altKey && e.key === 'r') {
        setAutoRefresh(!autoRefresh);
      }
      
      // Alt+S to run smoke test
      if (e.altKey && e.key === 's') {
        runSmokeTest();
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [autoRefresh]);
  
  // Run smoke test
  const runSmokeTest = async () => {
    mevState.labTests.status = 'running';
    
    try {
      const response = await fetch('/api/lab/smoke-test', { method: 'POST' });
      const result = await response.json();
      
      mevState.labTests.status = result.success ? 'success' : 'failed';
      mevState.labTests.lastRun = Date.now();
      mevState.labTests.testsRun = result.testsRun;
      mevState.labTests.testsPassed = result.testsPassed;
      mevState.labTests.testsFailed = result.testsFailed;
      mevState.labTests.coverage = result.coverage;
      mevState.labTests.results = result.results;
      
      mevState.alerts.push({
        id: `test_${Date.now()}`,
        level: result.success ? 'info' : 'error',
        source: 'Lab Tests',
        message: `Smoke test ${result.success ? 'passed' : 'failed'}: ${result.testsPassed}/${result.testsRun} tests passed`,
        timestamp: Date.now(),
        acknowledged: false
      });
    } catch (error) {
      mevState.labTests.status = 'failed';
      console.error('Smoke test error:', error);
    }
  };
  
  // Get health color
  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'running':
        return '#52c41a';
      case 'degraded':
      case 'idle':
        return '#faad14';
      case 'critical':
      case 'stopped':
      case 'error':
        return '#f5222d';
      default:
        return '#8c8c8c';
    }
  };
  
  // Overview Tab Component
  const OverviewTab = () => (
    <div>
      {/* System Health */}
      <Card 
        title={<><MonitorOutlined /> System Health</>} 
        style={{ marginBottom: 24 }}
        extra={
          <Badge 
            status={state.systemHealth.overall === 'healthy' ? 'success' : 
                   state.systemHealth.overall === 'degraded' ? 'warning' : 'error'}
            text={state.systemHealth.overall.toUpperCase()}
          />
        }
      >
        <Row gutter={16}>
          {Object.entries(state.systemHealth.services).map(([service, status]) => (
            <Col span={4} key={service}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <Statistic
                  title={service.toUpperCase()}
                  value={status.status}
                  valueStyle={{ 
                    color: getHealthColor(status.status),
                    fontSize: '14px'
                  }}
                  prefix={
                    status.status === 'healthy' || status.status === 'running' ? 
                    <CheckCircleOutlined /> : 
                    status.status === 'degraded' || status.status === 'idle' ?
                    <WarningOutlined /> :
                    <CloseCircleOutlined />
                  }
                />
              </Card>
            </Col>
          ))}
        </Row>
      </Card>
      
      {/* MEV Performance Metrics */}
      <Card title={<><RocketOutlined /> MEV Performance</> } style={{ marginBottom: 24 }}>
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="Lands/Min"
              value={state.mevPerformance.landsPerMinute}
              precision={0}
              valueStyle={{ color: '#3f8600' }}
              prefix={<ThunderboltOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="EV/Min (SOL)"
              value={state.mevPerformance.evPerMinute}
              precision={4}
              valueStyle={{ color: '#3f8600' }}
              prefix="◎"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Success Rate"
              value={state.mevPerformance.successRate * 100}
              precision={1}
              valueStyle={{ color: '#3f8600' }}
              suffix="%"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Tip Efficiency"
              value={state.mevPerformance.tipEfficiency * 100}
              precision={1}
              valueStyle={{ color: '#3f8600' }}
              suffix="%"
            />
          </Col>
        </Row>
        
        <Divider />
        
        <Row gutter={16}>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="Today's Profit"
                value={state.mevPerformance.profitToday}
                precision={4}
                valueStyle={{ color: '#52c41a' }}
                prefix="◎"
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="This Week"
                value={state.mevPerformance.profitWeek}
                precision={4}
                valueStyle={{ color: '#52c41a' }}
                prefix="◎"
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="This Month"
                value={state.mevPerformance.profitMonth}
                precision={4}
                valueStyle={{ color: '#52c41a' }}
                prefix="◎"
              />
            </Card>
          </Col>
        </Row>
      </Card>
      
      {/* System Metrics Charts */}
      <Row gutter={16}>
        <Col span={12}>
          <Card title="CPU Usage" size="small">
            <Line
              data={state.metrics.cpuUsage}
              xField="time"
              yField="value"
              smooth
              height={200}
              xAxis={{
                type: 'time',
                tickCount: 5
              }}
              yAxis={{
                max: 100,
                min: 0
              }}
              color="#1890ff"
              point={{ size: 0 }}
              animation={{ appear: { animation: 'path-in', duration: 300 } }}
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="Memory Usage" size="small">
            <Line
              data={state.metrics.memoryUsage}
              xField="time"
              yField="value"
              smooth
              height={200}
              xAxis={{
                type: 'time',
                tickCount: 5
              }}
              yAxis={{
                max: 100,
                min: 0
              }}
              color="#52c41a"
              point={{ size: 0 }}
              animation={{ appear: { animation: 'path-in', duration: 300 } }}
            />
          </Card>
        </Col>
      </Row>
      
      {/* Kafka Topics Status */}
      <Card title={<><ApiOutlined /> Kafka Topics</> } style={{ marginTop: 24 }}>
        <Row gutter={16}>
          {Object.entries(state.kafkaTopics).map(([topic, metrics]) => (
            <Col span={6} key={topic}>
              <Card size="small">
                <Title level={5}>{topic}</Title>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text type="secondary">Lag: </Text>
                    <Text strong>{metrics.lag}</Text>
                  </div>
                  <div>
                    <Text type="secondary">Throughput: </Text>
                    <Text strong>{metrics.throughput}/s</Text>
                  </div>
                  <div>
                    <Text type="secondary">Errors: </Text>
                    <Text strong style={{ color: metrics.errors > 0 ? '#f5222d' : '#52c41a' }}>
                      {metrics.errors}
                    </Text>
                  </div>
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>
    </div>
  );
  
  // Bandit Dashboard Tab
  const BanditTab = () => (
    <div>
      <Card title={<><SafetyOutlined /> Bandit Optimizer</> } style={{ marginBottom: 24 }}>
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="Total Pulls"
              value={state.banditMetrics.totalPulls}
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Best Arm"
              value={state.banditMetrics.bestArm}
              valueStyle={{ color: '#52c41a' }}
            />
          </Col>
          <Col span={6}>
            <Progress
              type="circle"
              percent={state.banditMetrics.exploration * 100}
              format={() => `Explore\n${(state.banditMetrics.exploration * 100).toFixed(0)}%`}
              width={80}
            />
          </Col>
          <Col span={6}>
            <Progress
              type="circle"
              percent={state.banditMetrics.exploitation * 100}
              format={() => `Exploit\n${(state.banditMetrics.exploitation * 100).toFixed(0)}%`}
              width={80}
              strokeColor="#52c41a"
            />
          </Col>
        </Row>
      </Card>
      
      {/* Arm Performance Table */}
      <Card title="Arm Performance">
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #d9d9d9' }}>
              <th style={{ padding: 8, textAlign: 'left' }}>Arm ID</th>
              <th style={{ padding: 8, textAlign: 'left' }}>Route</th>
              <th style={{ padding: 8, textAlign: 'right' }}>Tip Ladder</th>
              <th style={{ padding: 8, textAlign: 'right' }}>Pulls</th>
              <th style={{ padding: 8, textAlign: 'right' }}>Avg Reward</th>
              <th style={{ padding: 8, textAlign: 'right' }}>UCB Score</th>
              <th style={{ padding: 8, textAlign: 'right' }}>Land Rate</th>
            </tr>
          </thead>
          <tbody>
            {state.banditMetrics.armPerformance.slice(0, 10).map((arm, idx) => (
              <tr key={arm.id} style={{ 
                borderBottom: '1px solid #f0f0f0',
                backgroundColor: idx === 0 ? '#e6fffb' : 'transparent'
              }}>
                <td style={{ padding: 8 }}>{arm.id}</td>
                <td style={{ padding: 8 }}>
                  <Tag color={arm.route === 'Direct' ? 'green' : 'blue'}>
                    {arm.route}
                  </Tag>
                </td>
                <td style={{ padding: 8, textAlign: 'right' }}>{arm.tipLadder.toFixed(2)}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{arm.pulls}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{arm.avgReward.toFixed(6)}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>
                  <Text strong>{arm.ucb.toFixed(6)}</Text>
                </td>
                <td style={{ padding: 8, textAlign: 'right' }}>
                  <Progress
                    percent={arm.landRate * 100}
                    size="small"
                    format={p => `${p?.toFixed(1)}%`}
                    strokeColor={arm.landRate > 0.6 ? '#52c41a' : '#faad14'}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
  
  // Decision DNA Tab
  const DNATab = () => (
    <div>
      <Card title={<><ForkOutlined /> Decision DNA</> } style={{ marginBottom: 24 }}>
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="Total Decisions"
              value={state.decisionDNA.totalDecisions}
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Unique Strategies"
              value={state.decisionDNA.uniqueStrategies}
              valueStyle={{ color: '#722ed1' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Chain Height"
              value={state.decisionDNA.hashChainHeight}
              valueStyle={{ color: '#13c2c2' }}
              prefix="#"
            />
          </Col>
          <Col span={6}>
            <Badge
              status={state.decisionDNA.hashChainVerified ? 'success' : 'error'}
              text={state.decisionDNA.hashChainVerified ? 'Chain Verified' : 'Chain Invalid'}
              style={{ fontSize: '16px' }}
            />
          </Col>
        </Row>
        
        <Divider />
        
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text type="secondary">Current Hash: </Text>
            <Text code copyable>{state.decisionDNA.currentHash.slice(0, 32)}...</Text>
          </div>
          <div>
            <Text type="secondary">Previous Hash: </Text>
            <Text code>{state.decisionDNA.previousHash.slice(0, 32)}...</Text>
          </div>
        </Space>
      </Card>
      
      {/* Decision Lineage */}
      <Card title="Recent Decision Lineage">
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          {state.decisionDNA.lineage.map((decision, idx) => (
            <Card 
              key={decision.hash} 
              size="small" 
              style={{ marginBottom: 8 }}
              type={decision.outcome === 'success' ? 'inner' : undefined}
            >
              <Row>
                <Col span={4}>
                  <Text type="secondary">
                    {new Date(decision.timestamp).toLocaleTimeString()}
                  </Text>
                </Col>
                <Col span={8}>
                  <Tag color="purple">{decision.strategy}</Tag>
                </Col>
                <Col span={6}>
                  <Tag color={decision.outcome === 'success' ? 'success' : 'error'}>
                    {decision.outcome.toUpperCase()}
                  </Tag>
                </Col>
                <Col span={6} style={{ textAlign: 'right' }}>
                  <Text strong>◎ {decision.reward.toFixed(6)}</Text>
                </Col>
              </Row>
            </Card>
          ))}
        </div>
      </Card>
    </div>
  );
  
  // Lab Tests Tab
  const LabTab = () => (
    <div>
      <Card 
        title={<><DatabaseOutlined /> Lab Smoke Tests</> } 
        style={{ marginBottom: 24 }}
        extra={
          <Space>
            <Button
              type="primary"
              onClick={runSmokeTest}
              loading={state.labTests.status === 'running'}
              icon={<SyncOutlined />}
            >
              Run Smoke Test
            </Button>
            <Badge
              status={
                state.labTests.status === 'success' ? 'success' :
                state.labTests.status === 'failed' ? 'error' :
                state.labTests.status === 'running' ? 'processing' : 'default'
              }
              text={state.labTests.status.toUpperCase()}
            />
          </Space>
        }
      >
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="Tests Run"
              value={state.labTests.testsRun}
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Tests Passed"
              value={state.labTests.testsPassed}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Tests Failed"
              value={state.labTests.testsFailed}
              valueStyle={{ color: state.labTests.testsFailed > 0 ? '#f5222d' : '#8c8c8c' }}
              prefix={state.labTests.testsFailed > 0 ? <CloseCircleOutlined /> : null}
            />
          </Col>
          <Col span={6}>
            <Progress
              type="circle"
              percent={state.labTests.coverage}
              format={p => `${p}%\nCoverage`}
              width={80}
            />
          </Col>
        </Row>
        
        {state.labTests.lastRun > 0 && (
          <div style={{ marginTop: 16 }}>
            <Text type="secondary">
              Last run: {new Date(state.labTests.lastRun).toLocaleString()}
            </Text>
          </div>
        )}
      </Card>
      
      {/* Test Results */}
      {state.labTests.results.length > 0 && (
        <Card title="Test Results">
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            {state.labTests.results.map((test, idx) => (
              <div 
                key={idx}
                style={{ 
                  padding: '8px 12px',
                  marginBottom: 8,
                  backgroundColor: test.status === 'pass' ? '#f6ffed' : '#fff2e8',
                  border: `1px solid ${test.status === 'pass' ? '#b7eb8f' : '#ffbb96'}`,
                  borderRadius: 4
                }}
              >
                <Row>
                  <Col span={16}>
                    <Space>
                      {test.status === 'pass' ? 
                        <CheckCircleOutlined style={{ color: '#52c41a' }} /> :
                        <CloseCircleOutlined style={{ color: '#f5222d' }} />
                      }
                      <Text strong>{test.name}</Text>
                    </Space>
                  </Col>
                  <Col span={8} style={{ textAlign: 'right' }}>
                    <Text type="secondary">{test.duration}ms</Text>
                  </Col>
                </Row>
                {test.error && (
                  <div style={{ marginTop: 8 }}>
                    <Text type="danger" style={{ fontSize: 12 }}>{test.error}</Text>
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
  
  // Alerts Panel
  const AlertsPanel = () => (
    <Card 
      title="System Alerts" 
      size="small"
      style={{ marginBottom: 24 }}
      bodyStyle={{ maxHeight: 200, overflowY: 'auto' }}
    >
      {state.alerts.filter(a => !a.acknowledged).slice(0, 5).map(alert => (
        <Alert
          key={alert.id}
          message={`[${alert.source}] ${alert.message}`}
          type={
            alert.level === 'critical' ? 'error' :
            alert.level === 'error' ? 'error' :
            alert.level === 'warning' ? 'warning' : 'info'
          }
          showIcon
          closable
          onClose={() => {
            const a = mevState.alerts.find(a => a.id === alert.id);
            if (a) a.acknowledged = true;
          }}
          style={{ marginBottom: 8 }}
        />
      ))}
      {state.alerts.filter(a => !a.acknowledged).length === 0 && (
        <Text type="secondary">No active alerts</Text>
      )}
    </Card>
  );
  
  return (
    <div style={{ 
      padding: 24, 
      backgroundColor: '#f0f2f5',
      minHeight: '100vh'
    }}>
      {/* Header */}
      <div style={{ 
        marginBottom: 24,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <Title level={2} style={{ margin: 0 }}>
          <DashboardOutlined /> MEV Control Center
        </Title>
        
        <Space>
          <Select
            value={refreshInterval}
            onChange={setRefreshInterval}
            style={{ width: 120 }}
          >
            <Select.Option value={5000}>5 seconds</Select.Option>
            <Select.Option value={10000}>10 seconds</Select.Option>
            <Select.Option value={30000}>30 seconds</Select.Option>
            <Select.Option value={60000}>1 minute</Select.Option>
          </Select>
          
          <Button
            type={autoRefresh ? 'primary' : 'default'}
            icon={<SyncOutlined spin={autoRefresh} />}
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? 'Auto-Refresh ON' : 'Auto-Refresh OFF'}
          </Button>
        </Space>
      </div>
      
      {/* Alerts */}
      <AlertsPanel />
      
      {/* Main Tabs */}
      <Tabs
        activeKey={selectedTab}
        onChange={setSelectedTab}
        type="card"
        size="large"
      >
        <TabPane
          tab={<span><DashboardOutlined /> Overview</span>}
          key="overview"
        >
          <OverviewTab />
        </TabPane>
        
        <TabPane
          tab={<span><SafetyOutlined /> Bandit Optimizer</span>}
          key="bandit"
        >
          <BanditTab />
        </TabPane>
        
        <TabPane
          tab={<span><ForkOutlined /> Decision DNA</span>}
          key="dna"
        >
          <DNATab />
        </TabPane>
        
        <TabPane
          tab={<span><MonitorOutlined /> System Monitoring</span>}
          key="monitoring"
        >
          <OverviewTab />
        </TabPane>
        
        <TabPane
          tab={<span><DatabaseOutlined /> Lab Tests</span>}
          key="lab"
        >
          <LabTab />
        </TabPane>
      </Tabs>
      
      {/* Footer with keyboard shortcuts */}
      <div style={{ 
        marginTop: 24,
        padding: 16,
        backgroundColor: '#fff',
        borderRadius: 4,
        textAlign: 'center'
      }}>
        <Space split={<Divider type="vertical" />}>
          <Text type="secondary">Alt+1-5: Switch tabs</Text>
          <Text type="secondary">Alt+R: Toggle refresh</Text>
          <Text type="secondary">Alt+S: Run smoke test</Text>
          <Text type="secondary">ESC: Clear alerts</Text>
        </Space>
      </div>
    </div>
  );
}