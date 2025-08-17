import { useState, useEffect, useCallback } from 'react';
import { proxy, useSnapshot } from 'valtio';
import WTEcho from './pages/tools/wt-echo';
import BanditDashboard from './components/bandit/BanditDashboard';
import DecisionDNA from './components/dna/DecisionDNA';
import CommandCenter from './components/operator/CommandCenter';
import HashChainVerifier from './components/HashChainVerifier';
import ProtobufMonitor from './components/ProtobufMonitor';
import LeaderPhaseHeatmap from './components/LeaderPhaseHeatmap';
import MEVControlCenter from './components/MEVControlCenter';
import ClickHouseQueryBuilder from './components/ClickHouseQueryBuilder';
import GrafanaProvisioning from './components/GrafanaProvisioning';

// Global app state
const appState = proxy({
  currentView: 'mev-control' as 'mev-control' | 'command-center' | 'bandit' | 'dna' | 'wt-echo' | 'hash-chain' | 'protobuf' | 'leader-heatmap' | 'clickhouse' | 'grafana',
  theme: 'matrix' as 'matrix' | 'cyberpunk' | 'minimal',
  performance: {
    fps: 0,
    frameTime: 0,
    renderTime: 0,
    updateTime: 0
  },
  connected: false,
  version: '1.0.0-legendary'
});

// Performance monitoring with RAF
let lastTime = performance.now();
let frames = 0;

function measurePerformance() {
  const now = performance.now();
  const delta = now - lastTime;
  frames++;
  
  if (delta >= 1000) {
    appState.performance.fps = Math.round((frames * 1000) / delta);
    appState.performance.frameTime = delta / frames;
    frames = 0;
    lastTime = now;
  }
  
  requestAnimationFrame(measurePerformance);
}

// Start performance monitoring
measurePerformance();

export default function App() {
  const state = useSnapshot(appState);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  // WebSocket connection status
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch('/api/health');
        appState.connected = response.ok;
      } catch {
        appState.connected = false;
      }
    };
    
    checkConnection();
    const interval = setInterval(checkConnection, 5000);
    return () => clearInterval(interval);
  }, []);
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Alt+0-9 to switch views
      if (e.altKey) {
        switch (e.key) {
          case '1':
            appState.currentView = 'mev-control';
            break;
          case '2':
            appState.currentView = 'command-center';
            break;
          case '3':
            appState.currentView = 'bandit';
            break;
          case '4':
            appState.currentView = 'dna';
            break;
          case '5':
            appState.currentView = 'clickhouse';
            break;
          case '6':
            appState.currentView = 'grafana';
            break;
          case '7':
            appState.currentView = 'wt-echo';
            break;
          case '8':
            appState.currentView = 'hash-chain';
            break;
          case '9':
            appState.currentView = 'protobuf';
            break;
          case '0':
            appState.currentView = 'leader-heatmap';
            break;
        }
      }
      
      // Ctrl+B to toggle sidebar
      if (e.ctrlKey && e.key === 'b') {
        setSidebarOpen(prev => !prev);
        e.preventDefault();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
  
  const renderView = useCallback(() => {
    switch (state.currentView) {
      case 'mev-control':
        return <MEVControlCenter />;
      case 'command-center':
        return <CommandCenter />;
      case 'bandit':
        return <BanditDashboard />;
      case 'dna':
        return <DecisionDNA />;
      case 'wt-echo':
        return <WTEcho />;
      case 'hash-chain':
        return <HashChainVerifier />;
      case 'protobuf':
        return <ProtobufMonitor />;
      case 'leader-heatmap':
        return <LeaderPhaseHeatmap />;
      case 'clickhouse':
        return <ClickHouseQueryBuilder />;
      case 'grafana':
        return <GrafanaProvisioning />;
      default:
        return <MEVControlCenter />;
    }
  }, [state.currentView]);
  
  const getThemeStyles = useCallback(() => {
    switch (state.theme) {
      case 'matrix':
        return {
          primary: '#00ff00',
          secondary: '#00aa00',
          background: '#000',
          text: '#00ff00',
          border: '#00ff00'
        };
      case 'cyberpunk':
        return {
          primary: '#ff00ff',
          secondary: '#00ffff',
          background: '#0a0014',
          text: '#ff00ff',
          border: '#ff00ff'
        };
      case 'minimal':
        return {
          primary: '#ffffff',
          secondary: '#888888',
          background: '#111111',
          text: '#ffffff',
          border: '#333333'
        };
      default:
        return {
          primary: '#00ff00',
          secondary: '#00aa00',
          background: '#000',
          text: '#00ff00',
          border: '#00ff00'
        };
    }
  }, [state.theme]);
  
  const theme = getThemeStyles();
  
  return (
    <div style={{
      display: 'flex',
      height: '100vh',
      backgroundColor: theme.background,
      color: theme.text,
      fontFamily: 'Monaco, Consolas, monospace',
      overflow: 'hidden'
    }}>
      {/* Sidebar */}
      <div style={{
        width: sidebarOpen ? '250px' : '50px',
        backgroundColor: theme.background,
        borderRight: `1px solid ${theme.border}`,
        transition: 'width 0.3s ease',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column'
      }}>
        {/* Logo/Title */}
        <div style={{
          padding: '20px',
          borderBottom: `1px solid ${theme.border}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: sidebarOpen ? 'space-between' : 'center'
        }}>
          {sidebarOpen && (
            <div style={{
              fontSize: '16px',
              fontWeight: 'bold',
              textShadow: `0 0 10px ${theme.primary}`
            }}>
              SOL MEV HQ
            </div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            style={{
              background: 'transparent',
              border: `1px solid ${theme.border}`,
              color: theme.text,
              cursor: 'pointer',
              padding: '5px 10px',
              borderRadius: '4px'
            }}
          >
            {sidebarOpen ? '◀' : '▶'}
          </button>
        </div>
        
        {/* Navigation */}
        <nav style={{ flex: 1, padding: sidebarOpen ? '20px' : '10px' }}>
          {[
            { id: 'mev-control', label: 'MEV Control Center', icon: '🚀', key: '1' },
            { id: 'command-center', label: 'Command Center', icon: '⌘', key: '2' },
            { id: 'bandit', label: 'Bandit Optimizer', icon: '🎰', key: '3' },
            { id: 'dna', label: 'Decision DNA', icon: '🧬', key: '4' },
            { id: 'clickhouse', label: 'ClickHouse Query', icon: '🗄️', key: '5' },
            { id: 'grafana', label: 'Grafana Setup', icon: '📈', key: '6' },
            { id: 'wt-echo', label: 'WebTransport', icon: '📡', key: '7' },
            { id: 'hash-chain', label: 'Hash Chain', icon: '🔗', key: '8' },
            { id: 'protobuf', label: 'Protobuf Monitor', icon: '📊', key: '9' },
            { id: 'leader-heatmap', label: 'Leader Heatmap', icon: '🗺️', key: '0' }
          ].map(item => (
            <div
              key={item.id}
              onClick={() => appState.currentView = item.id as any}
              style={{
                padding: '10px',
                marginBottom: '5px',
                cursor: 'pointer',
                borderRadius: '4px',
                background: state.currentView === item.id ? `${theme.primary}22` : 'transparent',
                border: state.currentView === item.id ? `1px solid ${theme.primary}` : '1px solid transparent',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = `${theme.primary}11`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = state.currentView === item.id ? `${theme.primary}22` : 'transparent';
              }}
            >
              <span style={{ fontSize: '18px' }}>{item.icon}</span>
              {sidebarOpen && (
                <>
                  <span style={{ flex: 1 }}>{item.label}</span>
                  <span style={{ fontSize: '10px', opacity: 0.5 }}>Alt+{item.key}</span>
                </>
              )}
            </div>
          ))}
        </nav>
        
        {/* Status */}
        <div style={{
          padding: sidebarOpen ? '20px' : '10px',
          borderTop: `1px solid ${theme.border}`,
          fontSize: '12px'
        }}>
          {sidebarOpen && (
            <>
              <div style={{ marginBottom: '10px' }}>
                Status: <span style={{ color: state.connected ? theme.primary : '#ff0000' }}>
                  {state.connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <div style={{ marginBottom: '10px' }}>
                FPS: {state.performance.fps}
              </div>
              <div style={{ marginBottom: '10px' }}>
                Frame Time: {state.performance.frameTime.toFixed(2)}ms
              </div>
              <div>
                Version: {state.version}
              </div>
            </>
          )}
        </div>
        
        {/* Theme Selector */}
        {sidebarOpen && (
          <div style={{
            padding: '20px',
            borderTop: `1px solid ${theme.border}`
          }}>
            <label style={{ fontSize: '12px' }}>Theme:</label>
            <select
              value={appState.theme}
              onChange={(e) => appState.theme = e.target.value as any}
              style={{
                width: '100%',
                marginTop: '5px',
                padding: '5px',
                background: theme.background,
                color: theme.text,
                border: `1px solid ${theme.border}`,
                borderRadius: '4px'
              }}
            >
              <option value="matrix">Matrix</option>
              <option value="cyberpunk">Cyberpunk</option>
              <option value="minimal">Minimal</option>
            </select>
          </div>
        )}
      </div>
      
      {/* Main Content */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {renderView()}
      </div>
      
      {/* Performance Overlay */}
      <div style={{
        position: 'fixed',
        top: '10px',
        right: '10px',
        padding: '10px',
        background: 'rgba(0, 0, 0, 0.8)',
        border: `1px solid ${theme.border}`,
        borderRadius: '4px',
        fontSize: '10px',
        fontFamily: 'monospace',
        pointerEvents: 'none',
        zIndex: 9999
      }}>
        <div>FPS: {state.performance.fps}</div>
        <div>Frame: {state.performance.frameTime.toFixed(2)}ms</div>
      </div>
    </div>
  );
}