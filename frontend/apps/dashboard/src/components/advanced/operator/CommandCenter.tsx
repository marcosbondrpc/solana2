import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { proxy, useSnapshot } from 'valtio';

// Command center state
const commandState = proxy({
  panes: [
    { id: 'mev', title: 'MEV Agent', active: true, output: [], status: 'running' },
    { id: 'arb', title: 'ARB Agent', active: false, output: [], status: 'running' },
    { id: 'control', title: 'Control Plane', active: false, output: [], status: 'running' },
    { id: 'metrics', title: 'Metrics', active: false, output: [], status: 'running' },
    { id: 'logs', title: 'System Logs', active: false, output: [], status: 'running' },
    { id: 'shell', title: 'Shell', active: false, output: [], status: 'idle' }
  ] as Array<{
    id: string;
    title: string;
    active: boolean;
    output: string[];
    status: 'running' | 'stopped' | 'error' | 'idle';
  }>,
  
  activePane: 'mev',
  commandHistory: [] as string[],
  historyIndex: -1,
  currentCommand: '',
  
  systemMetrics: {
    cpuUsage: 0,
    memoryUsage: 0,
    networkIn: 0,
    networkOut: 0,
    diskIO: 0,
    uptime: 0,
    coreTemps: [] as number[],
    latencyP99: 0
  },
  
  alerts: [] as Array<{
    id: string;
    level: 'info' | 'warning' | 'error' | 'critical';
    message: string;
    timestamp: number;
  }>,
  
  hashChain: {
    current: '',
    previous: '',
    height: 0,
    verified: true
  }
});

// Terminal output buffer (ring buffer for performance)
class OutputBuffer {
  private buffer: string[];
  private capacity: number;
  private writePos = 0;
  
  constructor(capacity: number = 10000) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }
  
  push(line: string): void {
    this.buffer[this.writePos % this.capacity] = line;
    this.writePos++;
  }
  
  getLines(count: number): string[] {
    const start = Math.max(0, this.writePos - count);
    const lines: string[] = [];
    for (let i = start; i < this.writePos; i++) {
      const line = this.buffer[i % this.capacity];
      if (line !== undefined) lines.push(line);
    }
    return lines;
  }
  
  clear(): void {
    this.writePos = 0;
    this.buffer = new Array(this.capacity);
  }
}

// Output buffers for each pane
const outputBuffers = new Map<string, OutputBuffer>();
commandState.panes.forEach(pane => {
  outputBuffers.set(pane.id, new OutputBuffer());
});

// ANSI color codes to HTML
function ansiToHtml(text: string): string {
  const ansiRegex = /\x1b\[([0-9;]+)m/g;
  const colorMap: { [key: string]: string } = {
    '30': 'color: #000',
    '31': 'color: #ff0000',
    '32': 'color: #00ff00',
    '33': 'color: #ffff00',
    '34': 'color: #0000ff',
    '35': 'color: #ff00ff',
    '36': 'color: #00ffff',
    '37': 'color: #fff',
    '90': 'color: #666',
    '91': 'color: #ff6666',
    '92': 'color: #66ff66',
    '93': 'color: #ffff66',
    '94': 'color: #6666ff',
    '95': 'color: #ff66ff',
    '96': 'color: #66ffff',
    '97': 'color: #ffffff',
    '1': 'font-weight: bold',
    '3': 'font-style: italic',
    '4': 'text-decoration: underline'
  };
  
  return text.replace(ansiRegex, (match, codes) => {
    const styles = codes.split(';')
      .map(code => colorMap[code])
      .filter(Boolean)
      .join('; ');
    return styles ? `<span style="${styles}">` : '</span>';
  });
}

export default function CommandCenter() {
  const state = useSnapshot(commandState);
  const terminalRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const inputRef = useRef<HTMLInputElement>(null);
  const [layout, setLayout] = useState<'horizontal' | 'vertical' | 'grid'>('grid');
  const wsRef = useRef<Map<string, WebSocket>>(new Map());
  
  // Connect to WebSocket streams for each pane
  useEffect(() => {
    const endpoints = {
      mev: 'ws://localhost:8080/stream/mev',
      arb: 'ws://localhost:8080/stream/arb',
      control: 'ws://localhost:8080/stream/control',
      metrics: 'ws://localhost:8080/stream/metrics',
      logs: 'ws://localhost:8080/stream/logs'
    };
    
    Object.entries(endpoints).forEach(([id, url]) => {
      const ws = new WebSocket(url);
      wsRef.current.set(id, ws);
      
      ws.onmessage = (event) => {
        const buffer = outputBuffers.get(id);
        if (buffer) {
          const iso = new Date().toISOString();
          const tIndex = iso.indexOf('T');
          const dotIndex = iso.indexOf('.');
          const timestamp = tIndex !== -1 && dotIndex !== -1 ? iso.slice(tIndex + 1, dotIndex) : iso;
          buffer.push(`[${timestamp}] ${event.data}`);
          
          // Update pane output
          const pane = commandState.panes.find(p => p.id === id);
          if (pane) {
            pane.output = buffer.getLines(100);
          }
        }
      };
      
      ws.onerror = () => {
        const pane = commandState.panes.find(p => p.id === id);
        if (pane) pane.status = 'error';
      };
      
      ws.onclose = () => {
        const pane = commandState.panes.find(p => p.id === id);
        if (pane) pane.status = 'stopped';
      };
    });
    
    return () => {
      wsRef.current.forEach(ws => ws.close());
    };
  }, []);
  
  // System metrics updater
  useEffect(() => {
    const updateMetrics = async () => {
      try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        commandState.systemMetrics = {
          cpuUsage: data.cpu || 0,
          memoryUsage: data.memory || 0,
          networkIn: data.network?.in || 0,
          networkOut: data.network?.out || 0,
          diskIO: data.disk?.io || 0,
          uptime: data.uptime || 0,
          coreTemps: data.temps || [],
          latencyP99: data.latency?.p99 || 0
        };
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      }
    };
    
    const interval = setInterval(updateMetrics, 1000);
    return () => clearInterval(interval);
  }, []);
  
  // Hash chain verifier
  useEffect(() => {
    const verifyHashChain = async () => {
      try {
        const response = await fetch('/api/hashchain/latest');
        const data = await response.json();
        
        commandState.hashChain = {
          current: data.hash,
          previous: data.previousHash,
          height: data.height,
          verified: data.verified
        };
        
        if (!data.verified) {
          commandState.alerts.push({
            id: `alert_${Date.now()}`,
            level: 'critical',
            message: 'Hash chain verification failed!',
            timestamp: Date.now()
          });
        }
      } catch (error) {
        console.error('Hash chain verification error:', error);
      }
    };
    
    const interval = setInterval(verifyHashChain, 5000);
    return () => clearInterval(interval);
  }, []);
  
  // Command execution
  const executeCommand = useCallback(async (command: string) => {
    if (!command.trim()) return;
    
    // Add to history
    commandState.commandHistory.push(command);
    if (commandState.commandHistory.length > 100) {
      commandState.commandHistory.shift();
    }
    commandState.historyIndex = -1;
    commandState.currentCommand = '';
    
    const activePane = commandState.panes.find(p => p.id === commandState.activePane);
    if (!activePane) return;
    
    const buffer = outputBuffers.get(activePane.id);
    if (!buffer) return;
    
    // Add command to output
    buffer.push(`$ ${command}`);
    
    // Parse and execute command
    const [cmd, ...args] = command.split(' ');
    
    switch (cmd) {
      case 'clear':
        buffer.clear();
        activePane.output = [];
        break;
        
      case 'restart':
        const service = args[0];
        if (service) {
          buffer.push(`Restarting ${service}...`);
          // Send restart command via WebSocket
          const ws = wsRef.current.get('control');
          if (ws) {
            ws.send(JSON.stringify({ command: 'restart', service }));
          }
        }
        break;
        
      case 'status':
        commandState.panes.forEach(pane => {
          buffer.push(`${pane.title}: ${pane.status}`);
        });
        break;
        
      case 'perf':
        buffer.push(`CPU: ${commandState.systemMetrics.cpuUsage.toFixed(1)}%`);
        buffer.push(`Memory: ${commandState.systemMetrics.memoryUsage.toFixed(1)}%`);
        buffer.push(`Network In: ${(commandState.systemMetrics.networkIn / 1024 / 1024).toFixed(2)} MB/s`);
        buffer.push(`Network Out: ${(commandState.systemMetrics.networkOut / 1024 / 1024).toFixed(2)} MB/s`);
        buffer.push(`P99 Latency: ${commandState.systemMetrics.latencyP99.toFixed(2)}ms`);
        break;
        
      case 'help':
        buffer.push('Available commands:');
        buffer.push('  clear        - Clear current pane');
        buffer.push('  restart <service> - Restart a service');
        buffer.push('  status       - Show all service status');
        buffer.push('  perf         - Show performance metrics');
        buffer.push('  layout <mode> - Change layout (horizontal/vertical/grid)');
        buffer.push('  switch <pane> - Switch to pane');
        break;
        
      case 'layout':
        {
          const modeArg = args[0];
          if (modeArg === 'horizontal' || modeArg === 'vertical' || modeArg === 'grid') {
            setLayout(modeArg);
            buffer.push(`Layout changed to ${modeArg}`);
          }
        }
        break;
        
      case 'switch':
        {
          const paneId = args[0];
          if (!paneId) break;
          const targetPane = commandState.panes.find(p => p.id === paneId);
          if (targetPane) {
            commandState.activePane = paneId;
            commandState.panes.forEach(p => p.active = p.id === paneId);
            buffer.push(`Switched to ${targetPane.title}`);
          }
        }
        break;
        
      default:
        // Send raw command to shell pane
        if (activePane.id === 'shell') {
          const ws = wsRef.current.get('shell');
          if (ws) {
            ws.send(command);
          }
        } else {
          buffer.push(`Unknown command: ${cmd}`);
        }
    }
    
    // Update pane output
    activePane.output = buffer.getLines(100);
  }, []);
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+1-6 to switch panes
      if (e.ctrlKey && e.key >= '1' && e.key <= '6') {
        const index = parseInt(e.key) - 1;
        if (index < commandState.panes.length) {
          const pane = commandState.panes[index];
          commandState.activePane = pane.id;
          commandState.panes.forEach(p => p.active = p.id === pane.id);
        }
        e.preventDefault();
      }
      
      // Ctrl+L to clear
      if (e.ctrlKey && e.key === 'l') {
        const activePane = commandState.panes.find(p => p.id === commandState.activePane);
        if (activePane) {
          const buffer = outputBuffers.get(activePane.id);
          if (buffer) {
            buffer.clear();
            activePane.output = [];
          }
        }
        e.preventDefault();
      }
      
      // Up/Down for history
      if (e.key === 'ArrowUp' && inputRef.current === document.activeElement) {
        if (commandState.historyIndex < commandState.commandHistory.length - 1) {
          commandState.historyIndex++;
          const historyCommand = commandState.commandHistory[
            commandState.commandHistory.length - 1 - commandState.historyIndex
          ] ?? '';
          commandState.currentCommand = historyCommand;
          if (inputRef.current) {
            inputRef.current.value = historyCommand;
          }
        }
        e.preventDefault();
      }
      
      if (e.key === 'ArrowDown' && inputRef.current === document.activeElement) {
        if (commandState.historyIndex > 0) {
          commandState.historyIndex--;
          const historyCommand = commandState.commandHistory[
            commandState.commandHistory.length - 1 - commandState.historyIndex
          ] ?? '';
          commandState.currentCommand = historyCommand;
          if (inputRef.current) {
            inputRef.current.value = historyCommand;
          }
        } else if (commandState.historyIndex === 0) {
          commandState.historyIndex = -1;
          commandState.currentCommand = '';
          if (inputRef.current) {
            inputRef.current.value = '';
          }
        }
        e.preventDefault();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
  
  // Auto-scroll to bottom
  useEffect(() => {
    terminalRefs.current.forEach((ref, id) => {
      if (ref) {
        ref.scrollTop = ref.scrollHeight;
      }
    });
  }, [state.panes]);
  
  // Layout styles
  const getLayoutStyle = useMemo(() => {
    switch (layout) {
      case 'horizontal':
        return {
          display: 'grid',
          gridTemplateColumns: '1fr',
          gridTemplateRows: 'repeat(auto-fit, minmax(150px, 1fr))'
        };
      case 'vertical':
        return {
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gridTemplateRows: '1fr'
        };
      case 'grid':
      default:
        return {
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gridTemplateRows: 'repeat(2, 1fr)'
        };
    }
  }, [layout]);
  
  return (
    <div className="command-center" style={{
      height: '100vh',
      backgroundColor: '#000',
      color: '#00ff00',
      fontFamily: 'Monaco, monospace',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <div style={{
        height: '40px',
        borderBottom: '2px solid #00ff00',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 20px',
        background: 'linear-gradient(180deg, #001100 0%, #000 100%)'
      }}>
        <div style={{ fontSize: '18px', fontWeight: 'bold', textShadow: '0 0 10px #00ff00' }}>
          OPERATOR COMMAND CENTER
        </div>
        
        <div style={{ display: 'flex', gap: '20px', fontSize: '12px' }}>
          <div>CPU: {state.systemMetrics.cpuUsage.toFixed(1)}%</div>
          <div>MEM: {state.systemMetrics.memoryUsage.toFixed(1)}%</div>
          <div>NET: ↓{(state.systemMetrics.networkIn / 1024 / 1024).toFixed(1)} ↑{(state.systemMetrics.networkOut / 1024 / 1024).toFixed(1)} MB/s</div>
          <div>P99: {state.systemMetrics.latencyP99.toFixed(2)}ms</div>
          <div style={{ color: commandState.hashChain.verified ? '#00ff00' : '#ff0000' }}>
            CHAIN: #{commandState.hashChain.height}
          </div>
        </div>
      </div>
      
      {/* Alerts */}
      {state.alerts.slice(-3).map(alert => (
        <div key={alert.id} style={{
          padding: '5px 20px',
          background: alert.level === 'critical' ? '#ff000033' :
                     alert.level === 'error' ? '#ff660033' :
                     alert.level === 'warning' ? '#ffff0033' : '#00ff0033',
          borderBottom: `1px solid ${
            alert.level === 'critical' ? '#ff0000' :
            alert.level === 'error' ? '#ff6600' :
            alert.level === 'warning' ? '#ffff00' : '#00ff00'
          }`
        }}>
          [{alert.level.toUpperCase()}] {alert.message}
        </div>
      ))}
      
      {/* Pane Tabs */}
      <div style={{
        height: '30px',
        display: 'flex',
        borderBottom: '1px solid #00ff00',
        background: '#001100'
      }}>
        {state.panes.map((pane, index) => (
          <div
            key={pane.id}
            onClick={() => {
              commandState.activePane = pane.id;
              commandState.panes.forEach(p => p.active = p.id === pane.id);
            }}
            style={{
              padding: '5px 15px',
              cursor: 'pointer',
              borderRight: '1px solid #00ff00',
              background: pane.active ? '#00ff0022' : 'transparent',
              color: pane.status === 'error' ? '#ff0000' :
                     pane.status === 'stopped' ? '#ffff00' :
                     pane.active ? '#00ff00' : '#00ff0099',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: pane.status === 'running' ? '#00ff00' :
                         pane.status === 'error' ? '#ff0000' :
                         pane.status === 'stopped' ? '#ffff00' : '#666'
            }} />
            {pane.title} [{index + 1}]
          </div>
        ))}
      </div>
      
      {/* Terminal Panes */}
      <div style={{ ...getLayoutStyle, flex: 1, padding: '10px', gap: '10px' }}>
        {state.panes.map(pane => (
          <div
            key={pane.id}
            style={{
              border: `1px solid ${pane.active ? '#00ff00' : '#00ff0044'}`,
              borderRadius: '4px',
              display: pane.active || layout !== 'grid' ? 'flex' : 'none',
              flexDirection: 'column',
              overflow: 'hidden',
              background: '#000'
            }}
          >
            <div style={{
              padding: '5px 10px',
              borderBottom: '1px solid #00ff0044',
              fontSize: '12px',
              background: '#001100'
            }}>
              {pane.title}
            </div>
            
            <div
              ref={ref => {
                if (ref) terminalRefs.current.set(pane.id, ref);
              }}
              style={{
                flex: 1,
                padding: '10px',
                overflow: 'auto',
                fontSize: '12px',
                lineHeight: '1.4'
              }}
            >
              {pane.output.map((line, i) => (
                <div key={i} dangerouslySetInnerHTML={{ __html: ansiToHtml(line) }} />
              ))}
            </div>
          </div>
        ))}
      </div>
      
      {/* Command Input */}
      <div style={{
        height: '40px',
        borderTop: '2px solid #00ff00',
        display: 'flex',
        alignItems: 'center',
        padding: '0 20px',
        background: '#001100'
      }}>
        <span style={{ marginRight: '10px' }}>
          [{state.activePane}]$
        </span>
        <input
          ref={inputRef}
          type="text"
          value={state.currentCommand}
          onChange={(e) => commandState.currentCommand = e.target.value}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              executeCommand(state.currentCommand);
              commandState.currentCommand = '';
              e.currentTarget.value = '';
            }
          }}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            color: '#00ff00',
            fontSize: '14px',
            fontFamily: 'Monaco, monospace',
            outline: 'none'
          }}
          placeholder="Enter command... (help for commands)"
        />
      </div>
    </div>
  );
}