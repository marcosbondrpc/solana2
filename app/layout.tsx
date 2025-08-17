'use client';

import "../styles/globals.css";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSnapshot } from "valtio";
import { mevStore } from "../stores/mevStore";
import { clsx } from "clsx";
import { useEffect } from "react";
import { connectMEVWebSocket, disconnectMEVWebSocket } from "../lib/websocket";
import { startBatchFlushing, stopBatchFlushing } from "../stores/mevStore";

const navItems = [
  { href: "/", label: "Dashboard", icon: "📊" },
  { href: "/realtime", label: "Realtime", icon: "⚡" },
  { href: "/control", label: "Control", icon: "🎮" },
  { href: "/analytics", label: "Analytics", icon: "📈" },
  { href: "/datasets", label: "Datasets", icon: "💾" },
  { href: "/training", label: "Training", icon: "🧠" },
  { href: "/models", label: "Models", icon: "🤖" },
];

function ConnectionIndicator() {
  const snap = useSnapshot(mevStore);
  
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="flex items-center gap-1">
        <div className={clsx(
          "connection-indicator",
          snap.wsStatus === 'connected' && "connection-active",
          snap.wsStatus === 'connecting' && "connection-connecting",
          snap.wsStatus === 'disconnected' && "connection-inactive",
          snap.wsStatus === 'error' && "connection-inactive"
        )} />
        <span className="text-zinc-500">WS</span>
      </div>
      <div className="flex items-center gap-1">
        <div className={clsx(
          "connection-indicator",
          snap.clickhouseStatus === 'connected' && "connection-active",
          snap.clickhouseStatus === 'connecting' && "connection-connecting",
          snap.clickhouseStatus === 'disconnected' && "connection-inactive",
          snap.clickhouseStatus === 'error' && "connection-inactive"
        )} />
        <span className="text-zinc-500">CH</span>
      </div>
      <div className="flex items-center gap-1">
        <div className={clsx(
          "connection-indicator",
          snap.backendStatus === 'connected' && "connection-active",
          snap.backendStatus === 'connecting' && "connection-connecting",
          snap.backendStatus === 'disconnected' && "connection-inactive",
          snap.backendStatus === 'error' && "connection-inactive"
        )} />
        <span className="text-zinc-500">API</span>
      </div>
    </div>
  );
}

function SystemHealth() {
  const snap = useSnapshot(mevStore);
  const health = snap.health;
  
  return (
    <div className="flex items-center gap-4 text-xs">
      <div className="flex items-center gap-1">
        <span className="text-zinc-500">P50:</span>
        <span className={clsx(
          health.decisionLatencyP50 <= 8 ? "latency-excellent" : 
          health.decisionLatencyP50 <= 15 ? "latency-good" : "latency-poor"
        )}>
          {health.decisionLatencyP50.toFixed(1)}ms
        </span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-zinc-500">P99:</span>
        <span className={clsx(
          health.decisionLatencyP99 <= 20 ? "latency-excellent" : 
          health.decisionLatencyP99 <= 30 ? "latency-good" : "latency-poor"
        )}>
          {health.decisionLatencyP99.toFixed(1)}ms
        </span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-zinc-500">Land:</span>
        <span className={clsx(
          health.bundleLandRate >= 65 ? "text-green-400" : 
          health.bundleLandRate >= 55 ? "text-yellow-400" : "text-red-400"
        )}>
          {health.bundleLandRate.toFixed(1)}%
        </span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-zinc-500">Rate:</span>
        <span className={clsx(
          health.ingestionRate >= 200000 ? "text-green-400" : 
          health.ingestionRate >= 150000 ? "text-yellow-400" : "text-red-400"
        )}>
          {(health.ingestionRate / 1000).toFixed(0)}k/s
        </span>
      </div>
    </div>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  
  useEffect(() => {
    // Connect WebSocket on mount
    connectMEVWebSocket();
    startBatchFlushing();
    
    // Cleanup on unmount
    return () => {
      disconnectMEVWebSocket();
      stopBatchFlushing();
    };
  }, []);
  
  return (
    <html lang="en">
      <head>
        <title>LEGENDARY MEV Dashboard</title>
        <meta name="description" content="Ultra-high-performance Solana MEV infrastructure dashboard" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="min-h-screen bg-black text-white grid-pattern">
        <nav className="glass-dark px-6 py-3 border-b sticky top-0 z-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <Link href="/" className="flex items-center gap-2">
                <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  LEGENDARY
                </span>
                <span className="text-xs text-zinc-500">v1.0</span>
              </Link>
              
              <div className="flex items-center gap-1">
                {navItems.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={clsx(
                      "px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200",
                      pathname === item.href
                        ? "bg-zinc-800 text-white"
                        : "text-zinc-400 hover:text-white hover:bg-zinc-800/50"
                    )}
                  >
                    <span className="mr-1.5">{item.icon}</span>
                    {item.label}
                  </Link>
                ))}
              </div>
            </div>
            
            <div className="flex items-center gap-6">
              <SystemHealth />
              <ConnectionIndicator />
            </div>
          </div>
        </nav>
        
        <main className="relative">
          <div className="absolute inset-0 noise opacity-30 pointer-events-none" />
          <div className="relative z-10 p-6">
            {children}
          </div>
        </main>
        
        {/* Global alerts */}
        <div className="fixed bottom-4 right-4 z-50 space-y-2 max-w-md">
          {/* Alert container will be populated by toast notifications */}
        </div>
      </body>
    </html>
  );
}