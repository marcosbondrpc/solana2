'use client';

import { useEffect, useRef, useCallback } from 'react';
import io, { Socket } from 'socket.io-client';
import { useNodeStore } from '@/lib/store';
import { useToast } from '@/hooks/use-toast';

export function useWebSocket() {
  const socketRef = useRef<Socket | null>(null);
  const { toast } = useToast();
  const setConnected = useNodeStore((state) => state.setConnected);
  const updateNodeMetrics = useNodeStore((state) => state.updateNodeMetrics);
  const updateSystemMetrics = useNodeStore((state) => state.updateSystemMetrics);
  const updateJitoMetrics = useNodeStore((state) => state.updateJitoMetrics);
  const updateRPCMetrics = useNodeStore((state) => state.updateRPCMetrics);
  const addLog = useNodeStore((state) => state.addLog);
  const addSlotData = useNodeStore((state) => state.addSlotData);
  const addCpuData = useNodeStore((state) => state.addCpuData);
  const addMemoryData = useNodeStore((state) => state.addMemoryData);
  const addNetworkData = useNodeStore((state) => state.addNetworkData);
  const addTpsData = useNodeStore((state) => state.addTpsData);

  const connect = useCallback(() => {
    if (socketRef.current?.connected) return;

    socketRef.current = io('http://localhost:42392', {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socketRef.current.on('connect', () => {
      setConnected(true);
      toast({
        title: 'Connected',
        description: 'Connected to node monitoring server',
      });
    });

    socketRef.current.on('disconnect', () => {
      setConnected(false);
      toast({
        title: 'Disconnected',
        description: 'Lost connection to monitoring server',
        variant: 'destructive',
      });
    });

    socketRef.current.on('node-metrics', (data) => {
      updateNodeMetrics(data);
      if (data.slot) addSlotData(data.slot);
      if (data.transactionCount) {
        // Calculate TPS (simplified)
        addTpsData(data.transactionCount / 10);
      }
    });

    socketRef.current.on('system-metrics', (data) => {
      updateSystemMetrics(data);
      if (data.cpuUsage) addCpuData(data.cpuUsage);
      if (data.memoryPercent) addMemoryData(data.memoryPercent);
      if (data.networkRx && data.networkTx) {
        addNetworkData(data.networkRx, data.networkTx);
      }
    });

    socketRef.current.on('jito-metrics', (data) => {
      updateJitoMetrics(data);
    });

    socketRef.current.on('rpc-metrics', (data) => {
      updateRPCMetrics(data);
    });

    socketRef.current.on('log', (data) => {
      addLog({
        timestamp: new Date(data.timestamp),
        level: data.level,
        message: data.message,
        source: data.source || 'system',
      });
    });

    socketRef.current.on('error', (error) => {
      console.error('WebSocket error:', error);
      toast({
        title: 'Connection Error',
        description: error.message || 'WebSocket error occurred',
        variant: 'destructive',
      });
    });
  }, [
    setConnected,
    updateNodeMetrics,
    updateSystemMetrics,
    updateJitoMetrics,
    updateRPCMetrics,
    addLog,
    addSlotData,
    addCpuData,
    addMemoryData,
    addNetworkData,
    addTpsData,
    toast,
  ]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((event: string, data: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data);
    }
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return { connect, disconnect, sendMessage };
}