'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useNodeStore } from '@/lib/store';
import { switchNetwork } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import { Globe, Wifi, WifiOff, RefreshCw, Server } from 'lucide-react';

const networks = [
  {
    id: 'mainnet-beta',
    name: 'Mainnet Beta',
    endpoint: 'https://api.mainnet-beta.solana.com',
    color: 'text-green-500',
  },
  {
    id: 'testnet',
    name: 'Testnet',
    endpoint: 'https://api.testnet.solana.com',
    color: 'text-yellow-500',
  },
  {
    id: 'devnet',
    name: 'Devnet',
    endpoint: 'https://api.devnet.solana.com',
    color: 'text-blue-500',
  },
] as const;

export function NetworkStatus() {
  const { toast } = useToast();
  const currentNetwork = useNodeStore((state) => state.network);
  const setNetwork = useNodeStore((state) => state.setNetwork);
  const [switching, setSwitching] = useState(false);
  const [endpoints, setEndpoints] = useState<Record<string, boolean>>({});

  useEffect(() => {
    checkEndpoints();
    const interval = setInterval(checkEndpoints, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const checkEndpoints = async () => {
    const results: Record<string, boolean> = {};
    
    for (const network of networks) {
      try {
        const response = await fetch(network.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jsonrpc: '2.0',
            id: 1,
            method: 'getHealth',
          }),
        });
        results[network.id] = response.ok;
      } catch {
        results[network.id] = false;
      }
    }
    
    setEndpoints(results);
  };

  const handleNetworkSwitch = async (networkId: typeof currentNetwork) => {
    if (networkId === currentNetwork) return;
    
    setSwitching(true);
    try {
      const result = await switchNetwork(networkId);
      
      if (result.success) {
        setNetwork(networkId);
        toast({
          title: 'Network Switched',
          description: `Successfully switched to ${networkId}`,
        });
      } else {
        toast({
          title: 'Switch Failed',
          description: result.error || 'Failed to switch network',
          variant: 'destructive',
        });
      }
    } finally {
      setSwitching(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Current Network */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Current Network</CardTitle>
              <CardDescription>Active Solana network connection</CardDescription>
            </div>
            <Globe className="h-5 w-5 text-muted-foreground" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div
                className={cn(
                  'h-3 w-3 rounded-full',
                  endpoints[currentNetwork] ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                )}
              />
              <div>
                <div className="text-lg font-semibold capitalize">{currentNetwork}</div>
                <div className="text-sm text-muted-foreground">
                  {networks.find((n) => n.id === currentNetwork)?.endpoint}
                </div>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={checkEndpoints}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Network Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Switch Network</CardTitle>
          <CardDescription>Select a different Solana network</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {networks.map((network) => (
              <Card
                key={network.id}
                className={cn(
                  'cursor-pointer transition-all',
                  currentNetwork === network.id && 'ring-2 ring-primary'
                )}
                onClick={() => handleNetworkSwitch(network.id as typeof currentNetwork)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium">
                      {network.name}
                    </CardTitle>
                    {endpoints[network.id] ? (
                      <Wifi className={cn('h-4 w-4', network.color)} />
                    ) : (
                      <WifiOff className="h-4 w-4 text-red-500" />
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="text-xs text-muted-foreground truncate">
                      {network.endpoint}
                    </div>
                    <div className="flex items-center space-x-2">
                      <div
                        className={cn(
                          'h-2 w-2 rounded-full',
                          endpoints[network.id] ? 'bg-green-500' : 'bg-red-500'
                        )}
                      />
                      <span className="text-xs">
                        {endpoints[network.id] ? 'Online' : 'Offline'}
                      </span>
                    </div>
                    {currentNetwork === network.id && (
                      <div className="text-xs font-semibold text-primary">
                        Active
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          {switching && (
            <div className="mt-4 text-center text-sm text-muted-foreground">
              Switching network...
            </div>
          )}
        </CardContent>
      </Card>

      {/* Known Validators */}
      <Card>
        <CardHeader>
          <CardTitle>Known Validators</CardTitle>
          <CardDescription>Trusted validators for {currentNetwork}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {[
                '7Np41oeYqPefeNQEHSv1UDhYrehxin3NStELsSKCT4K2',
                'GdnSyH3YtwcxFvQrVVJMm1JhTS4QVX7MFsX56uJLUfiZ',
                'DE1bawNcRJB9rVm3buyMVfr8mBEoyyu73NBovf2oXJsJ',
                'CakcnaRDHka2gXyfbEd2d3xsvkJkqsLw2akB3zsN1D2S',
              ].map((validator) => (
                <div
                  key={validator}
                  className="flex items-center space-x-2 p-2 rounded bg-muted"
                >
                  <Server className="h-4 w-4 text-muted-foreground" />
                  <span className="text-xs font-mono truncate">{validator}</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}