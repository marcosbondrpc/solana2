import { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';

interface User {
  address: string;
  publicKey: string;
  connected: boolean;
  balance?: number;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  connect: () => Promise<void>;
  disconnect: () => void;
  refreshBalance: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    const checkSession = async () => {
      try {
        const storedUser = localStorage.getItem('solana-mev-user');
        if (storedUser) {
          const userData = JSON.parse(storedUser);
          setUser(userData);
        }
      } catch (error) {
        console.error('Failed to restore session:', error);
      } finally {
        setIsLoading(false);
      }
    };

    checkSession();
  }, []);

  const connect = useCallback(async () => {
    try {
      setIsLoading(true);
      
      // Simulate wallet connection (replace with actual wallet adapter)
      const mockUser: User = {
        address: '7VKz3mh3pYxJYHFbFmxVEBBdZTEeWwAr4q5CpWNmNUFf',
        publicKey: '7VKz3mh3pYxJYHFbFmxVEBBdZTEeWwAr4q5CpWNmNUFf',
        connected: true,
        balance: 10.5, // SOL balance
      };
      
      setUser(mockUser);
      localStorage.setItem('solana-mev-user', JSON.stringify(mockUser));
    } catch (error) {
      console.error('Failed to connect wallet:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const disconnect = useCallback(() => {
    setUser(null);
    localStorage.removeItem('solana-mev-user');
  }, []);

  const refreshBalance = useCallback(async () => {
    if (!user) return;
    
    try {
      // Simulate balance refresh (replace with actual RPC call)
      const newBalance = Math.random() * 20;
      const updatedUser = { ...user, balance: newBalance };
      setUser(updatedUser);
      localStorage.setItem('solana-mev-user', JSON.stringify(updatedUser));
    } catch (error) {
      console.error('Failed to refresh balance:', error);
    }
  }, [user]);

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user?.connected,
    isLoading,
    connect,
    disconnect,
    refreshBalance,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}