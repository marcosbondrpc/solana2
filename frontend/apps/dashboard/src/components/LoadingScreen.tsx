import { useEffect, useState } from 'react';

export function LoadingScreen({ message }: { message?: string }) {
  const [progress, setProgress] = useState(0);
  const [loadingText, setLoadingText] = useState('Initializing MEV Dashboard');

  useEffect(() => {
    const texts = [
      'Initializing MEV Dashboard',
      'Connecting to Solana RPC',
      'Loading Jito bundles',
      'Analyzing arbitrage opportunities',
      'Syncing blockchain state',
      'Optimizing performance metrics',
    ];

    let textIndex = 0;
    const textInterval = setInterval(() => {
      textIndex = (textIndex + 1) % texts.length;
      setLoadingText((texts[textIndex] as string));
    }, 1500);

    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return prev;
        return prev + Math.random() * 15;
      });
    }, 200);

    return () => {
      clearInterval(textInterval);
      clearInterval(progressInterval);
    };
  }, []);

  return (
    <div className="fixed inset-0 bg-gray-900 flex items-center justify-center z-50">
      <div className="max-w-md w-full px-8">
        <div className="text-center">
          {/* Animated Logo */}
          <div className="mb-8 relative">
            <div className="w-24 h-24 mx-auto relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full animate-pulse" />
              <div className="absolute inset-2 bg-gray-900 rounded-full" />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold text-white">MEV</span>
              </div>
            </div>
          </div>

          {/* Loading Text */}
          <h2 className="text-xl font-semibold text-white mb-2">
            {message ?? loadingText}
          </h2>
          
          {/* Progress Bar */}
          <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
          
          {/* Progress Percentage */}
          <p className="text-gray-400 text-sm mt-2">
            {Math.round(progress)}%
          </p>

          {/* Loading Dots */}
          <div className="flex justify-center mt-6 space-x-2">
            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        </div>
      </div>
    </div>
  );
}