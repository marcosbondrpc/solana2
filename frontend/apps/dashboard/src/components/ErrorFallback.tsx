import { FallbackProps } from 'react-error-boundary';

export function ErrorFallback({ error, resetErrorBoundary }: Partial<FallbackProps>) {
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-gray-800 rounded-lg shadow-xl p-6 border border-red-500/20">
        <div className="flex items-center mb-4">
          <div className="w-12 h-12 bg-red-500/10 rounded-full flex items-center justify-center mr-4">
            <svg
              className="w-6 h-6 text-red-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Something went wrong</h2>
            <p className="text-gray-400 text-sm">An unexpected error occurred</p>
          </div>
        </div>

        <div className="bg-gray-900 rounded p-4 mb-4">
          <p className="text-sm font-mono text-red-400 break-all">
            {error?.message || 'Unknown error'}
          </p>
          {error?.stack && (
            <details className="mt-2">
              <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">
                Show stack trace
              </summary>
              <pre className="mt-2 text-xs text-gray-500 overflow-auto max-h-40">
                {error.stack}
              </pre>
            </details>
          )}
        </div>

        <div className="flex space-x-3">
          <button
            onClick={resetErrorBoundary}
            className="flex-1 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded font-medium transition-colors"
          >
            Try Again
          </button>
          <button
            onClick={() => window.location.href = '/'}
            className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded font-medium transition-colors"
          >
            Go Home
          </button>
        </div>

        <div className="mt-4 pt-4 border-t border-gray-700">
          <p className="text-xs text-gray-500 text-center">
            If this error persists, please contact support or check the console for more details.
          </p>
        </div>
      </div>
    </div>
  );
}