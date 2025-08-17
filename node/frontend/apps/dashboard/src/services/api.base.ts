/**
 * Base API Service
 * Core HTTP client with retry logic, error handling, and request interceptors
 */

import { API_CONFIG } from '../config/api.config';
import { ApiResponse, ApiError, ApiErrorCode } from '../types/api.types';

export interface RequestOptions extends RequestInit {
  retries?: number;
  retryDelay?: number;
  timeout?: number;
  skipAuth?: boolean;
}

class ApiBase {
  private baseUrl: string;
  private defaultHeaders: HeadersInit;
  private abortControllers: Map<string, AbortController> = new Map();

  constructor() {
    this.baseUrl = API_CONFIG.API_BASE_URL;
    this.defaultHeaders = API_CONFIG.request.headers;
  }

  /**
   * Make an HTTP request with automatic retry and error handling
   */
  protected async request<T = any>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<ApiResponse<T>> {
    const {
      retries = API_CONFIG.request.retries,
      retryDelay = API_CONFIG.request.retryDelay,
      timeout = API_CONFIG.request.timeout,
      skipAuth = false,
      ...fetchOptions
    } = options;

    const url = `${this.baseUrl}${endpoint}`;
    const requestId = this.generateRequestId();
    
    // Cancel any existing request to the same endpoint
    this.cancelRequest(endpoint);
    
    // Create new abort controller for this request
    const abortController = new AbortController();
    this.abortControllers.set(endpoint, abortController);

    // Set up timeout
    const timeoutId = setTimeout(() => {
      abortController.abort();
    }, timeout);

    // Merge headers
    const headers = new Headers({
      ...this.defaultHeaders,
      ...(fetchOptions.headers as Record<string, string> || {}),
      'X-Request-ID': requestId,
    });

    // Add auth token if available and not skipped
    if (!skipAuth) {
      const token = this.getAuthToken();
      if (token) {
        headers.set('Authorization', `Bearer ${token}`);
      }
    }

    const finalOptions: RequestInit = {
      ...fetchOptions,
      headers,
      signal: abortController.signal,
    };

    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const response = await fetch(url, finalOptions);
        clearTimeout(timeoutId);
        
        // Handle rate limiting
        if (response.status === 429) {
          const retryAfter = response.headers.get('Retry-After');
          const delay = retryAfter ? parseInt(retryAfter) * 1000 : retryDelay * Math.pow(2, attempt);
          
          if (attempt < retries) {
            await this.delay(delay);
            continue;
          }
        }

        // Parse response
        const data = await this.parseResponse<T>(response);
        
        if (!response.ok) {
          const error: ApiError = {
            code: this.getErrorCode(response.status),
            message: data.message || `Request failed with status ${response.status}`,
            details: data,
          };
          
          return {
            success: false,
            error,
            timestamp: Date.now(),
            requestId,
          };
        }

        return {
          success: true,
          data: data.data || data,
          timestamp: Date.now(),
          requestId,
        };
      } catch (error) {
        lastError = error as Error;
        
        // Don't retry on abort
        if (error instanceof Error && error.name === 'AbortError') {
          clearTimeout(timeoutId);
          
          const apiError: ApiError = {
            code: ApiErrorCode.TIMEOUT,
            message: 'Request timeout',
            details: { endpoint, timeout },
          };
          
          return {
            success: false,
            error: apiError,
            timestamp: Date.now(),
            requestId,
          };
        }
        
        // Retry with exponential backoff
        if (attempt < retries) {
          const delay = retryDelay * Math.pow(2, attempt);
          await this.delay(delay);
          continue;
        }
      }
    }

    // All retries exhausted
    clearTimeout(timeoutId);
    this.abortControllers.delete(endpoint);
    
    const apiError: ApiError = {
      code: ApiErrorCode.NETWORK_ERROR,
      message: lastError?.message || 'Network request failed',
      details: { endpoint, attempts: retries + 1 },
    };
    
    return {
      success: false,
      error: apiError,
      timestamp: Date.now(),
      requestId,
    };
  }

  /**
   * GET request
   */
  protected get<T = any>(endpoint: string, options?: RequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...options, method: 'GET' });
  }

  /**
   * POST request
   */
  protected post<T = any>(
    endpoint: string,
    body?: any,
    options?: RequestOptions
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
    });
  }

  /**
   * PUT request
   */
  protected put<T = any>(
    endpoint: string,
    body?: any,
    options?: RequestOptions
  ): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PUT',
      body: body ? JSON.stringify(body) : undefined,
    });
  }

  /**
   * DELETE request
   */
  protected delete<T = any>(endpoint: string, options?: RequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' });
  }

  /**
   * Cancel a pending request
   */
  protected cancelRequest(endpoint: string): void {
    const controller = this.abortControllers.get(endpoint);
    if (controller) {
      controller.abort();
      this.abortControllers.delete(endpoint);
    }
  }

  /**
   * Cancel all pending requests
   */
  protected cancelAllRequests(): void {
    this.abortControllers.forEach(controller => controller.abort());
    this.abortControllers.clear();
  }

  /**
   * Parse response based on content type
   */
  private async parseResponse<T>(response: Response): Promise<any> {
    const contentType = response.headers.get('content-type');
    
    if (contentType?.includes('application/json')) {
      return response.json();
    } else if (contentType?.includes('text/')) {
      return response.text();
    } else if (contentType?.includes('application/octet-stream')) {
      return response.blob();
    }
    
    // Default to JSON
    try {
      return response.json();
    } catch {
      return response.text();
    }
  }

  /**
   * Get error code from status
   */
  private getErrorCode(status: number): ApiErrorCode {
    switch (status) {
      case 401:
        return ApiErrorCode.UNAUTHORIZED;
      case 403:
        return ApiErrorCode.FORBIDDEN;
      case 404:
        return ApiErrorCode.NOT_FOUND;
      case 422:
      case 400:
        return ApiErrorCode.VALIDATION_ERROR;
      case 429:
        return ApiErrorCode.RATE_LIMIT;
      case 503:
        return ApiErrorCode.MAINTENANCE;
      case 500:
      case 502:
      case 504:
        return ApiErrorCode.SERVER_ERROR;
      default:
        return ApiErrorCode.NETWORK_ERROR;
    }
  }

  /**
   * Get auth token from storage
   */
  private getAuthToken(): string | null {
    // Check multiple sources for auth token
    return (
      localStorage.getItem('auth_token') ||
      sessionStorage.getItem('auth_token') ||
      null
    );
  }

  /**
   * Generate unique request ID
   */
  private generateRequestId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Delay helper for retries
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

export default ApiBase;