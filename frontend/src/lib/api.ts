// Typed fetch wrapper.
// - Injects Bearer access token from authStore.
// - On 401: attempts ONE silent refresh, retries the original request.
// - If refresh also fails: clears auth state (triggers redirect via ProtectedRoute).
// - Never logs tokens.

import { authStore } from '@/lib/auth-store';
import type { TokenResponse } from '@/types/api';

const rawApiUrl = import.meta.env.VITE_API_URL as string | undefined;
export const API_BASE = rawApiUrl ? `${rawApiUrl.replace(/\/+$/, '')}/api/v1` : '/api/v1';
const BASE = API_BASE;

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// Single-flight refresh promise — shared across concurrent 401s so only one
// refresh call is made no matter how many requests race.
let refreshPromise: Promise<boolean> | null = null;

async function attemptRefresh(): Promise<boolean> {
  if (refreshPromise) return refreshPromise;

  refreshPromise = (async () => {
    try {
      const res = await fetch(`${BASE}/auth/refresh`, {
        method: 'POST',
        credentials: 'include', // sends the HttpOnly refresh cookie
      });
      if (!res.ok) {
        authStore.clear();
        return false;
      }
      const data = (await res.json()) as TokenResponse;
      authStore.set(data.access_token, data.user);
      return true;
    } catch {
      authStore.clear();
      return false;
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}

async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = (await res.json()) as { code?: string; message?: string };
    return new ApiError(
      res.status,
      body.code ?? 'UNKNOWN',
      body.message ?? res.statusText,
    );
  } catch {
    return new ApiError(res.status, 'UNKNOWN', res.statusText);
  }
}

interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: unknown;
  /** Skip 401 retry (used internally for the refresh call) */
  _retry?: boolean;
}

export async function apiFetch<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { body, _retry = false, ...rest } = options;

  const headers: Record<string, string> = {
    ...(rest.headers as Record<string, string> | undefined),
  };

  const isFormData = body instanceof FormData;
  if (body !== undefined && !isFormData) {
    headers['Content-Type'] = 'application/json';
  }

  const token = authStore.getToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE}${path}`, {
    ...rest,
    credentials: 'include',
    headers,
    body: isFormData ? body : body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (res.status === 401 && !_retry) {
    const refreshed = await attemptRefresh();
    if (refreshed) {
      return apiFetch<T>(path, { ...options, _retry: true });
    }
    throw await parseError(res);
  }

  if (!res.ok) {
    throw await parseError(res);
  }

  // 204 No Content
  if (res.status === 204) {
    return undefined as unknown as T;
  }

  return res.json() as Promise<T>;
}

// ── Convenience helpers ──────────────────────────────────────────────────────

export const api = {
  get<T>(path: string, options?: RequestOptions): Promise<T> {
    return apiFetch<T>(path, { ...options, method: 'GET' });
  },

  post<T>(path: string, body?: unknown, options?: RequestOptions): Promise<T> {
    return apiFetch<T>(path, { ...options, method: 'POST', body });
  },

  delete<T>(path: string, options?: RequestOptions): Promise<T> {
    return apiFetch<T>(path, { ...options, method: 'DELETE' });
  },

  uploadWithProgress<T>(
    path: string,
    formData: FormData,
    onProgress?: (progress: { loaded: number; total: number; percentage: number }) => void,
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${BASE}${path}`, true);
      xhr.withCredentials = true;

      const token = authStore.getToken();
      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      }

      if (xhr.upload && onProgress) {
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && e.total > 0) {
            const percentage = Math.min(100, Math.round((e.loaded / e.total) * 100));
            onProgress({ loaded: e.loaded, total: e.total, percentage });
          }
        };
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText) as T;
            resolve(data);
          } catch {
            resolve(undefined as unknown as T);
          }
        } else {
          try {
            const errorBody = JSON.parse(xhr.responseText) as { code?: string; message?: string };
            reject(
              new ApiError(
                xhr.status,
                errorBody.code ?? 'UPLOAD_ERROR',
                errorBody.message ?? xhr.statusText,
              ),
            );
          } catch {
            reject(new ApiError(xhr.status, 'UPLOAD_ERROR', xhr.statusText || 'Upload failed'));
          }
        }
      };

      xhr.onerror = () => {
        reject(new ApiError(0, 'NETWORK_ERROR', 'Network connection interrupted during upload'));
      };

      xhr.send(formData);
    });
  },
} as const;
