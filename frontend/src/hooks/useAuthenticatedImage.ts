/**
 * Hook to fetch forensic image artifacts using the in-memory Bearer access token.
 * Prevents HTTP 401 errors on native <img> tags while ensuring strict access control.
 */

import { useState, useEffect } from 'react';
import { authStore } from '../lib/auth-store';

export interface UseAuthenticatedImageResult {
  blobUrl: string | null;
  loading: boolean;
  error: boolean;
}

export function useAuthenticatedImage(url: string | null | undefined): UseAuthenticatedImageResult {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    const targetUrl = url;
    if (!targetUrl) {
      setBlobUrl(null);
      setLoading(false);
      setError(false);
      return;
    }

    let active = true;
    let localBlobUrl: string | null = null;

    async function fetchImage() {
      setLoading(true);
      setError(false);

      try {
        const token = authStore.getToken();
        const headers: Record<string, string> = {};
        if (token) {
          headers.Authorization = `Bearer ${token}`;
        }

        const requestUrl = targetUrl;
        if (!requestUrl) return;

        const res = await fetch(requestUrl, { headers, credentials: 'include' });
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }

        const blob = await res.blob();
        if (active) {
          localBlobUrl = URL.createObjectURL(blob);
          setBlobUrl(localBlobUrl);
          setLoading(false);
        }
      } catch {
        if (active) {
          setError(true);
          setLoading(false);
        }
      }
    }

    void fetchImage();

    return () => {
      active = false;
      if (localBlobUrl) {
        URL.revokeObjectURL(localBlobUrl);
      }
    };
  }, [url]);

  return { blobUrl, loading, error };
}
