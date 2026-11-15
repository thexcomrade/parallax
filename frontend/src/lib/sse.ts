/**
 * Fetch-stream based Server-Sent Events (SSE) subscriber.
 *
 * Traditional EventSource cannot send Authorization: Bearer headers.
 * We use fetch + ReadableStream to stream stage events securely.
 */

import { authStore } from './auth-store';
import type { StageEventPayload } from '../types/api';

export interface SSEOptions {
  onEvent: (event: StageEventPayload) => void;
  onError?: (err: Error) => void;
  onDone?: () => void;
}

export function subscribeToAnalysisEvents(
  analysisId: string,
  options: SSEOptions,
): () => void {
  const controller = new AbortController();
  let isDone = false;

  async function connect() {
    try {
      const token = authStore.getToken();
      const headers: Record<string, string> = {
        Accept: 'text/event-stream',
      };
      if (token) {
        headers.Authorization = `Bearer ${token}`;
      }

      const response = await fetch(`/api/v1/analyses/${analysisId}/events`, {
        headers,
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`SSE HTTP error: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('Response body is empty');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data:')) {
            const jsonStr = trimmed.slice(5).trim();
            if (jsonStr) {
              try {
                const data: StageEventPayload = JSON.parse(jsonStr);
                options.onEvent(data);
                if (data.stage === 'done') {
                  isDone = true;
                  options.onDone?.();
                  return;
                }
              } catch (e) {
                console.error('Failed to parse SSE payload:', e);
              }
            }
          }
        }
      }

      if (!isDone && !controller.signal.aborted) {
        options.onDone?.();
      }
    } catch (err: unknown) {
      if (!controller.signal.aborted) {
        options.onError?.(err instanceof Error ? err : new Error(String(err)));
      }
    }
  }

  void connect();

  return () => {
    controller.abort();
  };
}
