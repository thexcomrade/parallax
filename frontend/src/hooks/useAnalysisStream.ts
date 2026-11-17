/**
 * Hook to subscribe to live SSE stage events for an analysis job.
 * Handles live Server-Sent Events, automatic HTTP polling fallback,
 * reconnections, and honest state synchronization.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { subscribeToAnalysisEvents } from '../lib/sse';
import { api } from '../lib/api';
import type { AnalysisDetail, StageEventPayload } from '../types/api';

export type StreamConnectionStatus = 'connecting' | 'connected' | 'fallback' | 'disconnected';

export interface UseAnalysisStreamOptions {
  initialStage?: string;
  initialProgress?: number;
  initialMessage?: string;
  initialElapsed?: number;
  onDone?: () => void;
  onError?: (error: string) => void;
}

export interface UseAnalysisStreamResult {
  stage: string;
  progress: number;
  message: string;
  elapsed_s: number;
  isDone: boolean;
  error: string | null;
  connectionStatus: StreamConnectionStatus;
  reconnect: () => void;
}

export function useAnalysisStream(
  analysisId: string | null | undefined,
  onDoneOrOptions?: (() => void) | UseAnalysisStreamOptions,
): UseAnalysisStreamResult {
  const options: UseAnalysisStreamOptions =
    typeof onDoneOrOptions === 'function' ? { onDone: onDoneOrOptions } : onDoneOrOptions || {};

  const [event, setEvent] = useState<StageEventPayload>({
    stage: options.initialStage || 'queued',
    progress: options.initialProgress ?? 0.0,
    message: options.initialMessage || 'Waiting in processing queue…',
    elapsed_s: options.initialElapsed ?? 0.0,
  });
  const [isDone, setIsDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<StreamConnectionStatus>('connecting');
  const [reconnectKey, setReconnectKey] = useState(0);

  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const onDoneRef = useRef(options.onDone);
  const onErrorRef = useRef(options.onError);

  useEffect(() => {
    onDoneRef.current = options.onDone;
    onErrorRef.current = options.onError;
  }, [options.onDone, options.onError]);

  const reconnect = useCallback(() => {
    setError(null);
    setConnectionStatus('connecting');
    setReconnectKey((k) => k + 1);
  }, []);

  useEffect(() => {
    if (!analysisId) {
      setConnectionStatus('disconnected');
      return;
    }

    setIsDone(false);
    setError(null);
    setConnectionStatus('connecting');
    let doneCalled = false;

    const stopPolling = () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };

    const startPollingFallback = () => {
      if (pollingRef.current || doneCalled) return;
      setConnectionStatus('fallback');

      pollingRef.current = setInterval(async () => {
        try {
          const detail = await api.get<AnalysisDetail>(`/analyses/${analysisId}`);
          setEvent((prev) => ({
            ...prev,
            stage: detail.stage || detail.status,
            progress:
              detail.status === 'completed' || detail.status === 'done' || detail.stage === 'done'
                ? 1.0
                : prev.progress,
            message: detail.error || `Processing stage: ${detail.stage}`,
            elapsed_s: detail.duration_s || prev.elapsed_s,
          }));

          if (
            detail.status === 'completed' ||
            detail.status === 'done' ||
            detail.stage === 'done'
          ) {
            stopPolling();
            doneCalled = true;
            setIsDone(true);
            setConnectionStatus('connected');
            onDoneRef.current?.();
          } else if (
            detail.status === 'failed' ||
            detail.status === 'error' ||
            detail.stage === 'error'
          ) {
            stopPolling();
            const errMsg = detail.error || 'Forensic analysis pipeline failed';
            setError(errMsg);
            setConnectionStatus('disconnected');
            onErrorRef.current?.(errMsg);
          }
        } catch (pollErr: unknown) {
          if (pollErr instanceof Error && pollErr.message.includes('404')) {
            stopPolling();
            const notFoundMsg = 'Analysis job not found in records';
            setError(notFoundMsg);
            setConnectionStatus('disconnected');
            onErrorRef.current?.(notFoundMsg);
          }
        }
      }, 2000);
    };

    const unsubscribe = subscribeToAnalysisEvents(analysisId, {
      onEvent: (data) => {
        setConnectionStatus('connected');
        setEvent(data);
        if (data.stage === 'error') {
          setError(data.message);
          stopPolling();
          onErrorRef.current?.(data.message);
        }
      },
      onDone: () => {
        stopPolling();
        doneCalled = true;
        setIsDone(true);
        setConnectionStatus('connected');
        onDoneRef.current?.();
      },
      onError: () => {
        startPollingFallback();
      },
    });

    return () => {
      stopPolling();
      unsubscribe();
    };
  }, [analysisId, reconnectKey]);

  return {
    stage: event.stage,
    progress: event.progress,
    message: event.message,
    elapsed_s: event.elapsed_s,
    isDone,
    error,
    connectionStatus,
    reconnect,
  };
}
