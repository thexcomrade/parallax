/**
 * Hook to subscribe to live SSE stage events for an analysis job.
 * Handles live Server-Sent Events, automatic adaptive HTTP polling fallback,
 * in-flight request cancellation (AbortController), shared subscription
 * deduplication, and immediate termination on terminal states (Workstream E).
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

// Module-level deduplication to prevent multiple polling loops for the same analysis
const activePollers = new Map<string, number>();

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

  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
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
    let isTerminated = false;

    // Track active subscribers for this analysis ID
    const currentSubCount = activePollers.get(analysisId) || 0;
    activePollers.set(analysisId, currentSubCount + 1);

    const stopPolling = () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };

    const getAdaptiveDelayMs = (stage: string): number => {
      const lower = stage.toLowerCase();
      if (lower === 'queued' || lower === 'probe') {
        return 3000; // Moderate interval while waiting in queue
      }
      return 1500; // Fast cadence during active inference
    };

    const scheduleNextPoll = (delayMs: number) => {
      if (isTerminated) return;
      stopPolling();
      timeoutRef.current = setTimeout(async () => {
        if (isTerminated) return;

        abortControllerRef.current = new AbortController();
        const signal = abortControllerRef.current.signal;

        try {
          const detail = await api.get<AnalysisDetail>(`/analyses/${analysisId}`, { signal });
          if (isTerminated) return;

          const isTerminalSuccess =
            detail.status === 'completed' || detail.status === 'done' || detail.stage === 'done';
          const isTerminalFailure =
            detail.status === 'failed' || detail.status === 'error' || detail.stage === 'error';

          setEvent((prev) => ({
            ...prev,
            stage: detail.stage || detail.status,
            progress: isTerminalSuccess ? 1.0 : prev.progress,
            message: detail.error || `Processing stage: ${detail.stage}`,
            elapsed_s: detail.duration_s || prev.elapsed_s,
          }));

          if (isTerminalSuccess) {
            isTerminated = true;
            stopPolling();
            setIsDone(true);
            setConnectionStatus('connected');
            onDoneRef.current?.();
            return;
          }

          if (isTerminalFailure) {
            isTerminated = true;
            stopPolling();
            const errMsg = detail.error || 'Forensic analysis pipeline failed';
            setError(errMsg);
            setConnectionStatus('disconnected');
            onErrorRef.current?.(errMsg);
            return;
          }

          // Schedule next adaptive iteration based on current stage
          const nextDelay = getAdaptiveDelayMs(detail.stage || detail.status);
          scheduleNextPoll(nextDelay);
        } catch (pollErr: unknown) {
          if (isTerminated) return;
          if (pollErr instanceof Error && pollErr.name === 'AbortError') {
            return;
          }

          if (pollErr instanceof Error && pollErr.message.includes('404')) {
            isTerminated = true;
            stopPolling();
            const notFoundMsg = 'Analysis job not found in records';
            setError(notFoundMsg);
            setConnectionStatus('disconnected');
            onErrorRef.current?.(notFoundMsg);
            return;
          }

          // Transient network error: back off to 4000ms
          scheduleNextPoll(4000);
        }
      }, delayMs);
    };

    const startPollingFallback = () => {
      if (isTerminated) return;
      setConnectionStatus('fallback');
      scheduleNextPoll(1000); // Quick initial probe
    };

    const unsubscribe = subscribeToAnalysisEvents(analysisId, {
      onEvent: (data) => {
        if (isTerminated) return;
        setConnectionStatus('connected');
        setEvent(data);
        if (data.stage === 'error') {
          isTerminated = true;
          setError(data.message);
          stopPolling();
          onErrorRef.current?.(data.message);
        }
      },
      onDone: () => {
        isTerminated = true;
        stopPolling();
        setIsDone(true);
        setConnectionStatus('connected');
        onDoneRef.current?.();
      },
      onError: () => {
        startPollingFallback();
      },
    });

    return () => {
      isTerminated = true;
      stopPolling();
      unsubscribe();
      const remaining = (activePollers.get(analysisId) || 1) - 1;
      if (remaining <= 0) {
        activePollers.delete(analysisId);
      } else {
        activePollers.set(analysisId, remaining);
      }
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
