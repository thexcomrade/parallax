/**
 * Hook to subscribe to live SSE stage events for an analysis job.
 */

import { useState, useEffect, useRef } from 'react';
import { subscribeToAnalysisEvents } from '../lib/sse';
import { api } from '../lib/api';
import type { AnalysisDetail, StageEventPayload } from '../types/api';

export interface UseAnalysisStreamResult {
  stage: string;
  progress: number;
  message: string;
  elapsed_s: number;
  isDone: boolean;
  error: string | null;
}

export function useAnalysisStream(
  analysisId: string | null | undefined,
  onDone?: () => void,
): UseAnalysisStreamResult {
  const [event, setEvent] = useState<StageEventPayload>({
    stage: 'queued',
    progress: 0.0,
    message: 'Waiting in processing queue…',
    elapsed_s: 0.0,
  });
  const [isDone, setIsDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const onDoneRef = useRef(onDone);

  useEffect(() => {
    onDoneRef.current = onDone;
  }, [onDone]);

  useEffect(() => {
    if (!analysisId) return;

    setIsDone(false);
    setError(null);
    let doneCalled = false;

    const stopPolling = () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };

    const startPollingFallback = () => {
      if (pollingRef.current || doneCalled) return;

      pollingRef.current = setInterval(async () => {
        try {
          const detail = await api.get<AnalysisDetail>(`/analyses/${analysisId}`);
          setEvent((prev) => ({
            ...prev,
            stage: detail.stage || detail.status,
            message: detail.error || `Processing stage: ${detail.stage}`,
          }));

          if (detail.status === 'done') {
            stopPolling();
            doneCalled = true;
            setIsDone(true);
            onDoneRef.current?.();
          } else if (detail.status === 'error') {
            stopPolling();
            setError(detail.error || 'Analysis failed during execution');
          }
        } catch (pollErr: unknown) {
          if (pollErr instanceof Error && pollErr.message.includes('404')) {
            stopPolling();
            setError('Analysis job not found');
          }
        }
      }, 2000);
    };

    const unsubscribe = subscribeToAnalysisEvents(analysisId, {
      onEvent: (data) => {
        setEvent(data);
        if (data.stage === 'error') {
          setError(data.message);
          stopPolling();
        }
      },
      onDone: () => {
        stopPolling();
        doneCalled = true;
        setIsDone(true);
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
  }, [analysisId]);

  return {
    stage: event.stage,
    progress: event.progress,
    message: event.message,
    elapsed_s: event.elapsed_s,
    isDone,
    error,
  };
}
