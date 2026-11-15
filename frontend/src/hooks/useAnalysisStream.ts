/**
 * Hook to subscribe to live SSE stage events for an analysis job.
 */

import { useState, useEffect } from 'react';
import { subscribeToAnalysisEvents } from '../lib/sse';
import type { StageEventPayload } from '../types/api';

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

  useEffect(() => {
    if (!analysisId) return;

    setIsDone(false);
    setError(null);

    const unsubscribe = subscribeToAnalysisEvents(analysisId, {
      onEvent: (data) => {
        setEvent(data);
        if (data.stage === 'error') {
          setError(data.message);
        }
      },
      onDone: () => {
        setIsDone(true);
        onDone?.();
      },
      onError: (err) => {
        setError(err.message);
      },
    });

    return () => {
      unsubscribe();
    };
  }, [analysisId, onDone]);

  return {
    stage: event.stage,
    progress: event.progress,
    message: event.message,
    elapsed_s: event.elapsed_s,
    isDone,
    error,
  };
}
