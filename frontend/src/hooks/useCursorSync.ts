/**
 * Shared time-cursor store used by FrameStrip, SpectrogramView, and TimelineChart.
 * Synchronizes scrubbing timestamp across all three forensic views.
 */

import { useState, useCallback, useEffect } from 'react';

type Listener = (time: number | null) => void;

class CursorStore {
  private time: number | null = null;
  private listeners: Set<Listener> = new Set();

  getTime(): number | null {
    return this.time;
  }

  setTime(t: number | null) {
    if (this.time !== t) {
      this.time = t;
      this.listeners.forEach((fn) => fn(t));
    }
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }
}

export const cursorStore = new CursorStore();

export function useCursorSync(): [number | null, (t: number | null) => void] {
  const [time, setLocalTime] = useState<number | null>(cursorStore.getTime());

  useEffect(() => {
    const unsubscribe = cursorStore.subscribe((newTime) => {
      setLocalTime(newTime);
    });
    return unsubscribe;
  }, []);

  const setTime = useCallback((t: number | null) => {
    cursorStore.setTime(t);
  }, []);

  return [time, setTime];
}
