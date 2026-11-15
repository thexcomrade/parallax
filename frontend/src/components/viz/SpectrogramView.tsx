/**
 * Audio Spectrogram with Anomaly Bands and Shared Cursor.
 * Renders the backend log-mel spectrogram artifact with interactive scrubbing.
 */

import React, { useRef, useCallback } from 'react';
import { useCursorSync } from '../../hooks/useCursorSync';

interface SpectrogramViewProps {
  analysisId: string;
  duration_s?: number;
  hasAudio?: boolean;
}

export const SpectrogramView: React.FC<SpectrogramViewProps> = ({
  analysisId,
  duration_s = 3.0,
  hasAudio = true,
}) => {
  const [currentTime, setCurrentTime] = useCursorSync();
  const containerRef = useRef<HTMLDivElement>(null);

  const handlePointer = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!containerRef.current || !hasAudio) return;
      const rect = containerRef.current.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const ratio = Math.max(0, Math.min(1, clientX / rect.width));
      setCurrentTime(ratio * duration_s);
    },
    [duration_s, hasAudio, setCurrentTime],
  );

  if (!hasAudio) {
    return (
      <div className="flex flex-col gap-2 p-4 bg-black-900 border border-line rounded-lg">
        <div className="flex items-center justify-between pb-2 border-b border-line text-xs font-mono">
          <span className="text-lime-400 uppercase tracking-widest">
            CH-A // LOG-MEL SPECTROGRAM
          </span>
          <span className="text-text-500">NO AUDIO STREAM DETECTED</span>
        </div>
        <div className="p-8 text-center font-mono text-xs text-text-500 bg-black-950 rounded border border-line">
          AUDIO MODALITY MASKED (W_A = 0.00)
        </div>
      </div>
    );
  }

  const cursorRatio =
    currentTime !== null && duration_s > 0
      ? Math.max(0, Math.min(1, currentTime / duration_s))
      : null;

  return (
    <div className="flex flex-col gap-2 p-4 bg-black-900 border border-line rounded-lg">
      <div className="flex items-center justify-between pb-2 border-b border-line text-xs font-mono">
        <span className="text-lime-400 uppercase tracking-widest">
          CH-A // LOG-MEL SPECTROGRAM & ANOMALY BANDS
        </span>
        <span className="text-text-400">16 KHZ MONO · 64 MELS</span>
      </div>

      <div
        ref={containerRef}
        onPointerDown={handlePointer}
        onPointerMove={(e) => {
          if (e.buttons === 1) handlePointer(e);
        }}
        className="relative w-full h-32 bg-black-950 rounded border border-line overflow-hidden cursor-crosshair select-none"
      >
        <img
          src={`/api/v1/analyses/${analysisId}/artifacts/spectrogram.png`}
          alt="Audio Log-Mel Spectrogram with forensic anomaly overlay"
          className="w-full h-full object-fill"
          loading="lazy"
        />

        {/* Shared time cursor overlay */}
        {cursorRatio !== null && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-lime-400 pointer-events-none shadow-[0_0_8px_#C8FF2E]"
            style={{ left: `${cursorRatio * 100}%` }}
          />
        )}
      </div>

      <div className="flex items-center justify-between font-mono text-[10px] text-text-500 pt-1">
        <span>RED HIGHLIGHTS = TIME-WINDOW SYNTHESIS ANOMALIES</span>
        <span>0.0s — {duration_s.toFixed(1)}s</span>
      </div>
    </div>
  );
};
