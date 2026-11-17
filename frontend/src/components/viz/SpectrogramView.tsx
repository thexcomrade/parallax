/**
 * Audio Spectrogram with Anomaly Bands and Shared Cursor.
 * Renders the backend log-mel spectrogram artifact with interactive scrubbing
 * synchronized across all evidence components.
 */

import React, { useRef, useCallback } from 'react';
import { clsx } from 'clsx';
import { Mic, Activity, AlertCircle, Loader2 } from 'lucide-react';
import { useCursorSync } from '../../hooks/useCursorSync';
import { useAuthenticatedImage } from '../../hooks/useAuthenticatedImage';

export interface SpectrogramViewProps {
  analysisId: string;
  duration_s?: number;
  hasAudio?: boolean;
  className?: string;
}

export const SpectrogramView: React.FC<SpectrogramViewProps> = ({
  analysisId,
  duration_s = 3.0,
  hasAudio = true,
  className,
}) => {
  const [currentTime, setCurrentTime] = useCursorSync();
  const containerRef = useRef<HTMLDivElement>(null);
  const spectrogramUrl = hasAudio ? `/api/v1/analyses/${analysisId}/artifacts/spectrogram.png` : null;
  const { blobUrl, loading: imageLoading } = useAuthenticatedImage(spectrogramUrl);

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

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (!hasAudio || duration_s <= 0) return;
      const step = 0.1;
      const current = currentTime ?? 0;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setCurrentTime(Math.max(0, current - step));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setCurrentTime(Math.min(duration_s, current + step));
      }
    },
    [currentTime, duration_s, hasAudio, setCurrentTime],
  );

  if (!hasAudio) {
    return (
      <div
        role="region"
        aria-label="Acoustic Spectrogram"
        className={clsx(
          'flex flex-col gap-3 p-5 bg-[var(--surface-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] shadow-[var(--shadow-card)]',
          className,
        )}
      >
        <div className="flex items-center justify-between pb-2 border-b border-[var(--border-subtle)] text-xs font-mono">
          <div className="flex items-center gap-2">
            <Mic size={14} className="text-[var(--text-400)]" />
            <span className="font-sans font-bold text-xs text-[var(--text-100)]">
              CH-A Log-Mel Spectrogram
            </span>
          </div>
          <span className="text-[var(--text-400)] text-[11px]">NO AUDIO DETECTED</span>
        </div>

        <div className="p-8 text-center font-mono text-xs text-[var(--text-400)] bg-[var(--surface-base)] rounded-[var(--radius-md)] border border-[var(--border-subtle)] flex flex-col items-center justify-center gap-1.5">
          <AlertCircle size={18} className="text-amber-400 opacity-80" />
          <span className="font-semibold text-[var(--text-200)]">AUDIO CHANNEL MASKED (w_a = 0.00)</span>
          <span className="text-[11px] text-[var(--text-400)] max-w-sm">
            Source container contains no decodable audio stream. Forensic evaluation was executed on visual channel alone.
          </span>
        </div>
      </div>
    );
  }

  const cursorRatio =
    currentTime !== null && duration_s > 0
      ? Math.max(0, Math.min(1, currentTime / duration_s))
      : null;

  return (
    <div
      role="region"
      aria-label="Acoustic Spectrogram and Anomaly Bands"
      className={clsx(
        'flex flex-col gap-3 p-5 bg-[var(--surface-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] shadow-[var(--shadow-card)]',
        className,
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-[var(--border-subtle)] text-xs font-mono">
        <div className="flex items-center gap-2">
          <Mic size={14} className="text-[var(--lime-400)]" />
          <span className="font-sans font-bold text-xs text-[var(--text-100)]">
            CH-A Log-Mel Spectrogram & Anomaly Bands
          </span>
        </div>
        <span className="text-[var(--lime-400)] text-[11px] font-semibold">
          16 kHz Mono · 80 Filterbanks
        </span>
      </div>

      {/* Spectrogram image container with interactive cursor */}
      <div
        ref={containerRef}
        role="slider"
        tabIndex={0}
        aria-label="Acoustic time scrub bar (use left and right arrow keys to scrub)"
        aria-valuemin={0}
        aria-valuemax={duration_s}
        aria-valuenow={Number((currentTime ?? 0).toFixed(2))}
        aria-valuetext={`${(currentTime ?? 0).toFixed(2)} seconds of ${duration_s.toFixed(2)} seconds`}
        onKeyDown={handleKeyDown}
        onPointerDown={handlePointer}
        onPointerMove={(e) => {
          if (e.buttons === 1 || e.pressure > 0) handlePointer(e);
        }}
        className="relative w-full h-36 bg-[var(--surface-base)] rounded-[var(--radius-md)] border border-[var(--border-subtle)] overflow-hidden cursor-crosshair select-none focus-visible:outline-2 focus-visible:outline-[var(--lime-400)]"
      >
        {imageLoading ? (
          <div className="flex items-center justify-center w-full h-full font-mono text-xs text-[var(--text-400)] gap-2">
            <Loader2 size={16} className="animate-spin text-[var(--lime-400)]" />
            <span>DECODING SPECTROGRAM…</span>
          </div>
        ) : blobUrl ? (
          <img
            src={blobUrl}
            alt="Audio Log-Mel Spectrogram with forensic anomaly overlay"
            className="w-full h-full object-fill"
          />
        ) : (
          <div className="flex items-center justify-center w-full h-full font-mono text-xs text-[var(--text-400)]">
            Spectrogram unavailable
          </div>
        )}

        {/* Frequency scale overlay marks */}
        <div className="absolute top-1 left-2 font-mono text-[9px] text-white/50 bg-black/60 px-1 rounded pointer-events-none">
          8 kHz
        </div>
        <div className="absolute bottom-1 left-2 font-mono text-[9px] text-white/50 bg-black/60 px-1 rounded pointer-events-none">
          0 Hz
        </div>

        {/* Shared time cursor overlay */}
        {cursorRatio !== null && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-[var(--lime-400)] pointer-events-none shadow-[0_0_8px_#C8FF2E]"
            style={{ left: `${cursorRatio * 100}%` }}
          />
        )}
      </div>

      {/* Footer annotation */}
      <div className="flex items-center justify-between font-mono text-[11px] text-[var(--text-400)] pt-1">
        <span className="flex items-center gap-1.5">
          <Activity size={12} className="text-[var(--lime-400)]" />
          <span>Red Vertical Bands = Acoustic synthesis anomalies</span>
        </span>
        <span className="text-[var(--text-300)]">0.0s — {duration_s.toFixed(1)}s</span>
      </div>
    </div>
  );
};
