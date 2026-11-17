/**
 * Contact sheet of analysed video frames.
 * Supports scrubbing sync via useCursorSync and clicking to inspect frame heatmaps.
 */

import React from 'react';
import { clsx } from 'clsx';
import { Film, Check } from 'lucide-react';
import { useCursorSync } from '../../hooks/useCursorSync';
import { useAuthenticatedImage } from '../../hooks/useAuthenticatedImage';
import { formatTimecode } from '../../lib/format';
import type { ArtifactItem } from '../../types/api';

const FrameThumbnail: React.FC<{
  src: string;
  isSelected: boolean;
}> = ({ src, isSelected }) => {
  const { blobUrl, loading } = useAuthenticatedImage(src);

  return (
    <div className="relative w-28 h-20 bg-[var(--surface-base)] rounded overflow-hidden flex items-center justify-center">
      {loading ? (
        <span className="w-2.5 h-2.5 rounded-full bg-[var(--cobalt-400)] animate-pulse" />
      ) : blobUrl ? (
        <img src={blobUrl} alt="" className="w-full h-full object-cover" />
      ) : (
        <span className="font-mono text-[9px] text-[var(--text-400)]">FRAME</span>
      )}
      {isSelected && (
        <div
          aria-hidden="true"
          className="absolute top-1 left-1 px-1.5 py-0.2 bg-[var(--lime-400)] text-[var(--lime-ink)] font-mono text-[9px] font-bold rounded flex items-center gap-0.5"
        >
          <Check size={9} strokeWidth={3} />
          INSPECT
        </div>
      )}
    </div>
  );
};

export interface FrameStripProps {
  analysisId: string;
  artifacts: ArtifactItem[];
  duration_s?: number;
  onSelectFrame?: (artifactName: string) => void;
  selectedArtifact?: string | null;
  className?: string;
}

export const FrameStrip: React.FC<FrameStripProps> = ({
  analysisId,
  artifacts,
  duration_s = 3.0,
  onSelectFrame,
  selectedArtifact,
  className,
}) => {
  const [currentTime, setCurrentTime] = useCursorSync();

  const heatmaps = artifacts.filter(
    (a) => a.kind === 'heatmap' || a.name.startsWith('heatmap_'),
  );

  return (
    <div
      role="region"
      aria-label="Visual Modality Frame Evidence Strip"
      className={clsx(
        'flex flex-col gap-3 p-5 bg-[var(--surface-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] shadow-[var(--shadow-card)]',
        className,
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-[var(--border-subtle)] text-xs font-mono">
        <div className="flex items-center gap-2">
          <Film size={14} className="text-[var(--cobalt-400)]" />
          <span className="font-sans font-bold text-xs text-[var(--text-100)]">
            CH-V Frame Evidence Strip
          </span>
        </div>

        <span className="font-mono text-[11px] text-[var(--text-400)]">
          {heatmaps.length} Uniform Samples
        </span>
      </div>

      {/* Frame thumbnails row */}
      <div className="flex items-center gap-3 overflow-x-auto pb-2 scrollbar-thin">
        {heatmaps.length === 0 ? (
          <div className="font-mono text-xs text-[var(--text-400)] py-6 text-center w-full">
            No frame evidence artifacts recorded.
          </div>
        ) : (
          heatmaps.map((art, idx) => {
            const frameT = (idx / Math.max(1, heatmaps.length - 1)) * duration_s;
            const isSelected = selectedArtifact === art.name;
            const isActiveCursor =
              currentTime !== null &&
              Math.abs(currentTime - frameT) < duration_s / Math.max(1, heatmaps.length);

            return (
              <button
                key={art.name}
                type="button"
                onClick={() => {
                  setCurrentTime(frameT);
                  onSelectFrame?.(art.name);
                }}
                className={clsx(
                  'shrink-0 flex flex-col items-center gap-1.5 p-1.5 rounded-[var(--radius-sm)] border transition-all text-left outline-none focus-visible:ring-2 focus-visible:ring-[var(--cobalt-500)]',
                  isSelected
                    ? 'border-[var(--lime-400)] bg-[var(--surface-elevated)] shadow-xs'
                    : isActiveCursor
                    ? 'border-[var(--cobalt-500)] bg-[var(--surface-elevated)]/60'
                    : 'border-[var(--border-subtle)] hover:border-[var(--border-strong)] bg-[var(--surface-base)]',
                )}
              >
                <FrameThumbnail
                  src={`/api/v1/analyses/${analysisId}/artifacts/${art.name}`}
                  isSelected={isSelected}
                />

                <div className="flex items-center justify-between w-full font-mono text-[10px] px-0.5">
                  <span className="text-[var(--text-400)]">#{String(idx + 1).padStart(2, '0')}</span>
                  <span className="text-[var(--text-200)] font-medium">{formatTimecode(frameT)}</span>
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
};
