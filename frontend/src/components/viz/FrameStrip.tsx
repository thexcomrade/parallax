/**
 * Contact sheet of analysed video frames.
 * Supports scrubbing sync via useCursorSync and clicking to inspect frame heatmaps.
 */

import React from 'react';
import { useCursorSync } from '../../hooks/useCursorSync';
import { formatTimecode } from '../../lib/format';
import type { ArtifactItem } from '../../types/api';

interface FrameStripProps {
  analysisId: string;
  artifacts: ArtifactItem[];
  duration_s?: number;
  onSelectFrame?: (artifactName: string) => void;
  selectedArtifact?: string | null;
}

export const FrameStrip: React.FC<FrameStripProps> = ({
  analysisId,
  artifacts,
  duration_s = 3.0,
  onSelectFrame,
  selectedArtifact,
}) => {
  const [currentTime, setCurrentTime] = useCursorSync();

  const heatmaps = artifacts.filter((a) => a.kind === 'heatmap' || a.name.startsWith('heatmap_'));

  return (
    <div className="flex flex-col gap-2 p-4 bg-black-900 border border-line rounded-lg">
      <div className="flex items-center justify-between pb-2 border-b border-line text-xs font-mono">
        <span className="text-cobalt-300 uppercase tracking-widest">
          CH-V // CONTACT SHEET (SUSPICIOUS FRAMES)
        </span>
        <span className="text-text-400">
          {heatmaps.length} FRAMES EXTRACTED
        </span>
      </div>

      <div className="flex items-center gap-3 overflow-x-auto pb-2 scrollbar-thin">
        {heatmaps.length === 0 ? (
          <div className="font-mono text-xs text-text-500 py-4 text-center w-full">
            No frame evidence artifacts recorded.
          </div>
        ) : (
          heatmaps.map((art, idx) => {
            const frameT = (idx / Math.max(1, heatmaps.length - 1)) * duration_s;
            const isSelected = selectedArtifact === art.name;
            const isActive =
              currentTime !== null && Math.abs(currentTime - frameT) < (duration_s / heatmaps.length);

            return (
              <button
                key={art.name}
                type="button"
                onClick={() => {
                  setCurrentTime(frameT);
                  onSelectFrame?.(art.name);
                }}
                className={`flex-shrink-0 flex flex-col items-center gap-1 p-1 rounded border transition-all text-left group outline-none focus-visible:ring-2 focus-visible:ring-cobalt-500 ${
                  isSelected
                    ? 'border-lime-400 bg-black-800'
                    : isActive
                    ? 'border-cobalt-500 bg-black-800/60'
                    : 'border-line hover:border-line-strong bg-black-950'
                }`}
              >
                <div className="relative w-28 h-20 bg-black-800 rounded overflow-hidden">
                  <img
                    src={`/api/v1/analyses/${analysisId}/artifacts/${art.name}`}
                    alt={`Evidence frame ${idx + 1}`}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                  {isSelected && (
                    <div className="absolute top-1 left-1 px-1 bg-lime-400 text-lime-ink font-mono text-[9px] font-bold rounded">
                      ACTIVE
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between w-full font-mono text-[10px] px-0.5">
                  <span className="text-text-500">#{idx + 1}</span>
                  <span className="text-text-300">{formatTimecode(frameT)}</span>
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
};
