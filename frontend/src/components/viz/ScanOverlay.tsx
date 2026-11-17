/**
 * Forensic Scan Overlay inspector.
 * Renders high-resolution attention heatmap with opacity slider,
 * corner viewfinder marks, and explainability colormap annotations.
 */

import React, { useState } from 'react';
import { clsx } from 'clsx';
import { Eye, Sliders, Info, Loader2 } from 'lucide-react';
import { useAuthenticatedImage } from '../../hooks/useAuthenticatedImage';

export interface ScanOverlayProps {
  analysisId: string;
  artifactName: string | null;
  className?: string;
}

export const ScanOverlay: React.FC<ScanOverlayProps> = ({
  analysisId,
  artifactName,
  className,
}) => {
  const [opacity, setOpacity] = useState(0.85);
  const imageUrl = artifactName ? `/api/v1/analyses/${analysisId}/artifacts/${artifactName}` : null;
  const { blobUrl, loading: imageLoading } = useAuthenticatedImage(imageUrl);

  if (!artifactName) {
    return (
      <div
        className={clsx(
          'flex flex-col items-center justify-center p-8 bg-[var(--surface-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] text-[var(--text-400)] font-mono text-xs min-h-[360px] text-center gap-2',
          className,
        )}
      >
        <Eye size={24} className="text-[var(--cobalt-400)] opacity-60 mb-1" />
        <span className="font-semibold text-[var(--text-200)]">
          SELECT AN EVIDENCE FRAME TO INSPECT
        </span>
        <span className="text-[11px] text-[var(--text-400)] max-w-xs">
          Click any keyframe from the CH-V Contact Sheet above to inspect spatial ViT-B/16 attention rollouts.
        </span>
      </div>
    );
  }

  return (
    <div
      role="region"
      aria-label="XAI Attention Heatmap Inspector"
      className={clsx(
        'flex flex-col gap-4 p-5 bg-[var(--surface-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] shadow-[var(--shadow-card)]',
        className,
      )}
    >
      {/* Header with Title and Opacity Slider */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-[var(--border-subtle)] gap-2">
        <div className="flex items-center gap-2">
          <Eye size={15} className="text-[var(--cobalt-400)]" />
          <span className="font-sans font-bold text-xs text-[var(--text-100)]">
            XAI Attention Rollout
          </span>
          <span className="font-mono text-[10px] px-1.5 py-0.2 rounded bg-[var(--surface-elevated)] border border-[var(--border-subtle)] text-[var(--cobalt-300)]">
            {artifactName}
          </span>
        </div>

        {/* Opacity slider */}
        <div className="flex items-center gap-2.5 font-mono text-xs">
          <label htmlFor="heatmap-opacity" className="text-[var(--text-400)] text-[11px] uppercase flex items-center gap-1">
            <Sliders size={12} />
            <span>Intensity:</span>
          </label>
          <input
            id="heatmap-opacity"
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={opacity}
            aria-label="Heatmap overlay intensity"
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-24 accent-[var(--lime-400)] cursor-pointer focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)]"
          />
          <span className="text-[var(--lime-400)] font-bold text-[11px] w-9 text-right">
            {Math.round(opacity * 100)}%
          </span>
        </div>
      </div>

      {/* Main image viewer with viewfinder marks */}
      <div className="relative w-full max-h-[440px] min-h-[300px] bg-[var(--surface-base)] rounded-[var(--radius-md)] border border-[var(--border-subtle)] flex items-center justify-center overflow-hidden p-2">
        {/* Forensic viewfinder corner marks */}
        <div className="absolute top-3 left-3 w-3 h-3 border-t-2 border-l-2 border-[var(--lime-400)] z-10" />
        <div className="absolute top-3 right-3 w-3 h-3 border-t-2 border-r-2 border-[var(--lime-400)] z-10" />
        <div className="absolute bottom-3 left-3 w-3 h-3 border-b-2 border-l-2 border-[var(--lime-400)] z-10" />
        <div className="absolute bottom-3 right-3 w-3 h-3 border-b-2 border-r-2 border-[var(--lime-400)] z-10" />

        {imageLoading ? (
          <div className="flex flex-col items-center justify-center p-8 font-mono text-xs text-[var(--text-400)] gap-2">
            <Loader2 size={20} className="animate-spin text-[var(--cobalt-400)]" />
            <span>AUTHENTICATING & DECODING HEATMAP…</span>
          </div>
        ) : blobUrl ? (
          <img
            src={blobUrl}
            alt={`Attention heatmap for ${artifactName}`}
            className="max-h-[420px] w-auto max-w-full object-contain rounded transition-opacity duration-150"
            style={{ opacity }}
          />
        ) : (
          <div className="flex flex-col items-center justify-center p-8 font-mono text-xs text-[var(--text-400)]">
            Artifact unavailable
          </div>
        )}
      </div>

      {/* Colormap Legend Footer */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pt-2 border-t border-[var(--border-subtle)] text-[11px] font-mono text-[var(--text-400)]">
        <div className="flex items-center gap-1.5">
          <Info size={12} className="text-[var(--cobalt-400)] shrink-0" />
          <span>Inferno Spectrum: Bright yellow/white indicates localized spatial attention concentration.</span>
        </div>
        <span className="text-[10px] uppercase tracking-wider text-[var(--text-400)]">
          ViT-B/16 Self-Attention
        </span>
      </div>
    </div>
  );
};
