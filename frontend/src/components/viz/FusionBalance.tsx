/**
 * Multimodal Fusion Balance (Dynamic Weight Fusion).
 * Visualizes the learned gating balance between Audio (lime) and Video (cobalt),
 * cross-modal synchronization score, and channel presence indicators.
 */

import React from 'react';
import { clsx } from 'clsx';
import { Sliders, Eye, Mic } from 'lucide-react';

export interface FusionBalanceProps {
  w_audio: number | null;
  w_video: number | null;
  sync_score?: number | null;
  audio_present?: boolean;
  face_coverage?: number | null;
  className?: string;
}

export const FusionBalance: React.FC<FusionBalanceProps> = ({
  w_audio,
  w_video,
  sync_score,
  audio_present = true,
  face_coverage,
  className,
}) => {
  const wa = w_audio ?? (audio_present ? 0.5 : 0.0);
  const wv = w_video ?? (audio_present ? 0.5 : 1.0);

  const audioPercent = Math.round(wa * 100);
  const videoPercent = Math.round(wv * 100);

  let syncLabel = '—';
  let syncClass = 'text-[var(--text-400)]';
  if (sync_score !== undefined && sync_score !== null) {
    if (sync_score >= 0.7) {
      syncLabel = 'High Phase Coherence';
      syncClass = 'text-[var(--lime-400)]';
    } else if (sync_score >= 0.4) {
      syncLabel = 'Moderate Phase Alignment';
      syncClass = 'text-[var(--cobalt-300)]';
    } else {
      syncLabel = 'Low Cross-Modal Sync';
      syncClass = 'text-amber-400';
    }
  }

  return (
    <div
      role="region"
      aria-label="Dynamic Weight Fusion Balance"
      className={clsx(
        'flex flex-col gap-5 p-6 bg-[var(--surface-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] shadow-[var(--shadow-card)]',
        className,
      )}
    >
      {/* Header with Title and Synchronization Score */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-[var(--border-subtle)] gap-2">
        <div className="flex items-center gap-2">
          <Sliders size={15} className="text-[var(--cobalt-400)]" />
          <span className="font-sans font-bold text-sm text-[var(--text-100)]">
            Dynamic Weight Fusion (DWF)
          </span>
          <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-[var(--surface-elevated)] border border-[var(--border-subtle)] text-[var(--cobalt-300)] font-semibold">
            Gating Layer
          </span>
        </div>

        {sync_score !== undefined && sync_score !== null && (
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-[var(--text-400)]">Temporal Sync:</span>
            <span className="px-2 py-0.5 rounded bg-[var(--surface-elevated)] border border-[var(--border-subtle)] font-bold text-[var(--lime-400)]">
              {(sync_score).toFixed(2)}
            </span>
            <span className={clsx('text-[11px] font-sans hidden md:inline', syncClass)}>
              ({syncLabel})
            </span>
          </div>
        )}
      </div>

      {/* Dual Modality Balance Bar (Seesaw) */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between font-mono text-xs text-[var(--text-300)]">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-xs bg-[var(--lime-400)] shrink-0" />
            <span className="font-medium text-[var(--text-200)]">Acoustic Weight (w_a)</span>
            <span className="font-bold text-[var(--lime-400)]">{wa.toFixed(2)}</span>
            {!audio_present && (
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-[var(--alert-bg)] text-[var(--alert-500)] uppercase font-semibold">
                Masked
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="font-medium text-[var(--text-200)]">Visual Weight (w_v)</span>
            <span className="font-bold text-[var(--cobalt-300)]">{wv.toFixed(2)}</span>
            <span className="w-2.5 h-2.5 rounded-xs bg-[var(--cobalt-500)] shrink-0" />
          </div>
        </div>

        {/* Bar */}
        <div className="h-3 w-full bg-[var(--surface-base)] rounded-full overflow-hidden flex border border-[var(--border-subtle)] p-0.5">
          {/* Audio section */}
          <div
            className="h-full bg-[var(--lime-400)] rounded-l-full transition-all duration-500 ease-out"
            style={{ width: `${audioPercent}%` }}
            title={`Audio modality weight: ${wa.toFixed(2)} (${audioPercent}%)`}
          />
          {/* Video section */}
          <div
            className="h-full bg-[var(--cobalt-500)] rounded-r-full transition-all duration-500 ease-out"
            style={{ width: `${videoPercent}%` }}
            title={`Video modality weight: ${wv.toFixed(2)} (${videoPercent}%)`}
          />
        </div>
      </div>

      {/* Sub-channel Detail Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
        {/* CH-V Channel Summary */}
        <div className="p-3.5 rounded-[var(--radius-md)] bg-[var(--surface-elevated)] border border-[var(--border-subtle)] flex items-start gap-3">
          <div className="w-8 h-8 rounded-[var(--radius-sm)] bg-[var(--cobalt-500)]/15 border border-[var(--cobalt-500)]/30 flex items-center justify-center text-[var(--cobalt-400)] shrink-0 mt-0.5">
            <Eye size={16} />
          </div>
          <div className="flex flex-col min-w-0">
            <span className="font-sans font-bold text-xs text-[var(--text-100)] flex items-center gap-1.5">
              CH-V Visual Channel
              <span className="font-mono text-[10px] text-[var(--cobalt-300)] font-normal">
                ({videoPercent}%)
              </span>
            </span>
            <span className="font-mono text-[11px] text-[var(--text-300)] mt-0.5">
              ViT-B/16 Spatial Patch Encoder
            </span>
            <div className="flex items-center gap-2 mt-1.5 text-[11px] font-mono text-[var(--text-400)]">
              <span>Face Coverage:</span>
              <span className="font-semibold text-[var(--text-200)]">
                {face_coverage !== undefined && face_coverage !== null
                  ? `${(face_coverage * 100).toFixed(1)}%`
                  : 'Tracked'}
              </span>
            </div>
          </div>
        </div>

        {/* CH-A Channel Summary */}
        <div className="p-3.5 rounded-[var(--radius-md)] bg-[var(--surface-elevated)] border border-[var(--border-subtle)] flex items-start gap-3">
          <div
            className={clsx(
              'w-8 h-8 rounded-[var(--radius-sm)] flex items-center justify-center shrink-0 mt-0.5 border',
              audio_present
                ? 'bg-[var(--lime-400)]/15 border-[var(--lime-400)]/30 text-[var(--lime-400)]'
                : 'bg-[var(--surface-base)] border-[var(--border-strong)] text-[var(--text-500)]',
            )}
          >
            <Mic size={16} />
          </div>
          <div className="flex flex-col min-w-0">
            <span className="font-sans font-bold text-xs text-[var(--text-100)] flex items-center gap-1.5">
              CH-A Acoustic Channel
              <span className="font-mono text-[10px] text-[var(--lime-400)] font-normal">
                ({audioPercent}%)
              </span>
            </span>
            <span className="font-mono text-[11px] text-[var(--text-300)] mt-0.5">
              {audio_present ? 'Audio Transformer 4-Layer' : 'Audio Modality Inactive'}
            </span>
            <div className="flex items-center gap-2 mt-1.5 text-[11px] font-mono text-[var(--text-400)]">
              <span>Status:</span>
              <span
                className={clsx(
                  'font-semibold',
                  audio_present ? 'text-[var(--lime-400)]' : 'text-amber-400',
                )}
              >
                {audio_present ? '16 kHz Mono Active' : 'Silent / Absent (Masked)'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
