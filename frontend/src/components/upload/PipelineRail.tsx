import React from 'react';
import { clsx } from 'clsx';
import { Check, Loader2, Clock, AlertCircle } from 'lucide-react';

interface StageInfo {
  id: string;
  label: string;
  description: string;
}

const STAGES: StageInfo[] = [
  { id: 'probe', label: 'Media Container Probe', description: 'FFmpeg stream verification & metadata decode' },
  { id: 'frames', label: 'Frame Extraction', description: 'Uniform 32-frame optical sampling' },
  { id: 'faces', label: 'Facial Landmark Tracking', description: 'MediaPipe biometric landmark localization' },
  { id: 'audio', label: 'Acoustic Processing', description: '16kHz mono MFCC & 80-bin log-mel filterbanks' },
  { id: 'align', label: 'Temporal Synchronization', description: 'Audio-visual cross-modal frame alignment' },
  { id: 'face_enc', label: 'Vision Transformer Encoder', description: 'ViT-B/16 spatial feature extraction' },
  { id: 'audio_enc', label: 'Audio Transformer Encoder', description: 'Temporal acoustic attention embedding' },
  { id: 'fusion', label: 'Dynamic Weight Fusion', description: 'Learned gating cross-attention evaluation' },
  { id: 'explain', label: 'Explainability & Heatmaps', description: 'Grad-CAM attention maps & spectrogram analysis' },
  { id: 'done', label: 'Report Assembly', description: 'Dossier consolidation & calibration' },
];

export interface PipelineRailProps {
  currentStage: string;
  progress: number;
  message?: string;
  elapsed_s?: number;
  error?: string | null;
  className?: string;
}

export const PipelineRail: React.FC<PipelineRailProps> = ({
  currentStage,
  progress,
  message,
  elapsed_s = 0,
  error,
  className,
}) => {
  const isQueued = currentStage === 'queued';
  const isCompleted = currentStage === 'done' || currentStage === 'completed';
  const currentIdx = STAGES.findIndex((s) => s.id === currentStage);
  const progressPercent = Math.min(100, Math.max(0, Math.round(progress * 100)));

  return (
    <div
      role="region"
      aria-label="Analysis Pipeline Execution State"
      className={clsx(
        'flex flex-col gap-5 p-6 rounded-[var(--radius-lg)] bg-[var(--surface-card)] border border-[var(--border-default)] shadow-[var(--shadow-card)]',
        className,
      )}
    >
      {/* Header & Elapsed Time */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-[var(--border-subtle)]">
        <div>
          <span className="font-sans font-bold text-base text-[var(--text-primary)]">
            Forensic Execution Pipeline
          </span>
          <p className="font-sans text-xs text-[var(--text-300)] mt-0.5">
            Real-time SSE telemetry streamed from backend worker.
          </p>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-[var(--radius-sm)] bg-[var(--surface-elevated)] border border-[var(--border-subtle)] text-[var(--text-300)]">
            <Clock size={13} className="text-[var(--cobalt-400)]" />
            <span>T+{elapsed_s.toFixed(1)}s</span>
          </span>
          <span
            className={clsx(
              'px-2.5 py-1 rounded-[var(--radius-sm)] border font-bold',
              isCompleted
                ? 'bg-[var(--success-500)]/15 border-[var(--success-500)]/30 text-[var(--success-500)]'
                : isQueued
                ? 'bg-[var(--surface-elevated)] border-[var(--border-strong)] text-[var(--text-300)]'
                : 'bg-[var(--cobalt-500)]/15 border-[var(--cobalt-500)]/30 text-[var(--cobalt-300)]',
            )}
          >
            {isCompleted ? '100%' : isQueued ? 'QUEUED' : `${progressPercent}%`}
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div
        className="w-full h-2 rounded-full overflow-hidden bg-[var(--surface-base)] border border-[var(--border-subtle)] relative"
        role="progressbar"
        aria-valuenow={isQueued ? 0 : progressPercent}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        {isQueued ? (
          <div className="absolute inset-0 bg-[var(--cobalt-500)]/30 animate-pulse" />
        ) : (
          <div
            className="h-full bg-[var(--cobalt-400)] transition-[width] duration-[var(--motion-smooth)]"
            style={{ width: `${progressPercent}%` }}
          />
        )}
      </div>

      {/* Current Stage Announcement Message */}
      {message && (
        <div
          role="status"
          aria-live="polite"
          className="flex items-center gap-2.5 px-3.5 py-2.5 rounded-[var(--radius-sm)] bg-[var(--surface-elevated)] border border-[var(--border-strong)] text-xs font-sans"
        >
          {isCompleted ? (
            <Check size={15} className="text-[var(--success-500)] shrink-0" />
          ) : (
            <Loader2 size={15} className="animate-spin text-[var(--cobalt-400)] shrink-0" />
          )}
          <span className="text-[var(--text-200)] font-medium truncate">
            {message}
          </span>
        </div>
      )}

      {/* Stepper Stages List */}
      <div className="flex flex-col gap-1.5 max-h-[460px] overflow-y-auto pr-1">
        {STAGES.map((st, idx) => {
          const isDone = isCompleted || (currentIdx >= 0 && currentIdx > idx);
          const isActive = currentStage === st.id && !isCompleted;

          return (
            <div
              key={st.id}
              className={clsx(
                'flex items-center justify-between p-2.5 rounded-[var(--radius-sm)] border transition-all duration-150',
                isActive
                  ? 'border-[var(--cobalt-500)] bg-[var(--surface-elevated)] shadow-xs'
                  : isDone
                  ? 'border-[var(--border-subtle)] bg-[var(--surface-card)]/50'
                  : 'border-transparent text-[var(--text-400)] opacity-60',
              )}
            >
              <div className="flex items-center gap-3 min-w-0">
                {/* Step indicator circle */}
                <div
                  className={clsx(
                    'w-6 h-6 rounded-full flex items-center justify-center text-xs shrink-0 font-mono',
                    isDone
                      ? 'bg-[var(--success-500)]/15 text-[var(--success-500)] border border-[var(--success-500)]/30'
                      : isActive
                      ? 'bg-[var(--cobalt-500)] text-[var(--text-100)] animate-pulse'
                      : 'bg-[var(--surface-base)] text-[var(--text-500)] border border-[var(--border-subtle)]',
                  )}
                >
                  {isDone ? (
                    <Check size={13} strokeWidth={3} />
                  ) : isActive ? (
                    <Loader2 size={13} className="animate-spin" />
                  ) : (
                    <span>{idx + 1}</span>
                  )}
                </div>

                <div className="flex flex-col min-w-0">
                  <span
                    className={clsx(
                      'font-sans text-xs font-medium truncate',
                      isActive
                        ? 'text-[var(--text-100)] font-semibold'
                        : isDone
                        ? 'text-[var(--text-200)]'
                        : 'text-[var(--text-400)]',
                    )}
                  >
                    {st.label}
                  </span>
                  <span className="font-sans text-[11px] text-[var(--text-400)] truncate hidden sm:block">
                    {st.description}
                  </span>
                </div>
              </div>

              {/* Status Tag */}
              <span
                className={clsx(
                  'font-mono text-[10px] px-2 py-0.5 rounded uppercase tracking-wider shrink-0',
                  isDone
                    ? 'text-[var(--success-500)] font-semibold'
                    : isActive
                    ? 'text-[var(--cobalt-300)] font-semibold animate-pulse'
                    : 'text-[var(--text-500)]',
                )}
              >
                {isDone ? 'COMPLETE' : isActive ? 'PROCESSING' : 'PENDING'}
              </span>
            </div>
          );
        })}
      </div>

      {error && (
        <div
          role="alert"
          className="p-3 rounded-[var(--radius-sm)] bg-[var(--alert-bg)] border border-[var(--alert-500)]/40 text-[var(--alert-500)] text-xs flex items-center gap-2"
        >
          <AlertCircle size={15} className="shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
