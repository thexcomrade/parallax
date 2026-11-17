/**
 * ProcessingView — Premium AI Forensic Processing Experience.
 *
 * Implements Phase 08 of PARALLAX:
 * - Real-time SSE stage telemetry via useAnalysisStream
 * - Automatic HTTP polling fallback and reconnection recovery
 * - Dual-modality flux visualizer (CH-V Visual & CH-A Acoustic)
 * - Conceptual 6-stage execution stepper with honest progress
 * - Selected media metadata card
 * - Zero fabricated progress or synthetic timers
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { clsx } from 'clsx';
import {
  Copy,
  Check,
  Eye,
  Mic,
  Cpu,
  RefreshCw,
  AlertTriangle,
  ArrowLeft,
  Clock,
  ShieldCheck,
  FileVideo,
  CheckCircle2,
  Loader2,
  FolderArchive,
  History,
} from 'lucide-react';
import { useAnalysisStream } from '../../hooks/useAnalysisStream';
import { Button } from '../ui/Button';
import { formatTimecode } from '../../lib/format';
import type { AnalysisDetail } from '../../types/api';

export interface ProcessingViewProps {
  analysisId: string;
  initialDetail?: AnalysisDetail | null;
  onComplete?: () => void;
  className?: string;
}

interface PipelineStage {
  id: string;
  name: string;
  description: string;
  stageKeys: string[];
}

const CONCEPTUAL_STAGES: PipelineStage[] = [
  {
    id: 'prep',
    name: 'Video Preparation',
    description: 'Container verification & metadata decode',
    stageKeys: ['probe'],
  },
  {
    id: 'frames',
    name: 'Frame Extraction',
    description: '32-frame optical sampling & landmark tracking',
    stageKeys: ['frames', 'faces'],
  },
  {
    id: 'audio',
    name: 'Audio Extraction',
    description: '16kHz mono MFCC & 80-bin log-mel filterbanks',
    stageKeys: ['audio'],
  },
  {
    id: 'sync',
    name: 'Audio-Visual Synchronization',
    description: 'Cross-modal temporal alignment check',
    stageKeys: ['align'],
  },
  {
    id: 'inference',
    name: 'Model Analysis',
    description: 'Dual-stream ViT-B/16 & DWF fusion evaluation',
    stageKeys: ['face_enc', 'audio_enc', 'fusion'],
  },
  {
    id: 'report',
    name: 'Final Classification',
    description: 'Grad-CAM explainability & dossier assembly',
    stageKeys: ['explain', 'done'],
  },
];

const RAW_STAGES = [
  'queued',
  'probe',
  'frames',
  'faces',
  'audio',
  'align',
  'face_enc',
  'audio_enc',
  'fusion',
  'explain',
  'done',
];

export const ProcessingView: React.FC<ProcessingViewProps> = ({
  analysisId,
  initialDetail,
  onComplete,
  className,
}) => {
  const navigate = useNavigate();
  const [copied, setCopied] = useState(false);

  const {
    stage,
    progress,
    message,
    elapsed_s,
    isDone,
    error: streamError,
    connectionStatus,
    reconnect,
  } = useAnalysisStream(analysisId, {
    initialStage: initialDetail?.stage || initialDetail?.status || 'queued',
    initialProgress: initialDetail?.status === 'completed' ? 1.0 : 0.0,
    initialMessage: initialDetail?.error || 'Subscribed to forensic event stream…',
    initialElapsed: initialDetail?.duration_s || 0.0,
    onDone: onComplete,
  });

  const handleCopyId = () => {
    void navigator.clipboard.writeText(analysisId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isQueued = stage === 'queued';
  const effectiveError = streamError || initialDetail?.error || null;
  const isFailed = initialDetail?.status === 'failed' || !!effectiveError;
  const progressPercent = Math.min(100, Math.max(0, Math.round(progress * 100)));

  // Extract container format from filename
  const fileName = initialDetail?.original_name || 'video_clip.mp4';
  const fileExt = fileName.split('.').pop()?.toUpperCase() ?? 'VIDEO';

  return (
    <div className={clsx('w-full max-w-[780px] mx-auto pt-4 sm:pt-6 pb-12 flex flex-col gap-5 sm:gap-6', className)}>
      {/* ── 1. PROCESSING INTRODUCTION ── */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-2 border-b border-[#253247]">
        <div className="flex flex-col gap-1">
          <span className="text-[11px] font-mono font-medium uppercase tracking-widest text-[#648CFF]">
            FORENSIC ANALYSIS
          </span>
          <h1 className="text-2xl sm:text-[28px] font-semibold text-[#E8EDF5] tracking-tight leading-tight">
            Analyzing your media <span className="sr-only">Forensic Signal Processing Bench</span>
          </h1>
          <p className="text-sm text-[#91A0B8] leading-relaxed">
            PARALLAX is processing your video for audio-visual deepfake detection.
          </p>
        </div>

        {/* Telemetry Actions Bar */}
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {/* Analysis ID chip */}
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-[7px] bg-[#111927] border border-[#253247] font-mono text-xs text-[#91A0B8]">
            <span className="text-[#64748B]">ID:</span>
            <span>{analysisId.slice(0, 8)}…</span>
            <button
              type="button"
              onClick={handleCopyId}
              className="p-1 hover:text-[#E8EDF5] rounded transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#648CFF]"
              title="Copy Full Analysis ID"
              aria-label="Copy analysis ID"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-[#35C99A]" />
              ) : (
                <Copy className="w-3.5 h-3.5 text-[#91A0B8]" />
              )}
            </button>
          </div>

          {/* Connection Status Badge */}
          <div
            className={clsx(
              'inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-[7px] border font-mono text-xs',
              connectionStatus === 'connected'
                ? 'bg-[#35C99A]/10 border-[#35C99A]/30 text-[#35C99A]'
                : connectionStatus === 'fallback'
                ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                : 'bg-red-950/40 border-red-500/30 text-red-300',
            )}
            title={
              connectionStatus === 'connected'
                ? 'Live SSE event stream active'
                : connectionStatus === 'fallback'
                ? 'Polling fallback active (every 2s)'
                : 'Connection dropped'
            }
          >
            <span
              className={clsx(
                'w-1.5 h-1.5 rounded-full',
                connectionStatus === 'connected'
                  ? 'bg-[#35C99A] animate-pulse'
                  : connectionStatus === 'fallback'
                  ? 'bg-amber-400'
                  : 'bg-red-400',
              )}
            />
            <span className="text-[10px] font-semibold tracking-wider uppercase">
              {connectionStatus === 'connected'
                ? 'SSE Live'
                : connectionStatus === 'fallback'
                ? 'Polling'
                : 'Offline'}
            </span>
          </div>

          {/* Reconnect button if disconnected */}
          {connectionStatus !== 'connected' && (
            <button
              type="button"
              onClick={reconnect}
              className="h-[30px] px-2.5 rounded-[7px] text-xs font-medium bg-[#171F2D] border border-[#253247] hover:bg-[#1C2737] text-[#E8EDF5] inline-flex items-center gap-1.5 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#648CFF]"
              title="Reconnect Event Stream"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Retry</span>
            </button>
          )}
        </div>
      </div>

      {/* ── 2. FAILURE ERROR BANNER (if pipeline failed) ── */}
      {isFailed && (
        <div
          role="alert"
          className="rounded-[12px] bg-red-950/40 border border-red-500/30 p-4 sm:p-5 flex flex-col gap-3.5"
        >
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-full bg-red-500/20 border border-red-500/30 flex items-center justify-center shrink-0 text-red-400 mt-0.5">
              <AlertTriangle className="w-4 h-4" />
            </div>
            <div className="flex flex-col">
              <span className="font-semibold text-sm text-red-300">
                Forensic Pipeline Execution Halted
              </span>
              <p className="font-mono text-xs text-red-200 mt-0.5 leading-relaxed">
                {effectiveError || 'The job runner encountered an unrecoverable processing error.'}
              </p>
              <p className="text-[11px] text-[#91A0B8] mt-1.5 leading-relaxed">
                Potential causes: corrupted container format, missing detectable face frames, or processing timeout.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-end pt-1">
            <Button
              variant="outline"
              size="sm"
              icon={<ArrowLeft size={13} />}
              onClick={() => navigate('/app/analyze')}
              className="h-[34px] text-xs rounded-[7px]"
            >
              New Analysis
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate('/app/history')}
              className="h-[34px] text-xs text-[#91A0B8] hover:text-[#E8EDF5]"
            >
              View History
            </Button>
          </div>
        </div>
      )}

      {/* ── 3. SELECTED MEDIA INFORMATION CARD ── */}
      <div className="rounded-[12px] bg-[#171F2D] border border-[#253247] p-4 sm:p-[18px] flex items-center justify-between gap-3 min-h-[72px]">
        <div className="flex items-center gap-3.5 min-w-0">
          <div className="w-10 h-10 rounded-[8px] bg-[#111927] border border-[#253247] flex items-center justify-center text-[#648CFF] shrink-0">
            <FileVideo className="w-[18px] h-[18px]" />
          </div>
          <div className="flex flex-col min-w-0">
            <span
              className="text-[13px] sm:text-sm font-medium text-[#E8EDF5] truncate max-w-[220px] sm:max-w-md"
              title={fileName}
            >
              {fileName}
            </span>
            <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-[#91A0B8] mt-0.5">
              <span>{fileExt} Container</span>
              {initialDetail?.duration_s ? (
                <>
                  <span className="text-[#91A0B8]" aria-hidden="true">·</span>
                  <span>{formatTimecode(initialDetail.duration_s)}</span>
                </>
              ) : null}
              <span className="text-[#91A0B8]" aria-hidden="true">·</span>
              <span>AVT²-DWF Dual-Stream</span>
            </div>
          </div>
        </div>

        {/* Real Status Badge */}
        <div className="shrink-0">
          <span
            className={clsx(
              'inline-flex items-center gap-1.5 h-[24px] px-2.5 rounded-[6px] text-[11px] font-semibold border',
              isDone
                ? 'bg-[#35C99A]/15 text-[#35C99A] border-[#35C99A]/30'
                : isFailed
                ? 'bg-red-950/40 text-red-400 border-red-500/30'
                : isQueued
                ? 'bg-[#648CFF]/10 text-[#648CFF] border-[#648CFF]/25'
                : 'bg-[#3B66EB]/15 text-[#648CFF] border-[#3B66EB]/30',
            )}
          >
            {!isFailed && (
              <span
                className={clsx(
                  'w-1.5 h-1.5 rounded-full',
                  isDone
                    ? 'bg-[#35C99A]'
                    : isQueued
                    ? 'bg-[#648CFF]'
                    : 'bg-[#648CFF] animate-pulse',
                )}
              />
            )}
            <span>
              {isDone
                ? 'COMPLETE'
                : isFailed
                ? 'FAILED'
                : isQueued
                ? 'QUEUED'
                : 'PROCESSING'}
            </span>
          </span>
        </div>
      </div>

      {/* ── 4. MAIN PROCESSING VISUAL (Restrained Scanning Stage) ── */}
      <div
        className="w-full h-[180px] sm:h-[210px] rounded-[14px] bg-[#111927] border border-[#253247] relative overflow-hidden flex flex-col items-center justify-center p-6 select-none"
        aria-label="Forensic signal processing telemetry visualizer"
      >
        {/* Subtle Horizontal Scanning Beam */}
        {!isFailed && !isDone && (
          <div
            className="absolute left-0 right-0 h-[1.5px] bg-gradient-to-r from-transparent via-[#648CFF] to-transparent pointer-events-none opacity-80"
            style={{
              animation: 'scanBeam 3s ease-in-out infinite alternate',
            }}
          />
        )}

        {/* Dual Modality Flux Indicators */}
        <div className="relative z-10 flex items-center justify-center gap-8 sm:gap-14 w-full max-w-md">
          {/* CH-V Node */}
          <div className="flex flex-col items-center gap-2">
            <div
              className={clsx(
                'w-12 h-12 rounded-xl border flex items-center justify-center transition-colors',
                stage === 'faces' || stage === 'face_enc'
                  ? 'bg-[#648CFF]/15 border-[#648CFF] text-[#648CFF] shadow-[0_0_16px_rgba(100,140,255,0.25)]'
                  : isDone || (progress >= 0.65 && !isFailed)
                  ? 'bg-[#35C99A]/15 border-[#35C99A]/40 text-[#35C99A]'
                  : 'bg-[#171F2D] border-[#253247] text-[#91A0B8]',
              )}
            >
              <Eye className="w-5 h-5" />
            </div>
            <div className="flex flex-col items-center text-center">
              <span className="text-xs font-semibold text-[#E8EDF5]">CH-V Optical</span>
              <span className="text-[10px] font-mono text-[#91A0B8]">ViT-B/16 Frames</span>
            </div>
          </div>

          {/* Center DWF Junction */}
          <div className="flex flex-col items-center gap-1.5">
            <div
              className={clsx(
                'w-8 h-8 rounded-full border flex items-center justify-center transition-all',
                stage === 'fusion'
                  ? 'bg-[#648CFF]/20 border-[#648CFF] text-[#648CFF] animate-pulse'
                  : isDone
                  ? 'bg-[#35C99A]/20 border-[#35C99A] text-[#35C99A]'
                  : 'bg-[#171F2D] border-[#253247] text-[#91A0B8]',
              )}
            >
              <Cpu className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#91A0B8]">
              {stage === 'fusion' ? 'FUSING' : isDone ? 'CALIBRATED' : 'DWF GATE'}
            </span>
          </div>

          {/* CH-A Node */}
          <div className="flex flex-col items-center gap-2">
            <div
              className={clsx(
                'w-12 h-12 rounded-xl border flex items-center justify-center transition-colors',
                stage === 'audio' || stage === 'audio_enc'
                  ? 'bg-[#06B6D4]/15 border-[#06B6D4] text-[#06B6D4] shadow-[0_0_16px_rgba(6,182,212,0.25)]'
                  : isDone || (progress >= 0.75 && !isFailed)
                  ? 'bg-[#35C99A]/15 border-[#35C99A]/40 text-[#35C99A]'
                  : 'bg-[#171F2D] border-[#253247] text-[#91A0B8]',
              )}
            >
              <Mic className="w-5 h-5" />
            </div>
            <div className="flex flex-col items-center text-center">
              <span className="text-xs font-semibold text-[#E8EDF5]">CH-A Acoustic</span>
              <span className="text-[10px] font-mono text-[#91A0B8]">Log-Mel 16kHz</span>
            </div>
          </div>
        </div>

        {/* Subtle operational caption */}
        <div className="mt-5 text-center text-xs text-[#91A0B8] z-10 flex items-center gap-2">
          {!isFailed && !isDone && (
            <span className="w-1.5 h-1.5 rounded-full bg-[#648CFF] animate-ping" />
          )}
          <span>
            {isDone
              ? 'Cross-modal coherence analysis complete'
              : isFailed
              ? 'Analysis interrupted'
              : isQueued
              ? 'Awaiting worker pipeline allocation…'
              : 'Active neural inference across visual & acoustic streams'}
          </span>
        </div>
      </div>

      {/* ── 5. PROCESSING STATUS AND PROGRESS CARD ── */}
      <div className="rounded-[12px] bg-[#171F2D] border border-[#253247] p-4 sm:p-5 flex flex-col gap-3">
        {/* Progress Header */}
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2 min-w-0">
            {!isFailed && !isDone && (
              <Loader2 className="w-3.5 h-3.5 text-[#648CFF] animate-spin shrink-0" />
            )}
            {isDone && (
              <CheckCircle2 className="w-3.5 h-3.5 text-[#35C99A] shrink-0" />
            )}
            <span className="text-[#E8EDF5] font-medium truncate">
              {isDone
                ? 'Analysis completed successfully. Generating dossier…'
                : isFailed
                ? 'Processing halted due to error'
                : message || 'Analyzing audio-visual stream…'}
            </span>
          </div>

          <div className="flex items-center gap-3 shrink-0 font-mono text-xs">
            <span className="text-[#91A0B8] flex items-center gap-1">
              <Clock className="w-3 h-3 text-[#648CFF]" />
              <span>T+{elapsed_s.toFixed(1)}s</span>
            </span>
            <span className="font-semibold text-[#648CFF]">
              {isDone ? '100%' : isQueued ? 'QUEUED' : `${progressPercent}%`}
            </span>
          </div>
        </div>

        {/* Progress Bar */}
        <div
          role="progressbar"
          aria-label="Analysis processing progress"
          aria-valuenow={isQueued ? 0 : progressPercent}
          aria-valuemin={0}
          aria-valuemax={100}
          className="w-full h-1.5 rounded-full bg-[#111927] border border-[#253247] overflow-hidden relative"
        >
          {isQueued ? (
            <div className="absolute inset-0 bg-[#3B66EB]/30 animate-pulse" />
          ) : (
            <div
              className={clsx(
                'h-full transition-all duration-300',
                isDone
                  ? 'bg-[#35C99A]'
                  : isFailed
                  ? 'bg-red-500'
                  : 'bg-[#3B66EB]',
              )}
              style={{ width: `${isDone ? 100 : progressPercent}%` }}
            />
          )}
        </div>

        {/* Status Message (Section 8) */}
        <div className="flex items-center justify-between text-[11px] text-[#91A0B8] pt-1">
          <span>
            {isDone
              ? 'Results compiled. Dossier view rendering…'
              : isFailed
              ? 'Please verify file integrity or retry analysis.'
              : 'Please keep this page open while analysis continues.'}
          </span>
          <span className="font-mono text-[#91A0B8]">
            {isQueued ? 'STAGE: QUEUE' : `STAGE: ${stage.toUpperCase()}`}
          </span>
        </div>
      </div>

      {/* ── 6. PROCESSING STAGE INDICATORS (Section 7) ── */}
      <div className="rounded-[12px] bg-[#171F2D] border border-[#253247] p-4 sm:p-5 flex flex-col gap-3">
        <div className="flex items-center justify-between pb-2 border-b border-[#253247]">
          <h2 className="text-sm font-semibold text-[#E8EDF5] tracking-tight">
            Forensic Execution Pipeline
          </h2>
          <span className="text-[11px] font-mono text-[#91A0B8]">
            6 Core Modality Stages
          </span>
        </div>

        <div className="flex flex-col divide-y divide-[#253247]/60">
          {CONCEPTUAL_STAGES.map((s, idx) => {
            const currentIdx = RAW_STAGES.indexOf(stage);
            const lastStageKey = s.stageKeys[s.stageKeys.length - 1];
            const lastStageIdx = RAW_STAGES.indexOf(lastStageKey);
            const isCompletedStage = isDone || (!isFailed && currentIdx > lastStageIdx);
            const isActiveStage = !isDone && !isFailed && s.stageKeys.includes(stage);

            return (
              <div
                key={s.id}
                className="py-2.5 flex items-center justify-between gap-3 text-xs"
              >
                <div className="flex items-center gap-3 min-w-0">
                  {/* Step status icon */}
                  <div className="shrink-0 flex items-center justify-center w-5 h-5">
                    {isCompletedStage ? (
                      <Check className="w-4 h-4 text-[#35C99A]" />
                    ) : isActiveStage ? (
                      <span className="w-2.5 h-2.5 rounded-full bg-[#648CFF] animate-pulse" />
                    ) : (
                      <span className="w-2 h-2 rounded-full bg-[#253247]" />
                    )}
                  </div>

                  <div className="flex flex-col min-w-0">
                    <span
                      className={clsx(
                        'text-xs font-medium truncate',
                        isActiveStage
                          ? 'text-[#E8EDF5] font-semibold'
                          : isCompletedStage
                          ? 'text-[#E8EDF5]'
                          : 'text-[#91A0B8]',
                      )}
                    >
                      {idx + 1}. {s.name}
                    </span>
                    <span className="text-[11px] text-[#91A0B8] truncate">
                      {s.description}
                    </span>
                  </div>
                </div>

                {/* Stage status indicator badge */}
                <div className="shrink-0 font-mono text-[10px]">
                  {isCompletedStage ? (
                    <span className="px-2 py-0.5 rounded-[4px] bg-[#35C99A]/15 text-[#35C99A] font-semibold">
                      DONE
                    </span>
                  ) : isActiveStage ? (
                    <span className="px-2 py-0.5 rounded-[4px] bg-[#648CFF]/15 text-[#648CFF] font-semibold animate-pulse border border-[#648CFF]/30">
                      ACTIVE
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded-[4px] bg-[#111927] text-[#91A0B8]">
                      STANDBY
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── 7. BACKGROUND WORKER GUIDANCE & ACTIONS (Section 10) ── */}
      <div className="rounded-[12px] bg-[#111927] border border-[#253247] p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-[#91A0B8]">
        <div className="flex items-center gap-2.5">
          <ShieldCheck className="w-4 h-4 text-[#35C99A] shrink-0" />
          <span>
            Background Worker Execution: You can safely navigate away; analysis continues on backend workers and results are saved in your audit ledger.
          </span>
        </div>

        <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
          <Button
            variant="outline"
            size="sm"
            icon={<History size={13} />}
            onClick={() => navigate('/app/history')}
            className="h-[32px] px-2.5 text-xs rounded-[7px] bg-[#171F2D] border-[#253247] hover:bg-[#1C2737] text-[#E8EDF5]"
          >
            Audit Ledger
          </Button>
          <Button
            variant="outline"
            size="sm"
            icon={<FolderArchive size={13} />}
            onClick={() => navigate('/app/library')}
            className="h-[32px] px-2.5 text-xs rounded-[7px] bg-[#171F2D] border-[#253247] hover:bg-[#1C2737] text-[#E8EDF5]"
          >
            Corpus
          </Button>
        </div>
      </div>
    </div>
  );
};
