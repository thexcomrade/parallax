/**
 * Vertical forensic pipeline stage monitor.
 * Displays real-time SSE progress across the 10 analysis stages.
 */

import React from 'react';

interface StageInfo {
  id: string;
  label: string;
  code: string;
}

const STAGES: StageInfo[] = [
  { id: 'probe', label: 'Media Container Probe', code: 'ST-01' },
  { id: 'frames', label: 'Uniform Frame Sampling (32f)', code: 'ST-02' },
  { id: 'faces', label: 'Face Tracking & Alignment', code: 'ST-03' },
  { id: 'audio', label: 'Audio MFCC & Log-Mel Extraction', code: 'ST-04' },
  { id: 'align', label: 'Cross-Modal Temporal Alignment', code: 'ST-05' },
  { id: 'face_enc', label: 'Facial ViT-B/16 Encoder', code: 'ST-06' },
  { id: 'audio_enc', label: 'Audio Transformer Encoder', code: 'ST-07' },
  { id: 'fusion', label: 'Dynamic Weight Fusion (DWF)', code: 'ST-08' },
  { id: 'explain', label: 'Explainability & Heatmaps', code: 'ST-09' },
  { id: 'done', label: 'Forensic Report Generation', code: 'ST-10' },
];

interface PipelineRailProps {
  currentStage: string;
  progress: number;
  message?: string;
  elapsed_s?: number;
}

export const PipelineRail: React.FC<PipelineRailProps> = ({
  currentStage,
  progress,
  message,
  elapsed_s = 0,
}) => {
  const currentIdx = STAGES.findIndex((s) => s.id === currentStage);

  return (
    <div className="flex flex-col gap-3 p-4 bg-black-900 border border-line rounded-lg">
      <div className="flex items-center justify-between pb-2 border-b border-line text-xs font-mono">
        <span className="text-cobalt-300 tracking-wider">PIPELINE MONITOR // AVT²-DWF</span>
        <span className="text-text-400">T+{elapsed_s.toFixed(1)}s</span>
      </div>

      <div className="flex flex-col gap-2">
        {STAGES.map((st, idx) => {
          const isDone = currentIdx > idx || currentStage === 'done';
          const isActive = currentStage === st.id;

          let statusTag = 'WAIT';
          let statusColor = 'text-text-500 border-line';
          if (isDone) {
            statusTag = 'DONE';
            statusColor = 'text-lime-400 border-lime-400/40 bg-lime-400/10';
          } else if (isActive) {
            statusTag = 'EXEC';
            statusColor = 'text-cobalt-400 border-cobalt-500/60 bg-cobalt-500/10 animate-pulse';
          }

          return (
            <div
              key={st.id}
              className={`flex items-center justify-between p-2 rounded border transition-colors ${
                isActive
                  ? 'border-cobalt-500/50 bg-black-800'
                  : isDone
                  ? 'border-line/60 bg-black-950/40'
                  : 'border-transparent text-text-500'
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-mono text-[10px] text-text-500 w-8">{st.code}</span>
                <span
                  className={`text-xs font-mono uppercase tracking-wide ${
                    isActive ? 'text-text-100 font-semibold' : isDone ? 'text-text-300' : 'text-text-500'
                  }`}
                >
                  {st.label}
                </span>
              </div>

              <span
                className={`font-mono text-[10px] px-1.5 py-0.5 rounded border uppercase tracking-wider ${statusColor}`}
              >
                {statusTag}
              </span>
            </div>
          );
        })}
      </div>

      {message && (
        <div className="mt-2 pt-2 border-t border-line font-mono text-[11px] text-text-400 flex items-center justify-between">
          <span className="truncate pr-2">{message}</span>
          <span className="text-lime-400 font-bold">{Math.round(progress * 100)}%</span>
        </div>
      )}
    </div>
  );
};
