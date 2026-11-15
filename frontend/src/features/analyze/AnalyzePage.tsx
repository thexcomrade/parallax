/**
 * Analyze Page: Video upload, live pipeline stage streaming, and transition to report.
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Dropzone } from '../../components/upload/Dropzone';
import { PipelineRail } from '../../components/upload/PipelineRail';
import { useAnalysisStream } from '../../hooks/useAnalysisStream';
import { api } from '../../lib/api';
import type { AnalysisCreated } from '../../types/api';

export const AnalyzePage: React.FC = () => {
  const navigate = useNavigate();
  const [activeAnalysisId, setActiveAnalysisId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const { stage, progress, message, elapsed_s, error: streamError } = useAnalysisStream(
    activeAnalysisId,
    () => {
      // On completion, wait 1 second for visual satisfaction, then navigate to Report
      setTimeout(() => {
        if (activeAnalysisId) {
          navigate(`/app/analyses/${activeAnalysisId}`);
        }
      }, 1200);
    },
  );

  const handleFileSelected = async (file: File) => {
    setIsUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await api.post<AnalysisCreated>('/analyses', formData);
      setActiveAnalysisId(res.id);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed';
      setUploadError(msg);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 max-w-5xl mx-auto py-2">
      {/* Header Bar */}
      <div className="flex items-center justify-between pb-3 border-b border-line">
        <div>
          <div className="font-mono text-xs uppercase tracking-widest text-cobalt-300">
            SIGNAL INGESTION // FORENSIC SCAN
          </div>
          <h1 className="font-display text-2xl font-bold text-text-100 tracking-tight">
            Analyze Media Clip
          </h1>
        </div>
        <div className="font-mono text-xs text-text-500 uppercase tracking-wider hidden sm:block">
          ENGINE: AVT²-DWF // MULTIMODAL
        </div>
      </div>

      {uploadError && (
        <div
          role="alert"
          className="p-4 rounded border border-alert-500/40 bg-alert-500/10 text-alert-500 font-mono text-xs"
        >
          UPLOAD REJECTED // {uploadError}
        </div>
      )}

      {streamError && (
        <div
          role="alert"
          className="p-4 rounded border border-alert-500/40 bg-alert-500/10 text-alert-500 font-mono text-xs"
        >
          PIPELINE ERROR // {streamError}
        </div>
      )}

      {!activeAnalysisId ? (
        <div className="flex flex-col gap-6">
          <Dropzone onFileSelected={handleFileSelected} isUploading={isUploading} />

          {/* Quick guidance notes */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
            <div className="p-3 bg-black-900 border border-line rounded">
              <div className="text-cobalt-300 mb-1">01 // CONTAINER VERIFICATION</div>
              <div className="text-text-400">
                Magic bytes inspected and verified via ffprobe to prevent MIME spoofing.
              </div>
            </div>
            <div className="p-3 bg-black-900 border border-line rounded">
              <div className="text-cobalt-300 mb-1">02 // FACE & AUDIO EXTRACTION</div>
              <div className="text-text-400">
                32 uniform frames with face tracking + 16kHz mono MFCC/log-mel spectrograms.
              </div>
            </div>
            <div className="p-3 bg-black-900 border border-line rounded">
              <div className="text-cobalt-300 mb-1">03 // DYNAMIC WEIGHT FUSION</div>
              <div className="text-text-400">
                Learned gating seesaw calculates audio vs. visual modality trustworthiness.
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-between p-4 bg-black-900 border border-cobalt-500/40 rounded-lg font-mono text-xs">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-lime-400 animate-pulse inline-block" />
              <span className="text-text-100 font-bold uppercase tracking-wide">
                SCAN IN PROGRESS // {activeAnalysisId}
              </span>
            </div>
            <span className="text-text-400">STAGES 01—10</span>
          </div>

          <PipelineRail
            currentStage={stage}
            progress={progress}
            message={message}
            elapsed_s={elapsed_s}
          />
        </div>
      )}
    </div>
  );
};
