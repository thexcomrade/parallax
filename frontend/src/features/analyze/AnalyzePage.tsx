/**
 * Analyze Page: Video upload, staged video preview, live pipeline stage streaming, and transition to report.
 * Aligns with Level 5 Forensic Engineering & UI/UX Specifications.
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Dropzone } from '../../components/upload/Dropzone';
import { PipelineRail } from '../../components/upload/PipelineRail';
import { Button } from '../../components/ui/Button';
import { useAnalysisStream } from '../../hooks/useAnalysisStream';
import { api } from '../../lib/api';
import { formatBytes, formatTimecode } from '../../lib/format';
import type { AnalysisCreated } from '../../types/api';
import { Play, RotateCcw, ShieldCheck, Film, HardDrive, Clock, Monitor } from 'lucide-react';

interface VideoMeta {
  duration: number;
  width: number;
  height: number;
  previewUrl: string;
}

export const AnalyzePage: React.FC = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoMeta, setVideoMeta] = useState<VideoMeta | null>(null);
  const [activeAnalysisId, setActiveAnalysisId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Clean up object URL when file changes or component unmounts
  useEffect(() => {
    return () => {
      if (videoMeta?.previewUrl) {
        URL.revokeObjectURL(videoMeta.previewUrl);
      }
    };
  }, [videoMeta?.previewUrl]);

  const { stage, progress, message, elapsed_s, error: streamError } = useAnalysisStream(
    activeAnalysisId,
    () => {
      // On completion, brief hold for visual confirmation, then navigate to dossier
      setTimeout(() => {
        if (activeAnalysisId) {
          navigate(`/app/analyses/${activeAnalysisId}`);
        }
      }, 1200);
    },
  );

  const handleFilePicked = (file: File) => {
    setUploadError(null);
    if (videoMeta?.previewUrl) {
      URL.revokeObjectURL(videoMeta.previewUrl);
    }

    const previewUrl = URL.createObjectURL(file);
    const tempVideo = document.createElement('video');
    tempVideo.preload = 'metadata';
    tempVideo.src = previewUrl;

    const timer = setTimeout(() => {
      setVideoMeta((prev) => prev || {
        duration: 0,
        width: 0,
        height: 0,
        previewUrl,
      });
      setSelectedFile((prev) => prev || file);
    }, 400);

    tempVideo.onloadedmetadata = () => {
      clearTimeout(timer);
      setVideoMeta({
        duration: tempVideo.duration || 0,
        width: tempVideo.videoWidth || 0,
        height: tempVideo.videoHeight || 0,
        previewUrl,
      });
      setSelectedFile(file);
    };

    tempVideo.onerror = () => {
      clearTimeout(timer);
      setVideoMeta({
        duration: 0,
        width: 0,
        height: 0,
        previewUrl,
      });
      setSelectedFile(file);
    };
  };

  const handleClearSelection = () => {
    if (videoMeta?.previewUrl) {
      URL.revokeObjectURL(videoMeta.previewUrl);
    }
    setSelectedFile(null);
    setVideoMeta(null);
    setUploadError(null);
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

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
      <div className="flex items-center justify-between pb-3 border-b border-[var(--line)]">
        <div>
          <div className="font-mono text-xs uppercase tracking-widest text-[var(--cobalt-300)]">
            SIGNAL INGESTION // FORENSIC SCAN
          </div>
          <h1 className="font-display text-2xl font-bold text-white tracking-tight">
            Analyze Media Clip
          </h1>
        </div>
        <div className="font-mono text-xs text-[var(--text-400)] uppercase tracking-wider hidden sm:block">
          ENGINE: AVT²-DWF // MULTIMODAL
        </div>
      </div>

      {uploadError && (
        <div
          role="alert"
          className="p-4 rounded border border-[var(--alert-500)]/40 bg-[var(--alert-500)]/10 text-[var(--alert-500)] font-mono text-xs flex items-center justify-between"
        >
          <span>UPLOAD REJECTED // {uploadError}</span>
          <Button variant="ghost" size="sm" onClick={() => setUploadError(null)}>
            DISMISS
          </Button>
        </div>
      )}

      {streamError && (
        <div
          role="alert"
          className="p-4 rounded border border-[var(--alert-500)]/40 bg-[var(--alert-500)]/10 text-[var(--alert-500)] font-mono text-xs"
        >
          PIPELINE ERROR // {streamError}
        </div>
      )}

      {!activeAnalysisId ? (
        !selectedFile ? (
          <div className="flex flex-col gap-6">
            <Dropzone onFileSelected={handleFilePicked} isUploading={isUploading} />

            {/* Quick guidance notes */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
              <div className="p-3 bg-[var(--black-900)] border border-[var(--line)] rounded">
                <div className="text-[var(--cobalt-300)] mb-1 flex items-center gap-1.5">
                  <ShieldCheck size={14} /> 01 // CONTAINER VERIFICATION
                </div>
                <div className="text-[var(--text-400)]">
                  Magic bytes inspected and verified via ffprobe to prevent MIME spoofing.
                </div>
              </div>
              <div className="p-3 bg-[var(--black-900)] border border-[var(--line)] rounded">
                <div className="text-[var(--cobalt-300)] mb-1 flex items-center gap-1.5">
                  <Film size={14} /> 02 // FACE & AUDIO EXTRACTION
                </div>
                <div className="text-[var(--text-400)]">
                  32 uniform frames with face tracking + 16kHz mono MFCC/log-mel spectrograms.
                </div>
              </div>
              <div className="p-3 bg-[var(--black-900)] border border-[var(--line)] rounded">
                <div className="text-[var(--cobalt-300)] mb-1 flex items-center gap-1.5">
                  <Play size={14} /> 03 // DYNAMIC WEIGHT FUSION
                </div>
                <div className="text-[var(--text-400)]">
                  Learned gating seesaw calculates audio vs. visual modality trustworthiness.
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Staged Forensic Video Preview Bench */
          <div className="flex flex-col gap-6 bg-[var(--black-900)] border border-[var(--line)] rounded-lg p-5">
            <div className="flex items-center justify-between pb-3 border-b border-[var(--line)] font-mono text-xs">
              <div className="flex items-center gap-2 text-[var(--cobalt-300)] font-bold uppercase">
                <span className="w-2 h-2 rounded-full bg-[var(--lime-400)]" />
                STAGED MEDIA SPECIFICATION // {selectedFile.name}
              </div>
              <span className="text-[var(--text-400)]">READY FOR INGESTION</span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Native Video Player Preview */}
              <div className="lg:col-span-2 bg-[var(--black-950)] border border-[var(--line)] rounded flex items-center justify-center overflow-hidden min-h-[260px] max-h-[380px]">
                {videoMeta?.previewUrl ? (
                  <video
                    src={videoMeta.previewUrl}
                    controls
                    playsInline
                    className="w-full h-full object-contain max-h-[360px]"
                  />
                ) : (
                  <div className="text-[var(--text-400)] font-mono text-xs">
                    PREVIEW NOT AVAILABLE
                  </div>
                )}
              </div>

              {/* Forensic Metadata Readout */}
              <div className="flex flex-col justify-between gap-4 font-mono text-xs">
                <div className="flex flex-col gap-3 p-4 bg-[var(--black-950)] border border-[var(--line)] rounded">
                  <div className="text-[var(--cobalt-300)] font-bold text-[11px] uppercase tracking-wider mb-1">
                    STREAM CHARACTERISTICS
                  </div>

                  <div className="flex items-center justify-between py-1 border-b border-[var(--line)]">
                    <span className="text-[var(--text-400)] flex items-center gap-1.5">
                      <HardDrive size={13} /> FILE SIZE:
                    </span>
                    <span className="text-white font-bold">{formatBytes(selectedFile.size)}</span>
                  </div>

                  <div className="flex items-center justify-between py-1 border-b border-[var(--line)]">
                    <span className="text-[var(--text-400)] flex items-center gap-1.5">
                      <Clock size={13} /> DURATION:
                    </span>
                    <span className="text-white font-bold">
                      {videoMeta?.duration ? formatTimecode(videoMeta.duration) : 'PROBING…'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between py-1 border-b border-[var(--line)]">
                    <span className="text-[var(--text-400)] flex items-center gap-1.5">
                      <Monitor size={13} /> RESOLUTION:
                    </span>
                    <span className="text-white font-bold">
                      {videoMeta?.width && videoMeta?.height
                        ? `${videoMeta.width} × ${videoMeta.height}`
                        : 'DETECTING…'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between py-1">
                    <span className="text-[var(--text-400)] flex items-center gap-1.5">
                      <Film size={13} /> CONTAINER:
                    </span>
                    <span className="text-[var(--lime-400)] font-bold uppercase truncate max-w-[120px]">
                      {selectedFile.name.split('.').pop() ?? 'MP4'}
                    </span>
                  </div>
                </div>

                <div className="flex flex-col gap-2 pt-2">
                  <Button
                    type="button"
                    variant="primary"
                    size="lg"
                    className="w-full justify-center gap-2 bg-[var(--lime-400)] text-[var(--lime-ink)] font-bold"
                    onClick={handleStartAnalysis}
                    loading={isUploading}
                    disabled={isUploading}
                  >
                    <span>START FORENSIC ANALYSIS</span>
                  </Button>

                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="w-full justify-center text-[var(--text-400)] hover:text-white"
                    onClick={handleClearSelection}
                    disabled={isUploading}
                  >
                    <RotateCcw size={13} className="mr-1.5" /> DISCARD & SELECT DIFFERENT FILE
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )
      ) : (
        /* Live Execution Monitor */
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-between p-4 bg-[var(--black-900)] border border-[var(--cobalt-500)]/40 rounded-lg font-mono text-xs">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[var(--lime-400)] animate-pulse inline-block" />
              <span className="text-white font-bold uppercase tracking-wide">
                SCAN IN PROGRESS // {activeAnalysisId}
              </span>
            </div>
            <span className="text-[var(--text-400)]">STAGES 01—10</span>
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
