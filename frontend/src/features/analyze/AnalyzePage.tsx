import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Dropzone } from '@/components/upload/Dropzone';
import { PipelineRail } from '@/components/upload/PipelineRail';
import { Button } from '@/components/ui/Button';
import { ErrorState } from '@/components/ui/ErrorState';
import { useAnalysisStream } from '@/hooks/useAnalysisStream';
import { api } from '@/lib/api';
import { formatBytes, formatTimecode } from '@/lib/format';
import type { AnalysisCreated } from '@/types/api';
import {
  Play,
  RotateCcw,
  ShieldCheck,
  FileVideo,
  Clock,
  FolderArchive,
  Cpu,
  AlertCircle,
  X,
} from 'lucide-react';

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
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
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
      // Hold for 1 second on completion so operator sees the 100% check, then navigate to dossier
      setTimeout(() => {
        if (activeAnalysisId) {
          navigate(`/app/analyses/${activeAnalysisId}`);
        }
      }, 1000);
    },
  );

  const handleFilePicked = (file: File) => {
    setUploadError(null);
    setUploadProgress(null);
    setSelectedFile(file);

    if (videoMeta?.previewUrl) {
      URL.revokeObjectURL(videoMeta.previewUrl);
    }

    const previewUrl = URL.createObjectURL(file);
    const tempVideo = document.createElement('video');
    tempVideo.preload = 'metadata';
    tempVideo.src = previewUrl;

    const fallbackTimer = setTimeout(() => {
      setVideoMeta((prev) => prev || {
        duration: 0,
        width: 0,
        height: 0,
        previewUrl,
      });
    }, 500);

    tempVideo.onloadedmetadata = () => {
      clearTimeout(fallbackTimer);
      setVideoMeta({
        duration: tempVideo.duration || 0,
        width: tempVideo.videoWidth || 0,
        height: tempVideo.videoHeight || 0,
        previewUrl,
      });
    };

    tempVideo.onerror = () => {
      clearTimeout(fallbackTimer);
      setVideoMeta({
        duration: 0,
        width: 0,
        height: 0,
        previewUrl,
      });
    };
  };

  const handleClearSelection = () => {
    if (videoMeta?.previewUrl) {
      URL.revokeObjectURL(videoMeta.previewUrl);
    }
    setSelectedFile(null);
    setVideoMeta(null);
    setUploadError(null);
    setUploadProgress(null);
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile || isUploading) return;

    setIsUploading(true);
    setUploadProgress(0);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const res = await api.uploadWithProgress<AnalysisCreated>(
        '/analyses',
        formData,
        (p) => {
          setUploadProgress(p.percentage);
        },
      );

      // Clean up object preview
      if (videoMeta?.previewUrl) {
        URL.revokeObjectURL(videoMeta.previewUrl);
      }

      // Navigate to dedicated processing view with persistent canonical URL
      navigate(`/app/analyses/${res.id}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Media upload failed';
      setUploadError(msg);
      setUploadProgress(null);
      setIsUploading(false);
    }
  };

  const fileExt = selectedFile
    ? selectedFile.name.split('.').pop()?.toUpperCase() ?? 'VIDEO'
    : '';

  return (
    <div className="w-full max-w-[1040px] mx-auto flex flex-col gap-6 sm:gap-7">
      {/* 1. Page Introduction (Section 3) */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div className="flex flex-col gap-1">
          <span className="text-[11px] font-mono font-medium uppercase tracking-widest text-[#648CFF]">
            FORENSIC ANALYSIS
          </span>
          <h1 className="text-2xl sm:text-[28px] md:text-[30px] font-semibold text-[#E8EDF5] tracking-tight leading-tight">
            Analyze Media <span className="sr-only">Forensic Media Workbench</span>
          </h1>
          <p className="text-sm text-[#91A0B8] leading-relaxed">
            Upload a video to perform audio-visual deepfake detection.
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button
            variant="outline"
            size="sm"
            icon={<FolderArchive size={14} />}
            onClick={() => navigate('/app/library')}
            className="h-[36px] px-3.5 text-xs font-medium rounded-[8px] bg-[#111927] border-[#253247] hover:bg-[#171F2D] text-[#E8EDF5]"
          >
            Load Benchmark Sample
          </Button>
        </div>
      </div>

      {/* 2. Validation / Ingestion Error Notice */}
      {uploadError && (
        <div
          role="alert"
          className="p-3.5 rounded-[10px] bg-red-950/40 border border-red-500/30 text-red-300 text-xs flex items-center justify-between gap-3"
        >
          <div className="flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
            <span className="leading-relaxed">{uploadError}</span>
          </div>
          <button
            type="button"
            onClick={() => setUploadError(null)}
            className="text-red-300 hover:text-white text-xs underline font-medium p-1 shrink-0"
            aria-label="Dismiss error notice"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Stream Error Notice */}
      {streamError && (
        <ErrorState
          title="Pipeline Execution Interrupted"
          message={streamError}
          onRetry={() => {
            setActiveAnalysisId(null);
            handleClearSelection();
          }}
        />
      )}

      {/* 3. Main Workspace Area: Dropzone OR Selected File Presentation */}
      {!activeAnalysisId ? (
        !selectedFile ? (
          /* Step A: Refined Dropzone Workspace (Section 4 & 5) */
          <Dropzone
            onFileSelected={handleFilePicked}
            isUploading={isUploading}
            onErrorChange={(err) => {
              if (err) setUploadError(null);
            }}
          />
        ) : (
          /* Step B: Selected File Information Panel (Section 6) */
          <div className="rounded-[12px] bg-[#171F2D] border border-[#253247] p-4 sm:p-[18px] flex flex-col gap-4 shadow-sm">
            {/* File Info Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 min-h-[44px]">
              <div className="flex items-center gap-3.5 min-w-0">
                <div className="w-10 h-10 rounded-[8px] bg-[#111927] border border-[#253247] flex items-center justify-center text-[#648CFF] shrink-0">
                  <FileVideo className="w-[18px] h-[18px]" />
                </div>
                <div className="flex flex-col min-w-0">
                  <div className="flex items-center gap-2">
                    <span
                      className="text-[13px] sm:text-sm font-medium text-[#E8EDF5] truncate max-w-[200px] sm:max-w-md md:max-w-lg"
                      title={selectedFile.name}
                    >
                      {selectedFile.name}
                    </span>
                    <span className="inline-flex items-center gap-1.5 h-[22px] px-2 rounded-[6px] text-[10px] font-semibold bg-[#35C99A]/15 text-[#35C99A] border border-[#35C99A]/30 shrink-0">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#35C99A]" />
                      STAGED
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-[#91A0B8] mt-0.5">
                    <span>{formatBytes(selectedFile.size)}</span>
                    <span className="text-[#91A0B8]" aria-hidden="true">·</span>
                    <span>{fileExt}</span>
                    {videoMeta && videoMeta.duration > 0 && (
                      <>
                        <span className="text-[#91A0B8]" aria-hidden="true">·</span>
                        <span>{formatTimecode(videoMeta.duration)}</span>
                        {videoMeta.width > 0 && (
                          <>
                            <span className="text-[#91A0B8]" aria-hidden="true">·</span>
                            <span>{videoMeta.width}×{videoMeta.height}</span>
                          </>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* Action: Change / Replace File */}
              <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
                <Button
                  variant="outline"
                  size="sm"
                  icon={<RotateCcw size={13} />}
                  onClick={handleClearSelection}
                  disabled={isUploading}
                  className="h-[34px] px-3 text-xs font-medium rounded-[8px] bg-[#111927] border-[#253247] hover:bg-[#1C2737] text-[#E8EDF5]"
                >
                  Change File
                </Button>
              </div>
            </div>

            {/* Real Media Stream Upload Progress */}
            {uploadProgress !== null && (
              <div className="flex flex-col gap-2 pt-2 border-t border-[#253247]">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-[#91A0B8]">Transferring Media Stream</span>
                  <span className="text-[#648CFF] font-semibold">{uploadProgress}%</span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-[#111927] overflow-hidden">
                  <div
                    className="h-full bg-[#3B66EB] transition-all duration-150"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Video Preview Player */}
            {videoMeta?.previewUrl && (
              <div className="pt-3 border-t border-[#253247]">
                <div className="rounded-[8px] overflow-hidden bg-[#090D14] border border-[#253247] max-h-[280px] flex items-center justify-center">
                  <video
                    src={videoMeta.previewUrl}
                    controls
                    playsInline
                    className="max-h-[270px] w-auto max-w-full rounded-[6px]"
                  />
                </div>
              </div>
            )}
          </div>
        )
      ) : (
        /* Step C: In-Page Live Execution Pipeline Tracker */
        <div className="flex flex-col gap-6">
          <PipelineRail
            currentStage={stage}
            progress={progress}
            message={message}
            elapsed_s={elapsed_s}
            error={streamError}
          />
        </div>
      )}

      {/* 4. Analysis Action Area (Section 7) */}
      <div className="rounded-[12px] bg-[#111927] border border-[#253247] p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-2.5 text-xs text-[#91A0B8]">
          <ShieldCheck className="w-4 h-4 text-[#35C99A] shrink-0" />
          <span>Isolated sandbox decode · Dual-stream ViT-B/16 &amp; audio transformer inference</span>
        </div>

        <button
          type="button"
          disabled={!selectedFile || isUploading}
          onClick={handleStartAnalysis}
          aria-label={selectedFile ? 'Start Analysis — Run Forensic Analysis' : 'Start Analysis'}
          className="h-[40px] sm:h-[42px] px-5 text-[13px] font-semibold rounded-[8px] inline-flex items-center justify-center gap-2 transition-colors cursor-pointer w-full sm:w-fit disabled:opacity-45 disabled:cursor-not-allowed disabled:bg-[#1C2737] disabled:text-[#91A0B8] disabled:border disabled:border-[#253247] bg-[#3B66EB] hover:bg-[#3258D6] active:bg-[#284BBB] text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
        >
          {isUploading ? (
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin shrink-0" />
          ) : (
            <Play className="w-4 h-4 fill-current shrink-0" />
          )}
          <span>
            {isUploading
              ? `Streaming Ingestion (${uploadProgress ?? 0}%)…`
              : 'Start Analysis'}
          </span>
          {selectedFile && <span className="sr-only">Run Forensic Analysis</span>}
        </button>
      </div>

      {/* 5. Optional Supporting Information — Genuine Technical Specifications (Section 9) */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
        <div className="p-3.5 rounded-[10px] bg-[#111927] border border-[#253247] flex flex-col gap-1">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-[#E8EDF5]">
            <FileVideo className="w-3.5 h-3.5 text-[#648CFF]" />
            <span>Supported Containers</span>
          </div>
          <p className="text-[11px] text-[#91A0B8] leading-relaxed">
            MP4, MOV, WEBM, AVI, MKV. Verified via magic-byte container validation.
          </p>
        </div>

        <div className="p-3.5 rounded-[10px] bg-[#111927] border border-[#253247] flex flex-col gap-1">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-[#E8EDF5]">
            <Clock className="w-3.5 h-3.5 text-[#648CFF]" />
            <span>Size &amp; Duration</span>
          </div>
          <p className="text-[11px] text-[#91A0B8] leading-relaxed">
            Up to 200 MB per file. Recommended clip duration up to 60 seconds.
          </p>
        </div>

        <div className="p-3.5 rounded-[10px] bg-[#111927] border border-[#253247] flex flex-col gap-1">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-[#35C99A]">
            <Cpu className="w-3.5 h-3.5 text-[#35C99A]" />
            <span>Multimodal Extraction</span>
          </div>
          <p className="text-[11px] text-[#91A0B8] leading-relaxed">
            32 uniform optical frames (CH-V) + 16kHz spectral log-mel filterbanks (CH-A).
          </p>
        </div>
      </div>
    </div>
  );
};
