/**
 * Forensic Report Page.
 * Displays verdict, modality balance, timeline, heatmaps, spectrogram, and export options.
 */

import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { VerdictDial } from '../../components/viz/VerdictDial';
import { FusionBalance } from '../../components/viz/FusionBalance';
import { TimelineChart } from '../../components/viz/TimelineChart';
import { FrameStrip } from '../../components/viz/FrameStrip';
import { ScanOverlay } from '../../components/viz/ScanOverlay';
import { SpectrogramView } from '../../components/viz/SpectrogramView';
import { Button } from '../../components/ui/Button';
import { api } from '../../lib/api';
import { formatTimecode } from '../../lib/format';
import type { AnalysisDetail } from '../../types/api';

export const ReportPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [analysis, setAnalysis] = useState<AnalysisDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFrame, setSelectedFrame] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    if (!id) return;

    let mounted = true;
    async function loadReport() {
      setLoading(true);
      setError(null);
      try {
        const data = await api.get<AnalysisDetail>(`/analyses/${id}`);
        if (mounted) {
          setAnalysis(data);
          // Pre-select first heatmap artifact if available
          const firstHeatmap = data.artifacts.find(
            (a) => a.kind === 'heatmap' || a.name.startsWith('heatmap_'),
          );
          if (firstHeatmap) {
            setSelectedFrame(firstHeatmap.name);
          }
        }
      } catch (err: unknown) {
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Failed to load report');
        }
      } finally {
        if (mounted) setLoading(false);
      }
    }

    void loadReport();

    return () => {
      mounted = false;
    };
  }, [id]);

  const handleDelete = async () => {
    if (!id || !confirm('Permanently delete this analysis record and all evidence artifacts?')) return;
    setIsDeleting(true);
    try {
      await api.delete(`/analyses/${id}`);
      navigate('/app/history');
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to delete analysis');
      setIsDeleting(false);
    }
  };

  const handleExportJson = () => {
    if (!analysis) return;
    const blob = new Blob([JSON.stringify(analysis, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forensics_${analysis.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center p-12 min-h-[400px] font-mono text-xs text-text-400 gap-3">
        <span className="w-3 h-3 rounded-full bg-cobalt-500 animate-ping inline-block" />
        LOADING FORENSIC DOSSIER // {id}
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="flex flex-col items-center justify-center p-12 min-h-[400px] font-mono text-xs text-alert-500 gap-3">
        <div>ERR // {error ?? 'Report not found'}</div>
        <Button variant="secondary" size="sm" onClick={() => navigate('/app/analyze')}>
          RETURN TO INGESTION
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto py-2">
      {/* Top Dossier Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-line gap-4">
        <div>
          <div className="font-mono text-xs uppercase tracking-widest text-cobalt-300">
            DOSSIER // {analysis.id}
          </div>
          <h1 className="font-display text-2xl font-bold text-text-100 tracking-tight">
            {analysis.original_name}
          </h1>
          <div className="font-mono text-xs text-text-400 mt-1 flex flex-wrap gap-x-4 gap-y-1">
            <span>DURATION: {formatTimecode(analysis.duration_s)}</span>
            <span>MODEL: {analysis.model_version}</span>
            <span>STATUS: {analysis.status.toUpperCase()}</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Button variant="secondary" size="sm" onClick={handleExportJson}>
            EXPORT JSON
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDelete}
            disabled={isDeleting}
            className="text-alert-500 hover:text-alert-500"
          >
            DELETE
          </Button>
        </div>
      </div>

      {/* Untrained Banner if applicable */}
      {analysis.model_mode === 'untrained' && (
        <div
          role="alert"
          className="p-3 bg-black-900 border border-alert-500/40 rounded-lg flex items-center justify-between font-mono text-xs text-alert-500"
        >
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-alert-500 inline-block" />
            <span className="font-bold">UNCHECKED SYSTEM:</span>
            <span>Untrained weights — scores are not meaningful.</span>
          </div>
          <span className="text-text-500 text-[11px] hidden sm:inline">RESEARCH MODE ONLY</span>
        </div>
      )}

      {/* Primary Verdict & Balance Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <VerdictDial
            verdict={analysis.verdict}
            p_fake={analysis.p_fake}
            model_mode={analysis.model_mode}
          />
        </div>

        <div className="lg:col-span-2 flex flex-col gap-6">
          <FusionBalance
            w_audio={analysis.w_audio}
            w_video={analysis.w_video}
            sync_score={analysis.sync_score}
            audio_present={analysis.audio_present}
          />

          {analysis.reasons && analysis.reasons.length > 0 && (
            <div className="p-4 bg-black-900 border border-line rounded-lg font-mono text-xs">
              <div className="text-cobalt-300 uppercase tracking-widest mb-2">
                FORENSIC FINDINGS // REASONS
              </div>
              <ul className="list-disc list-inside space-y-1 text-text-300">
                {analysis.reasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Multimodal Timeline Chart */}
      <TimelineChart
        timeline={analysis.timeline ?? []}
        duration_s={analysis.duration_s}
      />

      {/* Frame Evidence & Heatmap Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="flex flex-col gap-6">
          <FrameStrip
            analysisId={analysis.id}
            artifacts={analysis.artifacts}
            duration_s={analysis.duration_s}
            onSelectFrame={(name) => setSelectedFrame(name)}
            selectedArtifact={selectedFrame}
          />
          <SpectrogramView
            analysisId={analysis.id}
            duration_s={analysis.duration_s}
            hasAudio={analysis.audio_present}
          />
        </div>

        <div>
          <ScanOverlay
            analysisId={analysis.id}
            artifactName={selectedFrame}
          />
        </div>
      </div>

      {/* Mandatory Forensic Integrity Disclaimer (Master Specification Level 5) */}
      <footer className="mt-2 p-4 bg-[var(--black-900)] border border-[var(--line)] rounded-lg flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs font-mono text-[var(--text-400)]">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[var(--cobalt-400)] shrink-0" />
          <span className="font-bold text-white uppercase tracking-wider">CRITICAL LIMITATION:</span>
          <span>Model-generated probability reflects statistical confidence on this media clip, not absolute forensic proof of authenticity or manipulation.</span>
        </div>
        <span className="text-[10px] text-[var(--text-500)] shrink-0">AVT²-DWF FORENSICS STANDARD</span>
      </footer>
    </div>
  );
};
