/**
 * Analysis Result Page — Premium REAL / FAKE Forensic Classification Experience.
 *
 * Implements Phase 09/15 Redesign:
 * - Direct, prominent REAL / FAKE classification focus
 * - Clean visual hierarchy with zero internal model score clutter
 * - Analyzed media information panel (format, duration, date, filename)
 * - Concise forensic summary / observations
 * - Content-fit action buttons (New Analysis, View History, Back to Dashboard, Delete)
 * - Full WCAG AA compliance, zero horizontal overflow, responsive layout
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion } from 'motion/react';
import { ProcessingView } from '../../components/upload/ProcessingView';
import { api } from '../../lib/api';
import { formatTimecode, formatRelativeTime } from '../../lib/format';
import type { AnalysisDetail } from '../../types/api';
import {
  ShieldCheck,
  AlertTriangle,
  HelpCircle,
  Video,
  PlusCircle,
  History,
  LayoutDashboard,
  Trash2,
  Copy,
  Check,
  FileDown,
  Printer,
  RotateCw,
} from 'lucide-react';

export const ReportPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [analysis, setAnalysis] = useState<AnalysisDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [copiedId, setCopiedId] = useState(false);

  const loadReport = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.get<AnalysisDetail>(`/analyses/${id}`);
      setAnalysis(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load report');
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    void loadReport();
  }, [loadReport]);

  const handleCopyId = () => {
    if (!analysis) return;
    void navigator.clipboard.writeText(analysis.id);
    setCopiedId(true);
    setTimeout(() => setCopiedId(false), 2000);
  };

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
    a.download = `parallax_dossier_${analysis.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePrint = () => {
    window.print();
  };

  // ─── 1. Loading State ─────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="w-full max-w-[940px] mx-auto px-4 sm:px-7 lg:px-8 pt-7 lg:pt-8 pb-14 space-y-6 font-sans">
        <div className="flex flex-col gap-2">
          <div className="h-3 w-28 bg-[#171F2D] rounded animate-pulse" />
          <div className="h-8 w-56 bg-[#171F2D] rounded-lg animate-pulse" />
          <div className="h-4 w-72 bg-[#111927] rounded animate-pulse" />
        </div>
        <div className="h-44 w-full bg-[#171F2D] rounded-[14px] border border-[#253247] animate-pulse" />
        <div className="h-28 w-full bg-[#111927] rounded-xl border border-[#253247] animate-pulse" />
        <div className="h-24 w-full bg-[#111927] rounded-xl border border-[#253247] animate-pulse" />
      </div>
    );
  }

  // ─── 2. Error State ───────────────────────────────────────────────────────
  if (error || !analysis) {
    return (
      <div className="w-full max-w-[940px] mx-auto px-4 sm:px-7 lg:px-8 pt-12 pb-14 text-center font-sans">
        <div className="max-w-md mx-auto p-6 rounded-2xl bg-[#111927] border border-[#253247] flex flex-col items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-red-950/40 border border-red-500/30 text-red-400 flex items-center justify-center">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-base font-semibold text-[#E8EDF5]">Failed to load result</h1>
            <p className="text-xs text-[#91A0B8] mt-1">{error ?? 'Analysis not found in ledger'}</p>
          </div>
          <div className="flex items-center gap-3 mt-2">
            <button
              type="button"
              onClick={() => void loadReport()}
              className="h-9 px-4 rounded-lg bg-[#3B66EB] hover:bg-[#4E77F3] text-white text-xs font-semibold inline-flex items-center gap-1.5 transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
            >
              <RotateCw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
            <Link
              to="/app/analyze"
              className="h-9 px-3.5 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-xs font-medium text-[#E8EDF5] inline-flex items-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
            >
              Return to Workbench
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // ─── 3. In-flight Processing State ────────────────────────────────────────
  const isCompleted = analysis.status === 'completed' || analysis.status === 'done';
  if (!isCompleted) {
    return (
      <ProcessingView
        analysisId={analysis.id}
        initialDetail={analysis}
        onComplete={() => {
          void loadReport();
        }}
      />
    );
  }

  // ─── 4. Final Classification Logic ────────────────────────────────────────
  const isReal = analysis.verdict === 'AUTHENTIC' || (analysis.verdict as string) === 'REAL';
  const isFake = analysis.verdict === 'FAKE';

  // Derive file extension
  const fileExt = analysis.original_name.includes('.')
    ? analysis.original_name.split('.').pop()?.toUpperCase() ?? 'MP4'
    : 'MP4';

  return (
    <div className="w-full max-w-[940px] mx-auto px-4 sm:px-7 lg:px-8 pt-7 lg:pt-8 pb-14 font-sans text-[#E8EDF5]">
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
        className="w-full space-y-6"
      >
        {/* ── Section 1: Page Introduction & Telemetry Strip ── */}
        <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-[#253247]/60 pb-5">
          <div>
            <span className="text-[11px] font-semibold tracking-wider text-[#648CFF] uppercase block">
              ANALYSIS RESULT
            </span>
            <h1 className="text-2xl sm:text-[28px] font-semibold text-[#E8EDF5] tracking-tight mt-0.5">
              Detection Result
            </h1>
            <p className="text-[13px] sm:text-sm text-[#91A0B8] mt-1 leading-normal">
              Your audio-visual analysis has been completed.
            </p>
          </div>

          {/* Secondary metadata: Identifier and Analysis Timestamp */}
          <div className="flex flex-wrap items-center gap-2 self-start sm:self-auto text-xs font-mono text-[#91A0B8]">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#111927] border border-[#253247]">
              <span className="text-[10px] text-[#91A0B8] uppercase">ID:</span>
              <span className="text-[#E8EDF5]">{analysis.id.slice(0, 8)}</span>
              <button
                type="button"
                onClick={handleCopyId}
                className="p-0.5 hover:text-[#E8EDF5] text-[#91A0B8] transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-[#648CFF] rounded"
                title="Copy full UUID"
                aria-label="Copy analysis ID"
              >
                {copiedId ? (
                  <Check className="w-3 h-3 text-[#35C99A]" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
              </button>
            </div>
            {analysis.completed_at && (
              <span className="px-2.5 py-1 rounded-md bg-[#111927] border border-[#253247] text-[11px] text-[#91A0B8]">
                {formatRelativeTime(analysis.completed_at)}
              </span>
            )}
          </div>
        </header>

        {/* ── Untrained Model Warning Banner (Honesty Standard) ── */}
        {analysis.model_mode === 'untrained' && (
          <div
            role="alert"
            className="p-3.5 bg-amber-950/30 border border-amber-500/30 rounded-xl flex items-center justify-between text-xs text-amber-300"
          >
            <div className="flex items-center gap-2.5">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <span className="font-semibold">UNCHECKED SYSTEM:</span>
              <span>Untrained weights — scores reflect statistical feature variance only, not calibrated synthetic probability.</span>
            </div>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-[#111927] border border-amber-500/30 text-amber-400 hidden sm:inline">
              Research Mode
            </span>
          </div>
        )}

        {/* ── Section 2: Primary Classification Card ── */}
        <section aria-label="Final Classification" className="w-full">
          <div
            className={`w-full rounded-[14px] p-6 sm:p-7 min-h-[150px] flex flex-col sm:flex-row sm:items-center justify-between gap-5 transition-all duration-200 ${
              isReal
                ? 'bg-[#171F2D] border border-[#35C99A]/30 shadow-sm'
                : isFake
                ? 'bg-[#171F2D] border border-[#F16F78]/30 shadow-sm'
                : 'bg-[#171F2D] border border-[#253247]'
            }`}
          >
            {/* Left side: Icon, Classification Label, Supporting Message */}
            <div className="flex items-center gap-4.5 sm:gap-5">
              {/* Classification Icon */}
              {isReal ? (
                <div className="w-12 h-12 rounded-xl bg-[#35C99A]/10 border border-[#35C99A]/25 text-[#35C99A] flex items-center justify-center shrink-0">
                  <ShieldCheck className="w-6 h-6" />
                </div>
              ) : isFake ? (
                <div className="w-12 h-12 rounded-xl bg-[#F16F78]/10 border border-[#F16F78]/25 text-[#F16F78] flex items-center justify-center shrink-0">
                  <AlertTriangle className="w-6 h-6" />
                </div>
              ) : (
                <div className="w-12 h-12 rounded-xl bg-[#1C2737] border border-[#253247] text-[#91A0B8] flex items-center justify-center shrink-0">
                  <HelpCircle className="w-6 h-6" />
                </div>
              )}

              {/* Main Classification Text */}
              <div className="flex flex-col">
                <span className="text-3xl sm:text-[34px] font-bold tracking-tight text-[#E8EDF5] leading-none">
                  {isReal ? 'REAL' : isFake ? 'FAKE' : 'INCONCLUSIVE'}
                </span>
                <p className="text-[13px] sm:text-sm text-[#91A0B8] mt-1.5 leading-normal">
                  {isReal
                    ? 'The submitted media was classified as REAL.'
                    : isFake
                    ? 'The submitted media was classified as FAKE.'
                    : 'The submitted media could not be conclusively classified.'}
                </p>
              </div>
            </div>

            {/* Right side: Restrained status badge */}
            <div className="self-start sm:self-center shrink-0">
              {isReal ? (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#35C99A]/15 border border-[#35C99A]/30 text-xs font-semibold text-[#35C99A] uppercase tracking-wider">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#35C99A]" />
                  AUTHENTIC
                </span>
              ) : isFake ? (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#F16F78]/15 border border-[#F16F78]/30 text-xs font-semibold text-[#F16F78] uppercase tracking-wider">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#F16F78]" />
                  SYNTHETIC DETECTED
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#1C2737] border border-[#253247] text-xs font-semibold text-[#91A0B8] uppercase tracking-wider">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#91A0B8]" />
                  UNRESOLVED
                </span>
              )}
            </div>
          </div>
        </section>

        {/* ── Section 3: Analyzed Media Information ── */}
        <section aria-labelledby="analyzed-media-heading">
          <h2 id="analyzed-media-heading" className="text-sm font-semibold text-[#E8EDF5] mb-2.5">
            Analyzed Media
          </h2>
          <div className="w-full bg-[#111927] border border-[#253247] rounded-xl p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3.5 min-w-0">
              {/* Media Icon Container (36-40px) */}
              <div className="w-10 h-10 rounded-lg bg-[#171F2D] border border-[#253247] flex items-center justify-center text-[#648CFF] shrink-0">
                <Video className="w-5 h-5" />
              </div>

              {/* Filename and Formats */}
              <div className="min-w-0">
                <span className="text-sm font-medium text-[#E8EDF5] block truncate max-w-[260px] sm:max-w-md md:max-w-lg">
                  {analysis.original_name}
                </span>
                <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 mt-1 text-xs text-[#91A0B8]">
                  <span>
                    FORMAT: <strong className="text-[#E8EDF5] font-mono">{fileExt}</strong>
                  </span>
                  <span>
                    DURATION: <strong className="text-[#E8EDF5] font-mono">{formatTimecode(analysis.duration_s)}</strong>
                  </span>
                  {analysis.completed_at && (
                    <span>
                      DATE: <strong className="text-[#E8EDF5] font-mono">{formatRelativeTime(analysis.completed_at)}</strong>
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Quick Export & Print Actions */}
            <div className="flex items-center gap-2 shrink-0 border-t sm:border-t-0 border-[#253247] pt-3 sm:pt-0 print:hidden">
              <button
                type="button"
                onClick={handlePrint}
                className="h-8 px-2.5 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-xs font-medium text-[#91A0B8] hover:text-[#E8EDF5] inline-flex items-center gap-1.5 transition-colors cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-[#648CFF]"
                title="Print report or save PDF"
              >
                <Printer className="w-3.5 h-3.5" />
                <span>Print</span>
              </button>
              <button
                type="button"
                onClick={handleExportJson}
                className="h-8 px-2.5 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-xs font-medium text-[#91A0B8] hover:text-[#E8EDF5] inline-flex items-center gap-1.5 transition-colors cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-[#648CFF]"
                title="Export analysis record as JSON"
              >
                <FileDown className="w-3.5 h-3.5" />
                <span>Export JSON</span>
              </button>
            </div>
          </div>
        </section>

        {/* ── Section 4: Supporting Information / Observations ── */}
        <section aria-labelledby="observations-heading" className="space-y-2.5">
          <h2 id="observations-heading" className="text-sm font-semibold text-[#E8EDF5]">
            {analysis.reasons && analysis.reasons.length > 0 ? 'Forensic Observations' : 'Forensic Summary'}
          </h2>
          <div className="w-full bg-[#111927] border border-[#253247] rounded-xl p-4 sm:p-5">
            {analysis.reasons && analysis.reasons.length > 0 ? (
              <ul className="space-y-2 text-[13px] text-[#91A0B8]">
                {analysis.reasons.map((reason, idx) => (
                  <li key={idx} className="flex items-start gap-2.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#648CFF] shrink-0 mt-2" />
                    <span className="leading-relaxed">{reason}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-[13px] text-[#91A0B8] leading-relaxed">
                {isReal
                  ? 'Multi-modal analysis concluded across acoustic and visual streams with zero decisive deepfake markers identified.'
                  : isFake
                  ? 'Multi-modal analysis flagged significant synthetic manipulation markers across evaluated streams.'
                  : 'Multi-modal analysis concluded without sufficient statistical confidence to render a definitive verdict.'}
              </p>
            )}
          </div>
        </section>

        {/* ── Section 5: Action Buttons ── */}
        <section
          aria-label="Result Navigation Actions"
          className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 print:hidden"
        >
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Primary Action: New Analysis */}
            <Link
              to="/app/analyze"
              className="h-[38px] px-4 rounded-lg bg-[#3B66EB] hover:bg-[#4E77F3] active:bg-[#3258D6] text-white text-[13px] font-semibold inline-flex items-center gap-2 shadow-sm transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF] focus-visible:ring-offset-2 focus-visible:ring-offset-[#090D14]"
            >
              <PlusCircle className="w-4 h-4" />
              <span>New Analysis</span>
            </Link>

            {/* Secondary Action: View History */}
            <Link
              to="/app/history"
              className="h-[36px] px-3.5 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-[#E8EDF5] text-[13px] font-medium inline-flex items-center gap-2 transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
            >
              <History className="w-4 h-4 text-[#91A0B8]" />
              <span>View History</span>
            </Link>

            {/* Optional Action: Back to Dashboard */}
            <Link
              to="/app"
              className="h-[36px] px-3.5 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-[#91A0B8] hover:text-[#E8EDF5] text-[13px] font-medium inline-flex items-center gap-2 transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
            >
              <LayoutDashboard className="w-4 h-4" />
              <span>Back to Dashboard</span>
            </Link>
          </div>

          {/* Delete Dossier Action */}
          <button
            type="button"
            onClick={handleDelete}
            disabled={isDeleting}
            className="h-[36px] px-3 rounded-lg hover:bg-red-950/30 border border-transparent hover:border-red-500/30 text-red-400 hover:text-red-300 text-[13px] font-medium inline-flex items-center gap-1.5 transition-colors cursor-pointer self-start sm:self-auto focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400"
            title="Permanently delete record"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Delete</span>
          </button>
        </section>
      </motion.div>
    </div>
  );
};
