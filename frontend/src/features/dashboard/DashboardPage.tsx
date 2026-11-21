import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '@/lib/api';
import { Skeleton } from '@/components/ui/Skeleton';
import { formatRelativeTime } from '@/lib/format';
import type {
  AnalysisListResponse,
  ReadyResponse,
  ModelCardResponse,
} from '@/types/api';
import {
  Plus,
  ArrowRight,
  Cpu,
  Video,
  Volume2,
  UploadCloud,
  FileText,
  ShieldCheck,
  AlertTriangle,
  Activity,
  FolderArchive,
  History,
  RotateCw,
} from 'lucide-react';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();

  // Queries backed strictly by real API responses
  const analysesQuery = useQuery({
    queryKey: ['analyses'],
    queryFn: () => api.get<AnalysisListResponse>('/analyses?page=1&size=10'),
    staleTime: 30_000,
  });

  const readyQuery = useQuery({
    queryKey: ['system', 'ready'],
    queryFn: () => api.get<ReadyResponse>('/ready'),
    staleTime: 30_000,
  });

  const modelQuery = useQuery({
    queryKey: ['system', 'model'],
    queryFn: () => api.get<ModelCardResponse>('/model'),
    staleTime: 60_000,
  });

  const analyses = analysesQuery.data?.items ?? [];
  const totalScans = analysesQuery.data?.total ?? analyses.length;

  // Genuine KPI calculations from real analyses
  const evaluatedAnalyses = analyses.filter(
    (a) => a.verdict && (a.status === 'done' || a.status === 'completed'),
  );

  const fakes = evaluatedAnalyses.filter((a) => a.verdict === 'FAKE').length;
  const authentic = evaluatedAnalyses.filter(
    (a) => a.verdict === 'AUTHENTIC' || (a.verdict as string) === 'REAL',
  ).length;

  const isUntrained = modelQuery.data?.model_mode === 'untrained';

  return (
    <div className="flex flex-col gap-6 sm:gap-7 w-full">
      {/* ── SECTION 1 — COMPACT INTRODUCTION ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <span className="text-[11px] font-mono font-semibold tracking-wider text-[var(--cobalt-300)] uppercase">
            WORKSPACE
          </span>
          <h1 className="text-2xl sm:text-[30px] font-semibold tracking-tight text-[var(--text-100)] mt-0.5 leading-tight">
            Dashboard <span className="sr-only">Forensic Command Overview</span>
          </h1>
          <p className="text-xs sm:text-[14px] text-[var(--text-300)] mt-1">
            Your audio-visual deepfake detection workspace.
          </p>
        </div>

        {/* Content-fit single primary action */}
        <button
          type="button"
          onClick={() => navigate('/app/analyze')}
          className="h-[38px] sm:h-10 inline-flex items-center justify-center gap-2 px-3.5 sm:px-4 rounded-[var(--radius-md)] bg-[var(--cobalt-500)] hover:bg-[var(--cobalt-400)] active:bg-[var(--cobalt-600)] text-[var(--text-100)] font-semibold text-[13px] transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-300)] shadow-xs cursor-pointer leading-none w-fit shrink-0"
        >
          <Plus size={16} strokeWidth={2.5} className="shrink-0" />
          <span>New Analysis</span>
        </button>
      </div>

      {/* Untrained Benchmark Banner (if running uncalibrated weights) */}
      {isUntrained && (
        <aside
          role="note"
          className="p-3.5 rounded-[var(--radius-lg)] bg-[var(--warning-500)]/10 border border-[var(--warning-500)]/30 flex items-center justify-between gap-3 text-xs text-[var(--warning-500)]"
        >
          <div className="flex items-center gap-2.5">
            <span className="w-2 h-2 rounded-full bg-[var(--warning-500)] shrink-0" />
            <span>
              <strong>Model Weights Uncalibrated:</strong> Detection output verifies architectural dataflow only.
            </span>
          </div>
        </aside>
      )}

      {/* ── SECTION 2 — OVERVIEW METRIC CARDS (4 equal fraction cards) ── */}
      <section
        aria-label="Overview Metrics"
        className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4"
      >
        {/* Card 1: Total Analyses */}
        <div className="p-4 sm:p-5 rounded-[var(--radius-cell)] bg-[var(--black-900)] border border-[var(--line)] hover:border-[var(--line-strong)] transition-colors flex flex-col justify-between min-h-[108px]">
          <div className="flex items-center justify-between">
            <span className="text-[12px] sm:text-[13px] font-medium text-[var(--text-300)]">
              Total Analyses
            </span>
            <div className="w-8 h-8 rounded-[var(--radius-sm)] bg-[var(--black-950)] border border-[var(--line)] flex items-center justify-center text-[var(--text-300)] shrink-0">
              <FileText size={16} />
            </div>
          </div>
          <div className="mt-2.5">
            <div className="text-2xl sm:text-[28px] font-semibold text-[var(--text-100)] tracking-tight leading-none">
              {analysesQuery.isLoading ? <Skeleton className="h-7 w-16" /> : totalScans}
            </div>
            <p className="text-[11px] sm:text-[12px] text-[var(--text-300)] mt-1.5 leading-none">
              Persisted scan dossiers
            </p>
          </div>
        </div>

        {/* Card 2: REAL Results */}
        <div className="p-4 sm:p-5 rounded-[var(--radius-cell)] bg-[var(--black-900)] border border-[var(--line)] hover:border-[var(--success-500)]/40 transition-colors flex flex-col justify-between min-h-[108px]">
          <div className="flex items-center justify-between">
            <span className="text-[12px] sm:text-[13px] font-medium text-[var(--text-300)]">
              REAL Results
            </span>
            <div className="w-8 h-8 rounded-[var(--radius-sm)] bg-[var(--success-500)]/10 border border-[var(--success-500)]/25 flex items-center justify-center text-[var(--success-500)] shrink-0">
              <ShieldCheck size={16} />
            </div>
          </div>
          <div className="mt-2.5">
            <div className="text-2xl sm:text-[28px] font-semibold text-[var(--success-500)] tracking-tight leading-none">
              {analysesQuery.isLoading ? <Skeleton className="h-7 w-16" /> : authentic}
            </div>
            <p className="text-[11px] sm:text-[12px] text-[var(--text-300)] mt-1.5 leading-none">
              Of the latest 10 dossiers
            </p>
          </div>
        </div>

        {/* Card 3: FAKE Results */}
        <div className="p-4 sm:p-5 rounded-[var(--radius-cell)] bg-[var(--black-900)] border border-[var(--line)] hover:border-[var(--alert-500)]/40 transition-colors flex flex-col justify-between min-h-[108px]">
          <div className="flex items-center justify-between">
            <span className="text-[12px] sm:text-[13px] font-medium text-[var(--text-300)]">
              FAKE Results
            </span>
            <div className="w-8 h-8 rounded-[var(--radius-sm)] bg-[var(--alert-500)]/10 border border-[var(--alert-500)]/25 flex items-center justify-center text-[var(--alert-500)] shrink-0">
              <AlertTriangle size={16} />
            </div>
          </div>
          <div className="mt-2.5">
            <div className="text-2xl sm:text-[28px] font-semibold text-[var(--alert-500)] tracking-tight leading-none">
              {analysesQuery.isLoading ? <Skeleton className="h-7 w-16" /> : fakes}
            </div>
            <p className="text-[11px] sm:text-[12px] text-[var(--text-300)] mt-1.5 leading-none">
              Of the latest 10 dossiers
            </p>
          </div>
        </div>

        {/* Card 4: Latest Analysis */}
        <div className="p-4 sm:p-5 rounded-[var(--radius-cell)] bg-[var(--black-900)] border border-[var(--line)] hover:border-[var(--line-strong)] transition-colors flex flex-col justify-between min-h-[108px]">
          <div className="flex items-center justify-between">
            <span className="text-[12px] sm:text-[13px] font-medium text-[var(--text-300)]">
              Latest Analysis
            </span>
            <div className="w-8 h-8 rounded-[var(--radius-sm)] bg-[var(--black-950)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)] shrink-0">
              <Activity size={16} />
            </div>
          </div>
          <div className="mt-2.5">
            <div className="text-2xl sm:text-[28px] font-semibold text-[var(--text-100)] tracking-tight leading-none truncate">
              {analysesQuery.isLoading ? (
                <Skeleton className="h-7 w-20" />
              ) : analyses[0] ? (
                formatRelativeTime(analyses[0].created_at)
              ) : (
                'Idle'
              )}
            </div>
            <p className="text-[11px] sm:text-[12px] text-[var(--text-300)] mt-1.5 leading-none">
              Most recent forensic scan
            </p>
          </div>
        </div>
      </section>

      {/* ── SECTION 3 — MAIN CONTENT GRID (8 cols / 4 cols) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 sm:gap-6 items-start">
        {/* Left Column (8 cols): Recent Analyses Panel */}
        <div className="lg:col-span-8 flex flex-col gap-4 min-w-0">
          <div className="rounded-[12px] bg-[var(--black-900)] border border-[var(--line)] overflow-hidden">
            {/* Header */}
            <div className="px-4 py-3.5 sm:px-5 sm:py-4 border-b border-[var(--line)] flex items-center justify-between">
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-[var(--text-100)]">
                  Recent Analyses
                </h2>
                <p className="text-xs text-[var(--text-300)] mt-0.5">
                  Your latest detection results.
                </p>
              </div>
              <Link
                to="/app/history"
                className="text-xs font-medium text-[var(--cobalt-300)] hover:text-[var(--cobalt-200)] inline-flex items-center gap-1 transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-300)] rounded"
              >
                <span>View All</span>
                <ArrowRight size={13} />
              </Link>
            </div>

            {/* List / Empty State / Error State */}
            <div className="p-0">
              {analysesQuery.isLoading ? (
                <div className="p-4 sm:p-5 flex flex-col gap-2.5">
                  <Skeleton className="h-14 w-full rounded-[8px]" />
                  <Skeleton className="h-14 w-full rounded-[8px]" />
                  <Skeleton className="h-14 w-full rounded-[8px]" />
                </div>
              ) : analysesQuery.isError ? (
                /* Compact Error State */
                <div className="p-8 sm:p-10 flex flex-col items-center justify-center text-center">
                  <div className="w-10 h-10 rounded-[10px] bg-[var(--alert-bg)] border border-[var(--alert-500)]/25 flex items-center justify-center text-[var(--alert-500)] mb-2.5">
                    <AlertTriangle size={18} />
                  </div>
                  <h3 className="text-sm font-semibold text-[var(--text-100)]">Unable to load analyses</h3>
                  <p className="text-xs text-[var(--text-300)] max-w-xs mt-1 mb-3.5">
                    The forensic telemetry service could not be reached.
                  </p>
                  <button
                    type="button"
                    onClick={() => analysesQuery.refetch()}
                    className="h-[34px] inline-flex items-center gap-1.5 px-3 rounded-[8px] bg-[var(--black-700)] hover:bg-[var(--line)] border border-[var(--line)] text-[var(--text-100)] text-xs font-medium transition-colors cursor-pointer"
                  >
                    <RotateCw size={13} />
                    <span>Retry connection</span>
                  </button>
                </div>
              ) : analyses.length === 0 ? (
                /* Compact Empty State */
                <div className="p-8 sm:p-12 flex flex-col items-center justify-center text-center">
                  <div className="w-11 h-11 rounded-[10px] bg-[var(--black-950)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)] mb-3">
                    <FileText size={18} />
                  </div>
                  <h3 className="text-sm sm:text-[15px] font-semibold text-[var(--text-100)]">No analyses yet</h3>
                  <p className="text-xs sm:text-[13px] text-[var(--text-300)] max-w-xs mt-1 mb-4">
                    Start your first audio-visual deepfake analysis.
                  </p>
                  <button
                    type="button"
                    onClick={() => navigate('/app/analyze')}
                    className="h-[36px] inline-flex items-center gap-1.5 px-3.5 rounded-[8px] bg-[var(--cobalt-500)] hover:bg-[var(--cobalt-400)] text-[var(--text-100)] text-xs font-semibold shadow-xs transition-colors cursor-pointer w-fit"
                  >
                    <Plus size={14} strokeWidth={2.5} />
                    <span>Start Analysis</span>
                  </button>
                </div>
              ) : (
                <div className="divide-y divide-[var(--line)]">
                  {analyses.slice(0, 5).map((item) => (
                    <Link
                      key={item.id}
                      to={`/app/analyses/${item.id}`}
                      className="px-4 py-3 sm:px-5 sm:py-3.5 flex items-center justify-between gap-3 hover:bg-[var(--black-700)]/60 transition-colors group min-h-[60px] sm:min-h-[64px]"
                    >
                      <div className="flex items-center gap-3 min-w-0 flex-1">
                        <div className="w-9 h-9 rounded-[8px] bg-[var(--black-950)] border border-[var(--line)] flex items-center justify-center text-[var(--text-300)] group-hover:text-[var(--cobalt-300)] group-hover:border-[var(--cobalt-300)]/30 shrink-0 transition-colors">
                          <Video size={17} />
                        </div>
                        <div className="flex flex-col min-w-0 flex-1">
                          <span
                            className="text-xs sm:text-[13px] font-medium text-[var(--text-100)] truncate group-hover:text-[var(--text-100)]"
                            title={item.original_name}
                          >
                            {item.original_name}
                          </span>
                          <span className="text-[11px] text-[var(--text-300)] mt-0.5">
                            {formatRelativeTime(item.created_at)}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center gap-3 shrink-0">
                        {item.verdict === 'FAKE' && (
                          <span className="h-[25px] inline-flex items-center gap-1.5 px-2.5 rounded-[7px] text-[11px] font-semibold bg-[var(--alert-500)]/15 text-[var(--alert-500)] border border-[var(--alert-500)]/30">
                            <span className="w-1.5 h-1.5 rounded-full bg-[var(--alert-500)]" />
                            FAKE
                          </span>
                        )}
                        {(item.verdict === 'AUTHENTIC' || (item.verdict as string) === 'REAL') && (
                          <span className="h-[25px] inline-flex items-center gap-1.5 px-2.5 rounded-[7px] text-[11px] font-semibold bg-[var(--success-500)]/15 text-[var(--success-500)] border border-[var(--success-500)]/30">
                            <span className="w-1.5 h-1.5 rounded-full bg-[var(--success-500)]" />
                            REAL
                          </span>
                        )}
                        {(!item.verdict || item.verdict === 'INCONCLUSIVE') && (
                          <span className="h-[25px] inline-flex items-center gap-1.5 px-2.5 rounded-[7px] text-[11px] font-semibold bg-[var(--black-950)] text-[var(--text-300)] border border-[var(--line)]">
                            <span className="w-1.5 h-1.5 rounded-full bg-[var(--text-300)]" />
                            {item.status === 'processing' || item.status === 'running'
                              ? 'PROCESSING'
                              : 'INCONCLUSIVE'}
                          </span>
                        )}
                        <ArrowRight
                          size={14}
                          className="text-[var(--text-300)] group-hover:text-[var(--text-100)] group-hover:translate-x-0.5 transition-all hidden sm:block shrink-0"
                        />
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </div>

            {analyses.length > 0 && (
              <div className="px-4 sm:px-5 py-2.5 border-t border-[var(--line)] bg-[var(--black-950)]/40 flex items-center justify-between text-[11px] text-[var(--text-300)]">
                <span>
                  Showing {Math.min(5, analyses.length)} most recent{' '}
                  {Math.min(5, analyses.length) === 1 ? 'record' : 'records'}
                </span>
                <span>Total indexed: {totalScans}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column (4 cols): Secondary Workspace & Telemetry */}
        <div className="lg:col-span-4 flex flex-col gap-4 min-w-0">
          {/* Media Ingestion Workbench Card */}
          <Link
            to="/app/analyze"
            className="p-4 sm:p-[18px] rounded-[12px] bg-[var(--black-900)] border border-[var(--line)] hover:border-[var(--cobalt-400)]/60 interactive-card group flex flex-col gap-3"
          >
            <div className="flex items-center justify-between">
              <div className="w-8.5 h-8.5 rounded-[8px] bg-[var(--cobalt-300)]/15 border border-[var(--cobalt-300)]/25 flex items-center justify-center text-[var(--cobalt-300)] group-hover:scale-105 transition-transform shrink-0">
                <UploadCloud size={17} />
              </div>
              <span className="h-[22px] inline-flex items-center gap-1 px-2 rounded-[6px] text-[10px] font-mono text-[var(--success-500)] bg-[var(--success-500)]/10 border border-[var(--success-500)]/20">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--success-500)]" />
                READY
              </span>
            </div>

            <div>
              <h3 className="text-[13px] sm:text-sm font-semibold text-[var(--text-100)] group-hover:text-[var(--text-100)] transition-colors">
                Media Ingestion Workbench
              </h3>
              <p className="text-xs text-[var(--text-300)] mt-1 leading-relaxed">
                Upload video media (MP4, WebM, MOV) for biometric and acoustic transformer verification.
              </p>
            </div>

            <div className="pt-2 border-t border-[var(--line)]/60 flex items-center justify-between text-xs text-[var(--cobalt-300)] font-medium group-hover:text-[var(--cobalt-200)]">
              <span>Launch Workbench</span>
              <ArrowRight size={13} className="group-hover:translate-x-0.5 transition-transform" />
            </div>
          </Link>

          {/* Inference Engine Telemetry Card */}
          <div className="p-4 sm:p-[18px] rounded-[12px] bg-[var(--black-900)] border border-[var(--line)] flex flex-col gap-3.5">
            <div className="flex items-center justify-between border-b border-[var(--line)] pb-2.5">
              <div className="flex items-center gap-2">
                <Cpu size={15} className="text-[var(--cobalt-300)]" />
                <h3 className="text-[13px] sm:text-sm font-semibold text-[var(--text-100)]">
                  Inference Engine
                </h3>
              </div>
              <span className="h-[22px] inline-flex items-center gap-1 px-2 rounded-[6px] text-[10px] font-medium bg-[var(--success-500)]/15 text-[var(--success-500)] border border-[var(--success-500)]/30">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--success-500)]" />
                {readyQuery.data?.status === 'ready' ? 'READY' : 'ONLINE'}
              </span>
            </div>

            <div className="flex flex-col gap-2 text-xs">
              <div className="flex items-center justify-between pb-1.5 border-b border-[var(--line)]/50">
                <span className="text-[var(--text-300)]">Architecture</span>
                <span className="font-mono text-xs font-semibold text-[var(--text-100)]">AVT²-DWF</span>
              </div>

              <div className="flex items-center justify-between pb-1.5 border-b border-[var(--line)]/50">
                <span className="text-[var(--text-300)]">Checkpoint</span>
                <span className="font-mono text-[11px] text-[var(--cobalt-300)] truncate max-w-[130px]">
                  {modelQuery.data?.checkpoint_hash
                    ? modelQuery.data.checkpoint_hash.slice(0, 12) + '...'
                    : 'LOCAL_WEIGHTS'}
                </span>
              </div>

              <div className="flex items-center justify-between pb-1.5 border-b border-[var(--line)]/50">
                <span className="text-[var(--text-300)]">Database Mode</span>
                <span className="font-mono text-xs text-[var(--success-500)] font-medium">
                  {readyQuery.data?.db_connected ? 'SQLite (WAL)' : 'DISCONNECTED'}
                </span>
              </div>

              <div className="flex flex-col gap-1.5 pt-0.5">
                <span className="text-[var(--text-300)] text-[11px]">Active Modality Channels</span>
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="p-2 rounded-[8px] bg-[var(--black-950)] border border-[var(--line)] flex items-center gap-1.5 text-[var(--text-100)]">
                    <Video size={13} className="text-[var(--cobalt-300)]" />
                    <span className="font-medium">ViT-B16</span>
                  </div>
                  <div className="p-2 rounded-[8px] bg-[var(--black-950)] border border-[var(--line)] flex items-center gap-1.5 text-[var(--text-100)]">
                    <Volume2 size={13} className="text-[var(--cobalt-300)]" />
                    <span className="font-medium">Log-Mel</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Resource Shortcuts */}
          <div className="p-3 rounded-[12px] bg-[var(--black-900)]/60 border border-[var(--line)] flex items-center justify-between text-xs text-[var(--text-300)]">
            <Link
              to="/app/library"
              className="flex items-center gap-1.5 hover:text-[var(--text-100)] transition-colors"
            >
              <FolderArchive size={14} className="text-[var(--cobalt-300)]" />
              <span>Reference Corpus</span>
            </Link>
            <span className="text-[var(--line)]">|</span>
            <Link
              to="/app/history"
              className="flex items-center gap-1.5 hover:text-[var(--text-100)] transition-colors"
            >
              <History size={14} className="text-[var(--cobalt-300)]" />
              <span>Full Audit Ledger</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};
