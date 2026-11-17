/**
 * HistoryPage — Premium Analysis Archive & Forensic History Ledger.
 *
 * Implements Phase 10/15 Redesign:
 * - Refined page introduction with compact "New Analysis" & "Export CSV" actions
 * - Compact search input and segmented classification filters (All, REAL, FAKE, INCONCLUSIVE)
 * - Single structured archive surface with consistent row heights (64–72px) and subtle separators
 * - Clean columns: Media, Analysis Date, Classification badge, and Action
 * - Responsive layout on viewports < 1024px (ensures no column squashing on 768px tablet)
 * - Zero horizontal overflow, full WCAG AA contrast, and resilient empty/error states
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { api } from '@/lib/api';
import { formatTimecode } from '@/lib/format';
import type { AnalysisListResponse, AnalysisSummary } from '@/types/api';
import {
  Video,
  Search,
  X,
  PlusCircle,
  FileDown,
  Trash2,
  ChevronLeft,
  ChevronRight,
  ArrowRight,
  RotateCw,
  FolderOpen,
} from 'lucide-react';

// Format helper for clean forensic timestamps
function formatAnalysisDate(dateString: string): string {
  try {
    const d = new Date(dateString);
    return d.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
  } catch {
    return dateString;
  }
}

export const HistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const [data, setData] = useState<AnalysisListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [verdictFilter, setVerdictFilter] = useState<string>('ALL');
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const isFirstMount = useRef(true);

  const loadHistory = useCallback(async (p: number, q: string, v: string) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        page: String(p),
        size: '15',
      });
      if (q.trim()) params.set('q', q.trim());
      if (v !== 'ALL') params.set('verdict', v);

      const res = await api.get<AnalysisListResponse>(`/analyses?${params.toString()}`);
      setData(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to query analysis ledger');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isFirstMount.current) {
      isFirstMount.current = false;
      void loadHistory(page, searchQuery, verdictFilter);
      return;
    }
    const timer = setTimeout(() => {
      void loadHistory(page, searchQuery, verdictFilter);
    }, 200);
    return () => clearTimeout(timer);
  }, [page, searchQuery, verdictFilter, loadHistory]);

  const handleFilterChange = (v: string) => {
    setVerdictFilter(v);
    setPage(1);
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    setPage(1);
  };

  const handleResetFilters = () => {
    setSearchQuery('');
    setVerdictFilter('ALL');
    setPage(1);
  };

  const handleDeleteItem = async (e: React.MouseEvent, item: AnalysisSummary) => {
    e.stopPropagation();
    if (!confirm(`Permanently delete dossier for "${item.original_name}"?`)) return;

    setDeletingId(item.id);
    try {
      await api.delete(`/analyses/${item.id}`);
      setData((prev) =>
        prev
          ? {
              ...prev,
              items: prev.items.filter((i) => i.id !== item.id),
              total: Math.max(0, prev.total - 1),
            }
          : null,
      );
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to delete record');
    } finally {
      setDeletingId(null);
    }
  };

  const handleExportCsv = () => {
    if (!data || data.items.length === 0) return;
    const headers = [
      'id',
      'original_name',
      'duration_s',
      'status',
      'stage',
      'verdict',
      'p_fake',
      'created_at',
    ];
    const rows = data.items.map((i) => [
      i.id,
      `"${i.original_name.replace(/"/g, '""')}"`,
      i.duration_s.toFixed(2),
      i.status,
      i.stage,
      i.verdict ?? 'NONE',
      i.p_fake !== null ? (i.p_fake * 100).toFixed(1) : '',
      i.created_at,
    ]);
    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `parallax_analysis_archive_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.ceil(total / 15);
  const isFiltersActive = searchQuery.trim() !== '' || verdictFilter !== 'ALL';

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-7 lg:px-8 pt-7 lg:pt-8 pb-14 font-sans text-[#E8EDF5]">
      <div className="w-full space-y-6">
        {/* ── Section 1: Page Introduction & Action Group ── */}
        <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-[#253247]/60 pb-5">
          <div>
            <span className="text-[11px] font-semibold tracking-wider text-[#648CFF] uppercase block">
              YOUR WORKSPACE
            </span>
            <div className="flex items-center gap-3 mt-0.5">
              <h1 className="text-2xl sm:text-[28px] font-semibold text-[#E8EDF5] tracking-tight">
                Analysis History
                <span className="sr-only">Forensic Audit Ledger</span>
              </h1>
              {total > 0 && (
                <span className="px-2.5 py-0.5 rounded-full bg-[#171F2D] border border-[#253247] text-xs font-mono font-medium text-[#648CFF]">
                  {total} {total === 1 ? 'record' : 'records'}
                </span>
              )}
            </div>
            <p className="text-[13px] sm:text-sm text-[#91A0B8] mt-1 leading-normal">
              Browse and review your previous analysis results.
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2.5 self-start sm:self-auto shrink-0 print:hidden">
            <button
              type="button"
              onClick={handleExportCsv}
              disabled={items.length === 0}
              className="h-[38px] px-3.5 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-[#91A0B8] hover:text-[#E8EDF5] text-[13px] font-medium inline-flex items-center gap-2 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
              title="Export ledger as CSV"
            >
              <FileDown className="w-4 h-4" />
              <span>Export CSV</span>
            </button>

            <Link
              to="/app/analyze"
              className="h-[38px] px-4 rounded-lg bg-[#3B66EB] hover:bg-[#4E77F3] active:bg-[#3258D6] text-white text-[13px] font-semibold inline-flex items-center gap-2 shadow-sm transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF] focus-visible:ring-offset-2 focus-visible:ring-offset-[#090D14]"
            >
              <PlusCircle className="w-4 h-4" />
              <span>New Analysis</span>
            </Link>
          </div>
        </header>

        {/* ── Section 2: Search & Filter Toolbar ── */}
        <section aria-label="Archive Search and Filters" className="w-full">
          <div className="w-full bg-[#111927] border border-[#253247] rounded-xl p-3 sm:p-3.5 flex flex-col md:flex-row md:items-center justify-between gap-3">
            {/* Search Input */}
            <div className="relative w-full md:w-80">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[#91A0B8] pointer-events-none" />
              <input
                type="text"
                placeholder="Search analyses..."
                aria-label="Search dossier by filename or ID"
                value={searchQuery}
                onChange={handleSearchChange}
                className="w-full h-[38px] pl-9 pr-9 bg-[#090D14] border border-[#253247] hover:border-[#374866] focus:border-[#648CFF] rounded-lg text-[13px] text-[#E8EDF5] placeholder-[#91A0B8]/60 transition-colors focus:outline-none focus:ring-2 focus:ring-[#648CFF]"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => {
                    setSearchQuery('');
                    setPage(1);
                  }}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-[#91A0B8] hover:text-[#E8EDF5] transition-colors rounded"
                  aria-label="Clear search input"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {/* Filter Group: Classification Tabs & Clear Filters */}
            <div className="flex flex-wrap items-center gap-2 self-start md:self-auto">
              {/* Segmented Control */}
              <div className="p-1 rounded-lg bg-[#090D14] border border-[#253247] flex items-center gap-1 text-xs">
                {[
                  { label: 'All Results', value: 'ALL' },
                  { label: 'REAL', value: 'AUTHENTIC' },
                  { label: 'FAKE', value: 'FAKE' },
                  { label: 'INCONCLUSIVE', value: 'INCONCLUSIVE' },
                ].map((tab) => {
                  const isSelected = verdictFilter === tab.value;
                  return (
                    <button
                      key={tab.value}
                      type="button"
                      onClick={() => handleFilterChange(tab.value)}
                      className={`h-7 px-2.5 rounded-md font-medium text-[11px] sm:text-xs tracking-wide transition-all cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-[#648CFF] ${
                        isSelected
                          ? 'bg-[#1C2737] text-[#E8EDF5] shadow-xs'
                          : 'text-[#91A0B8] hover:text-[#E8EDF5]'
                      }`}
                    >
                      {tab.label}
                    </button>
                  );
                })}
              </div>

              {/* Clear filters action */}
              {isFiltersActive && (
                <button
                  type="button"
                  onClick={handleResetFilters}
                  className="h-8 px-2.5 rounded-md text-xs font-medium text-[#91A0B8] hover:text-[#E8EDF5] hover:bg-[#171F2D] border border-transparent hover:border-[#253247] transition-all cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-[#648CFF]"
                >
                  Clear filters
                </button>
              )}
            </div>
          </div>
        </section>

        {/* ── Section 3: History Records — Main Surface ── */}
        <section aria-label="Analysis Records" className="w-full">
          {error ? (
            <div className="w-full bg-[#111927] border border-red-500/30 rounded-[14px] p-8 text-center flex flex-col items-center gap-3">
              <span className="text-sm font-semibold text-red-400">Unable to load records</span>
              <p className="text-xs text-[#91A0B8]">{error}</p>
              <button
                type="button"
                onClick={() => void loadHistory(page, searchQuery, verdictFilter)}
                className="mt-2 h-8 px-3.5 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-xs font-medium text-[#E8EDF5] inline-flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <RotateCw className="w-3.5 h-3.5" />
                <span>Retry</span>
              </button>
            </div>
          ) : loading ? (
            /* Skeleton Loading Rows */
            <div className="w-full bg-[#111927] border border-[#253247] rounded-[14px] overflow-hidden divide-y divide-[#253247]/60">
              <div className="h-11 bg-[#171F2D] border-b border-[#253247] hidden lg:block" />
              {[...Array(5)].map((_, idx) => (
                <div key={idx} className="h-16 px-5 flex items-center justify-between gap-4 animate-pulse">
                  <div className="flex items-center gap-3.5 flex-1">
                    <div className="w-9 h-9 rounded-lg bg-[#171F2D]" />
                    <div className="space-y-1.5">
                      <div className="h-3.5 w-48 bg-[#171F2D] rounded" />
                      <div className="h-2.5 w-28 bg-[#171F2D]/60 rounded" />
                    </div>
                  </div>
                  <div className="h-3 w-32 bg-[#171F2D] rounded hidden lg:block" />
                  <div className="h-6 w-20 bg-[#171F2D] rounded-[7px] hidden lg:block" />
                  <div className="h-4 w-24 bg-[#171F2D] rounded" />
                </div>
              ))}
            </div>
          ) : items.length === 0 ? (
            /* Empty State */
            <div className="w-full bg-[#111927] border border-[#253247] rounded-[14px] p-12 text-center flex flex-col items-center justify-center">
              <div className="w-12 h-12 rounded-xl bg-[#171F2D] border border-[#253247] flex items-center justify-center text-[#648CFF] mb-3">
                <FolderOpen className="w-6 h-6" />
              </div>

              {isFiltersActive ? (
                <>
                  <h2 className="text-base font-semibold text-[#E8EDF5]">
                    No matching analyses
                  </h2>
                  <p className="text-xs text-[#91A0B8] mt-1 max-w-sm leading-relaxed">
                    Try changing your search or filters.
                  </p>
                  <button
                    type="button"
                    onClick={handleResetFilters}
                    aria-label="Reset Filters"
                    className="mt-4 h-9 px-4 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-xs font-medium text-[#E8EDF5] inline-flex items-center gap-1.5 transition-colors cursor-pointer"
                  >
                    Clear filters
                  </button>
                </>
              ) : (
                <>
                  <h2 className="text-base font-semibold text-[#E8EDF5]">
                    No analyses found
                    <span className="sr-only">No forensic analyses in ledger</span>
                  </h2>
                  <p className="text-xs text-[#91A0B8] mt-1 max-w-sm leading-relaxed">
                    Your previous analysis records will appear here.
                  </p>
                  <Link
                    to="/app/analyze"
                    className="mt-4 h-9 px-4 rounded-lg bg-[#3B66EB] hover:bg-[#4E77F3] text-white text-xs font-semibold inline-flex items-center gap-1.5 shadow-sm transition-colors cursor-pointer"
                  >
                    <PlusCircle className="w-3.5 h-3.5" />
                    <span>Start New Analysis</span>
                  </Link>
                </>
              )}
            </div>
          ) : (
            /* Main Records Surface */
            <div className="w-full bg-[#111927] border border-[#253247] rounded-[14px] overflow-hidden shadow-xs">
              {/* Desktop Table Header (1024px+) */}
              <div className="hidden lg:flex items-center h-11 bg-[#171F2D] border-b border-[#253247] px-5 text-[11px] font-semibold text-[#91A0B8] uppercase tracking-wider">
                <div className="flex-1">Media</div>
                <div className="w-48 text-left">Analysis Date</div>
                <div className="w-36 text-left">Classification</div>
                <div className="w-36 text-right">Action</div>
              </div>

              {/* Records Rows */}
              <div className="divide-y divide-[#253247]/60">
                {items.map((item) => {
                  const isReal = item.verdict === 'AUTHENTIC' || (item.verdict as string) === 'REAL';
                  const isFake = item.verdict === 'FAKE';
                  const isCompleted = item.status === 'completed' || item.status === 'done';
                  const isFailed = item.status === 'failed';
                  const isDeleting = deletingId === item.id;

                  return (
                    <div
                      key={item.id}
                      onClick={() => navigate(`/app/analyses/${item.id}`)}
                      className="group flex flex-col lg:flex-row lg:items-center justify-between p-4 lg:px-5 lg:py-3.5 hover:bg-[#171F2D]/60 transition-colors cursor-pointer gap-3 sm:gap-4"
                    >
                      {/* Media column */}
                      <div className="flex items-center gap-3.5 flex-1 min-w-0">
                        <div className="w-9 h-9 rounded-lg bg-[#171F2D] group-hover:bg-[#1C2737] border border-[#253247] flex items-center justify-center text-[#648CFF] shrink-0 transition-colors">
                          <Video className="w-4 h-4" />
                        </div>

                        <div className="min-w-0">
                          <span
                            title={item.original_name}
                            className="text-[13px] sm:text-sm font-medium text-[#E8EDF5] group-hover:text-[#648CFF] transition-colors block truncate max-w-xs sm:max-w-md lg:max-w-lg"
                          >
                            {item.original_name}
                          </span>
                          <span className="text-[11px] text-[#91A0B8] font-mono block mt-0.5">
                            {formatTimecode(item.duration_s)} • #{item.id.slice(0, 8)}
                          </span>
                        </div>
                      </div>

                      {/* Date column (Desktop 1024px+) */}
                      <div className="hidden lg:block w-48 text-left shrink-0">
                        <span className="text-xs text-[#91A0B8] font-mono">
                          {formatAnalysisDate(item.created_at)}
                        </span>
                      </div>

                      {/* Classification column (Desktop 1024px+) */}
                      <div className="hidden lg:block w-36 text-left shrink-0">
                        {isCompleted && item.verdict ? (
                          <span
                            className={`inline-flex items-center px-2.5 py-0.5 rounded-[7px] text-[11px] font-semibold tracking-wider uppercase h-[25px] ${
                              isReal
                                ? 'bg-[#35C99A]/15 border border-[#35C99A]/30 text-[#35C99A]'
                                : isFake
                                ? 'bg-[#F16F78]/15 border border-[#F16F78]/30 text-[#F16F78]'
                                : 'bg-[#1C2737] border border-[#253247] text-[#91A0B8]'
                            }`}
                          >
                            {isReal ? 'REAL' : isFake ? 'FAKE' : item.verdict}
                          </span>
                        ) : isFailed ? (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-[7px] text-[11px] font-semibold tracking-wider uppercase h-[25px] bg-red-950/40 border border-red-500/30 text-red-300">
                            FAILED
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-[7px] text-[11px] font-semibold tracking-wider uppercase h-[25px] bg-[#1C2737] border border-[#253247] text-[#648CFF]">
                            {item.status.toUpperCase()}
                          </span>
                        )}
                      </div>

                      {/* Action column (Desktop 1024px+) */}
                      <div className="hidden lg:flex items-center justify-end gap-2 w-36 shrink-0">
                        <span className="text-xs font-semibold text-[#648CFF] group-hover:text-[#88A7FF] inline-flex items-center gap-1 transition-colors">
                          <span>View Result</span>
                          <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                        </span>

                        <button
                          type="button"
                          onClick={(e) => void handleDeleteItem(e, item)}
                          disabled={isDeleting}
                          title="Delete dossier"
                          aria-label={`Delete dossier for ${item.original_name}`}
                          className="p-1.5 rounded hover:bg-red-950/40 text-[#91A0B8] hover:text-red-400 transition-colors ml-2 focus:outline-none focus-visible:ring-1 focus-visible:ring-red-400"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      {/* Responsive Row Layout (< 1024px) */}
                      <div className="flex lg:hidden items-center justify-between border-t border-[#253247]/50 pt-2.5 mt-1">
                        <div className="flex items-center gap-2">
                          {isCompleted && item.verdict ? (
                            <span
                              className={`inline-flex items-center px-2 py-0.5 rounded-[6px] text-[10px] font-semibold tracking-wider uppercase ${
                                isReal
                                  ? 'bg-[#35C99A]/15 border border-[#35C99A]/30 text-[#35C99A]'
                                  : isFake
                                  ? 'bg-[#F16F78]/15 border border-[#F16F78]/30 text-[#F16F78]'
                                  : 'bg-[#1C2737] border border-[#253247] text-[#91A0B8]'
                              }`}
                            >
                              {isReal ? 'REAL' : isFake ? 'FAKE' : item.verdict}
                            </span>
                          ) : isFailed ? (
                            <span className="inline-flex items-center px-2 py-0.5 rounded-[6px] text-[10px] font-semibold tracking-wider uppercase bg-red-950/40 border border-red-500/30 text-red-300">
                              FAILED
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-2 py-0.5 rounded-[6px] text-[10px] font-semibold tracking-wider uppercase bg-[#1C2737] border border-[#253247] text-[#648CFF]">
                              {item.status.toUpperCase()}
                            </span>
                          )}
                          <span className="text-[11px] text-[#91A0B8] font-mono">
                            {formatAnalysisDate(item.created_at)}
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold text-[#648CFF] inline-flex items-center gap-1">
                            <span>View Result</span>
                            <ArrowRight className="w-3.5 h-3.5" />
                          </span>

                          <button
                            type="button"
                            onClick={(e) => void handleDeleteItem(e, item)}
                            disabled={isDeleting}
                            aria-label={`Delete dossier for ${item.original_name}`}
                            className="p-1 rounded text-[#91A0B8] hover:text-red-400"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* ── Section 4: Pagination Footer ── */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between p-3.5 sm:p-4 bg-[#111927] border-t border-[#253247] text-xs text-[#91A0B8] font-mono">
                  <span>
                    Showing page <strong className="text-[#E8EDF5]">{page}</strong> of{' '}
                    <strong className="text-[#E8EDF5]">{totalPages}</strong> ({total} analyses)
                  </span>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={page <= 1}
                      className="h-8 px-3 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-[#E8EDF5] inline-flex items-center gap-1 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-[#648CFF]"
                    >
                      <ChevronLeft className="w-3.5 h-3.5" />
                      <span>Previous</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                      disabled={page >= totalPages}
                      className="h-8 px-3 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-[#E8EDF5] inline-flex items-center gap-1 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-[#648CFF]"
                    >
                      <span>Next</span>
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
};
