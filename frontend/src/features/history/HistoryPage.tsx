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
import { Link } from 'react-router-dom';
import { api } from '@/lib/api';
import { formatTimecode } from '@/lib/format';
import { Dialog } from '@/components/ui/Dialog';
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
  const [data, setData] = useState<AnalysisListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [verdictFilter, setVerdictFilter] = useState<string>('ALL');
  const [deleteTarget, setDeleteTarget] = useState<AnalysisSummary | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
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

  const handleOpenDelete = (e: React.MouseEvent, item: AnalysisSummary) => {
    e.stopPropagation();
    e.preventDefault();
    setDeleteError(null);
    setDeleteTarget(item);
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    setIsDeleting(true);
    setDeleteError(null);
    try {
      await api.delete(`/analyses/${deleteTarget.id}`);
      setData((prev) =>
        prev
          ? {
              ...prev,
              items: prev.items.filter((i) => i.id !== deleteTarget.id),
              total: Math.max(0, prev.total - 1),
            }
          : null,
      );
      setDeleteTarget(null);
    } catch (err: unknown) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete record');
    } finally {
      setIsDeleting(false);
    }
  };

  const handleExportCsv = () => {
    if (!data || data.items.length === 0) return;
    // Public contract: only the categorical verdict and record metadata are
    // exported. Internal model probabilities never leave the backend.
    const headers = ['id', 'original_name', 'duration_s', 'status', 'stage', 'verdict', 'created_at'];
    const rows = data.items.map((i) => [
      i.id,
      `"${i.original_name.replace(/"/g, '""')}"`,
      i.duration_s.toFixed(2),
      i.status,
      i.stage,
      i.verdict ?? 'NONE',
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
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-7 lg:px-8 pt-7 lg:pt-8 pb-14 font-sans text-[var(--text-100)]">
      <div className="w-full space-y-6">
        {/* ── Section 1: Page Introduction & Action Group ── */}
        <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-[var(--line)]/60 pb-5">
          <div>
            <span className="text-[11px] font-semibold tracking-wider text-[var(--cobalt-300)] uppercase block">
              YOUR WORKSPACE
            </span>
            <div className="flex items-center gap-3 mt-0.5">
              <h1 className="text-2xl sm:text-[28px] font-semibold text-[var(--text-100)] tracking-tight">
                Analysis History
                <span className="sr-only">Forensic Audit Ledger</span>
              </h1>
              {total > 0 && (
                <span className="px-2.5 py-0.5 rounded-full bg-[var(--black-900)] border border-[var(--line)] text-xs font-mono font-medium text-[var(--cobalt-300)]">
                  {total} {total === 1 ? 'record' : 'records'}
                </span>
              )}
            </div>
            <p className="text-[13px] sm:text-sm text-[var(--text-300)] mt-1 leading-normal">
              Browse and review your previous analysis results.
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2.5 self-start sm:self-auto shrink-0 print:hidden">
            <button
              type="button"
              onClick={handleExportCsv}
              disabled={items.length === 0}
              className="h-[38px] px-3.5 rounded-lg bg-[var(--black-900)] hover:bg-[var(--black-700)] border border-[var(--line)] text-[var(--text-300)] hover:text-[var(--text-100)] text-[13px] font-medium inline-flex items-center gap-2 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--cobalt-300)]"
              title="Export ledger as CSV"
            >
              <FileDown className="w-4 h-4" />
              <span>Export CSV</span>
            </button>

            <Link
              to="/app/analyze"
              className="h-[38px] px-4 rounded-lg bg-[var(--cobalt-500)] hover:bg-[var(--cobalt-400)] active:bg-[var(--cobalt-600)] text-[var(--text-100)] text-[13px] font-semibold inline-flex items-center gap-2 shadow-sm transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--cobalt-300)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--surface-base)]"
            >
              <PlusCircle className="w-4 h-4" />
              <span>New Analysis</span>
            </Link>
          </div>
        </header>

        {/* ── Section 2: Search & Filter Toolbar ── */}
        <section aria-label="Archive Search and Filters" className="w-full">
          <div className="w-full bg-[var(--black-950)] border border-[var(--line)] rounded-xl p-3 sm:p-3.5 flex flex-col md:flex-row md:items-center justify-between gap-3">
            {/* Search Input */}
            <div className="relative w-full md:w-80">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-300)] pointer-events-none" />
              <input
                type="text"
                placeholder="Search analyses..."
                aria-label="Search dossier by filename or ID"
                value={searchQuery}
                onChange={handleSearchChange}
                className="w-full h-[38px] pl-9 pr-9 bg-[var(--surface-base)] border border-[var(--line)] hover:border-[var(--line-strong)] focus:border-[var(--cobalt-300)] rounded-lg text-[13px] text-[var(--text-100)] placeholder-[var(--text-300)]/60 transition-colors focus:outline-none focus:ring-2 focus:ring-[var(--cobalt-300)]"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => {
                    setSearchQuery('');
                    setPage(1);
                  }}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-[var(--text-300)] hover:text-[var(--text-100)] transition-colors rounded"
                  aria-label="Clear search input"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {/* Filter Group: Classification Tabs & Clear Filters */}
            <div className="flex flex-wrap items-center gap-2 self-start md:self-auto">
              {/* Segmented Control */}
              <div
                role="group"
                aria-label="Filter by verdict"
                className="p-1 rounded-lg bg-[var(--surface-base)] border border-[var(--line)] flex items-center gap-1 text-xs"
              >
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
                      className={`h-7 px-2.5 rounded-md font-medium text-[11px] sm:text-xs tracking-wide transition-all cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-[var(--cobalt-300)] ${
                        isSelected
                          ? 'bg-[var(--black-700)] text-[var(--text-100)] shadow-xs'
                          : 'text-[var(--text-300)] hover:text-[var(--text-100)]'
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
                  className="h-8 px-2.5 rounded-md text-xs font-medium text-[var(--text-300)] hover:text-[var(--text-100)] hover:bg-[var(--black-900)] border border-transparent hover:border-[var(--line)] transition-all cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-[var(--cobalt-300)]"
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
            <div className="w-full bg-[var(--black-950)] border border-[var(--alert-500)]/30 rounded-[14px] p-8 text-center flex flex-col items-center gap-3">
              <span className="text-sm font-semibold text-[var(--alert-400)]">Unable to load records</span>
              <p className="text-xs text-[var(--text-300)]">{error}</p>
              <button
                type="button"
                onClick={() => void loadHistory(page, searchQuery, verdictFilter)}
                className="mt-2 h-8 px-3.5 rounded-lg bg-[var(--black-900)] hover:bg-[var(--black-700)] border border-[var(--line)] text-xs font-medium text-[var(--text-100)] inline-flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <RotateCw className="w-3.5 h-3.5" />
                <span>Retry</span>
              </button>
            </div>
          ) : loading ? (
            /* Skeleton Loading Rows */
            <div className="w-full bg-[var(--black-950)] border border-[var(--line)] rounded-[14px] overflow-hidden divide-y divide-[var(--line)]/60">
              <div className="h-11 bg-[var(--black-900)] border-b border-[var(--line)] hidden lg:block" />
              {[...Array(5)].map((_, idx) => (
                <div key={idx} className="h-16 px-5 flex items-center justify-between gap-4 animate-pulse">
                  <div className="flex items-center gap-3.5 flex-1">
                    <div className="w-9 h-9 rounded-lg bg-[var(--black-900)]" />
                    <div className="space-y-1.5">
                      <div className="h-3.5 w-48 bg-[var(--black-900)] rounded" />
                      <div className="h-2.5 w-28 bg-[var(--black-900)]/60 rounded" />
                    </div>
                  </div>
                  <div className="h-3 w-32 bg-[var(--black-900)] rounded hidden lg:block" />
                  <div className="h-6 w-20 bg-[var(--black-900)] rounded-[7px] hidden lg:block" />
                  <div className="h-4 w-24 bg-[var(--black-900)] rounded" />
                </div>
              ))}
            </div>
          ) : items.length === 0 ? (
            /* Empty State */
            <div className="w-full bg-[var(--black-950)] border border-[var(--line)] rounded-[14px] p-12 text-center flex flex-col items-center justify-center">
              <div className="w-12 h-12 rounded-xl bg-[var(--black-900)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)] mb-3">
                <FolderOpen className="w-6 h-6" />
              </div>

              {isFiltersActive ? (
                <>
                  <h2 className="text-base font-semibold text-[var(--text-100)]">
                    No matching analyses
                  </h2>
                  <p className="text-xs text-[var(--text-300)] mt-1 max-w-sm leading-relaxed">
                    Try changing your search or filters.
                  </p>
                  <button
                    type="button"
                    onClick={handleResetFilters}
                    aria-label="Reset Filters"
                    className="mt-4 h-9 px-4 rounded-lg bg-[var(--black-900)] hover:bg-[var(--black-700)] border border-[var(--line)] text-xs font-medium text-[var(--text-100)] inline-flex items-center gap-1.5 transition-colors cursor-pointer"
                  >
                    Clear filters
                  </button>
                </>
              ) : (
                <>
                  <h2 className="text-base font-semibold text-[var(--text-100)]">
                    No analyses found
                    <span className="sr-only">No forensic analyses in ledger</span>
                  </h2>
                  <p className="text-xs text-[var(--text-300)] mt-1 max-w-sm leading-relaxed">
                    Your previous analysis records will appear here.
                  </p>
                  <Link
                    to="/app/analyze"
                    className="mt-4 h-9 px-4 rounded-lg bg-[var(--cobalt-500)] hover:bg-[var(--cobalt-400)] text-[var(--text-100)] text-xs font-semibold inline-flex items-center gap-1.5 shadow-sm transition-colors cursor-pointer"
                  >
                    <PlusCircle className="w-3.5 h-3.5" />
                    <span>Start New Analysis</span>
                  </Link>
                </>
              )}
            </div>
          ) : (
            /* Main Records Surface */
            <div className="w-full bg-[var(--black-950)] border border-[var(--line)] rounded-[14px] overflow-hidden shadow-xs">
              {/* Desktop Table Header (1024px+) */}
              <div className="hidden lg:flex items-center h-11 bg-[var(--black-900)] border-b border-[var(--line)] px-5 text-[11px] font-semibold text-[var(--text-300)] uppercase tracking-wider">
                <div className="flex-1">Media</div>
                <div className="w-48 text-left">Analysis Date</div>
                <div className="w-36 text-left">Classification</div>
                <div className="w-36 text-right">Action</div>
              </div>

              {/* Records Rows */}
              <div className="divide-y divide-[var(--line)]/60">
                {items.map((item) => {
                  const isReal = item.verdict === 'AUTHENTIC' || (item.verdict as string) === 'REAL';
                  const isFake = item.verdict === 'FAKE';
                  const isCompleted = item.status === 'completed' || item.status === 'done';
                  const isFailed = item.status === 'failed';

                  return (
                    <div
                      key={item.id}
                      className="group relative flex flex-col lg:flex-row lg:items-center justify-between p-4 lg:px-5 lg:py-3.5 hover:bg-[var(--black-900)]/60 transition-colors gap-3 sm:gap-4"
                    >

                      {/* Media column — the dossier link. A real link, not a
                          clickable row, so it is keyboard reachable and can be
                          announced with its destination filename. */}
                      <Link
                        to={`/app/analyses/${item.id}`}
                        className="flex items-center gap-3.5 flex-1 min-w-0 rounded-[var(--radius-sm)] focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-[var(--cobalt-400)]"
                      >
                        <div className="w-9 h-9 rounded-lg bg-[var(--black-900)] group-hover:bg-[var(--black-700)] border border-[var(--line)] flex items-center justify-center text-[var(--cobalt-300)] shrink-0 transition-colors">
                          <Video className="w-4 h-4" />
                        </div>

                        <div className="min-w-0">
                          <span
                            title={item.original_name}
                            className="text-[13px] sm:text-sm font-medium text-[var(--text-100)] group-hover:text-[var(--cobalt-300)] transition-colors block truncate max-w-xs sm:max-w-md lg:max-w-lg"
                          >
                            {item.original_name}
                          </span>
                          <span className="text-[11px] text-[var(--text-300)] font-mono block mt-0.5">
                            {formatTimecode(item.duration_s)} • #{item.id.slice(0, 8)}
                          </span>
                        </div>
                      </Link>

                      {/* Date column (Desktop 1024px+) */}
                      <div className="hidden lg:block w-48 text-left shrink-0">
                        <span className="text-xs text-[var(--text-300)] font-mono">
                          {formatAnalysisDate(item.created_at)}
                        </span>
                      </div>

                      {/* Classification column (Desktop 1024px+) */}
                      <div className="hidden lg:block w-36 text-left shrink-0">
                        {isCompleted && item.verdict ? (
                          <span
                            className={`inline-flex items-center px-2.5 py-0.5 rounded-[7px] text-[11px] font-semibold tracking-wider uppercase h-[25px] ${
                              isReal
                                ? 'bg-[var(--success-500)]/15 border border-[var(--success-500)]/30 text-[var(--success-500)]'
                                : isFake
                                ? 'bg-[var(--alert-500)]/15 border border-[var(--alert-500)]/30 text-[var(--alert-500)]'
                                : 'bg-[var(--black-700)] border border-[var(--line)] text-[var(--text-300)]'
                            }`}
                          >
                            {isReal ? 'REAL' : isFake ? 'FAKE' : item.verdict}
                          </span>
                        ) : isFailed ? (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-[7px] text-[11px] font-semibold tracking-wider uppercase h-[25px] bg-[var(--alert-bg)] border border-[var(--alert-500)]/30 text-[var(--alert-400)]">
                            FAILED
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-[7px] text-[11px] font-semibold tracking-wider uppercase h-[25px] bg-[var(--black-700)] border border-[var(--line)] text-[var(--cobalt-300)]">
                            {item.status.toUpperCase()}
                          </span>
                        )}
                      </div>

                      {/* Action column (Desktop 1024px+) */}
                      <div className="hidden lg:flex items-center justify-end gap-2 w-36 shrink-0">
                        <span className="text-xs font-semibold text-[var(--cobalt-300)] group-hover:text-[var(--cobalt-200)] inline-flex items-center gap-1 transition-colors">
                          <span>View Result</span>
                          <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                        </span>

                        <button
                          type="button"
                          onClick={(e) => handleOpenDelete(e, item)}
                          disabled={isDeleting}
                          title="Delete dossier"
                          aria-label={`Delete dossier for ${item.original_name}`}
                          className="relative z-20 p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--alert-bg)] text-[var(--text-300)] hover:text-[var(--alert-400)] transition-colors ml-2 focus:outline-none focus-visible:ring-1 focus-visible:ring-[var(--alert-500)]"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      {/* Responsive Row Layout (< 1024px) */}
                      <div className="flex lg:hidden items-center justify-between border-t border-[var(--line)]/50 pt-2.5 mt-1">
                        <div className="flex items-center gap-2">
                          {isCompleted && item.verdict ? (
                            <span
                              className={`inline-flex items-center px-2 py-0.5 rounded-[6px] text-[10px] font-semibold tracking-wider uppercase ${
                                isReal
                                  ? 'bg-[var(--success-500)]/15 border border-[var(--success-500)]/30 text-[var(--success-500)]'
                                  : isFake
                                  ? 'bg-[var(--alert-500)]/15 border border-[var(--alert-500)]/30 text-[var(--alert-500)]'
                                  : 'bg-[var(--black-700)] border border-[var(--line)] text-[var(--text-300)]'
                              }`}
                            >
                              {isReal ? 'REAL' : isFake ? 'FAKE' : item.verdict}
                            </span>
                          ) : isFailed ? (
                            <span className="inline-flex items-center px-2 py-0.5 rounded-[6px] text-[10px] font-semibold tracking-wider uppercase bg-[var(--alert-bg)] border border-[var(--alert-500)]/30 text-[var(--alert-400)]">
                              FAILED
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-2 py-0.5 rounded-[6px] text-[10px] font-semibold tracking-wider uppercase bg-[var(--black-700)] border border-[var(--line)] text-[var(--cobalt-300)]">
                              {item.status.toUpperCase()}
                            </span>
                          )}
                          <span className="text-[11px] text-[var(--text-300)] font-mono">
                            {formatAnalysisDate(item.created_at)}
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold text-[var(--cobalt-300)] inline-flex items-center gap-1">
                            <span>View Result</span>
                            <ArrowRight className="w-3.5 h-3.5" />
                          </span>

                          <button
                            type="button"
                            onClick={(e) => handleOpenDelete(e, item)}
                            disabled={isDeleting}
                            aria-label={`Delete dossier for ${item.original_name}`}
                            className="relative z-20 p-1.5 rounded-[var(--radius-sm)] text-[var(--text-300)] hover:text-[var(--alert-400)] transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-[var(--alert-500)]"
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
                <div className="flex items-center justify-between p-3.5 sm:p-4 bg-[var(--black-950)] border-t border-[var(--line)] text-xs text-[var(--text-300)] font-mono">
                  <span>
                    Showing page <strong className="text-[var(--text-100)]">{page}</strong> of{' '}
                    <strong className="text-[var(--text-100)]">{totalPages}</strong> ({total} analyses)
                  </span>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={page <= 1}
                      className="h-8 px-3 rounded-lg bg-[var(--black-900)] hover:bg-[var(--black-700)] border border-[var(--line)] text-[var(--text-100)] inline-flex items-center gap-1 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-[var(--cobalt-300)]"
                    >
                      <ChevronLeft className="w-3.5 h-3.5" />
                      <span>Previous</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                      disabled={page >= totalPages}
                      className="h-8 px-3 rounded-lg bg-[var(--black-900)] hover:bg-[var(--black-700)] border border-[var(--line)] text-[var(--text-100)] inline-flex items-center gap-1 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-[var(--cobalt-300)]"
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

        <Dialog
          open={deleteTarget !== null}
          title="Delete this analysis record?"
          description={
            deleteTarget
              ? `Permanently delete the dossier and artifacts for "${deleteTarget.original_name}"? This action cannot be undone.`
              : ''
          }
          confirmLabel="Delete record"
          cancelLabel="Keep record"
          variant="danger"
          loading={isDeleting}
          onConfirm={() => void handleConfirmDelete()}
          onClose={() => {
            if (!isDeleting) setDeleteTarget(null);
          }}
        >
          {deleteError && (
            <p role="alert" className="text-[12px] text-[var(--alert-400)] leading-relaxed">
              {deleteError}
            </p>
          )}
        </Dialog>
      </div>
    </div>
  );
};
