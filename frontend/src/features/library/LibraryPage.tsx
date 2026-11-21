/**
 * LibraryPage — Curated Forensic Benchmark Corpus & Ground-Truth Reference Samples.
 *
 * Implements Phase 11/15: PARALLAX LIBRARY PAGE REFINEMENT
 * - Consistent application typography (sans) matching Dashboard & History
 * - Ground Truth classification redesign: neutral label + restrained semantic value badge
 * - Balanced segmented filter controls using application selected style
 * - Refined card presentation with subtle dividers instead of heavy nested boxes
 * - Clean provenance banner with naturally wrapped text and zero misleading badges
 * - Preserves zero client third-party requests and 100% existing functionality
 */

import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '@/lib/api';
import { formatBytes, formatTimecode } from '@/lib/format';
import type { SampleAnalyzeResponse, SampleItem, SampleListResponse } from '@/types/api';
import {
  Film,
  Mic,
  Play,
  ExternalLink,
  ShieldCheck,
  AlertTriangle,
  Clock,
  HardDrive,
  Info,
  Search,
  X,
  Loader2,
  PlusCircle,
  RefreshCw,
  FolderArchive,
} from 'lucide-react';

export const LibraryPage: React.FC = () => {
  const navigate = useNavigate();
  const [filter, setFilter] = useState<'all' | 'authentic' | 'synthetic'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [analyzingId, setAnalyzingId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const samplesQuery = useQuery({
    queryKey: ['samples'],
    queryFn: () => api.get<SampleListResponse>('/samples'),
    staleTime: 120_000,
  });

  const handleAnalyzeSample = async (sample: SampleItem) => {
    setErrorMsg(null);
    setAnalyzingId(sample.id);
    try {
      const res = await api.post<SampleAnalyzeResponse>(`/samples/${sample.id}/analyze`);
      navigate(`/app/analyses/${res.id}`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Sample analysis could not be initiated';
      setErrorMsg(`Failed to stage sample "${sample.title}": ${message}`);
      setAnalyzingId(null);
    }
  };

  const samples = useMemo(
    () => samplesQuery.data?.items ?? [],
    [samplesQuery.data?.items]
  );

  const authenticCount = useMemo(
    () => samples.filter((s) => s.ground_truth === 'authentic').length,
    [samples]
  );
  const syntheticCount = useMemo(
    () => samples.filter((s) => s.ground_truth === 'synthetic').length,
    [samples]
  );

  const filteredSamples = useMemo(() => {
    return samples.filter((s) => {
      // Ground truth classification filter
      if (filter !== 'all' && s.ground_truth !== filter) {
        return false;
      }
      // Keyword search
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const matchesTitle = s.title.toLowerCase().includes(q);
        const matchesDesc = s.description.toLowerCase().includes(q);
        const matchesSource = s.source.toLowerCase().includes(q);
        const matchesAttribution = s.attribution.toLowerCase().includes(q);
        if (!matchesTitle && !matchesDesc && !matchesSource && !matchesAttribution) {
          return false;
        }
      }
      return true;
    });
  }, [samples, filter, searchQuery]);

  return (
    <div className="flex flex-col gap-6 w-full max-w-7xl mx-auto py-1 sm:py-2 text-[var(--text-100)]">
      {/* ── 1. Page Header (Matches Dashboard & History) ──────────────────── */}
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[var(--line)]/60">
        <div>
          <span className="text-[11px] font-semibold tracking-wider text-[var(--cobalt-300)] uppercase block">
            MEDIA WORKSPACE
          </span>
          <div className="flex items-center gap-3 mt-0.5">
            <h1 className="text-2xl sm:text-[28px] font-semibold text-[var(--text-100)] tracking-tight">
              Library <span className="sr-only">Forensic Benchmark Library</span>
            </h1>
            {samples.length > 0 && (
              <span className="px-2.5 py-0.5 rounded-full bg-[var(--black-900)] border border-[var(--line)] text-xs font-medium text-[var(--text-300)]">
                {samples.length} Curated Media
              </span>
            )}
          </div>
          <p className="text-[13px] sm:text-sm text-[var(--text-300)] mt-1 leading-normal max-w-2xl">
            Browse curated reference media and ground-truth benchmark samples for calibration and baseline evaluation.
          </p>
        </div>

        {/* Header Action: New Analysis */}
        <div className="flex items-center gap-2.5 self-start sm:self-auto shrink-0">
          <Link
            to="/app/analyze"
            className="h-[38px] px-4 rounded-lg bg-[var(--cobalt-500)] hover:bg-[var(--cobalt-400)] active:bg-[var(--cobalt-600)] text-[var(--text-100)] text-[13px] font-semibold inline-flex items-center gap-2 shadow-xs transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--cobalt-300)]"
          >
            <PlusCircle className="w-4 h-4" />
            <span>New Analysis</span>
          </Link>
        </div>
      </header>

      {/* ── 2. Security & Provenance Banner ───────────────────────────────── */}
      <aside
        aria-label="Security and provenance notice"
        className="p-3 sm:px-4 rounded-xl bg-[var(--black-950)] border border-[var(--line)] flex items-start sm:items-center gap-3 text-xs text-[var(--text-300)]"
      >
        <ShieldCheck className="w-4 h-4 text-[var(--cobalt-300)] shrink-0 mt-0.5 sm:mt-0" />
        <p className="leading-relaxed">
          <span className="font-semibold text-[var(--text-100)] mr-1.5">Restricted backend ingestion:</span>
          Benchmark media files are downloaded and evaluated strictly within backend isolated worker sandboxes with zero client cross-origin requests.
        </p>
      </aside>

      {/* ── 3. Search and Filter Controls Toolbar ─────────────────────────── */}
      <section aria-label="Library Controls" className="w-full">
        <div className="w-full p-2.5 sm:p-3 rounded-xl bg-[var(--black-950)] border border-[var(--line)] flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          {/* Search Input */}
          <div className="relative flex-1 max-w-md flex items-center">
            <Search className="w-4 h-4 text-[var(--text-300)] absolute left-3 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search benchmark media by title or keyword..."
              aria-label="Search benchmark media"
              className="w-full h-[38px] pl-9 pr-8 bg-[var(--surface-base)] border border-[var(--line)] hover:border-[var(--line-strong)] focus:border-[var(--cobalt-300)] focus:outline-none focus:ring-1 focus:ring-[var(--cobalt-300)] rounded-lg text-[13px] text-[var(--text-100)] placeholder-[var(--text-300)]/60 transition-colors"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                aria-label="Clear search"
                className="absolute right-2.5 p-1 text-[var(--text-300)] hover:text-[var(--text-100)] transition-colors cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Segmented Filter Control */}
          <div
            role="group"
            aria-label="Filter by ground truth classification"
            className="flex items-center gap-1 p-1 rounded-lg bg-[var(--surface-base)] border border-[var(--line)] shrink-0 text-xs overflow-x-auto"
          >
            <button
              type="button"
              onClick={() => setFilter('all')}
              className={`h-[30px] px-3.5 rounded-md text-xs font-medium transition-colors cursor-pointer shrink-0 ${
                filter === 'all'
                  ? 'bg-[var(--black-700)] text-[var(--text-100)] border border-[var(--line)] shadow-xs'
                  : 'text-[var(--text-300)] hover:text-[var(--text-100)]'
              }`}
            >
              All [{samples.length}]
            </button>
            <button
              type="button"
              onClick={() => setFilter('authentic')}
              className={`h-[30px] px-3.5 rounded-md text-xs font-medium transition-colors cursor-pointer shrink-0 ${
                filter === 'authentic'
                  ? 'bg-[var(--black-700)] text-[var(--text-100)] border border-[var(--line)] shadow-xs'
                  : 'text-[var(--text-300)] hover:text-[var(--text-100)]'
              }`}
            >
              Authentic [{authenticCount}]
            </button>
            <button
              type="button"
              onClick={() => setFilter('synthetic')}
              className={`h-[30px] px-3.5 rounded-md text-xs font-medium transition-colors cursor-pointer shrink-0 ${
                filter === 'synthetic'
                  ? 'bg-[var(--black-700)] text-[var(--text-100)] border border-[var(--line)] shadow-xs'
                  : 'text-[var(--text-300)] hover:text-[var(--text-100)]'
              }`}
            >
              Synthetic [{syntheticCount}]
            </button>
          </div>
        </div>
      </section>

      {/* ── 4. Error Message Banner if Staging Fails ──────────────────────── */}
      {errorMsg && (
        <div
          role="alert"
          className="p-3 rounded-xl bg-[var(--alert-bg)] border border-[var(--alert-500)]/30 text-xs text-[var(--alert-400)] flex items-center justify-between gap-2.5"
        >
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-[var(--alert-400)] shrink-0" />
            <span>{errorMsg}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorMsg(null)}
            aria-label="Dismiss error"
            className="text-[var(--text-300)] hover:text-[var(--text-100)] p-1 rounded-[var(--radius-sm)] cursor-pointer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* ── 5. Benchmark Samples Grid / Loading / Error / Empty ─────────────── */}
      {samplesQuery.isLoading ? (
        /* Loading Skeletons */
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-5 animate-pulse">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="rounded-[14px] bg-[var(--black-900)] border border-[var(--line)] p-5 flex flex-col justify-between min-h-[260px]"
            >
              <div>
                <div className="flex justify-between items-center mb-3">
                  <div className="w-16 h-4 bg-[var(--line)] rounded" />
                  <div className="w-28 h-4 bg-[var(--line)] rounded-full" />
                </div>
                <div className="w-3/4 h-5 bg-[var(--line)] rounded mb-2" />
                <div className="w-full h-3 bg-[var(--line)] rounded mb-1.5" />
                <div className="w-2/3 h-3 bg-[var(--line)] rounded mb-4" />
                <div className="w-full h-10 border-t border-b border-[var(--line)]/60 mb-3" />
              </div>
              <div className="pt-3 border-t border-[var(--line)] flex justify-between items-center">
                <div className="w-12 h-3 bg-[var(--line)] rounded" />
                <div className="w-28 h-8 bg-[var(--line)] rounded-lg" />
              </div>
            </div>
          ))}
        </div>
      ) : samplesQuery.isError ? (
        /* Error State */
        <div className="rounded-[14px] bg-[var(--black-950)] border border-[var(--line)] p-10 flex flex-col items-center justify-center text-center">
          <div className="w-10 h-10 rounded-xl bg-[var(--alert-bg)] border border-[var(--alert-500)]/30 flex items-center justify-center text-[var(--alert-400)] mb-3">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <h2 className="text-base font-semibold text-[var(--text-100)] mb-1">
            Failed to load benchmark library
          </h2>
          <p className="text-xs text-[var(--text-300)] max-w-md mb-4">
            The reference sample catalog could not be retrieved from the backend service. Check network or server readiness.
          </p>
          <button
            type="button"
            onClick={() => void samplesQuery.refetch()}
            className="h-8 px-3.5 rounded-lg bg-[var(--black-900)] hover:bg-[var(--black-700)] border border-[var(--line)] text-xs font-medium text-[var(--text-100)] inline-flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5 text-[var(--cobalt-300)]" />
            <span>Retry Request</span>
          </button>
        </div>
      ) : filteredSamples.length === 0 ? (
        /* Empty State */
        <div className="rounded-[14px] bg-[var(--black-950)] border border-[var(--line)] p-12 flex flex-col items-center justify-center text-center">
          <div className="w-10 h-10 rounded-xl bg-[var(--black-900)] border border-[var(--line)] flex items-center justify-center text-[var(--text-300)] mb-3">
            <FolderArchive className="w-5 h-5" />
          </div>
          <h2 className="text-base font-semibold text-[var(--text-100)] mb-1">
            No benchmark samples found
          </h2>
          <p className="text-xs text-[var(--text-300)] max-w-sm mb-4">
            {searchQuery
              ? `No benchmark reference media matched "${searchQuery}". Try a different search term or reset filters.`
              : 'No benchmark media items match the selected ground-truth classification filter.'}
          </p>
          <button
            type="button"
            onClick={() => {
              setFilter('all');
              setSearchQuery('');
            }}
            className="h-8 px-3.5 rounded-lg bg-[var(--black-900)] hover:bg-[var(--black-700)] border border-[var(--line)] text-xs font-medium text-[var(--text-100)] transition-colors cursor-pointer"
          >
            Reset Filters
          </button>
        </div>
      ) : (
        /* Samples Grid */
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-5">
          {filteredSamples.map((sample) => {
            const isAuthentic = sample.ground_truth === 'authentic';
            const isSynthetic = sample.ground_truth === 'synthetic';
            const isAnalyzing = analyzingId === sample.id;

            return (
              <article
                key={sample.id}
                className="rounded-[14px] bg-[var(--black-900)] border border-[var(--line)] hover:border-[var(--line-strong)] p-4 sm:p-5 flex flex-col justify-between transition-colors shadow-sm"
              >
                <div>
                  {/* Top Row: Media Type & Ground Truth Classification */}
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <div className="flex items-center gap-1.5 text-xs text-[var(--text-300)] font-medium">
                      {sample.media_type === 'video' ? (
                        <Film className="w-3.5 h-3.5 text-[var(--cobalt-300)]" />
                      ) : (
                        <Mic className="w-3.5 h-3.5 text-[var(--success-500)]" />
                      )}
                      <span className="capitalize">{sample.media_type}</span>
                    </div>

                    {/* Redesigned Ground Truth: neutral label + restrained badge */}
                    <div className="flex items-center gap-1.5 text-xs text-[var(--text-300)]">
                      <span>Ground truth</span>
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium ${
                          isAuthentic
                            ? 'bg-[var(--success-500)]/10 text-[var(--success-500)] border border-[var(--success-500)]/20'
                            : isSynthetic
                            ? 'bg-[var(--alert-500)]/10 text-[var(--alert-500)] border border-[var(--alert-500)]/20'
                            : 'bg-[var(--text-300)]/10 text-[var(--text-300)] border border-[var(--line)]'
                        }`}
                      >
                        {isAuthentic ? 'Authentic' : isSynthetic ? 'Synthetic' : sample.ground_truth}
                      </span>
                    </div>
                  </div>

                  {/* Sample Title */}
                  <h2 className="text-[15px] sm:text-base font-semibold text-[var(--text-100)] leading-snug tracking-tight mb-1.5 break-words">
                    {sample.title}
                  </h2>

                  {/* Sample Description */}
                  <p className="text-[13px] text-[var(--text-300)] leading-relaxed line-clamp-2 mb-3.5">
                    {sample.description}
                  </p>

                  {/* Subtle Metadata Strip (Without nested heavy boxes) */}
                  <div className="pt-2.5 pb-2.5 border-t border-b border-[var(--line)]/60 flex flex-wrap items-center justify-between gap-y-1.5 text-xs text-[var(--text-300)] mb-3">
                    <div className="flex items-center gap-2.5">
                      <div className="flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5 text-[var(--text-300)]" />
                        <span>{formatTimecode(sample.duration_s)}</span>
                      </div>
                      <span className="text-[var(--line)]">•</span>
                      <div className="flex items-center gap-1.5">
                        <HardDrive className="w-3.5 h-3.5 text-[var(--text-300)]" />
                        <span>{formatBytes(sample.file_size_bytes)}</span>
                      </div>
                    </div>
                    <div className="text-[12px] text-[var(--text-300)] truncate">
                      <span className="text-[var(--text-300)] mr-1">License:</span>
                      <span className="font-medium text-[var(--text-100)]">{sample.license}</span>
                    </div>
                  </div>

                  {/* Context Summary / Ground Truth Notes */}
                  {sample.context_summary && (
                    <div className="flex items-start gap-1.5 text-xs text-[var(--text-300)] leading-relaxed mb-4">
                      <Info className="w-3.5 h-3.5 text-[var(--cobalt-300)] shrink-0 mt-0.5" />
                      <span className="line-clamp-2">{sample.context_summary}</span>
                    </div>
                  )}
                </div>

                {/* Card Action Footer */}
                <div className="pt-3 border-t border-[var(--line)]/60 flex items-center justify-between gap-2 mt-auto">
                  <a
                    href={sample.page_url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-xs font-medium text-[var(--text-300)] hover:text-[var(--text-100)] transition-colors focus:outline-none focus-visible:underline"
                    title="View open access license and attribution"
                  >
                    <span>Source</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>

                  <button
                    type="button"
                    disabled={isAnalyzing}
                    onClick={() => handleAnalyzeSample(sample)}
                    className="h-[34px] px-3.5 rounded-lg bg-[var(--cobalt-500)] hover:bg-[var(--cobalt-400)] active:bg-[var(--cobalt-600)] text-[var(--text-100)] text-xs font-semibold inline-flex items-center gap-1.5 transition-colors shadow-xs disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--cobalt-300)]"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Staging...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-3.5 h-3.5 fill-current" />
                        <span>Analyze Sample</span>
                      </>
                    )}
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
};
