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
    <div className="flex flex-col gap-6 w-full max-w-7xl mx-auto py-1 sm:py-2 text-[#E8EDF5]">
      {/* ── 1. Page Header (Matches Dashboard & History) ──────────────────── */}
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[#253247]/60">
        <div>
          <span className="text-[11px] font-semibold tracking-wider text-[#648CFF] uppercase block">
            MEDIA WORKSPACE
          </span>
          <div className="flex items-center gap-3 mt-0.5">
            <h1 className="text-2xl sm:text-[28px] font-semibold text-[#E8EDF5] tracking-tight">
              Library <span className="sr-only">Forensic Benchmark Library</span>
            </h1>
            {samples.length > 0 && (
              <span className="px-2.5 py-0.5 rounded-full bg-[#171F2D] border border-[#253247] text-xs font-medium text-[#91A0B8]">
                {samples.length} Curated Media
              </span>
            )}
          </div>
          <p className="text-[13px] sm:text-sm text-[#91A0B8] mt-1 leading-normal max-w-2xl">
            Browse curated reference media and ground-truth benchmark samples for calibration and baseline evaluation.
          </p>
        </div>

        {/* Header Action: New Analysis */}
        <div className="flex items-center gap-2.5 self-start sm:self-auto shrink-0">
          <Link
            to="/app/analyze"
            className="h-[38px] px-4 rounded-lg bg-[#3B66EB] hover:bg-[#4E77F3] active:bg-[#3258D6] text-white text-[13px] font-semibold inline-flex items-center gap-2 shadow-xs transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
          >
            <PlusCircle className="w-4 h-4" />
            <span>New Analysis</span>
          </Link>
        </div>
      </header>

      {/* ── 2. Security & Provenance Banner ───────────────────────────────── */}
      <aside
        aria-label="Security and provenance notice"
        className="p-3 sm:px-4 rounded-xl bg-[#111927] border border-[#253247] flex items-start sm:items-center gap-3 text-xs text-[#91A0B8]"
      >
        <ShieldCheck className="w-4 h-4 text-[#648CFF] shrink-0 mt-0.5 sm:mt-0" />
        <p className="leading-relaxed">
          <span className="font-semibold text-[#E8EDF5] mr-1.5">Restricted backend ingestion:</span>
          Benchmark media files are downloaded and evaluated strictly within backend isolated worker sandboxes with zero client cross-origin requests.
        </p>
      </aside>

      {/* ── 3. Search and Filter Controls Toolbar ─────────────────────────── */}
      <section aria-label="Library Controls" className="w-full">
        <div className="w-full p-2.5 sm:p-3 rounded-xl bg-[#111927] border border-[#253247] flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          {/* Search Input */}
          <div className="relative flex-1 max-w-md flex items-center">
            <Search className="w-4 h-4 text-[#91A0B8] absolute left-3 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search benchmark media by title or keyword..."
              aria-label="Search benchmark media"
              className="w-full h-[38px] pl-9 pr-8 bg-[#090D14] border border-[#253247] hover:border-[#374866] focus:border-[#648CFF] focus:outline-none focus:ring-1 focus:ring-[#648CFF] rounded-lg text-[13px] text-[#E8EDF5] placeholder-[#91A0B8]/60 transition-colors"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                aria-label="Clear search"
                className="absolute right-2.5 p-1 text-[#91A0B8] hover:text-[#E8EDF5] transition-colors cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Segmented Filter Control */}
          <div
            aria-label="Filter by ground truth classification"
            className="flex items-center gap-1 p-1 rounded-lg bg-[#090D14] border border-[#253247] shrink-0 text-xs overflow-x-auto"
          >
            <button
              type="button"
              onClick={() => setFilter('all')}
              className={`h-[30px] px-3.5 rounded-md text-xs font-medium transition-colors cursor-pointer shrink-0 ${
                filter === 'all'
                  ? 'bg-[#1C2737] text-white border border-[#253247] shadow-xs'
                  : 'text-[#91A0B8] hover:text-[#E8EDF5]'
              }`}
            >
              All [{samples.length}]
            </button>
            <button
              type="button"
              onClick={() => setFilter('authentic')}
              className={`h-[30px] px-3.5 rounded-md text-xs font-medium transition-colors cursor-pointer shrink-0 ${
                filter === 'authentic'
                  ? 'bg-[#1C2737] text-white border border-[#253247] shadow-xs'
                  : 'text-[#91A0B8] hover:text-[#E8EDF5]'
              }`}
            >
              Authentic [{authenticCount}]
            </button>
            <button
              type="button"
              onClick={() => setFilter('synthetic')}
              className={`h-[30px] px-3.5 rounded-md text-xs font-medium transition-colors cursor-pointer shrink-0 ${
                filter === 'synthetic'
                  ? 'bg-[#1C2737] text-white border border-[#253247] shadow-xs'
                  : 'text-[#91A0B8] hover:text-[#E8EDF5]'
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
          className="p-3 rounded-xl bg-red-950/40 border border-red-500/30 text-xs text-red-300 flex items-center justify-between gap-2.5"
        >
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
            <span>{errorMsg}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorMsg(null)}
            aria-label="Dismiss error"
            className="text-[#91A0B8] hover:text-[#E8EDF5] font-bold text-xs p-1 cursor-pointer"
          >
            ✕
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
              className="rounded-[14px] bg-[#171F2D] border border-[#253247] p-5 flex flex-col justify-between min-h-[260px]"
            >
              <div>
                <div className="flex justify-between items-center mb-3">
                  <div className="w-16 h-4 bg-[#253247] rounded" />
                  <div className="w-28 h-4 bg-[#253247] rounded-full" />
                </div>
                <div className="w-3/4 h-5 bg-[#253247] rounded mb-2" />
                <div className="w-full h-3 bg-[#253247] rounded mb-1.5" />
                <div className="w-2/3 h-3 bg-[#253247] rounded mb-4" />
                <div className="w-full h-10 border-t border-b border-[#253247]/60 mb-3" />
              </div>
              <div className="pt-3 border-t border-[#253247] flex justify-between items-center">
                <div className="w-12 h-3 bg-[#253247] rounded" />
                <div className="w-28 h-8 bg-[#253247] rounded-lg" />
              </div>
            </div>
          ))}
        </div>
      ) : samplesQuery.isError ? (
        /* Error State */
        <div className="rounded-[14px] bg-[#111927] border border-[#253247] p-10 flex flex-col items-center justify-center text-center">
          <div className="w-10 h-10 rounded-xl bg-red-950/40 border border-red-500/30 flex items-center justify-center text-red-400 mb-3">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <h2 className="text-base font-semibold text-[#E8EDF5] mb-1">
            Failed to load benchmark library
          </h2>
          <p className="text-xs text-[#91A0B8] max-w-md mb-4">
            The reference sample catalog could not be retrieved from the backend service. Check network or server readiness.
          </p>
          <button
            type="button"
            onClick={() => void samplesQuery.refetch()}
            className="h-8 px-3.5 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-xs font-medium text-[#E8EDF5] inline-flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5 text-[#648CFF]" />
            <span>Retry Request</span>
          </button>
        </div>
      ) : filteredSamples.length === 0 ? (
        /* Empty State */
        <div className="rounded-[14px] bg-[#111927] border border-[#253247] p-12 flex flex-col items-center justify-center text-center">
          <div className="w-10 h-10 rounded-xl bg-[#171F2D] border border-[#253247] flex items-center justify-center text-[#91A0B8] mb-3">
            <FolderArchive className="w-5 h-5" />
          </div>
          <h2 className="text-base font-semibold text-[#E8EDF5] mb-1">
            No benchmark samples found
          </h2>
          <p className="text-xs text-[#91A0B8] max-w-sm mb-4">
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
            className="h-8 px-3.5 rounded-lg bg-[#171F2D] hover:bg-[#1C2737] border border-[#253247] text-xs font-medium text-[#E8EDF5] transition-colors cursor-pointer"
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
                className="rounded-[14px] bg-[#171F2D] border border-[#253247] hover:border-[#374866] p-4 sm:p-5 flex flex-col justify-between transition-colors shadow-sm"
              >
                <div>
                  {/* Top Row: Media Type & Ground Truth Classification */}
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <div className="flex items-center gap-1.5 text-xs text-[#91A0B8] font-medium">
                      {sample.media_type === 'video' ? (
                        <Film className="w-3.5 h-3.5 text-[#648CFF]" />
                      ) : (
                        <Mic className="w-3.5 h-3.5 text-[#35C99A]" />
                      )}
                      <span className="capitalize">{sample.media_type}</span>
                    </div>

                    {/* Redesigned Ground Truth: neutral label + restrained badge */}
                    <div className="flex items-center gap-1.5 text-xs text-[#91A0B8]">
                      <span>Ground truth</span>
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium ${
                          isAuthentic
                            ? 'bg-[#35C99A]/10 text-[#35C99A] border border-[#35C99A]/20'
                            : isSynthetic
                            ? 'bg-[#F16F78]/10 text-[#F16F78] border border-[#F16F78]/20'
                            : 'bg-[#91A0B8]/10 text-[#91A0B8] border border-[#253247]'
                        }`}
                      >
                        {isAuthentic ? 'Authentic' : isSynthetic ? 'Synthetic' : sample.ground_truth}
                      </span>
                    </div>
                  </div>

                  {/* Sample Title */}
                  <h2 className="text-[15px] sm:text-base font-semibold text-[#E8EDF5] leading-snug tracking-tight mb-1.5 break-words">
                    {sample.title}
                  </h2>

                  {/* Sample Description */}
                  <p className="text-[13px] text-[#91A0B8] leading-relaxed line-clamp-2 mb-3.5">
                    {sample.description}
                  </p>

                  {/* Subtle Metadata Strip (Without nested heavy boxes) */}
                  <div className="pt-2.5 pb-2.5 border-t border-b border-[#253247]/60 flex flex-wrap items-center justify-between gap-y-1.5 text-xs text-[#91A0B8] mb-3">
                    <div className="flex items-center gap-2.5">
                      <div className="flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5 text-[#91A0B8]" />
                        <span>{formatTimecode(sample.duration_s)}</span>
                      </div>
                      <span className="text-[#253247]">•</span>
                      <div className="flex items-center gap-1.5">
                        <HardDrive className="w-3.5 h-3.5 text-[#91A0B8]" />
                        <span>{formatBytes(sample.file_size_bytes)}</span>
                      </div>
                    </div>
                    <div className="text-[12px] text-[#91A0B8] truncate">
                      <span className="text-[#91A0B8] mr-1">License:</span>
                      <span className="font-medium text-[#E8EDF5]">{sample.license}</span>
                    </div>
                  </div>

                  {/* Context Summary / Ground Truth Notes */}
                  {sample.context_summary && (
                    <div className="flex items-start gap-1.5 text-xs text-[#91A0B8] leading-relaxed mb-4">
                      <Info className="w-3.5 h-3.5 text-[#648CFF] shrink-0 mt-0.5" />
                      <span className="line-clamp-2">{sample.context_summary}</span>
                    </div>
                  )}
                </div>

                {/* Card Action Footer */}
                <div className="pt-3 border-t border-[#253247]/60 flex items-center justify-between gap-2 mt-auto">
                  <a
                    href={sample.page_url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-xs font-medium text-[#91A0B8] hover:text-[#E8EDF5] transition-colors focus:outline-none focus-visible:underline"
                    title="View open access license and attribution"
                  >
                    <span>Source</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>

                  <button
                    type="button"
                    disabled={isAnalyzing}
                    onClick={() => handleAnalyzeSample(sample)}
                    className="h-[34px] px-3.5 rounded-lg bg-[#3B66EB] hover:bg-[#4E77F3] active:bg-[#3258D6] text-white text-xs font-semibold inline-flex items-center gap-1.5 transition-colors shadow-xs disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-[#648CFF]"
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
