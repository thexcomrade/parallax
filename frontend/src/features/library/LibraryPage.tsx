import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { api } from '@/lib/api';
import { BentoCell } from '@/components/bento/BentoCell';
import { Button } from '@/components/ui/Button';
import { formatBytes, formatTimecode } from '@/lib/format';
import type { SampleAnalyzeResponse, SampleItem, SampleListResponse } from '@/types/api';

export const LibraryPage: React.FC = () => {
  const navigate = useNavigate();
  const [filter, setFilter] = useState<'all' | 'authentic' | 'synthetic'>('all');
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
      setErrorMsg(`FAILED TO STAGE SAMPLE "${sample.id}": ${message}`);
      setAnalyzingId(null);
    }
  };

  const samples = samplesQuery.data?.items ?? [];
  const filteredSamples = samples.filter((s) => {
    if (filter === 'all') return true;
    return s.ground_truth === filter;
  });

  return (
    <div className="max-w-7xl mx-auto flex flex-col gap-6 py-4">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-4 border-b border-[var(--line)]">
        <div>
          <div className="font-mono text-xs uppercase text-[var(--cobalt-400)] tracking-widest flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[var(--cobalt-400)]" />
            CHANNEL // CH-04 // REFERENCE CORPUS
          </div>
          <h1 className="font-display font-black text-2xl md:text-3xl uppercase tracking-tight text-white mt-1">
            Forensic Benchmark Library
          </h1>
          <p className="font-sans text-xs md:text-sm text-[var(--text-300)] mt-1">
            Curated public-domain and CC-licensed media with verified ground-truth labels for algorithm evaluation.
          </p>
        </div>

        {/* Filter Tabs */}
        <div className="flex items-center gap-2 p-1 bg-[var(--black-900)] border border-[var(--line)] rounded-[var(--radius-sm)]">
          <button
            type="button"
            onClick={() => setFilter('all')}
            className={`px-3 py-1 text-xs font-mono uppercase tracking-wider rounded-[var(--radius-sm)] transition-colors ${
              filter === 'all'
                ? 'bg-[var(--cobalt-500)] text-white font-bold'
                : 'text-[var(--text-400)] hover:text-white'
            }`}
          >
            ALL [{samples.length}]
          </button>
          <button
            type="button"
            onClick={() => setFilter('authentic')}
            className={`px-3 py-1 text-xs font-mono uppercase tracking-wider rounded-[var(--radius-sm)] transition-colors ${
              filter === 'authentic'
                ? 'bg-[var(--lime-400)] text-[var(--lime-ink)] font-bold'
                : 'text-[var(--text-400)] hover:text-white'
            }`}
          >
            AUTHENTIC
          </button>
          <button
            type="button"
            onClick={() => setFilter('synthetic')}
            className={`px-3 py-1 text-xs font-mono uppercase tracking-wider rounded-[var(--radius-sm)] transition-colors ${
              filter === 'synthetic'
                ? 'bg-[var(--alert-500)] text-[var(--black-950)] font-black'
                : 'text-[var(--text-400)] hover:text-white'
            }`}
          >
            SYNTHETIC
          </button>
        </div>
      </header>

      {/* Provenance and Security Notice */}
      <aside
        className="p-3 bg-[var(--black-900)] border border-[var(--line)] rounded-[var(--radius-md)] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-xs font-mono text-[var(--text-400)]"
        aria-label="Security Notice"
      >
        <span className="flex items-center gap-2">
          <span className="text-[var(--lime-400)] font-bold">SSRF_PROTECTED //</span>
          Downloads are proxied through an allowlisted server-side sandbox. Zero direct browser third-party requests.
        </span>
        <span className="text-[var(--cobalt-400)] font-bold uppercase">
          STORAGE CAPACITY // {samples.length} SAMPLES CACHED
        </span>
      </aside>

      {/* Error alert if any */}
      {errorMsg && (
        <div
          role="alert"
          className="p-4 bg-[var(--black-900)] border border-[var(--alert-500)] text-[var(--alert-500)] rounded-[var(--radius-md)] font-mono text-xs flex items-center justify-between"
        >
          <span>{errorMsg}</span>
          <button
            type="button"
            onClick={() => setErrorMsg(null)}
            className="text-[var(--text-400)] hover:text-white font-mono uppercase"
          >
            DISMISS [×]
          </button>
        </div>
      )}

      {/* Loading state */}
      {samplesQuery.isLoading && (
        <div className="py-16 text-center font-mono text-sm text-[var(--text-400)]">
          INDEXING BENCHMARK SAMPLES FROM CORPUS REPOSITORY...
        </div>
      )}

      {/* Sample Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {filteredSamples.map((sample) => {
          const isAnalyzing = analyzingId === sample.id;
          const isFake = sample.ground_truth === 'synthetic';

          return (
            <BentoCell
              key={sample.id}
              label={`REF // ${sample.id.toUpperCase()}`}
              status={`${sample.media_type.toUpperCase()} // ${formatTimecode(sample.duration_s)}`}
              hazard={isFake}
              className="flex flex-col justify-between min-h-[380px]"
            >
              <div className="p-4 flex flex-col gap-3">
                {/* Ground Truth Badge */}
                <div className="flex items-center justify-between">
                  <span
                    className={`px-2 py-0.5 text-[10px] font-mono font-bold tracking-wider uppercase rounded-xs ${
                      isFake
                        ? 'bg-[var(--alert-500)] text-[var(--black-950)] font-black'
                        : 'bg-[var(--lime-400)] text-[var(--lime-ink)]'
                    }`}
                  >
                    GROUND TRUTH: {sample.ground_truth}
                  </span>
                  <span className="text-[10px] font-mono text-[var(--text-400)]">
                    {formatBytes(sample.file_size_bytes)}
                  </span>
                </div>

                {/* Title */}
                <h2 className="font-sans font-bold text-sm text-white leading-snug">
                  {sample.title}
                </h2>

                {/* Description */}
                <p className="font-sans text-xs text-[var(--text-300)] leading-relaxed">
                  {sample.description}
                </p>

                {/* Context card */}
                {sample.context_summary && (
                  <div className="p-2.5 bg-[var(--black-950)] border border-[var(--line)] rounded-[var(--radius-sm)] text-[11px] font-sans text-[var(--text-400)] leading-normal">
                    <span className="text-[var(--cobalt-300)] font-mono text-[10px] uppercase block mb-1">
                      CONTEXT INTEL:
                    </span>
                    {sample.context_summary}
                  </div>
                )}
              </div>

              {/* Card Footer: Metadata and CTA */}
              <div className="p-4 pt-3 border-t border-[var(--line)] bg-[var(--black-950)]/50 flex flex-col gap-3">
                <div className="flex flex-col gap-1 text-[10px] font-mono text-[var(--text-400)]">
                  <div className="flex items-center justify-between">
                    <span>LICENSE: {sample.license}</span>
                    <a
                      href={sample.page_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[var(--cobalt-400)] hover:text-[var(--cobalt-300)] underline"
                    >
                      {sample.source} ↗
                    </a>
                  </div>
                  <div className="truncate text-[var(--text-400)]">
                    ATTR: {sample.attribution}
                  </div>
                </div>

                <Button
                  variant="primary"
                  size="sm"
                  className="w-full"
                  onClick={() => handleAnalyzeSample(sample)}
                  loading={isAnalyzing}
                  disabled={analyzingId !== null}
                  aria-label={`Analyze this sample: ${sample.title}`}
                >
                  {isAnalyzing ? 'DOWNLOADING & STAGING...' : 'ANALYZE THIS SAMPLE'}
                </Button>
              </div>
            </BentoCell>
          );
        })}
      </div>
    </div>
  );
};
