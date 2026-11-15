import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { BentoCell } from '@/components/bento/BentoCell';
import { Button } from '@/components/ui/Button';
import { formatRelativeTime } from '@/lib/format';
import type {
  PulseFeedResponse,
  ResearchFeedResponse,
  ThreatFeedResponse,
} from '@/types/api';

export const IntelPage: React.FC = () => {
  const [refreshing, setRefreshing] = useState(false);

  // Queries for the three feeds
  const threatsQuery = useQuery({
    queryKey: ['feeds', 'threats'],
    queryFn: () => api.get<ThreatFeedResponse>('/feeds/threats'),
    staleTime: 60_000,
  });

  const researchQuery = useQuery({
    queryKey: ['feeds', 'research'],
    queryFn: () => api.get<ResearchFeedResponse>('/feeds/research'),
    staleTime: 60_000,
  });

  const pulseQuery = useQuery({
    queryKey: ['feeds', 'pulse'],
    queryFn: () => api.get<PulseFeedResponse>('/feeds/pulse'),
    staleTime: 60_000,
  });

  const handleRefreshAll = async () => {
    setRefreshing(true);
    try {
      await Promise.all([
        threatsQuery.refetch(),
        researchQuery.refetch(),
        pulseQuery.refetch(),
      ]);
    } finally {
      setRefreshing(false);
    }
  };

  const isLoading = threatsQuery.isLoading || researchQuery.isLoading || pulseQuery.isLoading;

  return (
    <div className="max-w-7xl mx-auto flex flex-col gap-6 py-4">
      {/* Page Header */}
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-4 border-b border-[var(--line)]">
        <div>
          <div className="font-mono text-xs uppercase text-[var(--cobalt-400)] tracking-widest flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[var(--lime-400)] animate-pulse" />
            CHANNEL // CH-05 // THREAT & RESEARCH RECON
          </div>
          <h1 className="font-display font-black text-2xl md:text-3xl uppercase tracking-tight text-white mt-1">
            Forensics Intelligence Radar
          </h1>
          <p className="font-sans text-xs md:text-sm text-[var(--text-300)] mt-1">
            Multi-source synthetic threat tracking, peer-reviewed arXiv literature, and developer signal.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="secondary"
            size="sm"
            onClick={handleRefreshAll}
            loading={refreshing || isLoading}
          >
            REFRESH TELEMETRY
          </Button>
        </div>
      </header>

      {/* Security provenance disclaimer */}
      <aside
        className="p-3 bg-[var(--black-900)] border border-[var(--line)] rounded-[var(--radius-md)] flex items-center justify-between text-xs font-mono text-[var(--text-400)]"
        aria-label="Provenance notice"
      >
        <span className="flex items-center gap-2">
          <span className="text-[var(--cobalt-400)] font-bold">PROVENANCE //</span>
          All feeds proxied server-side via restricted SSL backend. Zero browser third-party requests.
        </span>
        <span className="hidden sm:inline text-[var(--lime-400)]">SEC_CHECK // PASS</span>
      </aside>

      {/* Bento Grid layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Panel 1: Threat Pulse (GDELT) */}
        <BentoCell
          label="THREAT-PULSE // GDELT DOC 2.0"
          status={
            threatsQuery.data?.cached_at
              ? `SYNCED ${formatRelativeTime(threatsQuery.data.cached_at)}`
              : 'POLLING'
          }
          className="flex flex-col min-h-[500px]"
        >
          <div className="flex flex-col gap-3 p-4 flex-1 overflow-y-auto max-h-[600px]">
            {threatsQuery.isLoading ? (
              <div className="flex flex-col gap-2 py-8 text-center font-mono text-xs text-[var(--text-400)]">
                SCANNING GLOBAL MEDIA FEEDS...
              </div>
            ) : threatsQuery.data?.items.length ? (
              threatsQuery.data.items.map((item, idx) => (
                <article
                  key={`${item.url}-${idx}`}
                  className="p-3 bg-[var(--black-950)] border border-[var(--line)] rounded-[var(--radius-sm)] flex flex-col gap-2 hover:border-[var(--line-strong)] transition-colors"
                >
                  <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
                    <span className="text-[var(--cobalt-300)] uppercase font-semibold">
                      {item.domain || 'NEWS_SOURCE'}
                    </span>
                    {item.sourcecountry && (
                      <span className="text-[var(--text-500)] uppercase">
                        [{item.sourcecountry}]
                      </span>
                    )}
                  </div>
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-xs text-white hover:text-[var(--lime-400)] font-medium line-clamp-2 leading-relaxed"
                  >
                    {item.title}
                  </a>
                  <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-500)] pt-1 border-t border-[var(--line)]">
                    <span>{item.language || 'English'}</span>
                    <span className="text-[var(--cobalt-400)]">SOURCE ↗</span>
                  </div>
                </article>
              ))
            ) : (
              <div className="py-8 text-center font-mono text-xs text-[var(--text-400)]">
                NO ACTIVE THREAT DISPATCHES RECORDED
              </div>
            )}
          </div>
        </BentoCell>

        {/* Panel 2: Research Radar (arXiv) */}
        <BentoCell
          label="RESEARCH-RADAR // arXiv ATOM"
          status={
            researchQuery.data?.cached_at
              ? `SYNCED ${formatRelativeTime(researchQuery.data.cached_at)}`
              : 'POLLING'
          }
          className="flex flex-col min-h-[500px]"
        >
          <div className="flex flex-col gap-3 p-4 flex-1 overflow-y-auto max-h-[600px]">
            {researchQuery.isLoading ? (
              <div className="flex flex-col gap-2 py-8 text-center font-mono text-xs text-[var(--text-400)]">
                INGESTING LITERATURE PREPRINTS...
              </div>
            ) : researchQuery.data?.items.length ? (
              researchQuery.data.items.map((paper, idx) => (
                <article
                  key={`${paper.arxiv_id}-${idx}`}
                  className="p-3 bg-[var(--black-950)] border border-[var(--line)] rounded-[var(--radius-sm)] flex flex-col gap-2 hover:border-[var(--line-strong)] transition-colors"
                >
                  <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
                    <span className="px-1.5 py-0.5 bg-[var(--black-900)] border border-[var(--line)] text-[var(--lime-400)] font-semibold">
                      arXiv:{paper.arxiv_id}
                    </span>
                    <span className="text-[var(--text-500)]">
                      {paper.published_at ? formatRelativeTime(paper.published_at) : 'RECENT'}
                    </span>
                  </div>
                  <a
                    href={paper.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-xs text-white hover:text-[var(--lime-400)] font-medium leading-relaxed"
                  >
                    {paper.title}
                  </a>
                  {paper.authors.length > 0 && (
                    <div className="text-[10px] font-mono text-[var(--text-400)] truncate">
                      BY {paper.authors.join(', ')}
                    </div>
                  )}
                  <p className="text-[11px] font-sans text-[var(--text-300)] line-clamp-3 leading-snug">
                    {paper.summary}
                  </p>
                  <div className="flex items-center justify-end text-[10px] font-mono text-[var(--cobalt-400)] pt-1 border-t border-[var(--line)]">
                    <span>PDF / ABSTRACT ↗</span>
                  </div>
                </article>
              ))
            ) : (
              <div className="py-8 text-center font-mono text-xs text-[var(--text-400)]">
                NO RESEARCH PREPRINTS AVAILABLE
              </div>
            )}
          </div>
        </BentoCell>

        {/* Panel 3: Technical Pulse (Hacker News) */}
        <BentoCell
          label="COMMUNITY-PULSE // HN ALGOLIA"
          status={
            pulseQuery.data?.cached_at
              ? `SYNCED ${formatRelativeTime(pulseQuery.data.cached_at)}`
              : 'POLLING'
          }
          className="flex flex-col min-h-[500px]"
        >
          <div className="flex flex-col gap-3 p-4 flex-1 overflow-y-auto max-h-[600px]">
            {pulseQuery.isLoading ? (
              <div className="flex flex-col gap-2 py-8 text-center font-mono text-xs text-[var(--text-400)]">
                QUERYING TECHNICAL DISCUSSIONS...
              </div>
            ) : pulseQuery.data?.items.length ? (
              pulseQuery.data.items.map((story) => (
                <article
                  key={story.object_id}
                  className="p-3 bg-[var(--black-950)] border border-[var(--line)] rounded-[var(--radius-sm)] flex flex-col gap-2 hover:border-[var(--line-strong)] transition-colors"
                >
                  <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
                    <span className="text-[var(--lime-400)] font-bold">
                      ▲ {story.points} PTS
                    </span>
                    <span className="text-[var(--text-500)]">
                      {story.num_comments} COMMENTS
                    </span>
                  </div>
                  <a
                    href={story.url || `https://news.ycombinator.com/item?id=${story.object_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-xs text-white hover:text-[var(--lime-400)] font-medium line-clamp-2 leading-relaxed"
                  >
                    {story.title}
                  </a>
                  <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-500)] pt-1 border-t border-[var(--line)]">
                    <span>AUTHOR: {story.author}</span>
                    <span className="text-[var(--cobalt-400)]">THREAD ↗</span>
                  </div>
                </article>
              ))
            ) : (
              <div className="py-8 text-center font-mono text-xs text-[var(--text-400)]">
                NO DISCUSSION SIGNALS DETECTED
              </div>
            )}
          </div>
        </BentoCell>
      </div>
    </div>
  );
};
