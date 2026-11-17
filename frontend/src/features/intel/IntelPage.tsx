/**
 * IntelPage — Forensic Threat Reconnaissance & Academic Literature Radar.
 *
 * Implements Phase 8 of PARALLAX_FRONTEND_REDESIGN.md:
 * - Multi-source threat tracking (GDELT, arXiv, Hacker News)
 * - Strict server-side proxy isolation (zero browser third-party requests)
 * - Clear distinction between external research and PARALLAX model outputs
 * - Explicit cached vs live indicators (no fake live statuses)
 * - Comprehensive Loading, Empty, and Error states with retry
 */

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { PageHeader } from '@/components/layout/PageHeader';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { ErrorState } from '@/components/ui/ErrorState';
import { api } from '@/lib/api';
import { formatRelativeTime } from '@/lib/format';
import type {
  PulseFeedResponse,
  ResearchFeedResponse,
  ThreatFeedResponse,
} from '@/types/api';
import {
  RefreshCw,
  ShieldCheck,
  Globe,
  BookOpen,
  Radio,
  ExternalLink,
  Info,
  Clock,
  ArrowUpRight,
  MessageSquare,
} from 'lucide-react';

export const IntelPage: React.FC = () => {
  const [refreshing, setRefreshing] = useState(false);

  // Queries for the three feeds proxied server-side
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
    <div className="flex flex-col gap-6 max-w-7xl mx-auto py-2">
      {/* 1. Page Header */}
      <PageHeader
        title="Forensic Threat & Research Recon"
        description="Multi-source synthetic threat tracking, peer-reviewed arXiv preprints, and developer community pulse."
        actions={
          <Button
            variant="secondary"
            size="sm"
            onClick={handleRefreshAll}
            loading={refreshing || isLoading}
            icon={<RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />}
          >
            REFRESH RECON
          </Button>
        }
      />

      {/* 2. Security Provenance & External Disclaimer Strip */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
        {/* Network isolation banner */}
        <aside
          aria-label="Provenance notice"
          className="p-3 bg-[var(--surface-card)] border border-[var(--border-subtle)] rounded-[var(--radius-md)] flex items-center justify-between text-[var(--text-300)]"
        >
          <div className="flex items-center gap-2">
            <ShieldCheck size={16} className="text-[var(--cobalt-400)] shrink-0" />
            <span>
              <strong className="text-[var(--text-100)]">PROVENANCE:</strong> Proxied server-side via restricted SSL. Zero browser cross-origin leaks.
            </span>
          </div>
          <span className="hidden sm:inline-block px-1.5 py-0.5 rounded bg-[var(--surface-elevated)] text-[var(--lime-400)] text-[10px] uppercase font-bold shrink-0">
            ISOLATED
          </span>
        </aside>

        {/* Analytical boundary disclaimer */}
        <div className="p-3 bg-[var(--surface-card)] border border-[var(--border-subtle)] rounded-[var(--radius-md)] flex items-center justify-between text-[var(--text-300)]">
          <div className="flex items-center gap-2">
            <Info size={16} className="text-[var(--accent-primary)] shrink-0" />
            <span>
              <strong className="text-[var(--text-100)]">EXTERNAL DATA:</strong> Open intelligence signals are informational and distinct from PARALLAX model verdicts.
            </span>
          </div>
          <span className="hidden sm:inline-block px-1.5 py-0.5 rounded bg-[var(--surface-elevated)] text-[var(--text-400)] text-[10px] uppercase shrink-0">
            RESEARCH
          </span>
        </div>
      </div>

      {/* 3. Three-Column Recon Workbench */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* COLUMN 1: Global Media Threats (GDELT DOC 2.0) */}
        <Card variant="default" padding="none" className="flex flex-col h-[700px]">
          <div className="p-4 border-b border-[var(--border-subtle)] bg-[var(--surface-card)]">
            <CardHeader className="mb-0">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-full bg-[var(--surface-elevated)] border border-[var(--border-default)] flex items-center justify-center text-[var(--cobalt-400)]">
                  <Globe size={16} />
                </div>
                <div>
                  <CardTitle className="text-sm font-bold uppercase tracking-wider text-[var(--text-100)]">
                    Media Threats
                  </CardTitle>
                  <CardDescription className="text-[11px] font-mono text-[var(--text-400)]">
                    GDELT DOC 2.0 // GLOBAL BROADCAST
                  </CardDescription>
                </div>
              </div>

              {threatsQuery.data?.cached_at ? (
                <Badge variant="neutral" size="sm" className="font-mono text-[10px]">
                  CACHED {formatRelativeTime(threatsQuery.data.cached_at).toUpperCase()}
                </Badge>
              ) : (
                <Badge variant="neutral" size="sm" className="font-mono text-[10px]">
                  {threatsQuery.isLoading ? 'POLLING...' : 'STANDBY'}
                </Badge>
              )}
            </CardHeader>
          </div>

          <CardContent className="p-4 overflow-y-auto flex-1 flex flex-col gap-3">
            {threatsQuery.isLoading ? (
              <div className="flex flex-col gap-3 py-4">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="p-3 bg-[var(--surface-elevated)]/40 rounded border border-[var(--border-subtle)] animate-pulse flex flex-col gap-2">
                    <div className="h-3 bg-[var(--surface-elevated)] rounded w-1/3" />
                    <div className="h-4 bg-[var(--surface-elevated)] rounded w-5/6" />
                    <div className="h-3 bg-[var(--surface-elevated)] rounded w-1/2" />
                  </div>
                ))}
              </div>
            ) : threatsQuery.isError ? (
              <ErrorState
                title="Feed Unavailable"
                message="Unable to ingest GDELT threat stream. Verify backend upstream network status."
                onRetry={() => threatsQuery.refetch()}
              />
            ) : threatsQuery.data?.items && threatsQuery.data.items.length > 0 ? (
              threatsQuery.data.items.map((item, idx) => (
                <article
                  key={`${item.url}-${idx}`}
                  className="p-3.5 bg-[var(--surface-elevated)]/40 hover:bg-[var(--surface-elevated)] border border-[var(--border-subtle)] hover:border-[var(--border-strong)] rounded-[var(--radius-sm)] flex flex-col gap-2.5 transition-colors group"
                >
                  <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
                    <span className="text-[var(--cobalt-400)] font-bold uppercase truncate max-w-[160px]">
                      {item.domain || 'NEWS AGGREGATOR'}
                    </span>
                    {item.sourcecountry && (
                      <span className="text-[var(--text-400)] uppercase px-1 py-0.5 rounded bg-[var(--surface-card)] border border-[var(--border-subtle)]">
                        {item.sourcecountry}
                      </span>
                    )}
                  </div>

                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-xs text-[var(--text-100)] group-hover:text-[var(--accent-primary)] font-semibold line-clamp-2 leading-relaxed transition-colors flex items-start justify-between gap-2"
                  >
                    <span>{item.title}</span>
                    <ArrowUpRight size={13} className="shrink-0 opacity-40 group-hover:opacity-100 transition-opacity mt-0.5" />
                  </a>

                  <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-400)] pt-1.5 border-t border-[var(--border-subtle)]">
                    <span>{item.language || 'English'}</span>
                    <span className="text-[var(--text-400)] group-hover:text-[var(--accent-primary)] flex items-center gap-1 transition-colors">
                      VERIFY SOURCE
                    </span>
                  </div>
                </article>
              ))
            ) : (
              <EmptyState
                icon={<Globe size={24} />}
                title="No Threat Dispatches"
                description="Zero deepfake or synthetic identity incident telemetry currently reported by GDELT."
              />
            )}
          </CardContent>
        </Card>

        {/* COLUMN 2: Academic Literature (arXiv ATOM) */}
        <Card variant="default" padding="none" className="flex flex-col h-[700px]">
          <div className="p-4 border-b border-[var(--border-subtle)] bg-[var(--surface-card)]">
            <CardHeader className="mb-0">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-full bg-[var(--surface-elevated)] border border-[var(--border-default)] flex items-center justify-center text-[var(--accent-primary)]">
                  <BookOpen size={16} />
                </div>
                <div>
                  <CardTitle className="text-sm font-bold uppercase tracking-wider text-[var(--text-100)]">
                    Academic Literature
                  </CardTitle>
                  <CardDescription className="text-[11px] font-mono text-[var(--text-400)]">
                    arXiv ATOM // PREPRINT RADAR
                  </CardDescription>
                </div>
              </div>

              {researchQuery.data?.cached_at ? (
                <Badge variant="neutral" size="sm" className="font-mono text-[10px]">
                  CACHED {formatRelativeTime(researchQuery.data.cached_at).toUpperCase()}
                </Badge>
              ) : (
                <Badge variant="neutral" size="sm" className="font-mono text-[10px]">
                  {researchQuery.isLoading ? 'POLLING...' : 'STANDBY'}
                </Badge>
              )}
            </CardHeader>
          </div>

          <CardContent className="p-4 overflow-y-auto flex-1 flex flex-col gap-3">
            {researchQuery.isLoading ? (
              <div className="flex flex-col gap-3 py-4">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="p-3 bg-[var(--surface-elevated)]/40 rounded border border-[var(--border-subtle)] animate-pulse flex flex-col gap-2">
                    <div className="h-3 bg-[var(--surface-elevated)] rounded w-1/4" />
                    <div className="h-4 bg-[var(--surface-elevated)] rounded w-4/5" />
                    <div className="h-10 bg-[var(--surface-elevated)] rounded w-full" />
                  </div>
                ))}
              </div>
            ) : researchQuery.isError ? (
              <ErrorState
                title="Feed Unavailable"
                message="Unable to ingest arXiv ATOM stream. Upstream API may be rate-limiting."
                onRetry={() => researchQuery.refetch()}
              />
            ) : researchQuery.data?.items && researchQuery.data.items.length > 0 ? (
              researchQuery.data.items.map((paper, idx) => (
                <article
                  key={`${paper.arxiv_id}-${idx}`}
                  className="p-3.5 bg-[var(--surface-elevated)]/40 hover:bg-[var(--surface-elevated)] border border-[var(--border-subtle)] hover:border-[var(--border-strong)] rounded-[var(--radius-sm)] flex flex-col gap-2.5 transition-colors group"
                >
                  <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
                    <span className="px-1.5 py-0.5 rounded bg-[var(--surface-card)] border border-[var(--border-subtle)] text-[var(--accent-primary)] font-bold">
                      arXiv:{paper.arxiv_id}
                    </span>
                    <span className="text-[var(--text-400)] flex items-center gap-1">
                      <Clock size={11} />
                      {paper.published_at ? formatRelativeTime(paper.published_at).toUpperCase() : 'RECENT'}
                    </span>
                  </div>

                  <a
                    href={paper.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-xs text-[var(--text-100)] group-hover:text-[var(--accent-primary)] font-semibold leading-relaxed transition-colors flex items-start justify-between gap-2"
                  >
                    <span>{paper.title}</span>
                    <ExternalLink size={13} className="shrink-0 opacity-40 group-hover:opacity-100 transition-opacity mt-0.5" />
                  </a>

                  {paper.authors.length > 0 && (
                    <div className="text-[10px] font-mono text-[var(--text-400)] truncate">
                      AUTHORS: {paper.authors.slice(0, 3).join(', ')}{paper.authors.length > 3 ? ' et al.' : ''}
                    </div>
                  )}

                  <p className="text-[11px] font-sans text-[var(--text-300)] line-clamp-3 leading-relaxed">
                    {paper.summary}
                  </p>

                  <div className="flex items-center justify-end text-[10px] font-mono text-[var(--cobalt-400)] pt-1.5 border-t border-[var(--border-subtle)]">
                    <span className="group-hover:text-[var(--accent-primary)] transition-colors">
                      OPEN ABSTRACT & PDF ↗
                    </span>
                  </div>
                </article>
              ))
            ) : (
              <EmptyState
                icon={<BookOpen size={24} />}
                title="No Preprints Ingested"
                description="Zero recent forensic research papers retrieved from arXiv ATOM queries."
              />
            )}
          </CardContent>
        </Card>

        {/* COLUMN 3: Technical Pulse (HN Algolia) */}
        <Card variant="default" padding="none" className="flex flex-col h-[700px]">
          <div className="p-4 border-b border-[var(--border-subtle)] bg-[var(--surface-card)]">
            <CardHeader className="mb-0">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-full bg-[var(--surface-elevated)] border border-[var(--border-default)] flex items-center justify-center text-[var(--cobalt-400)]">
                  <Radio size={16} />
                </div>
                <div>
                  <CardTitle className="text-sm font-bold uppercase tracking-wider text-[var(--text-100)]">
                    Technical Pulse
                  </CardTitle>
                  <CardDescription className="text-[11px] font-mono text-[var(--text-400)]">
                    HN ALGOLIA // COMMUNITY SIGNALS
                  </CardDescription>
                </div>
              </div>

              {pulseQuery.data?.cached_at ? (
                <Badge variant="neutral" size="sm" className="font-mono text-[10px]">
                  CACHED {formatRelativeTime(pulseQuery.data.cached_at).toUpperCase()}
                </Badge>
              ) : (
                <Badge variant="neutral" size="sm" className="font-mono text-[10px]">
                  {pulseQuery.isLoading ? 'POLLING...' : 'STANDBY'}
                </Badge>
              )}
            </CardHeader>
          </div>

          <CardContent className="p-4 overflow-y-auto flex-1 flex flex-col gap-3">
            {pulseQuery.isLoading ? (
              <div className="flex flex-col gap-3 py-4">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="p-3 bg-[var(--surface-elevated)]/40 rounded border border-[var(--border-subtle)] animate-pulse flex flex-col gap-2">
                    <div className="h-3 bg-[var(--surface-elevated)] rounded w-1/4" />
                    <div className="h-4 bg-[var(--surface-elevated)] rounded w-4/5" />
                    <div className="h-3 bg-[var(--surface-elevated)] rounded w-1/3" />
                  </div>
                ))}
              </div>
            ) : pulseQuery.isError ? (
              <ErrorState
                title="Feed Unavailable"
                message="Unable to connect to Hacker News Algolia proxy. Try refreshing."
                onRetry={() => pulseQuery.refetch()}
              />
            ) : pulseQuery.data?.items && pulseQuery.data.items.length > 0 ? (
              pulseQuery.data.items.map((story) => (
                <article
                  key={story.object_id}
                  className="p-3.5 bg-[var(--surface-elevated)]/40 hover:bg-[var(--surface-elevated)] border border-[var(--border-subtle)] hover:border-[var(--border-strong)] rounded-[var(--radius-sm)] flex flex-col gap-2.5 transition-colors group"
                >
                  <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
                    <span className="px-1.5 py-0.5 rounded bg-[var(--surface-card)] border border-[var(--border-subtle)] text-[var(--lime-400)] font-bold">
                      ▲ {story.points} PTS
                    </span>
                    <span className="text-[var(--text-400)] flex items-center gap-1">
                      <MessageSquare size={11} />
                      {story.num_comments} COMMENTS
                    </span>
                  </div>

                  <a
                    href={story.url || `https://news.ycombinator.com/item?id=${story.object_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-sans text-xs text-[var(--text-100)] group-hover:text-[var(--accent-primary)] font-semibold line-clamp-2 leading-relaxed transition-colors flex items-start justify-between gap-2"
                  >
                    <span>{story.title}</span>
                    <ArrowUpRight size={13} className="shrink-0 opacity-40 group-hover:opacity-100 transition-opacity mt-0.5" />
                  </a>

                  <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-400)] pt-1.5 border-t border-[var(--border-subtle)]">
                    <span>OP: {story.author}</span>
                    <a
                      href={`https://news.ycombinator.com/item?id=${story.object_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[var(--cobalt-400)] group-hover:text-[var(--accent-primary)] transition-colors"
                    >
                      DISCUSSION ↗
                    </a>
                  </div>
                </article>
              ))
            ) : (
              <EmptyState
                icon={<Radio size={24} />}
                title="No Community Signals"
                description="Zero technical deepfake or media synthesis threads retrieved from Hacker News."
              />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
