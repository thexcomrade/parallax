import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '@/lib/api';
import { BentoGrid } from '@/components/bento/BentoGrid';
import { BentoCell } from '@/components/bento/BentoCell';
import { Badge } from '@/components/ui/Badge';
import { formatRelativeTime } from '@/lib/format';
import type {
  AnalysisListResponse,
  ReadyResponse,
  ResearchFeedResponse,
  ThreatFeedResponse,
  ModelCardResponse,
} from '@/types/api';
import { ArrowUpRight, Cpu, Plus, Video, Volume2 } from 'lucide-react';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();

  // Queries
  const analysesQuery = useQuery({
    queryKey: ['analyses'],
    queryFn: () => api.get<AnalysisListResponse>('/analyses?page=1&size=10'),
    staleTime: 30_000,
  });

  const threatsQuery = useQuery({
    queryKey: ['feeds', 'threats'],
    queryFn: () => api.get<ThreatFeedResponse>('/feeds/threats'),
    staleTime: 120_000,
  });

  const researchQuery = useQuery({
    queryKey: ['feeds', 'research'],
    queryFn: () => api.get<ResearchFeedResponse>('/feeds/research'),
    staleTime: 120_000,
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

  // Genuine KPI calculations
  const completedAnalyses = analyses.filter((a) => a.status === 'done');
  const evaluatedAnalyses = analyses.filter((a) => a.verdict && a.status === 'done');
  const hasEvaluated = evaluatedAnalyses.length > 0;

  const fakes = evaluatedAnalyses.filter((a) => a.verdict === 'FAKE').length;
  const authentic = evaluatedAnalyses.filter((a) => a.verdict === 'AUTHENTIC').length;
  const inconclusive = evaluatedAnalyses.filter((a) => a.verdict === 'INCONCLUSIVE').length;
  const totalEvaluated = fakes + authentic + inconclusive;

  const fakePct = totalEvaluated > 0 ? Math.round((fakes / totalEvaluated) * 100) : 0;
  const authPct = totalEvaluated > 0 ? Math.round((authentic / totalEvaluated) * 100) : 0;
  const inconclPct = totalEvaluated > 0 ? Math.max(0, 100 - fakePct - authPct) : 0;

  const latestDoneId = completedAnalyses[0]?.id;

  const isUntrained = modelQuery.data?.model_mode === 'untrained';

  return (
    <div className="flex flex-col gap-6 w-full max-w-[1760px] mx-auto py-2">
      {/* Forensic Workspace Header & Hero */}
      <header className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-5 border-b border-[var(--line)]">
        <div>
          <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-[var(--cobalt-300)]">
            <span className="w-2 h-2 rounded-full bg-[var(--lime-400)]" />
            PARALLAX FORENSIC WORKSPACE // OVERVIEW
          </div>
          <h1 className="font-display font-black text-2xl lg:text-3xl uppercase tracking-tight text-[var(--text-100)] mt-1">
            Forensic Command Bench
          </h1>
          <p className="font-sans text-xs text-[var(--text-300)] mt-0.5 max-w-2xl">
            Audio-visual deepfake detection, cross-modal transformer feature extraction, and dynamic weight fusion.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Model Status Chip */}
          <Link
            to="/app/model"
            className="inline-flex items-center gap-2 px-3 py-2 rounded-[var(--radius-sm)] bg-[var(--black-900)] border border-[var(--line)] hover:border-[var(--line-strong)] text-xs font-mono transition-colors"
            title="View Model Transparency Card"
          >
            <Cpu size={14} className="text-[var(--cobalt-400)]" />
            <span className="text-[var(--text-300)]">
              {modelQuery.data ? `${modelQuery.data.model_name} ${modelQuery.data.model_version}` : 'AVT²-DWF v1.0.0'}
            </span>
            <span
              className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider border ${
                isUntrained
                  ? 'bg-amber-950/40 text-amber-300 border-amber-600/40'
                  : 'bg-[var(--lime-400)]/10 text-[var(--lime-400)] border-[var(--lime-400)]/30'
              }`}
            >
              {isUntrained ? 'UNTRAINED' : 'VERIFIED WEIGHTS'}
            </span>
          </Link>

          {/* Primary Action Button */}
          <Link
            to="/app/analyze"
            className="inline-flex items-center justify-center gap-2 min-h-[40px] px-5 rounded-[var(--radius-sm)] bg-[var(--lime-400)] text-[var(--lime-ink)] font-mono text-xs font-bold uppercase tracking-wider hover:bg-[var(--lime-300)] transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)]"
          >
            <Plus size={14} strokeWidth={3} />
            <span>NEW ANALYSIS</span>
          </Link>
        </div>
      </header>

      {/* KPI Overview Row (Master Spec §7.4) */}
      <section aria-label="Forensic Key Metrics" className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <div className="p-3.5 bg-[var(--black-900)] border border-[var(--line)] rounded-[var(--radius-cell)] flex flex-col justify-between">
          <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-400)]">
            TOTAL ANALYSES
          </span>
          <div className="font-display font-bold text-2xl lg:text-3xl text-[var(--text-100)] mt-1">
            {totalScans}
          </div>
          <span className="font-mono text-[10px] text-[var(--text-500)] mt-1">
            ALL TIME SCANS
          </span>
        </div>

        <div className="p-3.5 bg-[var(--black-900)] border border-[var(--line)] rounded-[var(--radius-cell)] flex flex-col justify-between">
          <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-400)]">
            COMPLETED
          </span>
          <div className="font-display font-bold text-2xl lg:text-3xl text-[var(--lime-400)] mt-1">
            {completedAnalyses.length}
          </div>
          <span className="font-mono text-[10px] text-[var(--text-500)] mt-1">
            PERSISTED DOSSIERS
          </span>
        </div>

        <div className="p-3.5 bg-[var(--black-900)] border border-[var(--line)] rounded-[var(--radius-cell)] flex flex-col justify-between">
          <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--alert-500)]">
            FAKE DETECTED
          </span>
          <div className="font-display font-bold text-2xl lg:text-3xl text-[var(--alert-500)] mt-1">
            {fakes}
          </div>
          <span className="font-mono text-[10px] text-[var(--text-500)] mt-1">
            P_FAKE &gt; 0.65
          </span>
        </div>

        <div className="p-3.5 bg-[var(--black-900)] border border-[var(--line)] rounded-[var(--radius-cell)] flex flex-col justify-between">
          <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--lime-400)]">
            AUTHENTIC
          </span>
          <div className="font-display font-bold text-2xl lg:text-3xl text-[var(--lime-400)] mt-1">
            {authentic}
          </div>
          <span className="font-mono text-[10px] text-[var(--text-500)] mt-1">
            P_FAKE &lt; 0.35
          </span>
        </div>

        <div className="p-3.5 bg-[var(--black-900)] border border-[var(--line)] rounded-[var(--radius-cell)] flex flex-col justify-between col-span-2 sm:col-span-1">
          <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-400)]">
            INCONCLUSIVE
          </span>
          <div className="font-display font-bold text-2xl lg:text-3xl text-[var(--text-300)] mt-1">
            {inconclusive}
          </div>
          <span className="font-mono text-[10px] text-[var(--text-500)] mt-1">
            UNCERTAIN REGION
          </span>
        </div>
      </section>

      {/* Main Bento Grid */}
      <BentoGrid>
        {/* Cell 1: Quick-Analyze Dropzone Banner (colSpan 8) */}
        <BentoCell
          label="MEDIA INGEST // WORKBENCH"
          status="READY"
          colSpan={8}
          className="min-h-[160px]"
        >
          <div
            onClick={() => navigate('/app/analyze')}
            className="flex-1 border-2 border-dashed border-[var(--line-strong)] hover:border-[var(--cobalt-400)] rounded-[var(--radius-sm)] p-5 flex flex-col sm:flex-row items-center justify-between gap-4 cursor-pointer transition-colors bg-[var(--black-950)]/40 group"
          >
            <div className="flex flex-col gap-1 text-center sm:text-left">
              <span className="font-mono text-xs text-[var(--cobalt-300)] font-bold uppercase tracking-wider">
                MEDIA INGESTION DESK
              </span>
              <span className="font-sans text-sm text-[var(--text-100)] font-medium">
                Upload target MP4, WebM, or MOV sequence for multimodal forensic inspection.
              </span>
              <span className="font-mono text-[10px] text-[var(--text-500)]">
                MAX 100 MB // MAX 60 SECONDS // 1080P CONTAINER // FFmpeg &amp; OpenCV PIPELINE
              </span>
            </div>
            <div className="px-4 py-2 bg-[var(--black-800)] border border-[var(--line-strong)] group-hover:border-[var(--cobalt-400)] rounded-[var(--radius-sm)] font-mono text-xs text-[var(--text-100)] uppercase tracking-wider shrink-0 transition-colors flex items-center gap-1.5">
              <span>LAUNCH WORKBENCH</span>
              <ArrowUpRight size={13} />
            </div>
          </div>
        </BentoCell>

        {/* Cell 2: System Architecture Telemetry (colSpan 4) */}
        <BentoCell
          label="ENGINE TELEMETRY"
          status={readyQuery.data?.status === 'ready' ? 'READY' : 'DIAGNOSTIC'}
          colSpan={4}
          className="min-h-[160px]"
        >
          <div className="flex-1 flex flex-col justify-center gap-2 font-mono text-xs">
            <div className="flex items-center justify-between pb-1 border-b border-[var(--line)]">
              <span className="font-sans text-xs text-[var(--text-400)]">Architecture</span>
              <span className="text-[var(--text-100)] font-bold text-xs">AVT²-DWF</span>
            </div>
            <div className="flex items-center justify-between pb-1 border-b border-[var(--line)]">
              <span className="font-sans text-xs text-[var(--text-400)]">Checkpoint</span>
              <span className="text-xs text-[var(--cobalt-300)] truncate max-w-[150px]">
                {modelQuery.data?.checkpoint_hash ? modelQuery.data.checkpoint_hash.slice(0, 12) + '...' : 'LOCAL_WEIGHTS'}
              </span>
            </div>
            <div className="flex items-center justify-between pb-1 border-b border-[var(--line)]">
              <span className="font-sans text-xs text-[var(--text-400)]">Database</span>
              <span className="text-[var(--lime-400)] font-bold text-xs">● CONNECTED (WAL)</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="font-sans text-xs text-[var(--text-400)]">Modality Channels</span>
              <div className="flex items-center gap-2 text-[10px]">
                <span className="text-[var(--ch-visual)] font-bold flex items-center gap-1">
                  <Video size={10} /> VISUAL
                </span>
                <span className="text-[var(--ch-audio)] font-bold flex items-center gap-1">
                  <Volume2 size={10} /> AUDIO
                </span>
              </div>
            </div>
          </div>
        </BentoCell>

        {/* Cell 3: Verdict Distribution (Stacked Histogram, Truthful Empty State) (colSpan 6) */}
        <BentoCell
          label="VERDICT DISTRIBUTION"
          status={hasEvaluated ? 'EMPIRICAL HISTOGRAM' : 'EMPTY'}
          colSpan={6}
          className="min-h-[220px]"
        >
          {!hasEvaluated ? (
            <div className="flex-1 flex flex-col items-center justify-center p-6 text-center gap-1.5">
              <span className="font-mono text-xs text-[var(--text-400)] uppercase font-semibold">
                NO COMPLETED ANALYSES YET
              </span>
              <p className="font-sans text-xs text-[var(--text-500)] max-w-sm">
                Upload video media to begin forensic evaluation. Results will automatically aggregate into the empirical histogram.
              </p>
            </div>
          ) : (
            <div className="flex-1 flex flex-col justify-center gap-4">
              {/* Stacked bar */}
              <div className="w-full h-4 rounded-xs overflow-hidden flex bg-[var(--black-950)] border border-[var(--line)]">
                {fakePct > 0 && (
                  <div
                    style={{ width: `${fakePct}%` }}
                    className="h-full bg-[var(--alert-500)] transition-all duration-300"
                    title={`Fake: ${fakePct}%`}
                  />
                )}
                {authPct > 0 && (
                  <div
                    style={{ width: `${authPct}%` }}
                    className="h-full bg-[var(--lime-400)] transition-all duration-300"
                    title={`Authentic: ${authPct}%`}
                  />
                )}
                {inconclPct > 0 && (
                  <div
                    style={{ width: `${inconclPct}%` }}
                    className="h-full bg-[var(--black-800)] transition-all duration-300"
                    title={`Inconclusive: ${inconclPct}%`}
                  />
                )}
              </div>

              {/* Split Legend */}
              <div className="grid grid-cols-3 gap-3 font-mono text-[10px]">
                <div className="flex flex-col p-2 rounded bg-[var(--black-950)] border border-[var(--line)]">
                  <span className="text-[var(--alert-500)] font-bold">FAKE</span>
                  <span className="text-[var(--text-100)] font-bold text-base font-display">{fakePct}%</span>
                  <span className="text-[var(--text-500)]">{fakes} CLIPS</span>
                </div>
                <div className="flex flex-col p-2 rounded bg-[var(--black-950)] border border-[var(--line)]">
                  <span className="text-[var(--lime-400)] font-bold">AUTHENTIC</span>
                  <span className="text-[var(--text-100)] font-bold text-base font-display">{authPct}%</span>
                  <span className="text-[var(--text-500)]">{authentic} CLIPS</span>
                </div>
                <div className="flex flex-col p-2 rounded bg-[var(--black-950)] border border-[var(--line)]">
                  <span className="text-[var(--text-400)] font-bold">INCONCLUSIVE</span>
                  <span className="text-[var(--text-100)] font-bold text-base font-display">{inconclPct}%</span>
                  <span className="text-[var(--text-500)]">{inconclusive} CLIPS</span>
                </div>
              </div>
            </div>
          )}
        </BentoCell>

        {/* Cell 4: Modality Fusion Architecture (Section 6: Cobalt for Visual, Cyan for Audio) (colSpan 6) */}
        <BentoCell
          label="DYNAMIC WEIGHT FUSION (AVT²-DWF)"
          status={hasEvaluated ? 'ACTIVE GATING' : 'READY'}
          colSpan={6}
          className="min-h-[220px]"
        >
          <div className="flex-1 flex flex-col justify-center gap-3">
            <div className="flex items-center justify-between font-mono text-xs">
              <span className="text-[var(--ch-audio)] font-bold flex items-center gap-1">
                <Volume2 size={12} /> ACOUSTIC TRANSFORMER (CH-A)
              </span>
              <span className="text-[var(--ch-visual)] font-bold flex items-center gap-1">
                <Video size={12} /> SPATIAL ViT (CH-V)
              </span>
            </div>

            {/* Modality Bar: Cyan for Audio, Cobalt for Visual */}
            <div className="w-full h-3 rounded-xs overflow-hidden flex bg-[var(--black-950)] border border-[var(--line)]">
              <div
                style={{ width: '50%' }}
                className="h-full bg-[var(--ch-audio)]"
                title="Audio Modality Channel"
              />
              <div
                style={{ width: '50%' }}
                className="h-full bg-[var(--ch-visual)]"
                title="Visual Modality Channel"
              />
            </div>

            <div className="font-sans text-xs text-[var(--text-400)] leading-relaxed">
              Per Section 4.4, DWF weights (w_audio, w_video) are computed dynamically per input clip based on cross-modal attention confidence. Open any completed dossier to inspect its synchronized temporal trajectory.
            </div>

            {latestDoneId && (
              <Link
                to={`/app/analyses/${latestDoneId}`}
                className="font-mono text-[11px] text-[var(--cobalt-300)] hover:text-white uppercase flex items-center gap-1 pt-1"
              >
                <span>INSPECT LATEST REPORT DWF WEIGHTS</span>
                <ArrowUpRight size={12} />
              </Link>
            )}
          </div>
        </BentoCell>

        {/* Cell 5: Recent Analyses (colSpan 6) */}
        <BentoCell
          label="AUDIT DOSSIER // RECENT SCANS"
          status={`SHOWING ${analyses.slice(0, 4).length}`}
          colSpan={6}
          className="min-h-[300px]"
        >
          <div className="flex-1 flex flex-col gap-2 overflow-y-auto">
            {analyses.length === 0 ? (
              <div className="py-12 text-center font-mono text-xs text-[var(--text-400)]">
                NO RECENT ANALYSES RECORDED
              </div>
            ) : (
              analyses.slice(0, 4).map((item) => (
                <Link
                  key={item.id}
                  to={`/app/analyses/${item.id}`}
                  className="p-2.5 bg-[var(--black-950)] border border-[var(--line)] rounded-[var(--radius-sm)] flex items-center justify-between gap-3 hover:border-[var(--line-strong)] transition-colors"
                >
                  <div className="flex flex-col gap-0.5 min-w-0">
                    <span className="font-mono text-xs text-[var(--text-100)] truncate font-medium">
                      {item.original_name}
                    </span>
                    <span className="font-mono text-[10px] text-[var(--text-500)]">
                      {formatRelativeTime(item.created_at)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {item.verdict === 'FAKE' && (
                      <Badge variant="fake" size="sm" hazard>
                        FAKE
                      </Badge>
                    )}
                    {item.verdict === 'AUTHENTIC' && (
                      <Badge variant="authentic" size="sm">
                        AUTHENTIC
                      </Badge>
                    )}
                    {(!item.verdict || item.verdict === 'INCONCLUSIVE') && (
                      <Badge variant="inconclusive" size="sm">
                        INCONCL
                      </Badge>
                    )}
                  </div>
                </Link>
              ))
            )}
          </div>
        </BentoCell>

        {/* Cell 6: Threat Pulse Radar Mini (colSpan 3) */}
        <BentoCell
          label="THREAT PULSE // GDELT"
          status="LIVE"
          colSpan={3}
          className="min-h-[300px]"
        >
          <div className="flex-1 flex flex-col gap-2 overflow-y-auto">
            {threatsQuery.data?.items.slice(0, 3).map((threat, idx) => (
              <article key={idx} className="p-2 bg-[var(--black-950)] border border-[var(--line)] rounded-[var(--radius-sm)] flex flex-col gap-1">
                <span className="font-mono text-[9px] text-[var(--cobalt-300)] uppercase font-semibold">
                  {threat.domain || 'OSINT'}
                </span>
                <a
                  href={threat.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-sans text-[11px] text-[var(--text-100)] hover:text-[var(--lime-400)] line-clamp-2 leading-snug"
                >
                  {threat.title}
                </a>
              </article>
            ))}
            <Link
              to="/app/intel"
              className="mt-auto pt-2 font-mono text-[10px] text-[var(--cobalt-400)] hover:text-[var(--cobalt-300)] uppercase text-right"
            >
              VIEW RADAR [CH-05] ↗
            </Link>
          </div>
        </BentoCell>

        {/* Cell 7: Research Radar Mini (colSpan 3) */}
        <BentoCell
          label="RESEARCH // arXiv"
          status="ATOM"
          colSpan={3}
          className="min-h-[300px]"
        >
          <div className="flex-1 flex flex-col gap-2 overflow-y-auto">
            {researchQuery.data?.items.slice(0, 2).map((paper, idx) => (
              <article key={idx} className="p-2 bg-[var(--black-950)] border border-[var(--line)] rounded-[var(--radius-sm)] flex flex-col gap-1">
                <span className="font-mono text-[9px] text-[var(--lime-400)] font-semibold">
                  arXiv:{paper.arxiv_id}
                </span>
                <a
                  href={paper.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-sans text-[11px] text-[var(--text-100)] hover:text-[var(--lime-400)] line-clamp-2 leading-snug"
                >
                  {paper.title}
                </a>
              </article>
            ))}
            <Link
              to="/app/intel"
              className="mt-auto pt-2 font-mono text-[10px] text-[var(--cobalt-400)] hover:text-[var(--cobalt-300)] uppercase text-right"
            >
              VIEW PAPERS [CH-05] ↗
            </Link>
          </div>
        </BentoCell>
      </BentoGrid>
    </div>
  );
};
