/**
 * History Page: Paginated forensic log with verdict filtering and direct dossier navigation.
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../../components/ui/Button';
import { api } from '../../lib/api';
import { formatRelativeTime, formatTimecode } from '../../lib/format';
import type { AnalysisListResponse, AnalysisSummary } from '../../types/api';

export const HistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const [data, setData] = useState<AnalysisListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [verdictFilter, setVerdictFilter] = useState<string>('ALL');

  useEffect(() => {
    let mounted = true;
    async function loadHistory() {
      setLoading(true);
      try {
        const res = await api.get<AnalysisListResponse>(`/analyses?page=${page}&size=15`);
        if (mounted) setData(res);
      } catch {
        // Handled silently
      } finally {
        if (mounted) setLoading(false);
      }
    }
    void loadHistory();
    return () => {
      mounted = false;
    };
  }, [page]);

  const items = data?.items ?? [];
  const filteredItems = items.filter((item) => {
    if (verdictFilter === 'ALL') return true;
    return item.verdict === verdictFilter;
  });

  return (
    <div className="flex flex-col gap-6 max-w-6xl mx-auto py-2">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-line gap-4">
        <div>
          <div className="font-mono text-xs uppercase tracking-widest text-cobalt-300">
            LOG ARCHIVE // AUDIT TRAIL
          </div>
          <h1 className="font-display text-2xl font-bold text-text-100 tracking-tight">
            Analysis History
          </h1>
        </div>

        {/* Filter and Export tools */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1 p-1 bg-black-900 border border-line rounded font-mono text-xs">
            {['ALL', 'AUTHENTIC', 'FAKE', 'INCONCLUSIVE'].map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setVerdictFilter(v)}
                className={`px-3 py-1 rounded transition-colors ${
                  verdictFilter === v
                    ? 'bg-cobalt-500 text-white font-bold'
                    : 'text-text-400 hover:text-text-100'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
          {items.length > 0 && (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                const headers = ['id', 'original_name', 'duration_s', 'status', 'verdict', 'p_fake', 'created_at'];
                const rows = items.map((i) => [
                  i.id,
                  `"${i.original_name.replace(/"/g, '""')}"`,
                  i.duration_s.toFixed(2),
                  i.status,
                  i.verdict ?? 'NONE',
                  i.p_fake !== null ? `${(i.p_fake * 100).toFixed(1)}%` : 'N/A',
                  i.created_at,
                ]);
                const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `parallax_audit_history_${new Date().toISOString().slice(0, 10)}.csv`;
                a.click();
                URL.revokeObjectURL(url);
              }}
            >
              EXPORT CSV
            </Button>
          )}
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center p-12 min-h-[300px] font-mono text-xs text-text-400">
          LOADING ARCHIVE RECORDS…
        </div>
      ) : filteredItems.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-12 min-h-[300px] bg-black-900 border border-line rounded-lg text-center gap-3">
          <div className="font-mono text-xs text-text-400">NO ANALYSES RECORDED IN THIS VIEW</div>
          <Button variant="secondary" size="sm" onClick={() => navigate('/app/analyze')}>
            START NEW SCAN
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {filteredItems.map((item: AnalysisSummary) => {
            const isFake = item.verdict === 'FAKE';
            const isAuthentic = item.verdict === 'AUTHENTIC';
            let badgeClass = 'text-text-400 border-line bg-black-800';
            if (isFake) badgeClass = 'text-alert-500 border-alert-500/50 bg-alert-500/10 font-bold';
            if (isAuthentic) badgeClass = 'text-lime-400 border-lime-400/50 bg-lime-400/10 font-bold';

            return (
              <div
                key={item.id}
                role="button"
                tabIndex={0}
                onClick={() => navigate(`/app/analyses/${item.id}`)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') navigate(`/app/analyses/${item.id}`);
                }}
                className="flex items-center justify-between p-4 bg-black-900 hover:bg-black-800 border border-line rounded-lg transition-colors cursor-pointer group outline-none focus-visible:ring-2 focus-visible:ring-cobalt-500"
              >
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded border border-line bg-black-950 flex items-center justify-center font-mono text-xs text-cobalt-300">
                    MP4
                  </div>
                  <div>
                    <div className="font-medium text-text-100 group-hover:text-cobalt-300 transition-colors">
                      {item.original_name}
                    </div>
                    <div className="font-mono text-[11px] text-text-400 flex items-center gap-3 mt-0.5">
                      <span>ID: {item.id.slice(0, 8)}</span>
                      <span>DUR: {formatTimecode(item.duration_s)}</span>
                      <span>{formatRelativeTime(item.created_at)}</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  {item.p_fake !== null && (
                    <div className="font-mono text-xs text-right hidden sm:block">
                      <div className="text-text-400 text-[10px]">P(SYNTHETIC)</div>
                      <div className="text-text-200 font-bold">{(item.p_fake * 100).toFixed(0)}%</div>
                    </div>
                  )}

                  <span
                    className={`font-mono text-xs px-2.5 py-1 rounded border uppercase tracking-wider ${badgeClass}`}
                  >
                    {item.verdict ?? item.status}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Pagination controls */}
      {data && data.total > data.size && (
        <div className="flex items-center justify-between pt-4 border-t border-line font-mono text-xs text-text-400">
          <span>
            SHOWING {filteredItems.length} OF {data.total} DOSSIERS
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              PREV
            </Button>
            <span>PAGE {page}</span>
            <Button
              variant="ghost"
              size="sm"
              disabled={page * data.size >= data.total}
              onClick={() => setPage((p) => p + 1)}
            >
              NEXT
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
