/**
 * History Page: Paginated forensic log with search, verdict filtering, direct dossier navigation, and deletion.
 */

import React, { useState, useEffect, useCallback } from 'react';
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
  const [searchQuery, setSearchQuery] = useState('');
  const [verdictFilter, setVerdictFilter] = useState<string>('ALL');
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const loadHistory = useCallback(async (p: number, q: string, v: string) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(p),
        size: '15',
      });
      if (q.trim()) params.set('q', q.trim());
      if (v !== 'ALL') params.set('verdict', v);

      const res = await api.get<AnalysisListResponse>(`/analyses?${params.toString()}`);
      setData(res);
    } catch {
      // Handled silently
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
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

  const items = data?.items ?? [];

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
                onClick={() => handleFilterChange(v)}
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

      {/* Search Input Bar */}
      <div className="flex items-center justify-between gap-4">
        <div className="relative flex-1 max-w-md">
          <input
            type="text"
            placeholder="Search archive by filename or ID…"
            value={searchQuery}
            onChange={handleSearchChange}
            className="w-full px-3 py-2 bg-black-900 border border-line rounded font-mono text-xs text-text-100 placeholder:text-text-400 focus:outline-none focus:border-cobalt-500"
            aria-label="Search archive by filename or ID"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => {
                setSearchQuery('');
                setPage(1);
              }}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-text-400 hover:text-text-100 text-xs font-mono"
              aria-label="Clear search"
            >
              CLEAR
            </button>
          )}
        </div>

        <div className="font-mono text-xs text-text-400 hidden sm:block">
          {data ? `TOTAL: ${data.total} DOSSIERS` : 'LOADING…'}
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center p-12 min-h-[300px] font-mono text-xs text-text-400">
          LOADING ARCHIVE RECORDS…
        </div>
      ) : items.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-12 min-h-[300px] bg-black-900 border border-line rounded-lg text-center gap-3">
          <div className="font-mono text-xs text-text-400">
            {searchQuery || verdictFilter !== 'ALL'
              ? 'NO ANALYSES MATCH CURRENT QUERY / FILTER'
              : 'NO ANALYSES RECORDED IN ARCHIVE'}
          </div>
          {searchQuery || verdictFilter !== 'ALL' ? (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                setSearchQuery('');
                setVerdictFilter('ALL');
                setPage(1);
              }}
            >
              RESET FILTERS
            </Button>
          ) : (
            <Button variant="secondary" size="sm" onClick={() => navigate('/app/analyze')}>
              START NEW SCAN
            </Button>
          )}
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {items.map((item: AnalysisSummary) => {
            const isFake = item.verdict === 'FAKE';
            const isAuthentic = item.verdict === 'AUTHENTIC';
            const isFailed = item.status === 'failed';
            let badgeClass = 'text-text-400 border-line bg-black-800';
            if (isFake) badgeClass = 'text-alert-500 border-alert-500/50 bg-alert-500/10 font-bold';
            if (isAuthentic) badgeClass = 'text-lime-400 border-lime-400/50 bg-lime-400/10 font-bold';
            if (isFailed) badgeClass = 'text-alert-500 border-alert-500 bg-black-950 font-bold';

            return (
              <div
                key={item.id}
                className="flex items-center justify-between p-4 bg-black-900 hover:bg-black-800 border border-line rounded-lg transition-colors group"
              >
                <button
                  type="button"
                  onClick={() => navigate(`/app/analyses/${item.id}`)}
                  className="flex items-center gap-4 text-left flex-1 min-w-0 outline-none focus-visible:ring-2 focus-visible:ring-cobalt-500 rounded p-1"
                >
                  <div className="w-10 h-10 rounded border border-line bg-black-950 flex items-center justify-center font-mono text-xs text-cobalt-300 shrink-0">
                    MP4
                  </div>
                  <div className="min-w-0">
                    <div className="font-medium text-text-100 group-hover:text-cobalt-300 transition-colors truncate">
                      {item.original_name}
                    </div>
                    <div className="font-mono text-[11px] text-text-400 flex items-center gap-3 mt-0.5">
                      <span>ID: {item.id.slice(0, 8)}</span>
                      <span>DUR: {formatTimecode(item.duration_s)}</span>
                      <span>{formatRelativeTime(item.created_at)}</span>
                    </div>
                  </div>
                </button>

                <div className="flex items-center gap-4 shrink-0 ml-4">
                  {item.p_fake !== null && (
                    <div className="font-mono text-xs text-right hidden sm:block">
                      <div className="text-text-400 text-[10px]">P(SYNTHETIC)</div>
                      <div className="text-text-200 font-bold">{(item.p_fake * 100).toFixed(0)}%</div>
                    </div>
                  )}

                  <span
                    className={`font-mono text-xs px-2.5 py-1 rounded border uppercase tracking-wider ${badgeClass}`}
                  >
                    {isFailed ? 'FAILED' : item.verdict ?? item.status}
                  </span>

                  {/* Per-item Delete Action */}
                  <button
                    type="button"
                    onClick={(e) => handleDeleteItem(e, item)}
                    disabled={deletingId === item.id}
                    className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus:opacity-100 px-2.5 py-1 text-[11px] font-mono text-alert-500 hover:bg-alert-500/10 border border-line hover:border-alert-500/40 rounded transition-all outline-none focus-visible:ring-2 focus-visible:ring-alert-500"
                    aria-label={`Delete dossier ${item.original_name}`}
                  >
                    {deletingId === item.id ? 'PURGING…' : 'DELETE'}
                  </button>
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
            SHOWING {items.length} OF {data.total} DOSSIERS
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
