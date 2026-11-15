import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { ModelCardResponse, ReadyResponse } from '@/types/api';
import { useAuth } from '@/hooks/useAuth';
import { LogOut } from 'lucide-react';

export const StatusStrip: React.FC = () => {
  const { user, logout, isLoggingOut } = useAuth();

  const { data: readyData } = useQuery<ReadyResponse>({
    queryKey: ['system-ready'],
    queryFn: () => api.get<ReadyResponse>('/ready'),
    refetchInterval: 15_000,
  });

  const { data: modelData } = useQuery<ModelCardResponse>({
    queryKey: ['system-model'],
    queryFn: () => api.get<ModelCardResponse>('/model'),
    staleTime: 60_000,
  });

  const isDegraded = readyData?.status === 'degraded';
  const isUntrained = modelData?.model_mode === 'untrained';

  return (
    <header className="h-12 border-b border-[var(--line)] bg-[var(--black-900)] px-4 flex items-center justify-between z-20 shrink-0 select-none">
      <div className="flex items-center gap-3">
        {/* Signal bench live telemetry branding */}
        <div className="flex items-center gap-2">
          <span
            role="status"
            className={`w-2 h-2 rounded-full ${
              isDegraded ? 'bg-[var(--alert-500)] animate-pulse' : 'bg-[var(--lime-400)]'
            }`}
            aria-label={isDegraded ? 'System Degraded' : 'System Operational'}
          />
          <span className="font-display font-black text-sm tracking-widest text-[var(--text-100)]">
            PARALLAX
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-500)] hidden sm:inline">
            // FORENSIC BENCH
          </span>
        </div>

        <div className="h-4 w-px bg-[var(--line)] hidden md:block" />

        {/* Model mode badge */}
        <div className="hidden sm:flex items-center gap-2">
          <span className="font-mono text-[11px] text-[var(--text-300)]">
            {modelData ? `${modelData.model_name} ${modelData.model_version}` : 'AVT²-DWF v1.0.0'}
          </span>
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider font-semibold border ${
              isUntrained
                ? 'bg-amber-950/40 text-amber-300 border-amber-600/40'
                : 'bg-[var(--lime-400)]/10 text-[var(--lime-400)] border-[var(--lime-400)]/30'
            }`}
          >
            {isUntrained ? 'UNTRAINED WEIGHTS' : 'VERIFIED WEIGHTS'}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {user && (
          <div className="flex items-center gap-2.5">
            <span className="font-mono text-xs text-[var(--text-300)] hidden md:inline truncate max-w-[200px]">
              {user.email}
            </span>
            <button
              onClick={() => logout()}
              disabled={isLoggingOut}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono font-medium text-[var(--text-300)] hover:text-white hover:bg-[var(--black-800)] border border-[var(--line)] transition-colors focus-visible:outline-2 focus-visible:outline-[var(--cobalt-500)]"
              title="Exit session"
              aria-label="EXIT session"
            >
              <LogOut size={13} className="text-[var(--text-500)]" />
              <span className="hidden sm:inline">EXIT</span>
            </button>
          </div>
        )}
      </div>
    </header>
  );
};
