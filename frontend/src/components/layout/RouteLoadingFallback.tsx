import React from 'react';

export const RouteLoadingFallback: React.FC = () => (
  <div
    role="status"
    aria-label="Loading workspace module"
    className="w-full h-full min-h-[320px] flex items-center justify-center p-8"
  >
    <div className="flex flex-col items-center gap-3">
      <div className="w-6 h-6 border-2 border-[var(--cobalt-400)]/30 border-t-[var(--accent-primary)] rounded-full animate-spin" />
      <span className="font-mono text-[10px] text-[var(--text-400)] tracking-[0.12em] uppercase">
        Loading module
      </span>
    </div>
  </div>
);
