import React from 'react';
import { Navigate, useLocation, Outlet } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';

export const ProtectedRoute: React.FC = () => {
  const { isAuthenticated, isInitializing } = useAuth();
  const location = useLocation();

  if (isInitializing) {
    return (
      <div className="min-h-screen bg-[var(--black-950)] flex flex-col items-center justify-center gap-3 text-[var(--text-300)] select-none">
        <div className="w-8 h-8 border-2 border-[var(--cobalt-500)] border-t-[var(--lime-400)] rounded-full animate-spin" />
        <div className="font-mono text-xs uppercase tracking-widest text-[var(--text-500)]">
          // SYNCHRONIZING SECURE SESSION
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    const next = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/login?next=${next}`} replace />;
  }

  return <Outlet />;
};
